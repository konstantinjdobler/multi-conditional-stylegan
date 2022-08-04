# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import setuptools
from torch import tensor
from training.dataset import ImageFolderDataset
import functools
import os
import pathlib
from typing import Any, Dict, Iterable, List, Optional

import wandb
import click
import dnnlib
import json
import numpy as np
import PIL.Image
from time import time
import torch
import torchvision.models as models
from pathlib import Path

from training.networks import Generator
from generate_grid import condition_list, num_range, ModifiedPath
from training.training_loop import save_image_grid
from genart.wandb_helpers import WANDB_PROJECT_NAME, WANDB_TEAM_NAME
from img2vec_pytorch import Img2Vec

import legacy

#----------------------------------------------------------------------------

def mapping(G, seed, condition, num, device):
    z = np.random.RandomState(seed).randn(num, G.z_dim)
    z = torch.from_numpy(z).to(device)

    if len(condition.shape) == 1:
        condition = np.vstack([condition] * num)
    c = torch.from_numpy(condition).to(device)

    w = G.mapping(z, c)  # [N, L, C]
    w = w.cpu().numpy().astype(np.float32)
    return w

def create_style_mix(base_w, mix_w, layer, range_mix=True):
    assert base_w.shape == mix_w.shape
    w = base_w.copy()  # [L, C]
    if layer == len(w):
        # No layer to replace
        return w
    
    if range_mix:
        w[layer:,:] = mix_w[0]  # [L, C]
    else:
        w[layer,:] = mix_w[0]  # [L, C]
    return w

# spherical linear interpolation (slerp)


def slerp(val, low, high):
    low_norm = low/torch.norm(low, keepdim=True)
    high_norm = high/torch.norm(high, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

# uniform interpolation between two points in latent space


def spherical_interpolate_points(p1, p2, n_steps=10):
	# interpolate ratios between the points
	ratios = np.linspace(0, 1, num=n_steps)
	# linear interpolate vectors
	vectors = list()
	for ratio in ratios:
		v = slerp(ratio, torch.from_numpy(p1), torch.from_numpy(p2))
		vectors.append(v.numpy())
	return np.asarray(vectors)
#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename')
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--ipc', 'images_per_class', type=int, help='Number of images per class', default=8)
@click.option('--massive-multi-domain-conditions', '--mmdc', help="Specify the directory where the MMDC annotations (generated by the `create_label_json.py` script) are stored."
              "If this option is not provided, MMDC is not used. If provided without a value, a default value is used.", is_flag=False, flag_value="./annotations/emotions-artist-style-genre", type=ModifiedPath(file_okay=False, exists=True, path_type=pathlib.Path), default=False, is_eager=True)
@click.option('--conditions', type=condition_list, help='Class conditions in the form of <condition1>=<value1.1>,<value1.2>;<condition2>=<value2>')
@click.option('--use-wandb', '--wandb', help='run name in our wandb project')
@click.option('--avg-samples', help='How many images to draw the avg w-vectors from', default=10000)
@click.option('--interpolations', 'num_interpolations', help='How many images to create from the interpolation', default=10)
@click.option('--w-interpolation', help='Create interpolation in w space', is_flag=True)
@click.option('--style-mixing', help='Create style mix', is_flag=True)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: List[int],
    noise_mode: str,
    projected_w: Optional[str],
    outdir: str,
    images_per_class: int,
    massive_multi_domain_conditions: ModifiedPath,
    conditions: List[List[Any]],
    use_wandb: str,
    avg_samples: int,
    num_interpolations: int,
    w_interpolation: bool,
    style_mixing: bool,
):
    if w_interpolation and style_mixing:
        print("--w-interpolation and --style-mixing cannot be turned on at the same time")
        return

    if use_wandb:
        print("Downloading pkl from wandb")
        model_file = wandb.restore(
            network_pkl or "network-snapshot-latest.pkl", run_path=f"{WANDB_TEAM_NAME}{WANDB_PROJECT_NAME}/{use_wandb}", root=f"./models/{use_wandb}")
        network_pkl = model_file.name
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    assert network_pkl
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G: Generator = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if not torch.cuda.is_available():
        # https://github.com/NVlabs/stylegan2-ada-pytorch/pull/121/files
        G.forward = functools.partial(G.forward, force_fp32=True)
        G.synthesis.forward = functools.partial(G.synthesis.forward, force_fp32=True)

    if style_mixing:
        num_interpolations = G.num_ws + 1
        print(f"Using style-mixing with {num_interpolations} interpolations")
    
    # Parse c vector
    assert conditions is not None
    assert massive_multi_domain_conditions is not None
    gh = len(conditions)
    with open(massive_multi_domain_conditions / "prepared_dataset.json", 'r') as f:
        label_shape = json.load(f)["shapes"]
    cs = ImageFolderDataset.transform_multidomain_conditions(conditions, label_shape)
    
    # Adjust number of seeds and conditions
    if seeds is None:
        seeds = [0]
    if len(cs) == 1 and len(seeds) > 1:
        tile_shape = (len(seeds), 1)
        cs = np.tile(cs, tile_shape) # [N, C]
    if len(seeds) == 1 and len(conditions) > 1:
        seeds = seeds * len(conditions)

    assert len(cs) == len(seeds)
    w_avgs = []
    # Create average w vectors for conditions
    for seed, c in zip(seeds, cs):
        ws = mapping(G, seed+100, c, avg_samples, device)  # [N, L, C]
        w_avg = np.mean(ws, axis=0)  # [L, C]
        w_avgs.append(w_avg)

    assert num_interpolations >= 2
    if projected_w:
        # Use previously computed projected w
        print(f"Load w from {projected_w}")
        loaded_w = np.load(projected_w)['w']

    # Create interpolated w vectors
    all_ws = []
    for seed, c, w_avg in zip(seeds, cs, w_avgs):
        ws = loaded_w if projected_w else mapping(G, seed, c, 1, device)  # [1, L, C]
        w = np.squeeze(ws, axis=0)  # [L, C]

        if w_interpolation:
            # w interpolation
            # original_ws = np.linspace(w, G.mapping.w_avg.numpy(), num_interpolations)
            original_ws = spherical_interpolate_points(
                w, w_avg, num_interpolations)
            all_ws.append(original_ws)
            # ts = torch.from_numpy(np.linspace(0, 1, num_interpolations))
            # _w, _w_avg = torch.from_numpy(w), torch.from_numpy(w_avg)
            # ws = [slerp(_w, _w_avg, t).numpy() for t in ts]
            # ws = np.array(ws)
            # print(ws.shape)
            # all_ws.append(ws)
        else:
            # Style Mixing
            ws = np.array([create_style_mix(w, w_avg, len(w)-i) for i in range(num_interpolations)])
            all_ws.append(ws)
            # single_replace_ws = np.array([create_style_mix(w, w_avg, len(w)-i, False) for i in range(num_interpolations)])
            # all_ws.append(single_replace_ws)
        

    os.makedirs(outdir, exist_ok=True)
    timestamp = int(time())
    imgs_outdir = Path(outdir) / str(timestamp)
    os.makedirs(imgs_outdir)
    print(f"Images are saved in {imgs_outdir}")

    if torch.cuda.is_available():
        # Load VGG16 feature detector.
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)

        img_to_vec = lambda x: vgg16(x, resize_images=False, return_lpips=True)
    else:
        img_to_vec = Img2Vec()

    # Create images
    gw, gh = num_interpolations, len(all_ws)
    grid_imgs = []
    for ws in all_ws:
        ws = torch.from_numpy(ws).to(device)
        imgs = G.synthesis(ws, noise_mode=noise_mode)
        imgs = imgs.cpu().numpy()

        pils = [save_image_grid([img], None, drange=[-1,1], grid_size=(1,1)) for img in imgs]
        embs = img_to_vec.get_vec(pils, tensor=True).squeeze()
        original_sim = torch.nn.functional.cosine_similarity(embs[0], embs[-1], dim=0)
        print(f"Similarity between first and last image: {original_sim.item()}")
        
        sim = torch.nn.functional.cosine_similarity(embs[:-1], embs[1:])
        print(f"Similarities: {sim.tolist()}")
        k = min(3, len(sim))
        top = sim.topk(k, largest=False)
        print(top)
        for i, index in enumerate(top.indices):
            save_image_grid(imgs[index:index+2], os.path.join(imgs_outdir, f'cmp-seed{seed:04d}-{i}-{index}.jpg'), drange=[-1,1], grid_size=(2,1))

        grid_imgs.append(imgs)
    grid_imgs = np.array(grid_imgs)
    # save_image_grid(np.concatenate(grid_imgs[:,:6], axis=0), os.path.join(imgs_outdir, f'grid-seed{seed:04d}-1.jpg'), drange=[-1,1], grid_size=(6,1))
    # save_image_grid(np.concatenate(grid_imgs[:,6:12], axis=0), os.path.join(imgs_outdir, f'grid-seed{seed:04d}-2.jpg'), drange=[-1,1], grid_size=(6,1))
    # save_image_grid(np.concatenate(grid_imgs[:,12:], axis=0), os.path.join(imgs_outdir, f'grid-seed{seed:04d}-3.jpg'), drange=[-1,1], grid_size=(5,1))
    grid_imgs = np.concatenate(grid_imgs, axis=0)

    save_image_grid(grid_imgs, os.path.join(imgs_outdir, f'grid-seed{seed:04d}.jpg'), drange=[-1,1], grid_size=(gw,gh))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------