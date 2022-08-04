# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

from copy import Error
import os
import re
from typing import List, Optional

import click
from torch._C import device
import dnnlib
import numpy as np
import PIL.Image
from time import time
import torch
from training.training_loop import save_image_grid

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--classes', type=num_range, help='Two class labels to interpolate between', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--images', 'num_images', type=int, help='Number of different images to generate (height)', default=4)
@click.option('--interpolations', 'num_interpolations', type=int, help='Number of interpolation per image (width)', default=6)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    classes: List[int],
    projected_w: Optional[str],
    num_images: int,
    num_interpolations: int
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if len(classes) != 2:
        raise Error('Number of classes to interpolate need to be exactly 2')
    class_idx_1, class_idx_2 = tuple(classes)

    os.makedirs(outdir, exist_ok=True)

    gh, gw = num_images, num_interpolations
    grid_size = (gw, gh)

    labels = torch.zeros([gw, G.c_dim], device=device)
    for i in range(gw):
        labels[i, class_idx_1] = (gw - 1) - i
        labels[i, class_idx_2] = i
        labels[i] /= labels[i].sum()

    for seed_idx, seed in enumerate(seeds):
        images = np.array([])
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z_all = np.random.RandomState(seed).randn(gh, G.z_dim)

        for i in range(gh):
            z = np.repeat([z_all[i]], gw, axis=0)
            z = torch.from_numpy(z).to(device)
            
            interpolated_images = G(z, labels, truncation_psi=truncation_psi, noise_mode=noise_mode)
            if i == 0:
                images = interpolated_images.cpu().numpy()
            else:
                images = np.concatenate((images, interpolated_images.cpu().numpy()), axis=0)
        save_image_grid(images, os.path.join(outdir, f'interpolate-{seed:04d}-{class_idx_1}-{class_idx_2}-{int(time())}.png'), drange=[-1,1], grid_size=grid_size)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
