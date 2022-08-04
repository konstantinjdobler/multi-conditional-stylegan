# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

from training.dataset import ImageFolderDataset
import os
import pathlib
import re
from typing import Any, Dict, Iterable, List, Optional

import wandb
import click
import dnnlib
import itertools
import json
import numpy as np
import PIL.Image
from time import time
import torch
from sentence_transformers import SentenceTransformer, util
from training.training_loop import save_image_grid

from utils import load_model

# ----------------------------------------------------------------------------
class ModifiedPath(click.Path):
    def convert(self, value, param, ctx):
        if isinstance(value, bool):
            return value
        return super().convert(value, param, ctx)

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    try:
        return [int(x) for x in vals]
    except:
        return []


def condition_list(s: str):
    '''expect string in form of <condition1>=<value1.1>,<value1.2>;<condition2>=<value2> ... OR legacy num_range'''
    legacy_num_range = num_range(s)
    if len(legacy_num_range) > 0:
        return legacy_num_range

    annotation_dir = click.get_current_context(
    ).params["massive_multi_domain_conditions"]
    no_zero_vector_for_unknown = click.get_current_context().params.get("no_zero_unknown")
    zero_vector_for_unknown = not no_zero_vector_for_unknown
    print('Using zero vector for unknown' if zero_vector_for_unknown else 'Using special unknown token for unknown')

    if not isinstance(annotation_dir, pathlib.Path):
        print("Error: You need to specify the annotation directory with --massive-multi-domain-conditions / --mmdc ")
        raise Exception
    return _condition_list(s, annotation_dir, zero_vector_for_unknown)


def parse_conditions(s: str):
    '''expect <condition1>=<value1.1>,<value1.2>;<condition2>=<value2>'''
    def _parse_condition(s: str):
        '''expects <condition1>=<value1.1>,<value1.2>'''
        condition, _, raw_values = s.partition("=")
        values = raw_values.split(",")
        return condition, values

    conditions = s.split(';')
    condition_dict = dict(map(_parse_condition, conditions))
    return condition_dict


def get_fuzzy_match(s: str, possible_matches: Iterable[str], cond_name: str):
    if s == "unknown":
        return "AA__UNKNOWN__"
    if s in possible_matches:
        return s

    fuzzy_matches = [m for m in possible_matches if s.lower() in m.lower()]
    if len(fuzzy_matches) != 1:
        click.get_current_context().fail(
            f"Cannot finding unique match for specified condition {cond_name}={s}; Matches found: {fuzzy_matches}. Try an exact spelling if you have multiple possible matches.")
    fuzzy_match = fuzzy_matches[0]
    return fuzzy_match


def _condition_list(s: str, annotation_dir: pathlib.Path, zero_vector_for_unknown=False):
    with open(annotation_dir / "prepared_dataset.json") as f:
        condition_order = json.load(f)["condition_order"]
    print("Condition Order:", condition_order)

    condition_dict = parse_conditions(s)
    filtered_condition_dict = {
        c: condition_dict[c] for c in condition_order if c in condition_dict}

    print("Conditions:", filtered_condition_dict)
    original_dict = filtered_condition_dict.copy()

    for cond_name, values in filtered_condition_dict.items():
        index_filepath = f'{str(annotation_dir / cond_name)}_idx.json'
        occurrence_filepath = f'{str(annotation_dir / cond_name)}_occurrence.json'
        if os.path.isfile(index_filepath):
            print(f"Converting condition '{cond_name}' to index")
            with open(index_filepath) as f:
                index = json.load(f)
                matches = list(
                    map(lambda x: get_fuzzy_match(x, index, cond_name), values))
                original_dict[cond_name] = matches
                filtered_condition_dict[cond_name] = list(
                    map(lambda x: [] if zero_vector_for_unknown and x == "AA__UNKNOWN__" else index[x], matches))
        elif os.path.isfile(occurrence_filepath):
            print(f"Converting condition '{cond_name}' to BERT embedding")
            assert all([isinstance(v, str) for v in values])
            with open(occurrence_filepath) as f:
                text_to_occurrence = json.load(f)
                model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')
                texts = np.array(list(text_to_occurrence.keys()))

                embedded_texts = model.encode(texts)
                embedded_values = model.encode(values)

                cosine_scores = util.pytorch_cos_sim(
                    embedded_values, embedded_texts)
                all_similarities, all_top_indices = cosine_scores.topk(
                    3, dim=1)

                print(
                    "Similar Texts from the original dataset (similarity, occurrences):")
                for text, similarities, top_indices in zip(values, all_similarities, all_top_indices):
                    similar_text = texts[top_indices]
                    out = map(lambda x: f"{x[0]} ({str(np.round(x[1].item(), 4))}, {text_to_occurrence[x[0]]}x)", zip(
                        similar_text, similarities))
                    print(f"{text}:", ", ".join(out))

    final_conditions = {c: [[]] for c in condition_order}
    if filtered_condition_dict.get('keywords'):
        filtered_condition_dict['keywords'] = [
            [] if kw == "unknown" else kw for kw in filtered_condition_dict['keywords']]

    final_conditions.update(filtered_condition_dict)

    print(
        f"Using these conditions: {original_dict}\n Mapped to: {final_conditions}")
    print()
    cond_combinations = map(
        list, itertools.product(*final_conditions.values()))
    cond_combinations = list(cond_combinations)

    pretty_combinations = map(
        lambda x: [(x[0], v) for v in x[1]], original_dict.items())
    pretty_combinations = map(lambda c: ", ".join(
        [": ".join(x) for x in c]), itertools.product(*pretty_combinations))
    for i, c in enumerate(pretty_combinations):
        print(f"Row {i + 1}: {c}")
    print()

    return cond_combinations


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename')
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--conditional-truncation', '--ct', is_flag=True, help='Put seeds into same grid')

@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--ipc', 'images_per_class', type=int, help='Number of images per class / row', default=8)
@click.option('--umr', 'unconditional_multirow', is_flag=True, help='Use this if you do not specifiy conditions and want multiple seeds in the same grid.')
@click.option('--massive-multi-domain-conditions', '--mmdc', help="Specify the directory where the MMDC annotations (generated by the `create_label_json.py` script) are stored."
              "If this option is not provided, MMDC is not used. If provided without a value, a default value is used.", is_flag=False, flag_value="./annotations/emotions-artist-style-genre", type=ModifiedPath(file_okay=False, exists=True, path_type=pathlib.Path), default=False, is_eager=True)
@click.option('--conditions', type=condition_list, help='Class conditions in the form of <condition1>=<value1.1>,<value1.2>;<condition2>=<value2>')
@click.option('--use-wandb', '--wandb', help='Use .pkl from W&B, supply run name here.')
@click.option('--no-zero-unknown', '--nzu', help='Use a special trained unkown token instead of zero-vector for wildcard generation', is_eager=True, is_flag=True)
def generate_images(
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    conditional_truncation: bool,
    noise_mode: str,
    outdir: str,
    images_per_class: int,
    unconditional_multirow: bool,
    massive_multi_domain_conditions: ModifiedPath,
    conditions: List[List[Any]],
    use_wandb: str,
    no_zero_unknown: bool,
):
    """
    Example command. `unknown` specifies wildcard generation. 
    python generate_grid.py --conditions "style=unknown;painter=Monet;genre=landscape;keywords=trees,ocean" --seeds 1 --mmdc ./annotations/artist-style-genre-keywords 
    """
    G, device = load_model(network_pkl, use_wandb, use_source_forwards=True)
    print("Loaded model on device", device)

    os.makedirs(outdir, exist_ok=True)

    gh, gw = 1, images_per_class
    if G.c_dim > 0:
        assert conditions != None
        gh = len(conditions)
        if massive_multi_domain_conditions:
            with open(massive_multi_domain_conditions / "prepared_dataset.json", 'r') as f:
                label_shape = json.loads(f.read())["shapes"]
            embedded_conditions = ImageFolderDataset.transform_multidomain_conditions(
                conditions, label_shape)
        else:
            one_hot = torch.zeros([gh, G.c_dim], device=device)
            for o, l in zip(one_hot, conditions):
                o[l] = 1
            embedded_conditions = one_hot

    grid_size = (gw, gh)
    if unconditional_multirow:
        grid_size = (gw, len(seeds))
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' %
              (seed, seed_idx+1, len(seeds)))

        z = torch.from_numpy(np.random.RandomState(
            seed).randn(gw, G.z_dim)).to(device)
        for i in range(gh):
            c = None
            if G.c_dim > 0:
                c = np.vstack([embedded_conditions[i]] * gw)
                c = torch.from_numpy(c).to(device)
            class_images = G(
                z, c, truncation_psi=truncation_psi, noise_mode=noise_mode, conditional_truncation=conditional_truncation)
            
            if seed_idx == 0 and unconditional_multirow:
                images = class_images.cpu().numpy()
            elif i == 0 and not unconditional_multirow:
                images = class_images.cpu().numpy()
            else:
                images = np.concatenate(
                    (images, class_images.cpu().numpy()), axis=0)
        if not unconditional_multirow:
            save_image_grid(images, os.path.join(
                outdir, f'grid-seed{seed:05d}-{int(time())}.png'), drange=[-1, 1], grid_size=grid_size)
    if unconditional_multirow:
        save_image_grid(images, os.path.join(
            outdir, f'grid-seed{"_".join([str(see) for see in seeds])}-{int(time())}.png'), drange=[-1, 1], grid_size=grid_size)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
