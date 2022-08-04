import functools
import torch
import dnnlib
import wandb
from genart.wandb_helpers import WANDB_PROJECT_NAME, WANDB_TEAM_NAME
import legacy
from training.networks import Generator, MappingNetwork


def load_model(network_pkl, use_wandb, use_source_forwards=False):
    if use_wandb:
        print("Downloading pkl from wandb")
        model_file = wandb.restore(
            network_pkl or "network-snapshot-latest.pkl",
            run_path=f"{WANDB_TEAM_NAME}/{WANDB_PROJECT_NAME}/{use_wandb}",
            root=f"./models/{use_wandb}",
        )
        network_pkl = model_file.name

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert network_pkl
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    if use_source_forwards:
        # update pickled forward functions with new versions from current source code
        # enables new functionaloty for older checkpoints
        import types

        new_generator_forward = types.MethodType(Generator.forward, G)
        G.forward = new_generator_forward
        new_mapping_forward = types.MethodType(MappingNetwork.forward, G.mapping)
        G.mapping.forward = new_mapping_forward

    if not torch.cuda.is_available():
        # https://github.com/NVlabs/stylegan2-ada-pytorch/pull/121/files
        G.forward = functools.partial(G.forward, force_fp32=True)
        G.synthesis.forward = functools.partial(G.synthesis.forward, force_fp32=True)
    return G, device
