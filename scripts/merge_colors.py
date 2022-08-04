import click
import os
import glob
import json
from tqdm import tqdm


def out_path(path, outdir):
    filename = path.split("/")[-1].split(".")[0]
    return f"{outdir}/{filename}.json"

@click.command()
@click.pass_context
@click.option('--dataset', type=str)
@click.option('--outdir', type=str)
def merge(ctx, dataset, outdir):
    paths = glob.glob(f"{dataset}/*.json")
    os.makedirs(outdir, exist_ok=True)

    labels = []
    for path in tqdm(paths):
        with open(path, "r") as f:
            try:
                data = json.load(f)
            except:
                os.remove(path)
                print("Error in:", path)
            p, c = next(iter(data.items()))
            p = "/".join(p.split("/")[-2:])
            labels.append([p, c])
    
    result = {"condition_order": ["color1", "color2", "color3"], "shapes": [3, 3, 3], "labels": labels}
    with open(f"{outdir}/prepared_dataset.json", "w+") as f:
        json.dump(result, f)


if __name__ == "__main__":
    merge()
