import click
import os
import glob
import multiprocessing as mp
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import json
from tqdm import tqdm


def out_path(path, outdir):
    filename = path.split("/")[-1].split(".")[0]
    return f"{outdir}/{filename}.json"


def get_color(paths, n_colors, outdir, idx):
    for path in tqdm(paths, position=idx):
        img = Image.open(path)
        img = np.array(img, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        w, h, d = original_shape = tuple(img.shape)
        assert d == 3
        image_array = np.reshape(img, (w * h, d))

        np.random.shuffle(image_array)
        image_array_sample = image_array[:1000]
        kmeans = KMeans(n_clusters=n_colors).fit(image_array_sample)

        centers = kmeans.cluster_centers_
        labels = kmeans.predict(image_array)
        counter = Counter(labels)
        colors = [centers[i].tolist() for i in counter.keys()]
        
        with open(out_path(path, outdir), "w+") as f:
            json.dump({path: colors}, f)


def split_list_into_chunks(seq, num_chunks):
    return list((seq[i::num_chunks] for i in range(num_chunks)))

@click.command()
@click.pass_context
@click.option('--dataset', type=str)
@click.option('--outdir', type=str)
@click.option('--num-processes', '-n', type=int, default=8)
@click.option('--num-colors', type=int, default=3)
def color(ctx, dataset, outdir, num_processes, num_colors):
    paths = glob.glob(f"{dataset}/*/*.png")
    os.makedirs(outdir, exist_ok=True)

    filtered_paths = [p for p in paths if not os.path.isfile(out_path(p, outdir))]
    print(f"Reuse: Filtered from {len(paths)} to {len(filtered_paths)}")
    paths = filtered_paths

    if len(paths) == 0:
        print("Every possible img retrieved")
        return

    split_paths = split_list_into_chunks(paths, num_processes)

    with mp.Pool(num_processes) as pool:
        pool.starmap(get_color, zip(split_paths, [num_colors]*num_processes, [outdir]*num_processes, list(range(num_processes))))


if __name__ == "__main__":
    color()
