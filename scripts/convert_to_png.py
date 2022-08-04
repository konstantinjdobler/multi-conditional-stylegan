import os
import numpy as np

import click
import PIL.Image
from multiprocessing import Pool
from itertools import repeat
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = None


def convert_to_png(filename: str, source: str, dest: str):
    try:
        filepath = os.path.join(source, filename)
        img = PIL.Image.open(filepath)
        img = img.convert("RGB")
        base = os.path.splitext(filename)[0]
        img.save(f"{dest}/{base}.png", format="png", compress_level=1)
    except Exception as e:
        print(e)
        print(filename)

@click.command()
@click.pass_context
@click.option('--source', help='Directory of images', required=True, metavar='PATH')
@click.option('--dest', help='Directory of images', required=True, metavar='PATH')
def convert_images(
    ctx: click.Context,
    source: str,
    dest: str
):
    os.makedirs(dest, exist_ok=True)
    
    with Pool(os.cpu_count()) as p:
        p.starmap(convert_to_png, zip(os.listdir(source), repeat(source), repeat(dest)))


if __name__ == "__main__":
    convert_images() # pylint: disable=no-value-for-parameter
