from functools import partial
import requests
import shutil
import os
from multiprocessing import Pool
import pandas as pd
from pathlib import Path
import click


def _get_csv_entries(csv_file="./artemis_dataset_release_v0.csv"):

    static_link = "https://uploads7.wikiart.org/images/"

    pd_csv = pd.read_csv(csv_file)
    return (
        pd_csv["painting"]
        .apply(lambda s: static_link + s.replace("_", "/") + ".jpg")
        .unique()
    )


def _store_paintings_in_txt(link_list, path):
    with open(path, "w+") as f:
        for url in link_list:
            f.write(f"{url}\n")


def prepare_next_run(artemis_annotation_csv, out_dir):
    all_paintings = out_dir + "/all_paintings.txt"
    to_download_paintings = out_dir + "/to_download_paintings.txt"
    downloaded_paintings = out_dir + "/downloaded_paintings.txt"
    not_downloaded_paintings = out_dir + "/not_downloaded_paintings.txt"

    if not os.path.exists(all_paintings):
        csv_entries = _get_csv_entries(artemis_annotation_csv)
        _store_paintings_in_txt(csv_entries, all_paintings)

    crawled = []
    for file in [downloaded_paintings, not_downloaded_paintings]:
        if not os.path.exists(file):
            open(file, "w+").close()
        with open(file, "r") as f:
            crawled += f.readlines()

    with open(all_paintings, "r") as f:
        to_crawl = [x for x in f.readlines() if x not in set(crawled)]

    with open(to_download_paintings, "w+") as f:
        f.write("".join(to_crawl))

    return to_crawl


def _download_file(url, out_dir="./artemis-download"):
    try:
        url = url.replace("\n", "")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            # otherwise the downloaded image file's size will be zero
            r.raw.decode_content = True

            # should be underscore '_' if taken from original artwork name
            filename = "_".join(url.split("/")[-2:])

            external_directory = out_dir
            path = os.path.join(external_directory, filename)
            with open(path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

            # check file size and throw error if file size below 0!
            if Path(path).stat().st_size < 1:
                raise Exception("File size below 1 Byte")

            with open(out_dir + "-stats" + "/downloaded_paintings.txt", "a+") as f:
                f.write(f"{url}\n")

        else:
            print(f"{r.status_code} - {r.headers['Content-Type']} - {url}")

            with open(out_dir + "-stats" + "/not_downloaded_paintings.txt", "a+") as f:
                f.write(f"{url}\n")

    except Exception as e:
        print(f"Error: {str(e)} \n- by trying to download url (without 404): {url}")
        with open(out_dir + "-stats" + "/not_downloaded_paintings.txt", "a+") as f:
            f.write(f"{url}\n")


def download_in_parallel(link_list, out_dir, threads=16):
    download_partial = partial(_download_file, out_dir=out_dir)
    with Pool(threads) as p:
        p.map(download_partial, link_list)


@click.command()
@click.option(
    "--artemis-annotation-csv",
    default="./artemis_dataset_release_v0.csv",
    help="Path to the annotation csv from artemis",
)
@click.option(
    "--out-dir",
    default="./artemis-download",
    help="Where to save the results",
)
@click.option(
    "--threads",
    default=4,
    help="Number of threads for parallel download",
)
def main(artemis_annotation_csv, out_dir, threads):
    """
    Sometimes, we cannot find the c0rresponding image for an ArtEmis annotation.
    In this case, we do not use the image for training.

    You can restart this script if the downloading process is interrupted.
    We save state in the out_dir + "-stats" folder and resume from there.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + "-stats", exist_ok=True)

    csv_list = prepare_next_run(artemis_annotation_csv, out_dir + "-stats")
    download_in_parallel(csv_list, out_dir, threads)


if __name__ == "__main__":
    main()
