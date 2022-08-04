import json

import click
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from collections import defaultdict

@click.command()
@click.pass_context
@click.option('--bulletproof-json', '-bjson', '-b', type=click.Path(dir_okay=False, file_okay=True))
@click.option('--folder-mapping-json', '-fjson', '-f', type=click.Path(dir_okay=False, file_okay=True))
@click.option('--emotions-source-json', '-ejson', '-e', type=click.Path(dir_okay=False, file_okay=True))
@click.option('--out-file', '-ofile', '-o')
def main(
    ctx,
    bulletproof_json,
    folder_mapping_json,
    emotions_source_json,
    out_file):
    with open(bulletproof_json, 'r') as f:
        bulletproof = json.load(f)

    with open(folder_mapping_json, 'rb') as f:
        file_folder_names = [label_pair[0].split('/')
                             for label_pair in json.load(f)["labels"]]
    folder_mapping = {file: folder for folder, file in file_folder_names}

    model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')

    embedding_size = model.get_sentence_embedding_dimension()
    print("Embedding Size:", embedding_size)

    labels = []
    skipped = []
    num_empty = 0
    for entry in tqdm(bulletproof["entries"]):
        img_png_name = f"{entry['name'].split('.')[0]}.png"
        if folder_mapping.get(img_png_name) is None:
            skipped.append(img_png_name)
            continue
        img_path = f"{folder_mapping[img_png_name]}/{img_png_name}"

        keywords = entry["keywords"]
        labels.append([img_path, [keywords]])

    print("Empty:", num_empty)
    out_dict = {"labels": labels, "condition_order": "keywords", "shapes": [embedding_size]}
    with open(out_file, 'w+') as fp:
        json.dump(out_dict, fp, ensure_ascii=False)


if __name__ == "__main__":
    main()
