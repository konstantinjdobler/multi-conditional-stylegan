import click
import json
import os
from click.exceptions import ClickException
from collections import defaultdict
from sentence_transformers import SentenceTransformer

NONE_STR = "AA__UNKNOWN__"


def count_occurences(condition_type, enhanced_annotations_json, is_array_value=False):
    accum = defaultdict(int)
    for entry in enhanced_annotations_json["entries"]:
        condition_value = entry.get(condition_type) or NONE_STR
        if is_array_value and condition_value != NONE_STR:
            condition_value = condition_value[0]
        accum[condition_value] += 1

    return dict(sorted(accum.items(), key=lambda item: item[1], reverse=True))


def create_filtered_index(condition_type, enhanced_annotations_json, cutoff=100):
    is_array_value = isinstance(
        enhanced_annotations_json["entries"][0].get(condition_type), list
    )
    print(condition_type, is_array_value)
    occurrences = count_occurences(
        condition_type, enhanced_annotations_json, is_array_value
    )
    condition_values_per_sample = (
        entry.get(condition_type) or NONE_STR
        for entry in enhanced_annotations_json["entries"]
    )
    known_examples = set()
    if condition_type == "painter":
        cutoff = cutoff / 2
    for cond_value in condition_values_per_sample:
        if cond_value == NONE_STR:
            known_examples.add(cond_value)
            continue
        if is_array_value:
            cond_value = cond_value[0]
        if occurrences[cond_value] <= cutoff:
            known_examples.add(NONE_STR)
        else:
            known_examples.add(cond_value)
    return {value: i for i, value in enumerate(sorted(known_examples))}


def sanitize_keyword(keyword):
    return keyword.replace('"', "").replace("'", "").replace("-", " ").lower()


def create_filtered_keyword_count(enhanced_annotations_json):
    all_keywords = defaultdict(int)
    for entry in enhanced_annotations_json["entries"]:
        keywords = entry["keywords"]
        for k in keywords:
            k = sanitize_keyword(k)
            all_keywords[k] += 1
    out = dict(filter(lambda x: x[1] > 50, all_keywords.items()))
    out = dict(sorted(out.items(), key=lambda x: x[1], reverse=True))
    return out


emotion_idx = {
    "amusement": 0,
    "anger": 1,
    "awe": 2,
    "contentment": 3,
    "disgust": 4,
    "excitement": 5,
    "fear": 6,
    "sadness": 7,
    "something else": 8,
}


@click.command()
@click.option(
    "--enhanced-annotations-json",
    "-ajson",
    "-a",
    help="JSON file containing additional attributes (painter, art style, genre etc.) "
    "scraped from WikiArt for each painting.",
    type=click.Path(dir_okay=False, file_okay=True),
)
@click.option(
    "--emotions-dataset-json",
    "-ejson",
    "-e",
    help="dataset.json file produced by the dataset_tool.py. "
    "IMPORTANT: Specify the dataset.json in the OUTPUT and not the INPUT directory to dataset_tool.py, "
    "so that we can get the correct folder mapping."
    "You might need to unzip the output to access this file.",
    type=click.Path(dir_okay=False, file_okay=True),
)
@click.option("--out-dir", "-odir", "-o")
@click.option(
    "--conditions",
    "-c",
    multiple=True,
    help="Specify the conditions to include in the prepared_dataset.json."
    "To include all, do: -c painter -c emotions -c keywords -c style -c genre",
)
def main(
    enhanced_annotations_json,
    emotions_dataset_json,
    out_dir,
    conditions,
):
    ########### Load stuff ############
    folder_mapping_json = emotions_dataset_json
    with open(emotions_dataset_json, "rb") as f:
        emotions_dataset_json = dict(json.load(f)["labels"])
    with open(folder_mapping_json, "rb") as f:
        file_folder_names = [
            label_pair[0].split("/") for label_pair in json.load(f)["labels"]
        ]
    folder_mapping = {file: folder for folder, file in file_folder_names}
    with open(enhanced_annotations_json, "rb") as f:
        enhanced_annotations_json = json.load(f)
    conditions = list(conditions)
    for i, condition_type in enumerate(conditions):
        if condition_type == "artist":
            conditions[i] = condition_type = "painter"
        assert condition_type in [
            "painter",
            "emotions",
            "keywords",
            "style",
            "genre",
        ], f"Invalid condition: {condition_type}"

    out_dict = {"condition_order": [], "shapes": [], "labels": []}  # nine emotions

    ########### Generate indexes ########
    indexes = {}
    for condition_type in conditions:
        if condition_type == "emotions":
            index = emotion_idx
        elif condition_type == "keywords":
            index = create_filtered_keyword_count(enhanced_annotations_json)
        else:
            index = create_filtered_index(condition_type, enhanced_annotations_json)
        out_dict["shapes"].append(len(index))
        if condition_type == "keywords":
            # For textual annotations, use embedding dimension instead
            model = SentenceTransformer("paraphrase-TinyBERT-L6-v2")
            embedding_dimension = model.get_sentence_embedding_dimension()
            out_dict["shapes"][-1] = embedding_dimension

        out_dict["condition_order"].append(condition_type)
        indexes[condition_type] = index

    # print(indexes)
    print("Shapes:", out_dict["shapes"], "Order:", out_dict["condition_order"])

    ########### Create labels ###########
    skipped = []
    for entry in enhanced_annotations_json["entries"]:
        img_png_name = f"{entry['name'].split('.')[0]}.png"
        if folder_mapping.get(img_png_name) is None:
            skipped.append(img_png_name)
            continue
        img_path = f"{folder_mapping[img_png_name]}/{img_png_name}"
        labels = []
        for condition_type in conditions:
            if condition_type == "emotions":
                labels.append(emotions_dataset_json[img_path])
            elif condition_type == "keywords":
                if len(entry["keywords"]) == 0:
                    labels.append([])
                    continue
                filtered_keywords = [
                    sanitize_keyword(kw)
                    for kw in entry["keywords"]
                    if indexes["keywords"].get(sanitize_keyword(kw))
                ]
                labels.append(filtered_keywords)
            else:
                index = indexes[condition_type]
                condition_value = entry.get(condition_type) or NONE_STR
                if isinstance(condition_value, list):
                    condition_value = condition_value[0]
                label_value = index.get(condition_value) or index[NONE_STR]
                labels.append(label_value)

        out_dict["labels"].append([img_path, labels])

    # print(labels)
    print(f"Skipped {len(skipped)} files: {skipped}")

    ########### Write output ###########
    os.makedirs(out_dir, exist_ok=True)
    for condition_type in conditions:
        suffix = "occurrence" if condition_type == "keywords" else "idx"
        with open(f"{out_dir}/{condition_type}_{suffix}.json", "w") as fp:
            json.dump(indexes[condition_type], fp, indent=2, ensure_ascii=False)

    with open(f"{out_dir}/prepared_dataset.json", "w") as fp:
        json.dump(out_dict, fp, ensure_ascii=False)


if __name__ == "__main__":
    main()
