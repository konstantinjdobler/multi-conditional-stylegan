# 🎨 Art Creation With Multi-Conditional StyleGANs 🎨

This repository contains the code for [the paper](https://www.ijcai.org/proceedings/2022/684) "Art Creation With Multi-Conditional StyleGANs" accepted at IJCAI 2022.
The source code is based on [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch/) by Karras et al. from NVIDIA.

## Setup Instructions

In general, instructions are the same as for the original [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch/) code. For multi-conditional training, you will also need to supply a `prepared_dataset.json` file with the `--cond-path` flag to the `python train.py` command. The `prepared_dataset.json` is should contain the multi-conditions and have the following format:

```javascript
{
  "condition_order": ["painter", "keywords", "emotions"], // condition names in the order they appear in the concatenated vector
  "shapes": [352, 768, 9], // the size of the vector representation for each condition part (in the same order)
  "labels": [
    ["/path/to/image/within/data/dir", [<label>, <label>, <label>],
    ... // for all images in dataset
  ],
}
```

Each `<label>` is either:

- an integer (which will be converted to vector representation through one-hot-encoding)
- an array containing one or more strings (which will be converted to vector representation with a pretrained TinyBERT embedding). In case of multiple strings, a single representation will be randomly sampled each time the training sample is shown to the model.
- an array containing floats, which should be a probability distribution and are directly used as vector representation

<details>
<summary>
How to get the <b>EnhancedArtEmis</b> dataset?
</summary>

<br>

Unfortunately, we cannot host the dataset or emotion annotations due to copyright and licencing. We outline the process to reproduce the dataset and prepare it for training with StyleGAN2 here.

1. Follow the instructions in the [ArtEmis repository](https://github.com/optas/artemis) to download and preprocess the emotion annotations. We do not use their preprocessing for "deep" networks.
2. Download the actual **image files** for ArtEmis. You can use our [`download_artemis_images.py`](./scripts/download_artemis_images.py) script. This can take a while.
3. Create a `dataset.json` file compatible with the [`dataset_tool.py`](dataset_tool.py) that maps image names to emotion labels. We convert multiple emotion annotations per image directly into a probability distribution. The format for `dataset.json` should look like this:

```javascript
{
  "labels": [
    ["vincent-van-gogh_the-starry-night-1889.jpg", [0.0, 0.0, 0.2, 0.4, 0.0, 0.0, 0.2, 0.0, 0.2]],
    ["claude-monet_water-lilies-1919.jpg", [0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    ... //for every image in the dataset
    ["pablo-picasso_guernica-1937.jpg", [0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.4, 0.0, 0.0]],
  ]
}
```

4. Prepare the image files and `dataset.json` for training with the [`dataset_tool.py`](dataset_tool.py) (instructions in the original [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch/) repository). The `dataset.json` should be placed _inside_ the folder containing the images downloaded in step 2. We used a command like this (we scale non-square images instead of cropping):

```bash
python dataset_tool.py --source=./artemis-download/ --dest=./processed-artemis.zip --width=512 --height=512
```

5. To get additional annotations scraped from Wikiart use our [`scrape_additional_annotations.py`](./scripts/scrape_additional_annotations.py) script. Or you can use the [`enhanced_annotations.json`](annotations/enhanced_annotations.json) we provide (scraped as of July 2021).
6. Now, we only need to create the `prepared_dataset.json` to enable our multi-conditional training. We have prepared the [`create_label_json.py`](./scripts/create_label_json.py) script for this.
7. 🚀 Wow, you made it! 🚀 Time to create some 🎨 art 🎨.

</details>
<br>

You can start a multi-conditional training with a command like this:

```bash
python train.py --outdir=./results-out --data=</path/to/data.zip> --cond-path ./annotations/painter-style-keywords/prepared_dataset.json --gpus=1 --snap=50 --workers=4 --batch=64 --cond=1 -n my-multiconditional-stylegan --dataset-cache-dir </path/to/cache/if/wanted>
```

## Conditional Truncation Trick

In the paper, we propose the **conditional truncation trick** for StyleGAN. If you use the truncation trick together with conditional generation or on diverse datasets, give our conditional truncation trick a try (it's a drop-in replacement). The effect is illustrated below (figure taken from [the paper](https://www.ijcai.org/proceedings/2022/684)):

<img width="1464" alt="Comparison of conditional truncation trick and normal truncation trick" src="https://user-images.githubusercontent.com/28780372/182967022-13144d1b-9a18-43ef-8db9-b926be3ad43b.png">

The implementation is quite easy. The relevant lines of code can be found here:

<https://github.com/konstantinjdobler/multi-conditional-stylegan/blob/main/training/networks.py#L245-L264>

## License

The source code in this repository is distributed under the same [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html) as the original StyleGAN2-ADA repository.

## Citation

```
@inproceedings{dobler2022multiconditional,
  title     = {Art Creation with Multi-Conditional StyleGANs},
  author    = {Dobler, Konstantin and Hübscher, Florian and Westphal, Jan and Sierra-Múnera, Alejandro and de Melo, Gerard and Krestel, Ralf},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {4936--4942},
  year      = {2022},
  month     = {7},
  note      = {AI and Arts}
  doi       = {10.24963/ijcai.2022/684},
  url       = {https://doi.org/10.24963/ijcai.2022/684},
}
```

If you use the code in this repository, please also cite [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch/):

```
@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
```
