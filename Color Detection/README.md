# Color Detection

Pull the dominant colours out of any image with K-Means clustering, then map each one to the nearest human colour name. Validated against a synthetic image with known ground-truth colours before running on natural photos.

## Problem statement

Given an image, what are its main colours and what would a person call them? Useful for palette extraction, image tagging, and design tooling. The twist: extracting colours is easy and reliable, but *naming* them is fuzzy and only as good as your reference palette.

## Dataset

No external dataset. Three images are generated locally:

- `blocks.png`: a synthetic 4-colour image built from exact RGB values (red 220,30,30 / blue-ish 30,140,220 / green 40,180,60 / yellow 240,210,30). This is the ground-truth check.
- `china.png` and `flower.png`: [scikit-learn sample images](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_image.html) with real gradients and lighting.

For a labelled colour-name dataset, see [Kaggle color-recognition](https://www.kaggle.com/datasets/adikurniawan/color-dataset-for-color-recognition).

The `data/` folder is not committed. Regenerate the images before running the notebooks:

```
python make_data.py
```

## How it works

`utils.py` clusters the image's pixels in RGB space with K-Means. Each cluster centre is a dominant colour; the cluster's size is that colour's share of the image. A small 18-colour reference palette maps each centre to the nearest named colour by Euclidean distance in RGB.

## Project structure

```
Color Detection/
├── 01_dominant_colors.ipynb   # ground-truth check + palette strips
├── 02_named_colors.ipynb      # named colours across all three images
├── utils.py                   # K-Means extraction + name lookup
├── make_data.py               # regenerates the three images
├── requirements.txt
└── data/                      # created by make_data.py (not committed)
    ├── blocks.png             # synthetic, known colours
    ├── china.png              # sklearn sample photo
    └── flower.png             # sklearn sample photo
```

## Key findings (real output)

- On `blocks.png`, K-Means with k=4 recovers the planted colours exactly: **(240,210,30), (30,140,220), (220,30,30), (40,180,60)**, each a 0.25 share. The name lookup calls three of them yellow / red / green, but the blue-ish block (30,140,220) lands on **teal**, its nearest neighbour in the 18-colour palette. That mislabel on an obviously-blue block is the honest illustration of the point: extraction is exact, naming is only as good as the palette.
- On natural photos the cluster centres are blends (lit surfaces, shadows, gradients), so the palette strips look right but the *names* drift: a muted leaf can land on "green" or "olive" depending on lighting, and highlights collapse to "white" or "gray".
- The honest takeaway: dominant-colour extraction is solid and reproducible. Colour naming is the weak link, capped by how rich the reference palette is.

## Tech stack

Python, scikit-learn (KMeans), OpenCV, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_dominant_colors.ipynb
```
