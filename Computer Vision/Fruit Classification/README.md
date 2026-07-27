# Fruit Classification

Classify photos of fruit by type using transfer learning: a frozen ImageNet-pretrained ResNet18 as a feature extractor, plus a simple logistic-regression classifier on top. No fine-tuning, no GPU training loop.

## Problem statement

Given a photo of a single fruit, predict which fruit it is (apple, banana, orange, and so on). This is a clean, well-lit image-classification task, which makes it a good first look at how far frozen pretrained features get you.

## Dataset

**fruits-360**: studio photos of single fruits on a white background, 113 classes. Source: [HuggingFace `PedroSampaio/fruits-360`](https://huggingface.co/datasets/PedroSampaio/fruits-360) / [Kaggle moltean/fruits](https://www.kaggle.com/datasets/moltean/fruits).

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

This streams fruits-360 (cached in `~/.cache/huggingface`) and writes a 64x64 sample of 8 classes (apple, avocado, banana, kiwi, lemon, orange, pineapple, strawberry) to `data/sample/<fruit>/`, about 1,120 images.

## How it works

`utils.py` loads ResNet18 with its classifier head removed, so each image becomes a 512-dimensional embedding. Those embeddings feed a logistic-regression classifier trained on a stratified 70/30 split. The CNN itself is never trained.

## Project structure

```
Fruit Classification/
├── 01_eda.ipynb              # class balance, sample grids
├── 02_model_building.ipynb   # ResNet18 features + LogReg, confusion, per-class
├── utils.py                  # loader, feature extractor, classifier
├── make_data.py              # regenerates data/sample/ from fruits-360
├── requirements.txt
└── data/                     # created by make_data.py (not committed)
    └── sample/<fruit>/*.png  # 64x64 sample, 8 classes
```

## Key findings (real output)

- ResNet18 features + logistic regression hit **100% test accuracy** (338 / 338) on the held-out split. The 8 fruit classes are perfectly separated.
- The CNN was pretrained on ImageNet and never fine-tuned on fruit, yet the 512-d embeddings are already linearly separable for these classes.
- The reason is the data: clean, centered, single-object images on a white background are the easy end of image classification.
- For contrast, see the **Animal Species Classification** project, which runs the exact same recipe on messy 32x32 CIFAR-10 photos and scores about 75%. Same method, very different result, because the images are harder.

## Tech stack

Python, PyTorch, torchvision (ResNet18), scikit-learn (LogisticRegression), PIL, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_eda.ipynb
```

Run `01_eda`, then `02_model_building`. To scale to the full 113-class set, load `PedroSampaio/fruits-360` from HuggingFace and point the loader at it.
