# Dog vs Cat Classification

The classic binary image task, solved with transfer learning: a frozen ImageNet-pretrained ResNet18 feature extractor plus a logistic-regression classifier. No CNN fine-tuning.

## Problem statement

Given a photo of a pet, decide whether it is a dog or a cat. Simple to state, and a good check on how much a frozen ImageNet backbone already knows.

## Dataset

Natural pet photos, two classes. Source: [HuggingFace `Bingsu/Cat_and_Dog`](https://huggingface.co/datasets/Bingsu/Cat_and_Dog) / [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats).

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

This streams the dataset (cached in `~/.cache/huggingface`) and writes a 64x64 sample of 500 images per class (1,000 total) to `data/sample/<class>/`.

## How it works

`utils.py` turns each image into a 512-d ResNet18 embedding (classifier head removed), then trains a logistic-regression classifier on a stratified 70/30 split. The CNN is never fine-tuned.

## Project structure

```
Dog vs Cat Classification/
├── 01_eda.ipynb              # class balance, sample images
├── 02_model_building.ipynb   # ResNet18 features + LogReg, confusion
├── utils.py
├── make_data.py              # regenerates data/sample/
├── requirements.txt
└── data/                     # created by make_data.py (not committed)
    └── sample/<class>/*.png
```

## Key findings (real output)

- Frozen ResNet18 features + logistic regression separate cats from dogs at **88.33% test accuracy** (265 / 300).
- ImageNet pretraining already encodes strong cat and dog features (both are ImageNet super-classes), so a linear head on frozen features gets most of the way there.
- The gap to near-perfect comes from the 64x64 downscaling and natural clutter (odd poses, partial views); fine-tuning or higher resolution would close it.

## Tech stack

Python, PyTorch, torchvision (ResNet18), scikit-learn, PIL, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_eda.ipynb
```
