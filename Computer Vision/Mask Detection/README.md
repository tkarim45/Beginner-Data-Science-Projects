# Mask Detection

Detect whether a face is wearing a mask, using transfer learning: a frozen ImageNet-pretrained ResNet18 feature extractor plus a logistic-regression classifier. No CNN fine-tuning.

## Problem statement

Given a face photo, predict whether the person is wearing a mask (with_mask) or not (without_mask). A common pandemic-era computer-vision task.

## Dataset

Face photos in two classes (WithMask, WithoutMask). Source: [HuggingFace `sumitpardhiya/Face-Mask-Detection`](https://huggingface.co/datasets/sumitpardhiya/Face-Mask-Detection) / [Kaggle face-mask-dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset). It is a small set (200 images, 100 per class), so this is a compact demo.

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

This loads the dataset and writes a 64x64 sample of up to 100 images per class to `data/sample/<class>/`.

## How it works

`utils.py` turns each image into a 512-d ResNet18 embedding (classifier head removed), then trains a logistic-regression classifier on a stratified 70/30 split. No fine-tuning.

## Project structure

```
Mask Detection/
├── 01_eda.ipynb              # class balance, sample images
├── 02_model_building.ipynb   # ResNet18 features + LogReg, confusion
├── utils.py
├── make_data.py              # regenerates data/sample/
├── requirements.txt
└── data/                     # created by make_data.py (not committed)
    └── sample/<class>/*.png
```

## Key findings (real output)

- Frozen ResNet18 features + logistic regression reach **100% test accuracy** (60 / 60) on the held-out split.
- A mask is a large, high-contrast object over the lower face, so the two classes are very well separated in ImageNet feature space, even without fine-tuning.
- Caveat: this is a small, clean dataset (200 images total). A perfect score here reflects easy separability, not a hard real-world benchmark. Field deployment would need far more varied data (angles, mask types, occlusion).

## Tech stack

Python, PyTorch, torchvision (ResNet18), scikit-learn, PIL, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_eda.ipynb
```
