# Skin Cancer Classification

Classify dermatoscopic skin-lesion images into diagnosis categories using transfer learning (frozen ResNet18 features + logistic regression). A teaching demo, not a diagnostic device.

## Problem statement

Given a dermatoscopic image of a skin lesion, predict its diagnosis among 7 types (melanoma, melanocytic nevi, basal-cell carcinoma, actinic keratoses, benign keratosis-like lesions, dermatofibroma, vascular lesions). The classes look alike to a non-expert and are heavily imbalanced.

## Dataset

**HAM10000**, 7 lesion classes. Source: [HuggingFace `marmal88/skin_cancer`](https://huggingface.co/datasets/marmal88/skin_cancer) / [Kaggle HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

This streams HAM10000 (cached in `~/.cache/huggingface`) and writes a 64x64 sample of up to 250 images per class (1,496 total) to `data/sample/<diagnosis>/`. The two rarest classes are capped by availability (dermatofibroma ~110, vascular lesions ~136).

## How it works

`utils.py` extracts a 512-d ResNet18 embedding per lesion (classifier head removed) and trains a logistic-regression classifier on a stratified 70/30 split. No CNN fine-tuning.

## Project structure

```
Skin Cancer Classification/
├── 01_eda.ipynb              # class balance, sample per diagnosis
├── 02_model_building.ipynb   # ResNet18 features + LogReg, confusion, per-class
├── utils.py
├── make_data.py              # regenerates data/sample/ from HAM10000
├── requirements.txt
└── data/                     # created by make_data.py (not committed)
    └── sample/<diagnosis>/*.png
```

## Key findings (real output)

- Frozen ResNet18 features + logistic regression reach **55.90% test accuracy** (251 / 449) across 7 lesion types, well above the 14% chance rate but far from clinical quality.
- Per-class accuracy is uneven: melanocytic nevi (0.75) and vascular lesions (0.71) score best, while actinic keratoses and melanoma are hardest (0.45 each). Melanoma being one of the weakest is exactly the class you would least want to miss.
- The classes are visually similar and imbalanced, so overall accuracy flatters the majority class. Rare classes (dermatofibroma, vascular) have small test sets, so their numbers are noisy.
- **Caveat: dermatoscopy diagnosis is a specialist task. This demo shows the transfer-learning pipeline, not a clinically usable classifier. Do not use it for anything medical.**

## Tech stack

Python, PyTorch, torchvision (ResNet18), scikit-learn, PIL, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_eda.ipynb
```
