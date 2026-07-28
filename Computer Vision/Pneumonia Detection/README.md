# Pneumonia Detection

Detect pneumonia from a chest X-ray using transfer learning (frozen ResNet18 features + logistic regression). A teaching demo of medical image classification, not a diagnostic device.

## Problem statement

Given a chest X-ray, predict whether it shows pneumonia or is normal. Pneumonia appears as cloudy opacities in the lungs; normal X-rays are clearer.

## Dataset

Kermany chest X-ray set, two classes (NORMAL, PNEUMONIA). Source: [HuggingFace `hf-vision/chest-xray-pneumonia`](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) / [Kaggle chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

This streams the dataset (cached in `~/.cache/huggingface`) and writes a 64x64 sample of 500 images per class (1,000 total, class-balanced) to `data/sample/<class>/`.

## How it works

`utils.py` extracts a 512-d ResNet18 embedding per X-ray (classifier head removed) and trains a logistic-regression classifier on a stratified 70/30 split. No CNN fine-tuning.

## Project structure

```
Pneumonia Detection/
├── 01_eda.ipynb              # class balance, sample X-rays
├── 02_model_building.ipynb   # ResNet18 features + LogReg, confusion
├── utils.py
├── make_data.py              # regenerates data/sample/
├── requirements.txt
└── data/                     # created by make_data.py (not committed)
    └── sample/<class>/*.png
```

## Key findings (real output)

- Frozen ResNet18 features + logistic regression separate pneumonia from normal chest X-rays at **95.33% test accuracy** (286 / 300).
- ResNet was pretrained on natural photos, not radiographs, yet the signal (lung opacity) is coarse enough that generic edge and texture features transfer well.
- **Caveat: this is a demo on a small balanced sample. A real diagnostic tool needs a fine-tuned model, far more data, calibration, and clinical validation. Do not use this for anything medical.**

## Tech stack

Python, PyTorch, torchvision (ResNet18), scikit-learn, PIL, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_eda.ipynb
```
