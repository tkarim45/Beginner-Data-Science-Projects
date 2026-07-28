# Traffic Sign Recognition

Classify German traffic signs into their type using transfer learning: a frozen ImageNet-pretrained ResNet18 as a feature extractor plus a logistic-regression classifier. No CNN fine-tuning.

## Problem statement

Given a cropped photo of a traffic sign, predict which of 43 sign types it is (speed limits, warnings, mandatory-direction signs, and so on). This is the core perception task behind driver-assist and self-driving systems.

## Dataset

**GTSRB** (German Traffic Sign Recognition Benchmark), 43 classes. Source: [HuggingFace `tanganke/gtsrb`](https://huggingface.co/datasets/tanganke/gtsrb) / [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

This streams GTSRB (cached in `~/.cache/huggingface`) and writes a 64x64 sample of up to 60 images per class (43 classes, ~2,580 images) to `data/sample/<class>/`.

## How it works

`utils.py` loads ResNet18 with its classifier head removed, so each image becomes a 512-d embedding. Those embeddings feed a logistic-regression classifier trained on a stratified 70/30 split.

## Project structure

```
Traffic Sign Recognition/
├── 01_eda.ipynb              # class balance, sample per class
├── 02_model_building.ipynb   # ResNet18 features + LogReg, confusion, per-class
├── utils.py
├── make_data.py              # regenerates data/sample/ from GTSRB
├── requirements.txt
└── data/                     # created by make_data.py (not committed)
    └── sample/<class>/*.png
```

## Key findings (real output)

- Frozen ResNet18 features + logistic regression classify all 43 sign types at **99.35% test accuracy** (769 / 774).
- Most classes score a perfect 1.00; the few that dip to 0.94 are visually similar signs (some speed-limit numbers, a couple of triangular warning signs).
- Traffic signs are engineered to be distinct in shape and colour, so even generic ImageNet features separate them almost perfectly. No fine-tuning required.

## Tech stack

Python, PyTorch, torchvision (ResNet18), scikit-learn, PIL, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_eda.ipynb
```
