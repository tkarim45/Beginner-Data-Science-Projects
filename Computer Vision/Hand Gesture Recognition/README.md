# Hand Gesture Recognition

Recognise a hand gesture (rock, paper, or scissors) using transfer learning: a frozen ImageNet-pretrained ResNet18 feature extractor plus a logistic-regression classifier. No CNN fine-tuning.

## Problem statement

Given a photo of a hand, classify the gesture as rock (fist), paper (open hand), or scissors (two fingers). A compact multi-class image task and the basis of gesture-controlled interfaces.

## Dataset

Rock-paper-scissors hand images, three classes. Source: [HuggingFace `Javtor/rock-paper-scissors`](https://huggingface.co/datasets/Javtor/rock-paper-scissors) (2,520 images) / related to the [Kaggle leapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) gesture task.

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

This loads the dataset and writes a 64x64 sample of 300 images per class (900 total) to `data/sample/<gesture>/`.

## How it works

`utils.py` turns each image into a 512-d ResNet18 embedding (classifier head removed), then trains a logistic-regression classifier on a stratified 70/30 split. No fine-tuning.

## Project structure

```
Hand Gesture Recognition/
├── 01_eda.ipynb              # class balance, sample per gesture
├── 02_model_building.ipynb   # ResNet18 features + LogReg, confusion, per-class
├── utils.py
├── make_data.py              # regenerates data/sample/
├── requirements.txt
└── data/                     # created by make_data.py (not committed)
    └── sample/<gesture>/*.png
```

## Key findings (real output)

- Frozen ResNet18 features + logistic regression classify the three gestures at **100% test accuracy** (270 / 270).
- The three hand shapes differ strongly in silhouette and finger count, and the source images are clean studio shots, so the classes are linearly separable in ImageNet feature space, no fine-tuning needed.
- Like the Fruit project, a perfect score here reflects clean, well-separated data. On messy in-the-wild hand photos (varied backgrounds, angles, skin tones) you would expect confusion, especially scissors vs paper.

## Tech stack

Python, PyTorch, torchvision (ResNet18), scikit-learn, PIL, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_eda.ipynb
```
