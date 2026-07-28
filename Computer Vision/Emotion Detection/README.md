# Emotion Detection

Recognise the emotion in a facial photo using transfer learning (frozen ResNet18 features + logistic regression). This is the honest hard case of the computer-vision batch: the accuracy is low, and the README says why.

## Problem statement

Given a small grayscale photo of a face, predict the expressed emotion: angry, disgust, fear, happy, neutral, sad, or surprise. Facial-expression recognition is genuinely hard, even for people, on tiny low-resolution crops.

## Dataset

**FER2013**, 7 emotion classes, originally 48x48 grayscale faces. Source: [HuggingFace `AutumnQiu/fer2013`](https://huggingface.co/datasets/AutumnQiu/fer2013) / [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013).

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

This streams FER2013 (cached in `~/.cache/huggingface`) and writes a 64x64 sample of 250 images per class (7 classes, 1,750 images) to `data/sample/<emotion>/`.

## How it works

`utils.py` converts each grayscale face to RGB, extracts a 512-d ResNet18 embedding (classifier head removed), and trains a logistic-regression classifier on a stratified 70/30 split. No CNN fine-tuning.

## Project structure

```
Emotion Detection/
├── 01_eda.ipynb              # class balance, sample per emotion
├── 02_model_building.ipynb   # ResNet18 features + LogReg, confusion, per-class
├── utils.py
├── make_data.py              # regenerates data/sample/ from FER2013
├── requirements.txt
└── data/                     # created by make_data.py (not committed)
    └── sample/<emotion>/*.png
```

## Key findings (real output)

- Frozen ResNet18 features + logistic regression reach **33.14% test accuracy** (174 / 525) across 7 emotions. That is well above the 14% chance rate, but far below the clean-image projects in this folder.
- Per-class accuracy is uneven: angry is worst (0.16), fear and sad are weak (0.21 each), while surprise (0.45) and disgust (0.53) fare better. Subtle, overlapping expressions (fear vs surprise, sad vs neutral) are the main confusions.
- Why so low: FER faces are tiny, grayscale, and expressions are subtle. Human accuracy on FER2013 itself is only about 65%. Frozen ImageNet-RGB features are a poor match for 48x48 grayscale faces. A CNN fine-tuned on grayscale faces would do meaningfully better.
- The point of this project is the contrast: the same recipe that hits 99% on traffic signs and 88% on cats-vs-dogs lands at 33% here. The method is not the story, the data difficulty is.

## Tech stack

Python, PyTorch, torchvision (ResNet18), scikit-learn, PIL, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_eda.ipynb
```
