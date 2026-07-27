# Animal Species Classification

Classify animals into 6 species using transfer learning (frozen ResNet18 features + logistic regression). This is the deliberately hard counterpart to the Fruit Classification project: same recipe, much messier images, much lower accuracy. That gap is the whole point.

## Problem statement

Given a small photo of an animal, predict its species: bird, cat, deer, dog, frog, or horse. The images are low-resolution, cluttered, and shot in many poses, so this is a genuinely harder task than clean studio photos.

## Dataset

The animal classes of **CIFAR-10**: 32x32 real-world colour photos. Source: [HuggingFace `uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) / [original CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

This streams CIFAR-10 (cached in `~/.cache/huggingface`) and writes a 64x64 sample of 350 images per class for the 6 animal classes (2,100 total) to `data/sample/<animal>/`.

## How it works

Identical pipeline to the Fruit project: a frozen ImageNet-pretrained ResNet18 (classifier head removed) turns each image into a 512-d embedding, and a logistic-regression classifier is trained on a stratified 70/30 split. Nothing about the CNN is fine-tuned. Using the same method on purpose is what makes the comparison fair.

## Project structure

```
Animal Species Classification/
├── 01_eda.ipynb              # class balance, sample grids
├── 02_model_building.ipynb   # ResNet18 features + LogReg, confusion, per-class
├── utils.py                  # loader, feature extractor, classifier
├── make_data.py              # regenerates data/sample/ from CIFAR-10
├── requirements.txt
└── data/                     # created by make_data.py (not committed)
    └── sample/<animal>/*.png # 64x64 sample, 6 classes
```

## Key findings (real output)

- The same frozen-ResNet18 + logistic-regression recipe scores **75.24% test accuracy** (474 / 630), against **100%** on the Fruit project. Same method, harder images.
- Per-class accuracy: frog 0.83, horse 0.80, bird 0.77, deer 0.76, cat 0.70, dog 0.66. The weakest classes are the semantically confusable ones, cat and dog.
- The confusion matrix concentrates errors on close pairs (cat/dog, deer/horse), which is what you would expect from tiny 32x32 crops.
- Takeaway: transfer learning is powerful but not magic. Frozen ImageNet features carry a lot of signal, but low resolution, busy backgrounds, and fine-grained categories are exactly where you would need to fine-tune the CNN or use higher-resolution data to close the gap.

## Tech stack

Python, PyTorch, torchvision (ResNet18), scikit-learn (LogisticRegression), PIL, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_eda.ipynb
```

Run `01_eda`, then `02_model_building`.
