# Handwritten Digit Recognition (MNIST)

Train a small convolutional neural network from scratch to recognize handwritten digits 0-9. This is the "hello world" of computer vision, done properly: real training loop, held-out evaluation, confusion analysis.

## Problem statement

Given a 28x28 grayscale image of a handwritten digit, predict which digit (0-9) it is. MNIST is the classic benchmark for this.

## Dataset

**MNIST** handwritten digits. Source: [torchvision MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) / [Kaggle digit-recognizer](https://www.kaggle.com/c/digit-recognizer).

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

This downloads MNIST (cached in `/tmp`) and writes a fixed-seed 10,000 train / 2,000 test subset to `data/mnist_subset.npz`. The seed is fixed, so the numbers below reproduce exactly.

## Model

A compact CNN built in PyTorch (`utils.py`), about **207k parameters**:

- Conv(1->16, 3x3) + ReLU + MaxPool
- Conv(16->32, 3x3) + ReLU + MaxPool
- Flatten + Dense(1568->128) + ReLU + Dense(128->10)

Trained with Adam (lr 1e-3), cross-entropy loss, 3 epochs. Runs on Apple MPS, CUDA, or CPU automatically.

## Project structure

```
Handwritten Digit Recognition/
├── 01_eda.ipynb              # digit grid, class balance, mean images, pixel stats
├── 02_model_building.ipynb   # CNN train + evaluate + confusion analysis
├── utils.py                  # data loading, DigitCNN, train/eval
├── make_data.py              # regenerates data/mnist_subset.npz
├── requirements.txt
└── data/                     # created by make_data.py (not committed)
    └── mnist_subset.npz      # 10k train / 2k test
```

## Key findings (real output)

- The CNN reaches **97.20% test accuracy** (1,944 / 2,000 correct) after just 3 epochs on the 10k subset. Training loss falls 0.77 -> 0.21 -> 0.14 across the three epochs.
- The confusion matrix is nearly diagonal. Residual errors are the textbook MNIST confusions (4 vs 9, 3 vs 5, 7 vs 1), driven by genuinely ambiguous handwriting.
- Classes are close to balanced (~1,000 images per digit in the training subset).
- Trained on the full 60k MNIST set, the same network hits about **98.8%** in 2 epochs. The subset trades roughly a point of accuracy for a notebook that runs in seconds and needs no download.

## Tech stack

Python, PyTorch, torchvision, scikit-learn (metrics), NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_eda.ipynb
```

Run `01_eda` first, then `02_model_building` to train and evaluate.
