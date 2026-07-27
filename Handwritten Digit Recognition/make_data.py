"""
Regenerate data/mnist_subset.npz from torchvision MNIST.

Deterministic (fixed seed 42), so the committed notebook numbers reproduce
exactly. Run once before the notebooks:

    python make_data.py

The full MNIST download is cached outside the repo (/tmp), only the small
subset npz lands in data/.
"""
import os
import numpy as np
from torchvision import datasets

os.makedirs("data", exist_ok=True)
train = datasets.MNIST("/tmp/mnist_dl", train=True, download=True)
test = datasets.MNIST("/tmp/mnist_dl", train=False, download=True)

rng = np.random.RandomState(42)


def subset(ds, n):
    idx = rng.choice(len(ds), n, replace=False)
    X = np.stack([np.array(ds[i][0]) for i in idx]).astype("uint8")
    y = np.array([ds[i][1] for i in idx], np.int64)
    return X, y


Xtr, ytr = subset(train, 10000)
Xte, yte = subset(test, 2000)
np.savez_compressed("data/mnist_subset.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
print("wrote data/mnist_subset.npz", Xtr.shape, Xte.shape)
