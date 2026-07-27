"""
Utility functions for the Handwritten Digit Recognition project.

A small convolutional neural network trained from scratch on a committed MNIST
subset (self-contained, no download needed at run time).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_subset(path="data/mnist_subset.npz"):
    """Load the committed MNIST subset. Returns Xtr, ytr, Xte, yte (uint8/int)."""
    d = np.load(path)
    return d["Xtr"], d["ytr"], d["Xte"], d["yte"]


def to_tensor(X):
    """(N, 28, 28) uint8 -> (N, 1, 28, 28) float tensor, MNIST-normalized."""
    X = X.astype("float32") / 255.0
    X = (X - 0.1307) / 0.3081
    return torch.from_numpy(X).unsqueeze(1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DigitCNN(nn.Module):
    """Two conv blocks + two dense layers. ~207k parameters."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_model(model, Xtr, ytr, device, epochs=3, bs=128, lr=1e-3, seed=0):
    """Train in-place with Adam + cross-entropy. Returns per-epoch loss list."""
    torch.manual_seed(seed)
    model.to(device).train()
    Xt = to_tensor(Xtr).to(device)
    yt = torch.from_numpy(ytr).long().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(Xt)
    history = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history.append(total / n)
    return history


@torch.no_grad()
def predict(model, X, device, bs=256):
    """Return predicted class labels for X."""
    model.eval()
    Xt = to_tensor(X).to(device)
    preds = []
    for i in range(0, len(Xt), bs):
        preds.append(model(Xt[i:i + bs]).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate(model, X, y, device):
    """Return (accuracy, confusion_matrix, predictions)."""
    p = predict(model, X, device)
    return accuracy_score(y, p), confusion_matrix(y, p), p
