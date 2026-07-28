"""
Utility functions for image classification via transfer learning.

A frozen ImageNet-pretrained ResNet18 turns each image into a 512-d feature
vector; a logistic-regression classifier is trained on top. No CNN fine-tuning.
Images are loaded from a committed sample under data/sample/<class>/ (regenerate
with make_data.py).
"""

import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sample(root="data/sample"):
    """Return (list_of_PIL_images, list_of_labels, sorted_class_names)."""
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}"))
    imgs, labels = [], []
    for cls in classes:
        for p in sorted(glob.glob(f"{root}/{cls}/*.png")):
            imgs.append(Image.open(p).convert("RGB"))
            labels.append(cls)
    return imgs, labels, classes


_TF = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_extractor(device):
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    return net.eval().to(device)


@torch.no_grad()
def extract_features(net, imgs, device, bs=128):
    out = []
    for i in range(0, len(imgs), bs):
        batch = torch.stack([_TF(im) for im in imgs[i:i + bs]]).to(device)
        out.append(net(batch).cpu().numpy())
    return np.concatenate(out)


def train_classifier(Xtr, ytr):
    return LogisticRegression(max_iter=3000).fit(Xtr, ytr)


def evaluate(clf, Xte, yte, labels):
    p = clf.predict(Xte)
    return accuracy_score(yte, p), confusion_matrix(yte, p, labels=labels), p
