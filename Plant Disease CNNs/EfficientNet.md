# EfficientNet-B0 (timm) with PyTorch for Plant Disease Classification

*EfficientNet-B0 is a convolutional neural network (CNN) model from the Google AI EfficientNet family, designed for image classification tasks. It is the B0 variant in the EfficientNet series (B0–B7), offering the best balance of accuracy and computational efficiency for lightweight deployment.*

This repository provides a robust pipeline for classifying plant diseases using the EfficientNet architecture via the PyTorch Image Models (timm) library. The project uses **EfficientNet-B0** (`efficientnet_b0` in timm) as instantiated in `app.py`, which serves as an excellent balance between high classification accuracy and computational efficiency.

## Using EfficientNet-B0 via timm in PyTorch is excellent for plant disease classification.

### EfficientNet-B0 is:

- Lightweight (~5.3M parameters)
- High accuracy
- Perfect for transfer learning
- Faster than B3 but still powerful

## Training Setup (PyTorch + timm)

**Install timm**

```bash
pip install timm
```

**Load Pretrained EfficientNet-B0**
```python
import timm
import torch
import torch.nn as nn

num_classes = 10  # change according to your dataset

model = timm.create_model(
    'efficientnet_b0',
    pretrained=True,
    num_classes=num_classes
)
```

**Data Transforms**
```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
])
```

**For Validation**
```python
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

**Loss & Optimizer**
```python
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)
```

**Fine-Tuning Strategy**

#### step 1 : Freeze backbone

```python
for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True
```
#### Step 2: Unfreeze entire model
```python
for param in model.parameters():
    param.requires_grad = True
```
#### Train with lower LR:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

## EfficientNet Family Overview

| Model  | Parameters | Input Size  | Accuracy        | Use Case       |
| ------ | ---------- | ----------- | --------------- | -------------- |
| B0     | ~5M        | 224×224     | Base model      | Lightweight    |
| B1     | ~7.8M      | 240×240     | Higher accuracy | Mobile         |
| B2     | ~9.2M      | 260×260     | Balanced        | Moderate tasks |
| **B3** | **~12M**   | **300×300** | High accuracy   | Production     |
| B4–B7  | 19M–66M    | 380–600     | Very high       | Heavy tasks    |
