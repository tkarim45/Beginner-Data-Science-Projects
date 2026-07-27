"""
Regenerate data/sample/ from the CIFAR-10 dataset on HuggingFace.

Streams `uoft-cs/cifar10`, keeps the 6 animal classes, saves 350 64x64 images
per class into data/sample/<animal>/. The HF download is cached outside the
repo (~/.cache/huggingface). Run once before the notebooks:

    python make_data.py
"""
import os
from datasets import load_dataset

ROOT = "data/sample"
ANIMALS = {2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse"}
PER = 350

got = {}
stream = load_dataset("uoft-cs/cifar10", split="train", streaming=True)
for r in stream:
    lab = r["label"]
    if lab not in ANIMALS or got.get(lab, 0) >= PER:
        continue
    name = ANIMALS[lab]
    d = f"{ROOT}/{name}"
    os.makedirs(d, exist_ok=True)
    i = got.get(lab, 0)
    r["img"].convert("RGB").resize((64, 64)).save(f"{d}/{i}.png")
    got[lab] = i + 1
    if all(got.get(k, 0) >= PER for k in ANIMALS):
        break
print("saved per class:", {ANIMALS[k]: got[k] for k in ANIMALS})
