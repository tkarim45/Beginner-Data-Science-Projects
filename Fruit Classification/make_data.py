"""
Regenerate data/sample/ from the fruits-360 dataset on HuggingFace.

Streams `PedroSampaio/fruits-360`, keeps 8 fruit classes, saves up to ~140
64x64 images per class into data/sample/<fruit>/. The HF download is cached
outside the repo (~/.cache/huggingface). Run once before the notebooks:

    python make_data.py
"""
import os
from datasets import load_dataset, load_dataset_builder

ROOT = "data/sample"
KEEP = ["Apple", "Banana", "Orange", "Strawberry", "Lemon", "Avocado", "Kiwi", "Pineapple"]
PER = 140

names = load_dataset_builder("PedroSampaio/fruits-360").info.features["label"].names


def keyof(label):
    n = names[label]
    return next((k for k in KEEP if n.startswith(k)), None)


got = {}
stream = load_dataset("PedroSampaio/fruits-360", split="train", streaming=True)
for r in stream:
    k = keyof(r["label"])
    if k is None or got.get(k, 0) >= PER:
        continue
    d = f"{ROOT}/{k}"
    os.makedirs(d, exist_ok=True)
    i = got.get(k, 0)
    r["image"].convert("RGB").resize((64, 64)).save(f"{d}/{i}.png")
    got[k] = i + 1
    if all(got.get(x, 0) >= PER for x in KEEP):
        break
print("saved per class:", got)
