"""
Regenerate data/sample/ from the HuggingFace dataset `AutumnQiu/fer2013`.

Streams the dataset, saves up to 250 64x64 images per class into
data/sample/<class>/. The download is cached in ~/.cache/huggingface (not in
the repo). Run once before the notebooks:

    python make_data.py
"""
import os
from datasets import load_dataset, load_dataset_builder

HF = "AutumnQiu/fer2013"
SPLIT = "train"
IMG_FIELD = "image"
LABEL_FIELD = "label"
CAP = 250
EXPECTED_CLASSES = 7

try:
    names = getattr(load_dataset_builder(HF).info.features[LABEL_FIELD], "names", None)
except Exception:
    names = None

root = "data/sample"
got = {}
stream = load_dataset(HF, split=SPLIT, streaming=True)
for r in stream:
    lab = r[LABEL_FIELD]
    cls = str(names[lab] if names else lab).replace("/", "-").strip()
    if got.get(cls, 0) >= CAP:
        continue
    d = f"{root}/{cls}"
    os.makedirs(d, exist_ok=True)
    i = got.get(cls, 0)
    try:
        r[IMG_FIELD].convert("RGB").resize((64, 64)).save(f"{d}/{i}.png")
    except Exception:
        continue
    got[cls] = i + 1
    if sum(got.values()) >= CAP * EXPECTED_CLASSES:
        break
print("saved per class:", dict(sorted(got.items())))
