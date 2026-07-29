"""
Regenerate data/sample/ from the HuggingFace dataset `sumitpardhiya/Face-Mask-Detection`.

Loads the dataset (non-streaming: these ImageFolder datasets have broken
streaming label metadata), shuffles with a fixed seed, and saves up to 500
64x64 images per class into data/sample/<class>/. Download cached in
~/.cache/huggingface (not in repo). Run once before the notebooks:

    python make_data.py
"""
import os
import numpy as np
from datasets import load_dataset

HF="sumitpardhiya/Face-Mask-Detection"; SPLIT="train"; IMG="image"; LAB="label"; CAP=100
ds=load_dataset(HF, split=SPLIT)
names=ds.features[LAB].names if hasattr(ds.features[LAB],"names") else None
order=np.random.RandomState(42).permutation(len(ds))
root="data/sample"; got={}
for idx in order:
    r=ds[int(idx)]
    cls=str(names[r[LAB]] if names else r[LAB]).replace("/","-").strip()
    if got.get(cls,0)>=CAP: continue
    d=f"{root}/{cls}"; os.makedirs(d,exist_ok=True); i=got.get(cls,0)
    try: r[IMG].convert("RGB").resize((64,64)).save(f"{d}/{i}.png")
    except Exception: continue
    got[cls]=i+1
print("saved per class:", dict(sorted(got.items())))
