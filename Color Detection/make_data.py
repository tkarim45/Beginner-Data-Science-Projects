"""
Regenerate the sample images in data/.

Two are scikit-learn sample photos (china, flower); one is a synthetic 4-colour
block image with known ground-truth RGBs, used to validate the extractor. Run
once before the notebooks:

    python make_data.py
"""
import os
import cv2
import numpy as np
from sklearn.datasets import load_sample_image

os.makedirs("data", exist_ok=True)

# Synthetic 4-colour blocks with exact, known RGB values.
img = np.zeros((400, 400, 3), np.uint8)
img[:200, :200] = [220, 30, 30]    # red  (top-left)
img[:200, 200:] = [30, 140, 220]   # blue (top-right)
img[200:, :200] = [40, 180, 60]    # green (bottom-left)
img[200:, 200:] = [240, 210, 30]   # yellow (bottom-right)
cv2.imwrite("data/blocks.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# scikit-learn sample photos.
for name in ["china", "flower"]:
    im = load_sample_image(f"{name}.jpg")  # returns RGB
    cv2.imwrite(f"data/{name}.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

print("wrote data/blocks.png, data/china.png, data/flower.png")
