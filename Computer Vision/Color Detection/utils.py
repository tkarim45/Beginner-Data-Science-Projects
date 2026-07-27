"""
Utility functions for the Color Detection project.

K-Means dominant-colour extraction plus a nearest-named-colour lookup over a
small reference palette.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Reference palette: name -> RGB. Nearest match by Euclidean distance in RGB.
NAMED_COLORS = {
    "red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "orange": (255, 165, 0), "purple": (128, 0, 128),
    "pink": (255, 105, 180), "brown": (139, 69, 19), "black": (0, 0, 0),
    "white": (255, 255, 255), "gray": (128, 128, 128), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255), "teal": (0, 128, 128), "navy": (0, 0, 128),
    "olive": (128, 128, 0), "maroon": (128, 0, 0), "lime": (0, 255, 0),
}


def load_image(path, max_side=400):
    """Load an image as RGB, downscaled so K-Means stays fast."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def dominant_colors(rgb, k=5, seed=42):
    """
    Return (centers, proportions) for the k dominant colours, sorted by
    proportion descending. `centers` are uint8 RGB rows.
    """
    pixels = rgb.reshape(-1, 3).astype("float32")
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(pixels)
    centers = km.cluster_centers_.round().astype("uint8")
    counts = np.bincount(km.labels_, minlength=k).astype(float)
    props = counts / counts.sum()
    order = np.argsort(-props)
    return centers[order], props[order]


def closest_color_name(rgb_triplet):
    """Nearest named colour to an (R, G, B) triplet."""
    r, g, b = [int(v) for v in rgb_triplet]
    best, best_d = None, 1e18
    for name, (cr, cg, cb) in NAMED_COLORS.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_d:
            best_d, best = d, name
    return best


def palette_strip(centers, props, width=400, height=50):
    """Build a horizontal palette strip image, band widths ~ proportions."""
    strip = np.zeros((height, width, 3), dtype="uint8")
    x = 0
    for c, p in zip(centers, props):
        end = x + int(round(p * width))
        strip[:, x:end] = c
        x = end
    if x < width:
        strip[:, x:] = centers[-1]
    return strip
