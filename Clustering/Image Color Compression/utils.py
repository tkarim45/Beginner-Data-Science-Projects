"""
Image Color Compression — Utility Functions
Compress an image's colour palette with K-Means clustering on pixel RGB values.
Each pixel is a point in 3-D RGB space; K-Means finds k representative colours
(centroids) and every pixel is recoloured to its nearest centroid, so the output
uses only k distinct colours.

This is an unsupervised project — no target variable.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_image(filepath):
    """
    Load an image from disk into a (H, W, 3) uint8 RGB array.
    Uses matplotlib so no extra image library is required.
    PNGs are returned by matplotlib as float in [0, 1]; convert to uint8.
    """
    img = plt.imread(filepath)
    if img.dtype != np.uint8:
        img = (img[..., :3] * 255).round().astype(np.uint8)
    else:
        img = img[..., :3]
    return img


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def image_to_pixels(img):
    """
    Reshape an (H, W, 3) image into an (H*W, 3) array of pixels scaled to [0, 1].
    Returns: X (float pixel matrix), original shape (H, W, 3).
    """
    h, w, c = img.shape
    X = img.reshape(-1, c).astype(np.float64) / 255.0
    return X, img.shape


def pixels_to_image(X, shape):
    """Reshape a flat [0,1] pixel matrix back to a uint8 (H, W, 3) image."""
    arr = (np.clip(X, 0, 1) * 255).round().astype(np.uint8)
    return arr.reshape(shape)


# ─────────────────────────────────────────────
# Compression (K-Means)
# ─────────────────────────────────────────────

def compress_image(img, k, sample_size=10000, random_state=42):
    """
    Compress an image to k colours with K-Means.
      - Fit K-Means on a random sample of pixels (fast), then assign ALL pixels.
      - Replace every pixel with its cluster centroid colour.
    Returns: recoloured uint8 image, fitted KMeans, labels for all pixels.
    """
    X, shape = image_to_pixels(img)
    rng = np.random.RandomState(random_state)
    if sample_size and sample_size < len(X):
        idx = rng.choice(len(X), sample_size, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X
    km = KMeans(n_clusters=k, n_init=4, random_state=random_state)
    km.fit(X_fit)
    labels = km.predict(X)
    X_compressed = km.cluster_centers_[labels]
    return pixels_to_image(X_compressed, shape), km, labels


# ─────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────

def mse(original, compressed):
    """Mean squared error between two uint8 images (per-pixel, all channels)."""
    a = original.astype(np.float64)
    b = compressed.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def psnr(original, compressed):
    """
    Peak Signal-to-Noise Ratio in dB. Higher = closer to original.
    PSNR = 10 * log10(MAX^2 / MSE), MAX = 255 for 8-bit images.
    """
    err = mse(original, compressed)
    if err == 0:
        return float("inf")
    return float(10 * np.log10((255.0 ** 2) / err))


def compression_stats(img, k):
    """
    Bit-level compression for a k-colour palette image vs raw 24-bit RGB.
      raw        = H*W * 24 bits
      compressed = palette (k * 24 bits) + index map (H*W * ceil(log2 k) bits)
    Returns dict with raw_bits, compressed_bits, ratio, bpp (bits per pixel).
    """
    h, w = img.shape[:2]
    n = h * w
    bits_per_index = max(1, int(np.ceil(np.log2(k))))
    raw_bits = n * 24
    compressed_bits = k * 24 + n * bits_per_index
    return {
        "k": k,
        "bits_per_index": bits_per_index,
        "raw_bits": raw_bits,
        "compressed_bits": compressed_bits,
        "ratio": raw_bits / compressed_bits,
        "bpp": compressed_bits / n,
    }


def evaluate_k_range(img, k_values, sample_size=10000, random_state=42):
    """
    Compress `img` for each k in k_values and collect quality + size metrics.
    Returns a list of dicts (one per k): k, n_colors, mse, psnr, ratio, bpp.
    """
    import pandas as pd
    rows = []
    for k in k_values:
        comp, km, labels = compress_image(img, k, sample_size, random_state)
        stats = compression_stats(img, k)
        rows.append({
            "k": k,
            "n_colors": len(np.unique(labels)),
            "mse": round(mse(img, comp), 3),
            "psnr": round(psnr(img, comp), 3),
            "ratio": round(stats["ratio"], 3),
            "bpp": round(stats["bpp"], 3),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def count_unique_colors(img):
    """Number of distinct RGB triples in an image."""
    return len(np.unique(img.reshape(-1, img.shape[-1]), axis=0))


def plot_comparison(original, compressed, k, ax=None):
    """Show a compressed image with its colour count in the title."""
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(compressed)
    ax.set_title(f"k = {k} colours")
    ax.axis("off")
    return ax
