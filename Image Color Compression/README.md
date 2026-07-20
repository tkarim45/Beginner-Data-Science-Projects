# Image Color Compression (K-Means)

A beginner-level **unsupervised clustering** project that compresses an image's colour
palette using K-Means. Every pixel is a point in 3-D RGB space; K-Means finds `k`
representative colours and recolours each pixel to its nearest centroid, so the output
uses only `k` distinct colours instead of the tens of thousands in the original.

## Problem Statement

A typical photo contains tens of thousands of unique colours, but most are visually
redundant. Can we represent the same image with only a handful of colours, say 16, 
without an obvious loss of quality? This is **genuine unsupervised learning**: there is no
target column. K-Means partitions the pixel cloud in RGB space, and we then measure how
faithfully the `k`-colour reconstruction matches the original.

Quality is judged with **MSE** (reconstruction error, lower is better) and **PSNR** in
decibels (higher is better; > 30 dB is visually near-lossless), traded off against the
**compression ratio** of a `k`-colour palette + index map versus raw 24-bit RGB.

## Dataset

- **Source**: the two `load_sample_image` photos shipped with scikit-learn (`china.jpg`,
  `flower.jpg`), saved as PNGs in `data/`. Self-contained and fully reproducible, no
  external download or Kaggle account required.
- **Dimensions**: both images are **427 × 640 = 273,280 pixels**.
- **Unique colours**: `china.png` = **96,615**, `flower.png` = **62,941**.

| Item | Definition |
|---|---|
| Pixel matrix | Image reshaped to `(H*W, 3)` and scaled to `[0, 1]`, the input to K-Means |
| `k` | Number of clusters = number of colours in the compressed image |
| Centroid | A cluster centre = one colour in the learned palette |

K-Means is fit on a random **10,000-pixel sample** for speed, then **all 273,280 pixels**
are assigned to the learned centroids. The reshape is verified loss-less (round-trip
MSE = 0), so the only quality loss comes from clustering.

## Project Structure

```
Image Color Compression/
├── 01_eda.ipynb              # Explore images, colour counts, RGB-space scatter
├── 02_data_cleaning.ipynb    # Reshape image → pixel matrix, sampling, round-trip check
├── 03_model_building.ipynb   # K-Means compression across k, quality vs size metrics
├── utils.py                  # Reusable load / compress / metric helpers
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/
    ├── china.png             # Sample image 1 (427 × 640)
    └── flower.png            # Sample image 2 (427 × 640)
```

Run notebooks in order: `01_eda.ipynb` → `02_data_cleaning.ipynb` → `03_model_building.ipynb`.

## Results

All figures below are produced by executing `03_model_building.ipynb`, not assumed.
`bpp` = bits per pixel of the compressed representation (raw RGB is 24 bpp); `ratio` =
raw bits ÷ compressed bits.

### china.png (96,615 → k colours)

| k | MSE | PSNR (dB) | bpp | Compression ratio |
|---|---|---|---|---|
| 2 | 1285.186 | 17.041 | 1.000 | 24.0× |
| 4 | 456.907 | 21.533 | 2.000 | 12.0× |
| 8 | 211.036 | 24.887 | 3.001 | 8.0× |
| **16** | **114.991** | **27.524** | **4.001** | **6.0×** |
| 32 | 64.765 | 30.017 | 5.003 | 4.8× |
| 64 | 38.824 | 32.240 | 6.006 | 4.0× |

### flower.png (62,941 → k colours)

| k | MSE | PSNR (dB) | bpp | Compression ratio |
|---|---|---|---|---|
| 2 | 719.035 | 19.563 | 1.000 | 24.0× |
| 4 | 295.000 | 23.433 | 2.000 | 12.0× |
| 8 | 145.753 | 26.495 | 3.001 | 8.0× |
| **16** | **72.434** | **29.531** | **4.001** | **6.0×** |
| 32 | 42.407 | 31.856 | 5.003 | 4.8× |
| 64 | 25.667 | 34.037 | 6.006 | 4.0× |

## Key Findings

- **16 colours is the sweet spot for these photos.** At k = 16 the image is compressed
  **6×** (4 bpp vs 24 bpp) while reaching **27.5 dB (china)** and **29.5 dB (flower)**, 
  visually close to the original despite a 96,615 → 16 colour reduction.
- **PSNR rises steeply then flattens.** For china it climbs 17.0 → 24.9 → 30.0 dB going
  k = 2 → 8 → 32, but each doubling past k = 16 adds progressively less; the error is
  already small.
- **Doubling k costs ~1 bit per pixel.** bpp goes 1 → 2 → 3 → 4 → 5 → 6 for
  k = 2 → 4 → 8 → 16 → 32 → 64, because the index map needs `ceil(log2 k)` bits per pixel;
  the palette itself (`k × 24` bits) is negligible.
- **The trade-off is consistent across both images.** flower compresses a little better at
  every k (it has fewer unique colours to begin with), but the shape of the curve, steep
  gains up to k = 16, diminishing returns after, is identical.
- **Fitting on a 10,000-pixel sample is enough.** Assigning all 273,280 pixels to centroids
  learned from < 4 % of them produces these results, confirming the colour distribution is
  dense and well-sampled.

## Tech Stack

- numpy
- matplotlib
- scikit-learn

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
