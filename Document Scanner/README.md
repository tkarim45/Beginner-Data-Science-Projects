# Document Scanner

Turn a photo of a page into a clean, flattened "scan" using classic computer vision, no deep learning. This is the same core trick mobile scanner apps use: find the page, correct its perspective, and binarize it.

## Problem statement

You snap a photo of a receipt or a printed page. It comes out skewed, at an angle, sitting on a cluttered background. The goal is to detect the page automatically, warp it to a flat top-down view, and threshold it into a crisp black-on-white document.

## Dataset

No external dataset. The test image is synthetic: a white invoice page rendered with OpenCV, then perspective-warped onto a grey background so it looks like a phone photo taken at an angle. For a real-world scanned-document benchmark, see [SmartDoc / Zenodo 3966026](https://zenodo.org/record/3966026).

The `data/` folder is not committed. Regenerate the test image before running the notebooks:

```
python make_data.py
```

Swap in your own photo (any page against a contrasting background) and the pipeline runs the same way.

## How it works

The pipeline is four classic-CV stages, all in `utils.py`:

1. **Edges**: grayscale, Gaussian blur, Canny, then dilate to close small gaps.
2. **Contour**: keep the largest contours, approximate each to a polygon, take the first with exactly 4 corners. That is the page.
3. **Perspective warp**: order the 4 corners (top-left, top-right, bottom-right, bottom-left), compute a homography, and warp to a flat rectangle.
4. **Scan**: adaptive threshold the flattened page to drop the background to white and keep the text.

## Project structure

```
Document Scanner/
├── 01_edge_and_contour.ipynb      # edges + 4-corner page detection
├── 02_perspective_and_scan.ipynb  # homography warp + adaptive threshold
├── utils.py                       # the full pipeline
├── make_data.py                   # regenerates the synthetic test image
├── requirements.txt
└── data/                          # created by make_data.py (not committed)
    └── document_photo.jpg         # skewed "photographed page"
```

## Key findings (real output)

- The detector recovers a single 4-corner contour that covers about **51.6%** of the frame (area 464,567 px of 900,000), which is the page boundary.
- Ordering the corners and applying the homography flattens the slanted photo into a rectangular top-down page.
- Adaptive thresholding then removes the grey background cleanly and leaves the text readable.
- No training, no labels, no GPU. Pure OpenCV geometry.

## Tech stack

Python, OpenCV, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_edge_and_contour.ipynb
```

Run `01` for the detection half, then `02` for the warp and scan.
