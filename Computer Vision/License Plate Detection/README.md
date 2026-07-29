# License Plate Detection

Locate the license plate in a car image using classic computer vision, no deep learning. Edges plus contour geometry (a plate is a bright, roughly 4:1 rectangle) find the plate region, which you would then crop and pass to OCR.

## Problem statement

Given a photo of a car, find the rectangular license-plate region. This is the localization step that comes before any plate-reading (OCR) system.

## Dataset

No external dataset. The image is a synthetic car drawn with OpenCV: a body, windows, wheels, and a bright rectangular plate with text. The true plate box is stored in `data/plate_bbox.txt`. For a real-world set, see [Kaggle car-plate-detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection).

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

## How it works

`utils.py`:

1. **Edges**: grayscale, bilateral filter (smooths while keeping edges), Canny.
2. **Candidates** (`plate_candidates`): contour bounding boxes filtered by aspect ratio (3:1 to 6:1) and area. The area cap is what rejects the whole car body, which is also a wide rectangle; the aspect filter rejects the near-square windows.
3. **Selection** (`detect_plate`): among the candidates, pick the brightest region, since a license plate is a bright white rectangle. Return the box and the cropped plate.

## Project structure

```
License Plate Detection/
├── 01_edges_and_candidates.ipynb  # edges + candidate rectangles
├── 02_detect_plate.ipynb          # pick + crop the plate, IoU vs ground truth
├── utils.py
├── make_data.py                   # regenerates the synthetic car image
├── requirements.txt
└── data/                          # created by make_data.py (not committed)
    ├── car.jpg
    └── plate_bbox.txt
```

## Key findings (real output)

- The detector localizes the plate with **IoU 0.93** against the ground-truth box (detected `(201, 211, 128, 32)` vs true `(200, 210, 130, 34)`).
- Two filters do the work: the area cap removes the car body (a wide rectangle that would otherwise win on size), and the aspect-ratio band removes the near-square windows. Brightness then picks the white plate over anything left.
- No training, no labels, no GPU. Pure geometry.
- On real photos you would add plate-colour models and skew correction, then feed the crop to an OCR step, but the localization is this same classic pipeline.

## Tech stack

Python, OpenCV, NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_edges_and_candidates.ipynb
```
