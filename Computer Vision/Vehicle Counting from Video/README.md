# Vehicle Counting from Video

Count vehicles crossing a line in a traffic clip using classic computer vision, no deep learning. Background subtraction isolates moving vehicles; a per-lane line-crossing rule counts each one exactly once.

## Problem statement

Given a fixed-camera traffic video, count how many vehicles pass a point on the road. This is the core of traffic-flow monitoring and is a natural fit for motion-based classic CV.

## Dataset

No external dataset. The clip is synthetic and generated deterministically: a grey road with textured coloured rectangles ("vehicles") driving downward across a counting line, stored as a frame sequence in `data/frames/`. The true count is written to `data/ground_truth.txt`.

The `data/` folder is not committed. Regenerate it before running the notebooks:

```
python make_data.py
```

## How it works

`utils.py`:

1. **Background subtraction** (`cv2.createBackgroundSubtractorMOG2`) learns the static road and flags moving pixels.
2. **Blob cleanup** (`_foreground_boxes`): threshold, morphological open/close, contour bounding boxes, with a minimum-height filter that drops thin slivers (entering/leaving vehicles, internal bands).
3. **Per-lane counting** (`count_vehicles`): each blob is bucketed into a lane by its x-centre; a lane counts once when a blob crosses below the line, and re-arms when the lane empties, so the next vehicle in that lane is also counted.

## Project structure

```
Vehicle Counting from Video/
├── 01_motion_detection.ipynb   # frames + MOG2 foreground mask
├── 02_counting.ipynb           # run counter, count vs ground truth
├── utils.py
├── make_data.py                # regenerates the synthetic clip
├── requirements.txt
└── data/                       # created by make_data.py (not committed)
    ├── frames/*.png
    └── ground_truth.txt
```

## Key findings (real output)

- The counter recovers **all 8 vehicles exactly (8 / 8)** on the synthetic clip.
- One implementation detail mattered: a flat-colour rectangle only shows motion at its leading and trailing edges (the interior pixels do not change frame to frame), so a fully-entered car shrinks to thin bands and gets filtered out. Adding light texture stripes to the vehicles makes the whole body register as one moving blob, which is realistic since real vehicles have texture.
- The per-lane state machine (count on crossing, re-arm on empty) is what lets a lane count a second vehicle without double-counting the first.
- On real footage you would add shadow suppression and a stronger tracker for occlusion, but the counting logic is identical.

## Tech stack

Python, OpenCV (MOG2 background subtractor), NumPy, Matplotlib.

## Getting started

```
pip install -r requirements.txt
python make_data.py
jupyter notebook 01_motion_detection.ipynb
```
