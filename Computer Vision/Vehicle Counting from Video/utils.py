"""
Utility functions for Vehicle Counting from Video.

Classic computer vision, no deep learning: background subtraction to find moving
blobs, a light centroid tracker to follow them across frames, and a line-crossing
rule to count each vehicle once. Runs on a synthetic traffic clip stored as a
frame sequence (regenerate with make_data.py).
"""

import glob
import os

import cv2
import numpy as np


def load_frames(frames_dir="data/frames"):
    """Return the traffic clip as a list of BGR frames, in order."""
    paths = sorted(glob.glob(f"{frames_dir}/*.png"))
    return [cv2.imread(p) for p in paths]


def ground_truth(path="data/ground_truth.txt"):
    """The true number of vehicles the synthetic clip contains."""
    with open(path) as f:
        return int(f.read().strip())


def _foreground_boxes(fgmask, min_w=25, min_h=30):
    """
    Clean a foreground mask and return bounding boxes of moving blobs. The
    min_h filter drops thin slivers (a vehicle entering/leaving the frame, or an
    internal window band) so each vehicle is one box, not several.
    """
    _, m = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_w and h >= min_h:
            boxes.append((x, y, w, h))
    return boxes


def count_vehicles(frames, line_y, lane_w=60):
    """
    Count vehicles crossing the horizontal line at `line_y`.

    Uses a per-lane state machine: a moving blob is bucketed into a lane by its
    x-centre; a lane counts once when a blob in it crosses below the line, and
    re-arms when the lane empties (the vehicle leaves the frame), so the next
    vehicle in that lane is counted too. Robust for well-separated lanes.

    Returns (count, overlays): overlays are BGR frames with the line, boxes, and
    running count drawn on, for visualization.
    """
    bg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
    lane_state = {}      # lane key -> "above" | "below"
    count = 0
    overlays = []
    for frame in frames:
        fg = bg.apply(frame)
        boxes = _foreground_boxes(fg)
        seen = set()
        for (x, y, w, h) in boxes:
            cx, cy = x + w // 2, y + h // 2
            key = round(cx / lane_w)
            seen.add(key)
            state = lane_state.get(key, "above")
            if state == "above" and cy >= line_y:
                count += 1
                lane_state[key] = "below"
            elif cy < line_y:
                lane_state[key] = "above"
        # re-arm any lane with no blob this frame (its vehicle has left)
        for key in list(lane_state):
            if key not in seen:
                lane_state[key] = "above"

        vis = frame.copy()
        cv2.line(vis, (0, line_y), (vis.shape[1], line_y), (0, 0, 255), 2)
        for (x, y, w, h) in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, f"count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        overlays.append(vis)
    return count, overlays


def to_rgb(bgr):
    """BGR -> RGB for matplotlib display."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
