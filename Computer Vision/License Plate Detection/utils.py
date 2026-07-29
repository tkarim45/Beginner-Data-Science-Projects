"""
Utility functions for License Plate Detection.

Classic computer vision, no deep learning: find the rectangular license-plate
region in a car photo using edges and contour geometry (a plate is a bright,
roughly 2:1-to-6:1 rectangle). Runs on a synthetic car image (regenerate with
make_data.py).
"""

import cv2
import numpy as np


def load_image(path="data/car.jpg"):
    """Load a car image as RGB."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_edges(rgb):
    """Grayscale -> bilateral smooth (keeps edges) -> Canny."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    return cv2.Canny(gray, 30, 200)


def plate_candidates(edges, ar_range=(3.0, 6.0), area_range=(800, 12000)):
    """
    Return candidate plate boxes (x, y, w, h, area, ar) whose bounding-rect
    aspect ratio and area are plate-like. The upper area bound is what rejects
    the whole car body (which also happens to be a wide rectangle).
    """
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        ar, area = w / h, w * h
        if ar_range[0] <= ar <= ar_range[1] and area_range[0] <= area <= area_range[1]:
            out.append((x, y, w, h, area, ar))
    out.sort(key=lambda t: t[4], reverse=True)
    return out


def detect_plate(rgb):
    """
    Locate the plate. Among aspect/area candidates, pick the brightest region
    (a license plate is a bright white rectangle). Returns dict with edges, all
    candidates, the chosen box (x, y, w, h) or None, and the cropped plate.
    """
    edges = find_edges(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cands = plate_candidates(edges)
    out = {"edges": edges, "candidates": cands, "box": None, "crop": None}
    best, best_bright = None, -1
    for (x, y, w, h, area, ar) in cands:
        bright = float(gray[y:y + h, x:x + w].mean())
        if bright > best_bright:
            best_bright, best = bright, (x, y, w, h)
    if best is not None:
        x, y, w, h = best
        out["box"] = best
        out["crop"] = rgb[y:y + h, x:x + w]
    return out
