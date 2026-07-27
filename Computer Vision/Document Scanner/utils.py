"""
Utility functions for the Document Scanner project.

A classic-CV pipeline (no deep learning): find the page in a photo, correct its
perspective, and produce a clean, thresholded "scanned" image.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_image(path):
    """Load an image as RGB (OpenCV loads BGR by default)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Edge / contour detection
# ---------------------------------------------------------------------------
def find_edges(rgb, blur=5, low=50, high=150):
    """Grayscale -> blur -> Canny edges, then dilate to close gaps."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, low, high)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    return edges


def find_document_contour(edges):
    """
    Return the 4-point contour of the largest quadrilateral in the edge map,
    or None if no 4-corner shape is found. Points are (N, 1, 2) as OpenCV gives.
    """
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None


# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------
def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left  = smallest x+y
    rect[2] = pts[np.argmax(s)]      # bottom-right = largest x+y
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]      # top-right = smallest y-x
    rect[3] = pts[np.argmax(d)]      # bottom-left = largest y-x
    return rect


def four_point_transform(rgb, quad):
    """Warp the quad region of `rgb` to a top-down rectangle."""
    rect = order_points(quad)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(rgb, M, (maxW, maxH))


def to_scanned(warped):
    """Adaptive-threshold a warped page to a clean black-on-white 'scan'."""
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )


def scan(rgb):
    """
    Full pipeline. Returns a dict with every intermediate stage so the notebook
    can visualize each step. `scanned` is None if no 4-corner page was found.
    """
    edges = find_edges(rgb)
    quad = find_document_contour(edges)
    out = {"edges": edges, "quad": quad, "warped": None, "scanned": None}
    if quad is not None:
        warped = four_point_transform(rgb, quad)
        out["warped"] = warped
        out["scanned"] = to_scanned(warped)
    return out
