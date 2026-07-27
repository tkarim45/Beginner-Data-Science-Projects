"""
Regenerate data/document_photo.jpg.

There is no external dataset for this project. The test image is synthetic: a
white invoice page rendered with OpenCV, then perspective-warped onto a grey
background so it looks like a phone photo taken at an angle. Run once before the
notebooks:

    python make_data.py
"""
import os
import cv2
import numpy as np

os.makedirs("data", exist_ok=True)

# Build a white "page" with invoice-like text.
pw, ph = 520, 700
page = np.full((ph, pw, 3), 255, np.uint8)
cv2.putText(page, "INVOICE", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
cv2.line(page, (40, 100), (pw - 40, 100), (0, 0, 0), 2)
lines = [
    "Bill To: Acme Corp", "Date: 2026-07-27", "",
    "Item            Qty    Price",
    "Widget A         2    $40.00",
    "Widget B         1    $15.50",
    "Service fee      -    $25.00", "",
    "Subtotal             $80.50",
    "Tax (8%)              $6.44",
    "Total                $86.94",
]
y = 150
for ln in lines:
    cv2.putText(page, ln, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    y += 45

# Warp the page onto a grey canvas at a skew.
W, H = 900, 1000
canvas = np.full((H, W, 3), 150, np.uint8)
src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
dst = np.float32([[180, 120], [760, 60], [820, 860], [120, 760]])
M = cv2.getPerspectiveTransform(src, dst)
warp = cv2.warpPerspective(page, M, (W, H), borderValue=(150, 150, 150))
mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (W, H))
canvas[mask > 0] = warp[mask > 0]

cv2.imwrite("data/document_photo.jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
print("wrote data/document_photo.jpg", canvas.shape)
