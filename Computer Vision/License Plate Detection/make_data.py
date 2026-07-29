"""
Regenerate the synthetic car image in data/.

There is no external dataset. The image is a simple car drawn with OpenCV: a body,
windows, wheels, and a bright rectangular license plate with text. The plate is a
high-contrast ~4:1 rectangle, which is exactly the geometry the detector looks for.
Run once before the notebooks:

    python make_data.py
"""
import os
import cv2
import numpy as np

os.makedirs("data", exist_ok=True)

W, H = 500, 340
img = np.full((H, W, 3), (150, 170, 185), np.uint8)          # sky-ish background
cv2.rectangle(img, (0, 250), (W, H), (90, 90, 90), -1)       # road

# car body
cv2.rectangle(img, (70, 150), (430, 260), (40, 40, 160), -1)      # lower body
cv2.rectangle(img, (130, 95), (360, 155), (40, 40, 160), -1)      # cabin
cv2.rectangle(img, (145, 105), (250, 150), (200, 220, 230), -1)   # window 1
cv2.rectangle(img, (260, 105), (350, 150), (200, 220, 230), -1)   # window 2
cv2.circle(img, (140, 262), 30, (20, 20, 20), -1)                 # wheel
cv2.circle(img, (360, 262), 30, (20, 20, 20), -1)
cv2.circle(img, (140, 262), 12, (120, 120, 120), -1)
cv2.circle(img, (360, 262), 12, (120, 120, 120), -1)

# license plate: bright white rectangle with dark text, ~4:1 aspect
px, py, pw, ph = 200, 210, 130, 34
cv2.rectangle(img, (px, py), (px + pw, py + ph), (245, 245, 245), -1)
cv2.rectangle(img, (px, py), (px + pw, py + ph), (20, 20, 20), 2)
cv2.putText(img, "DL7C1234", (px + 6, py + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)

cv2.imwrite("data/car.jpg", img)
with open("data/plate_bbox.txt", "w") as f:
    f.write(f"{px},{py},{pw},{ph}")   # ground-truth plate box
print(f"wrote data/car.jpg with plate at ({px},{py},{pw},{ph}), aspect {pw/ph:.1f}:1")
