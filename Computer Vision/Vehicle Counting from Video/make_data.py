"""
Regenerate the synthetic traffic clip in data/.

There is no external dataset. The clip is generated deterministically: a grey road
with a fixed number of coloured rectangles ("vehicles") driving downward across a
counting line at staggered times and speeds. Frames are written to data/frames/,
and the true vehicle count to data/ground_truth.txt. Run once before the notebooks:

    python make_data.py
"""
import os
import cv2
import numpy as np

os.makedirs("data/frames", exist_ok=True)

W, H = 480, 320
N_FRAMES = 280
ROAD = (70, 70, 70)

# Four well-separated lanes, two vehicles each (staggered in time so a lane is
# never occupied by two vehicles at once). Each: (start_frame, x, w, h, speed, BGR).
LANE_X = [40, 160, 280, 400]
# two vehicles per lane, the second starting well after the first has left frame
VEHICLES = [
    (0,   LANE_X[0], 50, 68, 3.0, (0, 0, 200)),
    (20,  LANE_X[1], 50, 66, 3.0, (200, 120, 0)),
    (40,  LANE_X[2], 50, 70, 3.0, (0, 180, 0)),
    (60,  LANE_X[3], 50, 68, 3.0, (180, 0, 180)),
    (140, LANE_X[0], 50, 66, 3.0, (0, 160, 200)),
    (150, LANE_X[1], 50, 70, 3.0, (200, 200, 0)),
    (160, LANE_X[2], 50, 68, 3.0, (120, 120, 200)),
    (170, LANE_X[3], 50, 66, 3.0, (60, 200, 120)),
]
GROUND_TRUTH = len(VEHICLES)

# lane divider markings (static) so the scene is not a flat colour
def draw_background():
    img = np.full((H, W, 3), ROAD, np.uint8)
    for x in (100, 220, 340, 460):
        for y in range(0, H, 40):
            cv2.rectangle(img, (x - 3, y), (x + 3, y + 22), (200, 200, 200), -1)
    return img


for f in range(N_FRAMES):
    frame = draw_background()
    for (start, x, w, h, spd, color) in VEHICLES:
        if f < start:
            continue
        y = int((f - start) * spd) - h  # enter from the top
        if y > H:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        # horizontal texture stripes so the whole moving body registers as
        # foreground (a flat-colour rectangle only shows motion at its edges).
        for sy in range(y + 6, y + h - 4, 8):
            cv2.line(frame, (x + 4, sy), (x + w - 4, sy),
                     tuple(int(c * 0.6) for c in color), 2)
        cv2.rectangle(frame, (x + 6, y + 8), (x + w - 6, y + 20), (230, 230, 230), -1)  # windshield
    cv2.imwrite(f"data/frames/{f:03d}.png", frame)

with open("data/ground_truth.txt", "w") as fh:
    fh.write(str(GROUND_TRUTH))

print(f"wrote {N_FRAMES} frames to data/frames/, ground truth = {GROUND_TRUTH} vehicles")
