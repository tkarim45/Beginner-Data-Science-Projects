# Face Recognition

Recognizes known faces in a live webcam feed using OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer. A training script processes labeled face images to build a recognition model, and a separate script performs real-time face, eye, and smile detection with identity labeling via the webcam.

## Dataset

Custom face images organized in the `images/` folder with one subfolder per person. Haar Cascade XML classifiers for face, eye, and smile detection are stored in the `cascades/` folder.

## Tech Stack

- OpenCV (with `cv2.face` contrib module)
- NumPy
- Pillow
- pickle

## Results

Demonstrates real-time face, eye, and smile detection with identity labeling using OpenCV's LBPH recognizer via a live webcam feed. The system trains on labeled face images and performs recognition in real time.

## How to Run

1. Install dependencies: `pip install opencv-python opencv-contrib-python numpy Pillow`
2. Place training images in `images/`, with a subfolder for each person's name.
3. Update file paths in `faces-train.py` and `Face Recognition.py` to match your local setup.
4. Run `python faces-train.py` to train the LBPH recognizer and generate `trainner.yml` and `labels.pickle`.
5. Run `python "Face Recognition.py"` to start real-time face recognition via webcam. Press **q** to quit.
