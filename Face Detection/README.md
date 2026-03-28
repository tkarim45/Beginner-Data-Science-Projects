# Face Detection

Detects faces in images using two approaches: OpenCV Haar Cascades and MTCNN (Multi-task Cascaded Convolutional Networks). Draws bounding boxes around detected faces and compares the results of both methods.

## Dataset

A collection of 1,800 face images stored in the `Dataset/` folder. The Haar Cascade XML classifier file (`haarcascade_frontalface_default.xml`) is downloaded from Kaggle within the notebook.

## Tech Stack

- OpenCV
- MTCNN
- TensorFlow / Keras
- NumPy
- pandas
- matplotlib

## Results

Demonstrates face detection using Haar Cascades and MTCNN on a dataset of 1,800 images. Both methods draw bounding boxes around detected faces. MTCNN generally provides more accurate detections with fewer false positives.

## How to Run

1. Install dependencies: `pip install opencv-python mtcnn tensorflow numpy pandas matplotlib tqdm`
2. Place face images in the `Dataset/` folder.
3. Open `face-recognition.ipynb` and run all cells.
