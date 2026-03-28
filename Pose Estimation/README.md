# Pose Estimation and Activity Classification

Uses pose estimation models to extract body keypoints from video frames and classifies human activities (Cooking, Dancing, Gym) using machine learning. Compares three pose estimation approaches -- YOLOv8, MediaPipe Pose, and MediaPipe Holistic -- paired with SVM, Random Forest, and XGBoost classifiers.

## Dataset

Video frames organized into class folders (Cooking, Dancing, Gym). Videos are loaded from Google Drive and processed frame-by-frame to extract pose keypoints.

## Tech Stack

- OpenCV (cv2)
- Ultralytics (YOLOv8 pose model)
- MediaPipe (Pose and Holistic models)
- scikit-learn (SVM, Random Forest, train/test split, metrics)
- NumPy, Pandas
- Matplotlib, Seaborn
- Google Colab

## Results

Activity classification accuracy (Cooking, Dancing, Gym) using Random Forest with different pose estimation backends:

| Pose Model | SVM | Random Forest | XGBoost |
|------------|-----|---------------|---------|
| YOLOv8 | 80.0% | 94.3% | 88.6% |
| MediaPipe Pose | 81.8% | 94.6% | 90.9% |
| MediaPipe Holistic | 81.8% | 94.6% | 90.9% |

Random Forest consistently performs best across all pose estimation approaches.

## How to Run

1. Open `PoseEstimation.ipynb` in Google Colab.
2. Mount your Google Drive and place video data in the expected directory structure with class subfolders (Cooking, Dancing, Gym).
3. Run all cells to install dependencies (`ultralytics`, `mediapipe`), extract frames, compute pose keypoints, train classifiers, and evaluate results.
