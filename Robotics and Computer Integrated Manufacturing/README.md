# Weld Seam Classification

Classifies weld seam images into four categories (seam1, seam2, seam3, seam4) using transfer learning with MobileNetV2. Includes data augmentation, model training, confusion matrix evaluation, and seam width analysis via OpenCV edge detection.

## Dataset

2,000 weld seam images (500 per class) stored in the `training dataset/` folder with subfolders for each seam type. A `label color folder/` contains corresponding color-labeled reference images. The accompanying PDF paper provides the research context for the classification task.

## Tech Stack

- TensorFlow / Keras
- TensorFlow Hub (MobileNetV2 feature extractor)
- OpenCV (cv2)
- scikit-learn (confusion matrix)
- Pandas, NumPy
- Matplotlib, Seaborn

## Results

The MobileNetV2 transfer learning model achieves up to 98.75% validation accuracy on weld seam classification (4 classes) after 21 epochs of training. A confusion matrix is used for detailed per-class evaluation.

## How to Run

1. Ensure the `training dataset/` and `label color folder/` directories are in the same location as the notebook.
2. Open `main.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells to train the MobileNetV2-based classifier, evaluate with a confusion matrix, and visualize seam width distributions.
