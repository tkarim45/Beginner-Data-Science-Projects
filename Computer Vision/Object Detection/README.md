# Object Detection

Performs object detection using three model architectures: YOLOv8, Faster R-CNN, and RetinaNet. YOLOv8 is trained via the Ultralytics library, while Faster R-CNN and RetinaNet use Facebook's Detectron2 framework. Models are trained on a custom dataset sourced from Roboflow and evaluated with confusion matrices, precision-recall curves, and F1 curves.

## Dataset

A custom object detection dataset downloaded from Roboflow in YOLOv8 and COCO formats. The dataset is used for training and evaluation across all three model architectures.

## Tech Stack

- Ultralytics (YOLOv8)
- Detectron2 (Faster R-CNN, RetinaNet)
- PyTorch
- OpenCV (cv2)
- Roboflow
- MediaPipe
- Matplotlib
- Plotly
- scikit-learn
- Pandas
- NumPy

## Results

YOLOv8 achieves mAP50 of 0.426 and mAP50-95 of 0.299 after 5 epochs of training. Faster R-CNN and RetinaNet are trained using Detectron2 with decreasing loss over training iterations. All models are evaluated with confusion matrices and precision-recall curves.

## How to Run

1. This project is designed to run in **Google Colab** with GPU support.
2. Open `Yolov8.ipynb` for the YOLOv8 training and inference pipeline.
3. Open `RetinaNet_and_Faster_R_CNN.ipynb` for the Detectron2-based Faster R-CNN and RetinaNet pipelines.
4. Follow the in-notebook instructions to install dependencies (ultralytics, detectron2, roboflow) and download the dataset.
