# Computer Vision

Image tasks: a from-scratch CNN, frozen-ResNet18 transfer learning, and classic-OpenCV pipelines. Sources chosen to run offline on a laptop.

[Back to all projects](../README.md)

| Project | Description |
|---|---|
| [Handwritten Digit Recognition](Handwritten%20Digit%20Recognition) | From-scratch PyTorch CNN (~207k params) on an MNIST subset, 97.2% test accuracy. |
| [Fruit Classification](Fruit%20Classification) | Frozen ResNet18 features + LogReg on 8 fruits-360 classes, 100% test accuracy. |
| [Animal Species Classification](Animal%20Species%20Classification) | Same recipe on 6 messy CIFAR-10 animals, 75.2% (deliberate hard contrast to Fruit). |
| [Document Scanner](Document%20Scanner) | Classic OpenCV: Canny -> largest 4-corner contour -> homography warp -> adaptive threshold. |
| [Color Detection](Color%20Detection) | K-Means dominant-colour extraction + nearest-named-colour lookup. |
| [Alzheimer Detection](Alzheimer%20Detection) | Medical-imaging classification. |
| [Eye Disease Detection](Eye%20Disease%20Detection) | ResNet34 with a data-augmentation pipeline, medical imaging. |
| [Face Detection](Face%20Detection) | Haar cascades, MTCNN, OpenCV. |
| [Face Recognition](Face%20Recognition) | LBPH algorithm, real-time webcam recognition. |
| [Object Detection](Object%20Detection) | YOLOv8, Faster R-CNN, RetinaNet, Detectron2. |
| [Pose Estimation](Pose%20Estimation) | YOLOv8, MediaPipe, activity classification. |
| [Plant Disease CNNs](Plant%20Disease%20CNNs) | EfficientNet plant-disease classifier with an app. |
| [Gender Classification](Gender%20Classification) | EfficientNetV2 transfer learning on face images, ~91% accuracy. |

_13 projects in this category._
