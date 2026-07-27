# Iris Flower Classification

Classifies images of flowers (daisy, dandelion, rose, sunflower, tulip) into five categories using a Convolutional Neural Network (CNN). The model is trained on resized 150x150 images and evaluated with accuracy, classification reports, and ROC curves.

## Dataset

This project uses the **Flowers Recognition** dataset from Kaggle:
[kaggle.com/datasets/alxmamaev/flowers-recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

It contains ~4,300 images across five classes (daisy, dandelion, rose, sunflower, tulip).
The images are **not committed to this repository** (to keep it lightweight), download them
and place them under a local `flowers/` directory with one labeled subdirectory per class:

```
Iris Flower Classification/
└── flowers/
    ├── daisy/
    ├── dandelion/
    ├── rose/
    ├── sunflower/
    └── tulip/
```

You can fetch the dataset with the Kaggle CLI:

```bash
kaggle datasets download -d alxmamaev/flowers-recognition -p "Iris Flower Classification" --unzip
```

## Tech Stack

- NumPy
- Pandas
- OpenCV (cv2)
- Matplotlib
- Seaborn
- scikit-learn
- Keras / TensorFlow

## Results

The CNN achieves approximately 50% accuracy on the test set (5 flower classes). Training accuracy reaches ~83% by epoch 10, but validation accuracy plateaus around 50-54%, indicating overfitting. See notebook for classification reports and ROC curves.

## How to Run

1. Download the dataset (see **Dataset** above) so the `flowers/` directory is present with subdirectories for each flower type.
2. Install dependencies: `pip install numpy pandas opencv-python matplotlib seaborn scikit-learn tensorflow`
3. Open and run `Iris_Flower.ipynb` in Jupyter Notebook or JupyterLab.
