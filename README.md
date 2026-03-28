<h1 align="center">Beginner Level Data Science Projects</h1>

<p align="center">
  <img src="assets/data-science.png" alt="Project Overview" width="150">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/tkarim45/Beginner-Data-Science-Projects/stargazers"><img src="https://img.shields.io/github/stars/tkarim45/Beginner-Data-Science-Projects" alt="Stars"></a>
  <a href="https://github.com/tkarim45/Beginner-Data-Science-Projects/network/members"><img src="https://img.shields.io/github/forks/tkarim45/Beginner-Data-Science-Projects" alt="Forks"></a>
  <a href="https://github.com/tkarim45/Beginner-Data-Science-Projects/issues"><img src="https://img.shields.io/github/issues/tkarim45/Beginner-Data-Science-Projects" alt="Issues"></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"></a>
</p>

<p align="center">
  A curated collection of beginner-friendly data science projects with real datasets, clear explanations, and working code. Learn by building.
</p>

---

## Table of Contents

- [Why This Repo?](#why-this-repo)
- [Learning Path](#learning-path)
- [All Projects](#all-projects)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

## Why This Repo?

This repository is designed for **anyone getting started with data science** -- students, career switchers, and self-learners. Each project is a standalone Jupyter notebook that you can clone and run immediately.

You will learn:
- **Data Cleaning and Preprocessing** -- preparing real-world messy data
- **Exploratory Data Analysis** -- visualizations and statistical insights
- **Machine Learning** -- classification, regression, and anomaly detection
- **Deep Learning** -- CNNs, transfer learning, and NLP models
- **Computer Vision** -- detection, recognition, and pose estimation

## Learning Path

Start from the top and work your way down. Projects are ordered by difficulty within each level.

### Level 1 -- Fundamentals

Get comfortable with pandas, sklearn, and basic ML workflows.

| # | Project | What You'll Learn | Category |
|---|---------|-------------------|----------|
| 1 | [Iris Flower Classification](Iris%20Flower%20Classification) | Image classification with CNNs, data loading | Classification |
| 2 | [Customer Churn](Customer%20Churn) | Logistic regression from scratch, prediction on new data | Classification |
| 3 | [Heart Failure Prediction](Heart%20Failure%20Prediction) | Feature analysis, multiple classifiers, model evaluation | Classification |
| 4 | [Rental Prices of AirBnb](Rental%20Prices%20of%20AirBnb) | Linear regression, outlier analysis, label encoding | Regression |

### Level 2 -- Text and NLP

Learn to work with text data, preprocessing pipelines, and NLP techniques.

| # | Project | What You'll Learn | Category |
|---|---------|-------------------|----------|
| 5 | [Message Spam Filtering](Message%20Spam%20Filtering) | TF-IDF, text preprocessing, SVM classification | NLP |
| 6 | [Cyber-Bullying Prediction](Cyber-Bullying%20Prediction) | NLP pipeline, GridSearchCV, model comparison | NLP |
| 7 | [Sentiment Analysis](Sentiment%20Analysis) | Logistic regression from scratch, Twitter data, NLTK | NLP |
| 8 | [AirBnb Reviews Sentimental Analysis](AirBnb%20reviews%20Sentimental%20Analysis) | Full NLP pipeline: preprocessing, ML, deep learning, LLMs | NLP |

### Level 3 -- Computer Vision and Deep Learning

Work with images, neural networks, and pre-trained models.

| # | Project | What You'll Learn | Category |
|---|---------|-------------------|----------|
| 9 | [Gender Classification](Gender%20Classification) | EfficientNetV2, transfer learning, Keras | Classification |
| 10 | [Face Detection](Face%20Detection) | Haar cascades, MTCNN, OpenCV | Computer Vision |
| 11 | [Face Recognition](Face%20Recognition) | LBPH algorithm, real-time webcam recognition | Computer Vision |
| 12 | [Eye Disease Detection](Eye%20Disease%20Detection) | ResNet34, data augmentation pipeline, medical imaging | Computer Vision |
| 13 | [Alzheimer Detection](Alzheimer%20Detection) | Clinical data analysis, Random Forest on medical data | Computer Vision |

### Level 4 -- Advanced Topics

Tackle more complex real-world problems.

| # | Project | What You'll Learn | Category |
|---|---------|-------------------|----------|
| 14 | [Network Intrusion Detection System](Network%20Intrusion%20Detection%20System) | Ensemble methods, XGBoost, KDD Cup dataset | Anomaly Detection |
| 15 | [Object Detection](Object%20Detection) | YOLOv8, Faster R-CNN, RetinaNet, Detectron2 | Computer Vision |
| 16 | [Pose Estimation](Pose%20Estimation) | YOLOv8, MediaPipe, activity classification | Computer Vision |
| 17 | [Robotics and Computer Integrated Manufacturing](Robotics%20and%20Computer%20Integrated%20Manufacturing) | MobileNetV2, transfer learning, industrial imaging | Robotics |

## All Projects

| # | Project | Category | Difficulty |
|---|---------|----------|------------|
| 1 | [Iris Flower Classification](Iris%20Flower%20Classification) | Classification | Beginner |
| 2 | [Customer Churn](Customer%20Churn) | Classification | Beginner |
| 3 | [Heart Failure Prediction](Heart%20Failure%20Prediction) | Classification | Beginner |
| 4 | [Rental Prices of AirBnb](Rental%20Prices%20of%20AirBnb) | Regression | Beginner |
| 5 | [Message Spam Filtering](Message%20Spam%20Filtering) | NLP | Beginner |
| 6 | [Cyber-Bullying Prediction](Cyber-Bullying%20Prediction) | NLP | Beginner |
| 7 | [Sentiment Analysis](Sentiment%20Analysis) | NLP | Intermediate |
| 8 | [AirBnb Reviews Sentimental Analysis](AirBnb%20reviews%20Sentimental%20Analysis) | NLP | Intermediate |
| 9 | [Gender Classification](Gender%20Classification) | Classification | Intermediate |
| 10 | [Face Detection](Face%20Detection) | Computer Vision | Intermediate |
| 11 | [Face Recognition](Face%20Recognition) | Computer Vision | Intermediate |
| 12 | [Eye Disease Detection](Eye%20Disease%20Detection) | Computer Vision | Intermediate |
| 13 | [Alzheimer Detection](Alzheimer%20Detection) | Computer Vision | Intermediate |
| 14 | [Network Intrusion Detection System](Network%20Intrusion%20Detection%20System) | Anomaly Detection | Advanced |
| 15 | [Object Detection](Object%20Detection) | Computer Vision | Advanced |
| 16 | [Pose Estimation](Pose%20Estimation) | Computer Vision | Advanced |
| 17 | [Robotics and Computer Integrated Manufacturing](Robotics%20and%20Computer%20Integrated%20Manufacturing) | Robotics | Advanced |

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Core Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Additional Libraries (by project type)

| Project Type | Install |
|-------------|---------|
| Deep Learning | `pip install tensorflow keras` |
| Computer Vision | `pip install opencv-python` |
| NLP | `pip install nltk` |
| Object Detection | `pip install ultralytics` |

Each project has its own `requirements.txt` for exact dependencies:

```bash
cd "Project Folder Name"
pip install -r requirements.txt
jupyter notebook
```

### Quick Start

```bash
git clone https://github.com/tkarim45/Beginner-Data-Science-Projects.git
cd Beginner-Data-Science-Projects

# Pick a project and run it
cd "Iris Flower Classification"
pip install -r requirements.txt
jupyter notebook
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) before submitting a PR.

**Quick rules:**
- One project per PR
- Include a README, requirements.txt, and working notebook
- Host large datasets externally (>10 MB)
- Do not commit model binaries

## License

This project is licensed under the [MIT License](LICENSE) -- use it freely for learning, teaching, or building.

---

<p align="center">
  If this repo helped you, consider giving it a star -- it helps others find it too.
</p>
