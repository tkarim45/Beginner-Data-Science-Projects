# AirBnb Reviews Sentimental Analysis

Analyzes AirBnb user reviews to classify sentiment using a multi-stage pipeline: data preprocessing, classical machine learning, deep learning, and LLM-based approaches. Includes an interactive dashboard for exploring results and visualizations.

## Dataset

AirBnb user reviews (`Dataset/User Review/User_reviews.csv`), cleaned and stored in `Dataset/Cleaned Data/reviews.csv`.

## Tech Stack

- pandas, numpy
- scikit-learn (Logistic Regression, Random Forest, Decision Tree, KNN, AdaBoost, Gradient Boosting, SVM)
- TF-IDF vectorization
- Deep learning (notebook 3)
- LLM-based classification (notebook 4)
- Dash/Plotly for the interactive dashboard
- joblib/pickle for model persistence

## Results

The project compares multiple ML approaches for sentiment classification on AirBnb reviews, including Logistic Regression, Random Forest, Decision Tree, KNN, AdaBoost, Gradient Boosting, and SVM using TF-IDF features, as well as deep learning and LLM-based methods. See notebook for detailed results.

## How to Run

1. Install dependencies: `pip install pandas numpy scikit-learn plotly dash`
2. Run `1_dataPreprocessing.ipynb` to clean and prepare the review data.
3. Run `2_machineLearning.ipynb` to train and evaluate ML models.
4. Run `3_deepLearning.ipynb` for deep learning experiments.
5. Run `4_LLM.ipynb` for LLM-based sentiment analysis.
6. Launch the dashboard: `python dashboard.py`
