# Message Spam Filtering

Classifies SMS messages as spam or ham (not spam) using NLP preprocessing and machine learning. Text is cleaned by removing stopwords and applying stemming, then vectorized with TF-IDF. Multiple classifiers (SVM, Naive Bayes, Decision Tree) are trained and evaluated.

## Dataset

`spam.csv` -- a labeled SMS spam dataset with message text and spam/ham labels (commonly known as the SMS Spam Collection).

## Tech Stack

- Pandas
- NLTK
- scikit-learn (TfidfVectorizer, SVM, classification metrics)

## Results

Three classifiers are evaluated on the SMS spam dataset using TF-IDF features: SVM (sigmoid kernel), Multinomial Naive Bayes, and Decision Tree. See notebook for detailed accuracy, precision, recall, and F1-score comparisons.

## How to Run

1. Ensure `spam.csv` is in the project directory.
2. Install dependencies: `pip install pandas nltk scikit-learn`
3. Download NLTK data: run `nltk.download('stopwords')` in Python.
4. Open and run `messageSpamFiltering.ipynb` in Jupyter Notebook, or run `python messageSpamFiltering.py`.
