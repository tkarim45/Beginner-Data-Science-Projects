# Twitter Sentiment Analysis

Implements logistic regression from scratch to classify tweets as positive or negative. Covers the sigmoid function, gradient descent, feature extraction (positive/negative word frequency counts), and model evaluation on the NLTK Twitter dataset. Achieves approximately 99.5% accuracy.

## Dataset

NLTK `twitter_samples` corpus -- 5,000 positive tweets and 5,000 negative tweets, split 80/20 for training and testing.

## Tech Stack

- NumPy
- Pandas
- NLTK (twitter_samples, stopwords)
- Custom utilities (utils.py for tweet processing and frequency dictionary building)

## Results

The logistic regression model (implemented from scratch) achieves 99.5% accuracy on the test set of 2,000 tweets.

## How to Run

1. Install dependencies: `pip install nltk numpy pandas`.
2. Download NLTK data: `nltk.download('twitter_samples')` and `nltk.download('stopwords')`.
3. Open `MT.ipynb` (exercise notebook with blanks to fill in) or `sentimentalAnalysis.ipynb` in Jupyter Notebook.
4. Run all cells to build the frequency dictionary, train logistic regression, and evaluate on the test set.
