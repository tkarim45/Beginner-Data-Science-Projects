# AirBnB Rental Price Prediction

Predicts AirBnB rental prices based on listing features such as neighbourhood group, room type, location, minimum nights, and availability. Performs data preprocessing, outlier analysis, label encoding, and trains a Linear Regression model.

## Dataset

`dataSP23.csv` -- contains 27,379 AirBnB listings with columns including neighbourhood group, latitude, longitude, room type, price, minimum nights, host listing count, and availability.

## Tech Stack

- Pandas
- NumPy
- scikit-learn (LinearRegression, LabelEncoder, StandardScaler, train_test_split, r2_score)
- Matplotlib

## Results

The Linear Regression model achieves an R2 score of 0.14 on the test set. The low score is attributed to significant outliers in the dataset, which negatively impact model performance.

## How to Run

1. Ensure `dataSP23.csv` is in the same directory as the notebook.
2. Open `AirBnB.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells to preprocess data, train the model, and view predictions and visualizations.
