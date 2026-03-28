# Customer Churn Prediction

Predicts whether customers of a marketing agency will churn (stop using the service) using logistic regression implemented from scratch. Includes custom sigmoid, cost function, and gradient descent. The trained model is then used to predict churn risk for new incoming customers.

## Dataset

- - `customer_churn.csv` -- historical customer data with fields: Name, Age, Total_Purchase, Account_Manager, Years, Num_Sites, Onboard_date, Location, Company, Churn
- `new_customers_1.csv` -- new customers to predict churn for (no labels)

## Tech Stack

- pandas, numpy
- matplotlib, seaborn
- scikit-learn (train_test_split, StandardScaler)
- Custom logistic regression (sigmoid, cost function, gradient descent implemented manually)

## Results

The custom logistic regression model achieves 88.9% accuracy on the test set.

## How to Run

1. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
2. Place `customer_churn.csv` and `new_customers_1.csv` in the project directory.
3. Run `customerChurn.ipynb` from start to finish. Predictions for new customers are displayed at the end.
