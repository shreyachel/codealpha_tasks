# Task 1: Credit Scoring Model

## Objective

Predict whether a customer represents a lower-risk or higher-risk credit case using financial and payment-history information.

## Features

The demonstration dataset contains:

- age
- annual_income
- employment_years
- debt
- number_of_open_accounts
- payment_history_score
- late_payments
- credit_utilization
- loan_amount
- previous_defaults
- savings
- target: creditworthy (0/1)

## Models

Three classifiers are compared:

- Logistic Regression
- Decision Tree
- Random Forest

The workflow includes:

1. Data generation/loading
2. Basic preprocessing
3. Feature engineering
4. Train/test split
5. Model training
6. Accuracy, precision, recall and F1-score
7. ROC-AUC
8. Confusion matrix
9. ROC curve
10. Feature importance for the Random Forest
11. Saving the best model

## Dataset note

For a classroom demonstration, the script automatically creates a reproducible synthetic dataset if `credit_data.csv` is absent. It is designed to exercise the complete ML pipeline, but it is not a substitute for a real-world credit dataset.

For an academic submission that requires a public dataset, place the approved CSV at:

```text
task1_credit_scoring/credit_data.csv
```

and adjust `TARGET_COLUMN` if the target column has a different name.
