# Fraud Detection in Credit Card Transactions

This project builds a machine learning system to detect fraudulent credit card transactions using real-world financial data.

## Objective
To identify fraudulent transactions while balancing detection accuracy and minimizing false alarms.

## Dataset
- Credit card transaction dataset
- Highly imbalanced (fraud cases are very rare)
- 30 numerical features + target label (Class)

## Techniques Used 
- Data preprocessing
- Feature scaling (StandardScaler)
- Train-test split (stratified)
- Logistic Regression (baseline model)
- Random Forest Classifier (final model)

## Evaluation Metrics
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC Score

## Key Insight
Fraud detection is a highly imbalanced classification problem where recall is more important than accuracy.

## Author
Built as a fintech machine learning project for portfolio and job applications.

## Project Structure

fraud-detection-ml/
│
├── run.py                  # End-to-end pipeline execution
├── README.md
│
├── notebooks/
│   └── fraud_detection.ipynb   # Exploratory analysis & model development
│
├── src/
│   └── train_model.py          # ML training pipeline (modular code)
│
├── data/                       # Dataset (not always stored in repo)

## How to Run

Install dependencies:
pip install pandas numpy scikit-learn

Run the full pipeline:
python run.py

## Real-World Relevance

This project simulates a real fintech fraud detection pipeline used in banking systems:

- Handles highly imbalanced financial data
- Implements model comparison (Logistic Regression vs Random Forest)
- Uses evaluation metrics suitable for fraud detection (Recall, ROC-AUC)
- Demonstrates trade-off between fraud detection and false positives
