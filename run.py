from src.train_model import load_data, preprocess, train_model
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset (update path if needed)
df = load_data("data/creditcard.csv")

# Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess(df)

# Train
model = train_model(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_proba))
