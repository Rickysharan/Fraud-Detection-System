import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Fraud Detection System (Real ML Model)")

st.write("Enter transaction features (PCA values simplified for demo)")

# We simulate 30 features input (since dataset has 30 features)
features = []

for i in range(30):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(val)

features = np.array(features).reshape(1, -1)

if st.button("Predict Fraud"):
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected | Probability: {prob:.4f}")
    else:
        st.success(f"✅ Legitimate Transaction | Probability: {prob:.4f}")
