import streamlit as st
import requests

# API endpoints
logistic_endpoint = "http://127.0.0.1:8000/predict/logistic_regression/"
random_forest_endpoint = "http://127.0.0.1:8000/predict/random_forest/"

# Page Title
st.title("Fraud Detection Prediction UI")

# Input Form
st.header("Enter Transaction Details")

# Mapping form inputs to model inputs
form_data = {
    "CustomerId": st.number_input("CustomerId", min_value=0.0),
    "TransactionId": st.number_input("TransactionId", min_value=0.0),
    "TransactionStartTime": st.number_input("TransactionStartTime", min_value=0.0),
    "Amount": st.number_input("Amount", min_value=0.0),
    "Frequency_Score": st.number_input("Frequency_Score", min_value=0.0),
    "Monetary_Score": st.number_input("Monetary_Score", min_value=0.0),
    "Recency_Score": st.number_input("Recency_Score", min_value=0.0),
    "RFMS_Score": st.number_input("RFMS_Score", min_value=0.0),
}

# Button to trigger predictions
st.subheader("Choose a Model for Prediction")
if st.button("Predict with Logistic Regression"):
    try:
        response = requests.post(logistic_endpoint, json=form_data)
        if response.status_code == 200:
            prediction = response.json().get("prediction", "No prediction received")
            st.success(f"Logistic Regression Prediction: {prediction}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.button("Predict with Random Forest"):
    try:
        response = requests.post(random_forest_endpoint, json=form_data)
        if response.status_code == 200:
            prediction = response.json().get("prediction", "No prediction received")
            st.success(f"Random Forest Prediction: {prediction}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown(
    """
    ---
    **Note:** Ensure the FastAPI backend is running locally on `127.0.0.1:8000` for predictions to work.
    """
)
