import streamlit as st
import requests

# Page Config
st.set_page_config(
    page_title="Fraud Detection Prediction",
    page_icon="🔍",
    layout="centered",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .footer {
        font-size: 0.9em;
        text-align: center;
        color: #888;
        padding: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page Title
st.title("🔍 Fraud Detection Prediction")

# Description
st.markdown(
    """
    Welcome to the **Fraud Detection Prediction Interface**. This tool predicts the likelihood of fraud for a given transaction.  
    Simply enter the required details below and choose a model to get started.
    """
)

# Input Form
st.subheader("🔢 Transaction Details")
st.markdown("Fill out the details of the transaction:")

with st.form("input_form"):
    form_data = {
        "CustomerId": st.number_input("Customer ID", min_value=0.0),
        "TransactionId": st.number_input("Transaction ID", min_value=0.0),
        "TransactionStartTime": st.number_input("Transaction Start Time (Unix Timestamp)", min_value=0.0),
        "Amount": st.number_input("Amount", min_value=0.0),
        "Frequency_Score": st.number_input("Frequency Score", min_value=0.0),
        "Monetary_Score": st.number_input("Monetary Score", min_value=0.0),
        "Recency_Score": st.number_input("Recency Score", min_value=0.0),
        "RFMS_Score": st.number_input("RFMS Score", min_value=0.0),
    }

    submitted = st.form_submit_button("Submit Details")

# Display options after form submission
if submitted:
    st.success("Transaction details submitted successfully!")
    st.subheader("🚀 Choose Prediction Model")
    st.markdown("Select a model below to predict if the transaction is fraudulent.")

    # Prediction Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔗 Logistic Regression"):
            try:
                response = requests.post("http://127.0.0.1:8000/predict/logistic_regression/", json=form_data)
                if response.status_code == 200:
                    prediction = response.json().get("prediction", "No prediction received")
                    st.success(f"🔍 Logistic Regression Prediction: **{prediction}**")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    with col2:
        if st.button("🌳 Random Forest"):
            try:
                response = requests.post("http://127.0.0.1:8000/predict/random_forest/", json=form_data)
                if response.status_code == 200:
                    prediction = response.json().get("prediction", "No prediction received")
                    st.success(f"🌳 Random Forest Prediction: **{prediction}**")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.markdown(
    """
    ---
    <div class="footer">
    **Note:** Ensure the FastAPI backend is running locally on `127.0.0.1:8000`.  
    Created with ❤️ by [Your Name](https://your-portfolio-link.com)
    </div>
    """,
    unsafe_allow_html=True,
)
