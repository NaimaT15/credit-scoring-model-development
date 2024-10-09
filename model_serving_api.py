from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Define the FastAPI app
app = FastAPI()

# Load the trained models and feature columns
logistic_regression_model = joblib.load(r"notebooks\logistic_regression_model.pkl")
random_forest_model = joblib.load(r"notebooks\random_forest_model.pkl")

# Load the feature columns used for training
feature_columns = joblib.load(r"notebooks\feature_columns.pkl")

# Define the input data structure using Pydantic
class ModelInput(BaseModel):
    TransactionId: float
    BatchId: float
    AccountId: float
    SubscriptionId: float
    CustomerId: float
    CountryCode: float
    ProductId: float
    Amount: float
    Value: float
    TransactionStartTime: float
    PricingStrategy: float
    FraudResult: float
    Transaction_Hour: float
    Transaction_Day: float
    Transaction_Month: float
    Transaction_Year: float
    Transaction_DayOfWeek: float
    Transaction_WeekOfYear: float
    ProviderId_ProviderId_2: float
    ProviderId_ProviderId_3: float
    ProviderId_ProviderId_4: float
    ProviderId_ProviderId_5: float
    ProviderId_ProviderId_6: float
    ProductCategory_data_bundles: float
    ProductCategory_movies: float
    ProductCategory_other: float
    ProductCategory_ticket: float
    ProductCategory_transport: float
    ProductCategory_tv: float
    ProductCategory_utility_bill: float
    ChannelId_ChannelId_2: float
    ChannelId_ChannelId_3: float
    ChannelId_ChannelId_5: float

# Define a function to preprocess and align the input data for model prediction
def preprocess_data(input_data: ModelInput):
    # Convert the input data to a pandas DataFrame
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])

    # Ensure that the input DataFrame has the same columns as the training data
    missing_cols = set(feature_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Set missing columns to zero or a default value

    # Reorder the columns to match the training data
    input_df = input_df[feature_columns]

    return input_df

# API endpoint for logistic regression predictions
@app.post("/predict/logistic_regression/")
async def predict_logistic_regression(input_data: ModelInput):
    try:
        # Preprocess the input data
        data = preprocess_data(input_data)

        # Make the prediction
        prediction = logistic_regression_model.predict(data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# API endpoint for random forest predictions
@app.post("/predict/random_forest/")
async def predict_random_forest(input_data: ModelInput):
    try:
        # Preprocess the input data
        data = preprocess_data(input_data)

        # Make the prediction
        prediction = random_forest_model.predict(data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Health check endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Model Serving API"}