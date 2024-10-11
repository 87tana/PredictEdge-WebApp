from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd


# Initialize FastAPI
app = FastAPI()


# Define the request body for input data
class InputData(BaseModel):
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    

# Define a POST endpoint for predictions
@app.post("/predict")
def predict(input_data: InputData):
    # Create DataFrame from the input data
    data = pd.DataFrame([input_data.dict()])

    # Load the saved machine learning model and scaler
    with open("best_xgb.pkl", "rb") as f:
        model = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Categorical Encoding
    cat_cols = [col for col in data.columns if data[col].dtype == 'object']
    categorical_encoded = pd.DataFrame(encoder.transform(data[cat_cols]), index=data.index)

    # Drop categorical columns from original data
    numerical_data = data.drop(cat_cols, axis=1)

    # Convert numeric feature names (header) to string
    categorical_encoded.columns = [str(col) for col in categorical_encoded.columns]

    # Combine categorical and numerical data
    X = pd.concat([categorical_encoded, numerical_data], axis=1)

    # Apply scaler
    X_scaled = scaler.transform(X)

    # Make predictions
    prediction = model.predict(X_scaled)

    # Return the prediction
    return {"prediction": int(prediction[0]), "message": "Customer will Churn" if prediction[0] == 1 else "Customer will NOT Churn"}
   