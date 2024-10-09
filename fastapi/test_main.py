from fastapi.testclient import TestClient
from fatsapi_churnapp import app, InputData  # Import the FastAPI app and InputData schema
import pytest

# Initialize the TestClient with the FastAPI app
client = TestClient(app)

# Sample test data for the API
sample_input = {
    "gender": "Male",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 830.75
}

# Test the /predict/ endpoint with valid input data
def test_predict_churn():
    response = client.post("/predict/", json=sample_input)
    assert response.status_code == 200  # Ensure the request was successful
    data = response.json()  # Get the JSON response

    # Check if the response contains a prediction
    assert "prediction" in data
    assert "message" in data

    # Check if the prediction is either 0 (No Churn) or 1 (Churn)
    assert data["prediction"] in [0, 1]

    # Verify the message corresponds to the prediction
    if data["prediction"] == 1:
        assert data["message"] == "Customer will Churn"
    else:
        assert data["message"] == "Customer will NOT Churn"

# Test invalid input or missing fields (Edge case)
def test_predict_invalid_data():
    invalid_input = sample_input.copy()
    invalid_input.pop("gender")  # Remove a required field

    response = client.post("/predict/", json=invalid_input)
    assert response.status_code == 422  # 422 is returned for validation errors

# Test if the model and endpoints work for different inputs
@pytest.mark.parametrize("gender, expected_status", [
    ("Male", 200),
    ("Female", 200),
    (1, 422)  # Assuming "Other" is invalid for your model
])
def test_predict_gender_variations(gender, expected_status):
    input_data = sample_input.copy()
    input_data["gender"] = gender

    response = client.post("/predict/", json=input_data)
    assert response.status_code == expected_status
