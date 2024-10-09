import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load the encoder and model once at the start
with open('encoder.pkl', 'rb') as fp:
    encoder = pickle.load(fp)

with open('scaler.pkl', 'rb') as fp:
    scaler = pickle.load(fp)

with open('best_xgb.pkl', 'rb') as fp:
    best_xgb = pickle.load(fp)

# Title of the app
st.title("Customer Churn Prediction")

# Sidebar for user input features
st.sidebar.header("User Input Features")

# Create input fields in the sidebar
with st.sidebar.expander("Customer Information"):
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.slider('Tenure (months)', 0, 100, 1)

with st.sidebar.expander("Service Information"):
    phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
    
with st.sidebar.expander("Billing Information"):
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, format="%.2f")
    total_charges = st.number_input('Total Charges', min_value=0.0, format="%.2f")

# Create DataFrame for input features
input_df = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Display input summary
st.subheader("Your Input Summary:")
st.write(input_df)

# Predict button
if st.button('Predict'):    
    # Categorical Encoding
    cat_cols = [col for col in input_df.columns if input_df[col].dtype == 'object']

    categorical_encoded = pd.DataFrame(encoder.transform(input_df[cat_cols]), index=input_df.index)
    numerical_data = input_df.drop(cat_cols, axis=1)

    # Convert numeric feature names (header) to string.
    categorical_encoded.columns = [str(col) for col in categorical_encoded.columns]

    X = pd.concat([categorical_encoded, numerical_data], axis=1)

    # Feature Scaling
    X_scaled = scaler.transform(X)

    # Make prediction
    xgb_prediction = best_xgb.predict(X_scaled)

    # Display the prediction
    st.subheader("Prediction Result:")
    if xgb_prediction[0] == 1:
        st.write('**Customer will Churn**')
    else:
        st.write('**Customer will NOT Churn**')
