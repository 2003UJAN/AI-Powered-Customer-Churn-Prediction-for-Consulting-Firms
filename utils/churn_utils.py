import pandas as pd
import numpy as np
import pickle
import os

# Load the XGBoost model
def load_xgboost_model(model_path='models/xgboost_churn_model.pkl'):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the Scaler
def load_scaler(scaler_path='models/scaler.pkl'):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Preprocess input data for prediction
def preprocess_input(input_data, scaler):
    """
    input_data: dict
        e.g., {
            'Tenure': 12,
            'MonthlySpend': 500,
            'SatisfactionScore': 3,
            'SupportTickets': 2,
            'ContractType': 'Fixed'
        }
    """
    df = pd.DataFrame([input_data])
    
    # Convert categorical to numeric
    df['ContractType'] = df['ContractType'].apply(lambda x: 1 if x == 'Fixed' else 0)
    
    # Scale numerical features
    features = scaler.transform(df)
    return features

# Predict churn probability
def predict_churn(features, model):
    """
    features: np.array of shape (1, num_features)
    model: trained XGBoost model
    Returns: probability of churn
    """
    prob = model.predict_proba(features)[0][1]
    prediction = int(prob >= 0.5)
    return prediction, prob
