# utils/churn_utils.py

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ---------------------------
# Load XGBoost model
# ---------------------------
def load_xgboost_model(path='models/xgboost_churn_model.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ XGBoost model not found at {path}. Run generate_data.py first.")
    with open(path, 'rb') as f:
        return pickle.load(f)

# ---------------------------
# Load LSTM model
# ---------------------------
def load_lstm_model(path='models/lstm_churn_model.h5'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ LSTM model not found at {path}. Run generate_data.py first.")
    return load_model(path)

# ---------------------------
# Load Scaler
# ---------------------------
def load_scaler(path='models/scaler.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Scaler not found at {path}. Run generate_data.py first.")
    with open(path, 'rb') as f:
        return pickle.load(f)

# ---------------------------
# Load Feature Names
# ---------------------------
def load_feature_names(path='models/feature_names.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Feature names not found at {path}. Run generate_data.py first.")
    with open(path, 'rb') as f:
        return pickle.load(f)

# ---------------------------
# Preprocess input for prediction
# ---------------------------
def preprocess_input(input_data: dict, scaler):
    df = pd.DataFrame([input_data])

    # Optional: Handle categorical variables or missing values here
    df.fillna(0, inplace=True)

    # Transform using the loaded scaler
    try:
        features = scaler.transform(df)
    except ValueError as e:
        raise ValueError(f"Scaler mismatch! Check input features. Error: {str(e)}")
    
    return features

# ---------------------------
# Predict using XGBoost
# ---------------------------
def predict_xgboost(model, features):
    return model.predict(features)[0]

# ---------------------------
# Predict using LSTM
# ---------------------------
def predict_lstm(model, features):
    features_reshaped = np.reshape(features, (features.shape[0], 1, features.shape[1]))
    return model.predict(features_reshaped)[0][0]
