# churn_utils.py

import pickle
import numpy as np
import pandas as pd

# 📥 Load model
def load_xgboost_model(path='models/xgb_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

# 📥 Load scaler
def load_scaler(path='models/scaler.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

# 📥 Load feature names
def load_feature_names(path='models/feature_names.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

# 🧼 Preprocess input
def preprocess_input(input_data, scaler):
    df = pd.DataFrame([input_data])
    df['ContractType'] = df['ContractType'].apply(lambda x: 1 if x == 'Fixed' else 0)

    # Align feature order
    feature_names = load_feature_names()
    df = df.reindex(columns=feature_names)

    return scaler.transform(df)

# 🔮 Predict churn
def predict_churn(features, model):
    prob = model.predict_proba(features)[0][1]
    prediction = int(prob > 0.5)
    return prediction, prob
