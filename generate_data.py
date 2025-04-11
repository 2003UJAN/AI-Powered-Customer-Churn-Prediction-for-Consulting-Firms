# generate_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import os

# 📂 Create models folder if not exists
os.makedirs("models", exist_ok=True)

# 📊 Generate dummy customer churn dataset
def generate_dummy_data(n=500):
    data = {
        "Tenure": np.random.randint(1, 60, n),
        "MonthlySpend": np.random.randint(100, 2000, n),
        "SatisfactionScore": np.random.randint(1, 6, n),
        "SupportTickets": np.random.randint(0, 10, n),
        "ContractType": np.random.choice(["Fixed", "Flexible"], n),
        "Churn": np.random.choice([0, 1], n, p=[0.7, 0.3])  # 30% churn
    }
    return pd.DataFrame(data)

df = generate_dummy_data()

# 🧹 Encode categorical
df['ContractType'] = df['ContractType'].apply(lambda x: 1 if x == 'Fixed' else 0)

# 🏷️ Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 🎯 Save feature names
feature_names = X.columns.tolist()
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# ✂️ Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔢 Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 💾 Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 🤖 Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# 💾 Save model
with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Data, model, and scaler saved!")
