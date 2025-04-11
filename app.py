import streamlit as st
import numpy as np
from utils.churn_utils import load_xgboost_model, load_scaler, preprocess_input, predict_churn

# 🎨 Page Config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# 🎯 Load Model & Scaler
model = load_xgboost_model()
scaler = load_scaler()

# 🧑‍💼 Sidebar Inputs
st.sidebar.title("🔧 Customer Profile Inputs")
tenure = st.sidebar.slider("📆 Tenure (months)", 1, 60, 12)
monthly_spend = st.sidebar.slider("💰 Monthly Spend ($)", 100, 2000, 500)
satisfaction_score = st.sidebar.slider("😊 Satisfaction Score (1-5)", 1, 5, 3)
support_tickets = st.sidebar.slider("🎫 Support Tickets Last 6 Months", 0, 10, 2)
contract_type = st.sidebar.selectbox("📝 Contract Type", ["Fixed", "Flexible"])

# 🎛️ Main UI
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🚨 AI-Powered Churn Prediction</h1>
    <p style='text-align: center; color: #6c757d;'>Get customer churn risk instantly and plan smart retention strategies.</p>
    """, unsafe_allow_html=True)

# 🧮 Prediction Trigger
if st.button("📊 Predict Churn"):
    input_data = {
        'Tenure': tenure,
        'MonthlySpend': monthly_spend,
        'SatisfactionScore': satisfaction_score,
        'SupportTickets': support_tickets,
        'ContractType': contract_type
    }

    features = preprocess_input(input_data, scaler)
    prediction, probability = predict_churn(features, model)

    # 🎉 Output
    if prediction == 1:
        st.error(f"❌ High Risk: Customer likely to churn. Probability: {probability:.2f}")
        st.markdown("<span style='color:red;'>💡 Consider reaching out with a retention offer.</span>", unsafe_allow_html=True)
    else:
        st.success(f"✅ Low Risk: Customer likely to stay. Probability: {probability:.2f}")
        st.markdown("<span style='color:green;'>🎉 Great! Focus on customer satisfaction.</span>", unsafe_allow_html=True)

# 💡 Footer
st.markdown("""
    <hr>
    <small style='color: #999;'>Built with ❤️ using Streamlit, XGBoost, and Python</small>
""", unsafe_allow_html=True)
