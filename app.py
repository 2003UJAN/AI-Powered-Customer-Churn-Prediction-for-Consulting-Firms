import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="M&A Success Predictor", layout="centered", page_icon="📈")

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>💼 M&A Deal Success Predictor 🔮</h1>", unsafe_allow_html=True)
st.write("Fill in the financials of the M&A deal to predict its success probability.")

model = joblib.load("model/ma_model.pkl")

# User Inputs
acquirer_revenue = st.number_input("Acquirer Revenue ($)", value=50000000)
target_revenue = st.number_input("Target Revenue ($)", value=20000000)
deal_value = st.number_input("Deal Value ($)", value=100000000)
industry_match = st.selectbox("Industry Match?", ["Yes", "No"])
market_sentiment = st.slider("Market Sentiment (0-1)", 0.0, 1.0, 0.75)
prior_deals = st.number_input("Prior Deals by Acquirer", value=3)

# Convert to binary
industry_match_val = 1 if industry_match == "Yes" else 0

# Predict
if st.button("💡 Predict Deal Success"):
    input_data = pd.DataFrame([[acquirer_revenue, target_revenue, deal_value, industry_match_val, market_sentiment, prior_deals]],
                              columns=["acquirer_revenue", "target_revenue", "deal_value", "industry_match", "market_sentiment", "prior_deals_acquirer"])
    prob = model.predict_proba(input_data)[0][1]
    st.success(f"🧠 Predicted Success Probability: **{prob * 100:.2f}%**")

    if prob > 0.7:
        st.balloons()
