import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title("📊 Telecom Customer Churn Predictor")
st.write("Predict customer churn risk in real time using a balanced Random Forest model.")

# Auto-train if model file doesn't exist yet
if not os.path.exists("churn_model.pkl"):
    import train

model = joblib.load("churn_model.pkl")

st.sidebar.header("Customer Details")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges (₹)", 299.0, 2499.0, 799.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "1 Year", "2 Year"])
tech_support = st.sidebar.selectbox("Tech Support Active?", ["No", "Yes"])
support_calls = st.sidebar.slider("Customer Support Calls (Last 3 Months)", 0, 10, 1)

contract_map = {"Month-to-Month": 0, "1 Year": 1, "2 Year": 2}
tech_map = {"No": 0, "Yes": 1}

input_df = pd.DataFrame([{
    "tenure_months": tenure,
    "monthly_charges": monthly_charges,
    "contract_type": contract_map[contract],
    "tech_support": tech_map[tech_support],
    "support_calls": support_calls,
    "charge_per_month_ratio": monthly_charges / (tenure + 1)
}])

if st.button("Predict Churn Risk"):
    prob = model.predict_proba(input_df)[0][1]
    st.subheader(f"Churn Probability: {prob * 100:.1f}%")
    if prob > 0.5:
        st.error("High Risk: Recommend proactive retention offer or priority support callback.")
    else:
        st.success("Low Risk: Customer is likely to stay.")
