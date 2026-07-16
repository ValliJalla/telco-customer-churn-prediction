import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction Project")
st.write("""
Welcome! Use the sidebar to navigate:
- **Prediction** — Enter customer details and get a churn risk prediction
- **Business Impact** — See the estimated business value of this model
""")