# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model = joblib.load(os.path.join(BASE_DIR, 'models', 'churn_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
feature_columns = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_columns.pkl'))

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Predict customer churn risk and understand key drivers.")

st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Has Partner", ["Yes", "No"])
    dependents      = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    online_security  = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
    tech_support     = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract         = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method   = st.selectbox("Payment Method",
                           ["Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges  = st.number_input("Monthly Charges (₹)", 0.0, 200.0, 70.0)


# ─── PREPROCESSING (must match training pipeline exactly) ─────────────────────
def preprocess_input(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    # --- Feature engineering ---
    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = sum(1 for col in services if raw.get(col) == 'Yes')

    total_charges            = raw['MonthlyCharges'] * max(raw['tenure'], 1)
    df['TotalCharges']       = total_charges
    df['avg_monthly_spend']  = total_charges / max(raw['tenure'], 1)

    t = raw['tenure']
    df['tenure_group'] = ("0-1yr"  if t <= 12 else
                          "1-2yr"  if t <= 24 else
                          "2-4yr"  if t <= 48 else "4yr+")

    # --- Binary encoding ---
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['gender']       = df['gender'].map({'Male': 1, 'Female': 0})
    df['SeniorCitizen'] = raw['SeniorCitizen']   # already 0/1

    # --- One-hot encoding ---
    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaymentMethod', 'tenure_group']
    ohe_cols = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # --- Align to training columns (fills missing dummies with 0) ---
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


# ─── PREDICT ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn Risk"):

    raw_input = {
        'gender':           gender,
        'SeniorCitizen':    1 if senior_citizen == "Yes" else 0,
        'Partner':          partner,
        'Dependents':       dependents,
        'tenure':           tenure,
        'PhoneService':     phone_service,
        'MultipleLines':    multiple_lines,
        'InternetService':  internet_service,
        'OnlineSecurity':   online_security,
        'OnlineBackup':     'No',           # not in UI → default
        'DeviceProtection': 'No',           # not in UI → default
        'TechSupport':      tech_support,
        'StreamingTV':      streaming_tv,
        'StreamingMovies':  streaming_movies,
        'Contract':         contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod':    payment_method,
        'MonthlyCharges':   monthly_charges,
    }

    X_input        = preprocess_input(raw_input)
    X_scaled       = scaler.transform(X_input)
    proba          = model.predict_proba(X_scaled)[0][1]

    # ── Result display ──
    st.divider()
    st.subheader("🎯 Prediction Result")

    if proba >= 0.6:
        risk_label, risk_color = "🔴 HIGH RISK — likely to churn", "red"
    elif proba >= 0.3:
        risk_label, risk_color = "🟡 MEDIUM RISK — monitor this customer", "orange"
    else:
        risk_label, risk_color = "🟢 LOW RISK — likely to stay", "green"

    st.metric("Churn Probability", f"{proba*100:.1f}%")
    st.progress(min(int(proba * 100), 100))
    st.markdown(f"**{risk_label}**")

    # ── Feature importance (coefficient-based, correct direction) ──
    st.divider()
    st.subheader("🔍 Key Factors for This Customer")
    st.caption("Based on logistic regression weights × scaled feature values")

    coefs         = model.coef_[0]
    raw_values    = X_input.values[0]        # unscaled → 0 or 1 for dummies
    contributions = X_scaled[0] * coefs

    contrib_df = pd.DataFrame({
        'Feature':      feature_columns,
        'RawValue':     raw_values,
        'Contribution': contributions,
        'AbsContrib':   np.abs(contributions)
    })

    # Only show features this customer ACTUALLY HAS (raw value != 0)
    # Showing absent dummy features confuses the interpretation
    contrib_df = (
        contrib_df[contrib_df['RawValue'] != 0]
        .sort_values('AbsContrib', ascending=False)
        .head(5)
    )

    if contrib_df.empty:
        st.write("No significant active features found.")
    else:
        for _, row in contrib_df.iterrows():
            if row['Contribution'] > 0:
                st.write(f"🔴 **{row['Feature']}** — *increases* churn risk "
                         f"(impact: +{row['Contribution']:.3f})")
            else:
                st.write(f"🟢 **{row['Feature']}** — *decreases* churn risk "
                         f"(impact: {row['Contribution']:.3f})")

    st.caption(
        "ℹ️ Only showing features active for this customer. "
        "🔴 Positive impact → pushing toward churn. "
        "🟢 Negative impact → pushing away from churn."
    )
