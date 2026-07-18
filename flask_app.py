from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model           = joblib.load(os.path.join(BASE_DIR, 'models', 'churn_model.pkl'))
scaler          = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
feature_columns = joblib.load(os.path.join(BASE_DIR, 'models', 'feature_columns.pkl'))


def preprocess_input(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = sum(1 for col in services if raw.get(col) == 'Yes')

    total_charges           = raw['MonthlyCharges'] * max(raw['tenure'], 1)
    df['TotalCharges']      = total_charges
    df['avg_monthly_spend'] = total_charges / max(raw['tenure'], 1)

    t = raw['tenure']
    df['tenure_group'] = ("0-1yr" if t <= 12 else
                          "1-2yr" if t <= 24 else
                          "2-4yr" if t <= 48 else "4yr+")

    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['gender']        = df['gender'].map({'Male': 1, 'Female': 0})
    df['SeniorCitizen'] = raw['SeniorCitizen']

    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaymentMethod', 'tenure_group']
    ohe_cols = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/business')
def business():
    return render_template('business.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    raw_input = {
        'gender':           data.get('gender'),
        'SeniorCitizen':    1 if data.get('seniorCitizen') == 'Yes' else 0,
        'Partner':          data.get('partner'),
        'Dependents':       data.get('dependents'),
        'tenure':           int(data.get('tenure', 1)),
        'PhoneService':     data.get('phoneService'),
        'MultipleLines':    data.get('multipleLines'),
        'InternetService':  data.get('internetService'),
        'OnlineSecurity':   data.get('onlineSecurity'),
        'OnlineBackup':     'No',
        'DeviceProtection': 'No',
        'TechSupport':      data.get('techSupport'),
        'StreamingTV':      data.get('streamingTV'),
        'StreamingMovies':  data.get('streamingMovies'),
        'Contract':         data.get('contract'),
        'PaperlessBilling': data.get('paperlessBilling'),
        'PaymentMethod':    data.get('paymentMethod'),
        'MonthlyCharges':   float(data.get('monthlyCharges', 70)),
    }

    X_input  = preprocess_input(raw_input)
    X_scaled = scaler.transform(X_input)
    proba    = float(model.predict_proba(X_scaled)[0][1])

    if proba >= 0.6:
        risk = 'HIGH'
    elif proba >= 0.3:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'

    # Key factors (only active features)
    coefs         = model.coef_[0]
    raw_values    = X_input.values[0]
    contributions = X_scaled[0] * coefs

    contrib_df = pd.DataFrame({
        'feature':      feature_columns,
        'raw_value':    raw_values,
        'contribution': contributions,
    })
    contrib_df = (contrib_df[contrib_df['raw_value'] != 0]
                  .assign(abs_contrib=lambda d: d['contribution'].abs())
                  .sort_values('abs_contrib', ascending=False)
                  .head(5))

    factors = [
        {
            'name':      row['feature'].replace('_', ' '),
            'impact':    round(float(row['contribution']), 3),
            'direction': 'increase' if row['contribution'] > 0 else 'decrease'
        }
        for _, row in contrib_df.iterrows()
    ]

    return jsonify({
        'probability': round(proba * 100, 1),
        'risk':        risk,
        'factors':     factors,
        'tenure':      raw_input['tenure'],
        'monthly':     raw_input['MonthlyCharges'],
        'contract':    raw_input['Contract'],
    })


if __name__ == '__main__':
    app.run(debug=True)