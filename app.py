import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess(df):
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    simple_cols = ['Partner', 'Dependents', 'PhoneService',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'PaperlessBilling']
    for col in simple_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0,
                               'No internet service': 0,
                               'No phone service': 0})
    df = pd.get_dummies(df, columns=['gender', 'MultipleLines',
                                      'InternetService', 'Contract',
                                      'PaymentMethod'], drop_first=True)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

st.title("Customer Churn Predictor")
st.markdown("### CSV upload karo — saare customers ka churn predict hoga!")
st.markdown("---")

uploaded_file = st.file_uploader("Customer CSV file upload karo", type=['csv'])

if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df_input.head())
    st.markdown("---")
    
    if st.button("Predict"):
        df_processed = preprocess(df_input.copy())
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)[:, 1]
        df_processed['Churn_Prediction'] = predictions
        df_processed['Churn_Probability'] = (probabilities * 100).round(1)
        df_processed['Risk'] = df_processed['Churn_Probability'].apply(
            lambda x: 'High Risk' if x > 70 else ('Medium Risk' if x > 40 else 'Safe')
        )
        st.subheader("Prediction Results")
        st.dataframe(df_processed[['Churn_Prediction', 'Churn_Probability', 'Risk']])
        high_risk = (df_processed['Churn_Probability'] > 70).sum()
        st.error(f"High Risk Customers: {high_risk}")
        st.success(f"Safe Customers: {(predictions == 0).sum()}")
        csv = df_processed.to_csv(index=False)
        st.download_button(
            label="Download Results",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )