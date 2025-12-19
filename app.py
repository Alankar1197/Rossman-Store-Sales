import streamlit as st
import pandas as pd
import joblib
import os
import requests

st.set_page_config(page_title="Rossmann Store Sales Prediction", layout="wide")

MODEL_PATH = "rf_sales_model_19-12-2025-11-39-25.pkl"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1y6lUKBK6sEtm3lwJ2FZsg3OdCu8fJHqQ" 

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Rossmann Store Sales Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    try:
        preds = model.predict(df)
        df["Predicted_Sales"] = preds

        st.dataframe(df.head())

        st.download_button(
            "Download Predictions",
            df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(e)

