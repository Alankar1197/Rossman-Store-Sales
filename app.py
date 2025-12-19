import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os

st.set_page_config(page_title="Rossmann Store Sales Prediction", layout="wide")

MODEL_PATH = "rf_sales_model_19-12-2025-11-39-25.pkl"
GDRIVE_FILE_ID = "1y6lUKBK6sEtm3lwJ2FZsg3OdCu8fJHqQ"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={1y6lUKBK6sEtm3lwJ2FZsg3OdCu8fJHqQ}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Rossmann Store Sales Prediction")

st.sidebar.header("Input Parameters")
store_id = st.sidebar.number_input("Store ID", min_value=1, step=1)

uploaded_file = st.file_uploader(
    "Upload CSV (with date based features)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # 🔒 Ensure correct preprocessing
    if "Store" not in df.columns:
        df["Store"] = store_id

    # Drop target columns if accidentally present
    for col in ["Sales", "Customers"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    try:
        predictions = model.predict(df)
        df["Predicted_Sales"] = predictions

        st.subheader("Predictions")
        st.dataframe(df.head())

        st.download_button(
            "Download Predictions",
            df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
