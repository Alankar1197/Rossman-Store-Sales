import os
import joblib
import pandas as pd
import streamlit as st
import gdown

MODEL_FILE = "rf_sales_model_19-12-2025-11-39-25.pkl"

GDRIVE_FILE_ID = "1y6lUKBK6sEtm3lwJ2FZsg3OdCu8fJHqQ"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)
    return joblib.load(MODEL_FILE)

model = load_model()

st.title("Rossmann Store Sales Prediction")

st.sidebar.header("Input Parameters")
store_id = st.sidebar.number_input("Store ID", min_value=1, value=1)

uploaded_file = st.file_uploader(
    "Upload CSV (with date based features)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    try:
        predictions = model.predict(df)
        df["Predicted_Sales"] = predictions

        st.subheader("Predictions")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions",
            csv,
            "sales_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")




