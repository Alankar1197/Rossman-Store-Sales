import streamlit as st
import pandas as pd
import joblib
import gdown
import os

MODEL_NAME = "rf_sales_model_19-12-2025-11-39-25.pkl"
MODEL_URL = "https://drive.google.com/uc?id=1y6lUKBK6sEtm3lwJ2FZsg3OdCu8fJHqQ"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_NAME):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_NAME, quiet=False)
    return joblib.load(MODEL_NAME)

model = load_model()

st.title("Rossmann Store Sales Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    try:
        predictions = model.predict(df)
        df["Predicted_Sales"] = predictions

        st.subheader("Predictions")
        st.dataframe(df[["Predicted_Sales"]].head())

        st.download_button(
            "Download Predictions",
            df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")



