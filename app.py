import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rossmann Sales Predictor", layout="wide")

st.title("Rossmann Store Sales Prediction")

@st.cache_resource
def load_model():
    return joblib.load("rf_sales_model_19-12-2025-11-39-25.pkl")

model = load_model()

st.sidebar.header("Input Parameters")

store_id = st.sidebar.number_input("Store ID", min_value=1, step=1)

uploaded_file = st.file_uploader(
    "Upload CSV (with date based features)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    if "Store" not in df.columns:
        df["Store"] = store_id

    predictions = model.predict(df)

    df["Predicted_Sales"] = predictions

    st.subheader("Predictions")
    st.dataframe(df)

    st.subheader("Sales Trend")
    fig, ax = plt.subplots()
    ax.plot(df["Predicted_Sales"])
    ax.set_xlabel("Index")
    ax.set_ylabel("Predicted Sales")
    st.pyplot(fig)

    st.download_button(
        "Download Predictions",
        df.to_csv(index=False),
        "predictions.csv",
        "text/csv"
    )