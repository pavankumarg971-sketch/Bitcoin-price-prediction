# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import os

API_BASE = "http://127.0.0.1:8000"

# ---------- Helper function ----------
def clean_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a pandas DataFrame so Streamlit can display it without Arrow errors.
    """
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).fillna('')
        elif pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        else:
            df_clean[col] = df_clean[col].fillna('')
    return df_clean
# -------------------------------------

def get_prediction(model_name, endpoint):
    """Helper to fetch and safely display prediction"""
    try:
        resp = requests.get(f"{API_BASE}/{endpoint}", timeout=10)
        if resp.ok:
            data = resp.json()
            price = data.get("next_day_price")
            if price is not None:
                st.success(f"{model_name} predicted next-day price: {price:.2f} USD")
            else:
                st.error(f"{model_name} prediction failed: {data.get('error', 'Unknown error')}")
        else:
            st.error(f"{model_name} error: {resp.text}")
    except Exception as e:
        st.error(f"{model_name} request failed: {e}")

try:
    test = requests.get(f"{API_BASE}/", timeout=3)
    st.sidebar.success("FastAPI: Connected")
except:
    st.sidebar.error("FastAPI NOT running — Start uvicorn!")

FEATURES_CSV = os.path.join("data", "btc_features.csv")

st.set_page_config(page_title="Bitcoin Prediction Dashboard", layout="wide")

st.title("Bitcoin Price Prediction — Streamlit Dashboard")
st.markdown("FastAPI backend + Streamlit frontend demo")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Price history")
    if os.path.exists(FEATURES_CSV):
        df = pd.read_csv(FEATURES_CSV, index_col=0, parse_dates=True)
        df = clean_dataframe_for_streamlit(df)

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_cols = st.multiselect(
            "Select columns to plot:",
            numeric_cols,
            default=["Close"] if "Close" in numeric_cols else numeric_cols
        )

        if selected_cols:
            st.line_chart(df[selected_cols].tail(500))
        else:
            st.info("Please select at least one column to display.")
    else:
        st.info("Run `features.py` to generate data/btc_features.csv.")

with col2:
    st.subheader("Predict Next Day")
    st.write("Backend URL:", API_BASE)

    if st.button("Get Random Forest Prediction"):
        get_prediction("RF", "predict/rf")

    if st.button("Get LSTM Prediction"):
        get_prediction("LSTM", "predict/lstm")

    st.markdown("---")
    st.write("More actions")

    if st.button("Show last data row"):
        if os.path.exists(FEATURES_CSV):
            df_last = pd.read_csv(FEATURES_CSV)
            df_last = clean_dataframe_for_streamlit(df_last)
            st.write(df_last.tail(1).T)
        else:
            st.info("data/btc_features.csv not found.")

st.markdown("**Notes:** Start FastAPI first with uvicorn, then run Streamlit.**")
