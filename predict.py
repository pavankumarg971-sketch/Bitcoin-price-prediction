# predict.py
import os
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model

MODELS_DIR = "models"
FEATURES = ["Close", "Volume", "HL_PCT", "PCT_change", "MA7", "MA21", "MA50", "MA200", "STD21"]
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
RF_SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LSTM_MODEL_KERAS = os.path.join(MODELS_DIR, "lstm_model.keras")
LSTM_MODEL_H5 = os.path.join(MODELS_DIR, "lstm_model.h5")
LSTM_SCALER = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
FEATURES_CSV = os.path.join("data", "btc_features.csv")  # file created by features.py


def _load_df():
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"{FEATURES_CSV} not found. Run features.py first.")
    df = pd.read_csv(FEATURES_CSV, index_col=0, parse_dates=True)
    return df


def predict_rf():
    """Predict next-day price using RandomForest. Returns float or None if error."""
    try:
        if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(RF_SCALER_PATH):
            raise FileNotFoundError("RF model or scaler not found. Run train_rf.py first.")
        rf = load(RF_MODEL_PATH)
        scaler = load(RF_SCALER_PATH)

        df = _load_df()
        last_row = df[FEATURES].iloc[-1:].copy()
        X_scaled = scaler.transform(last_row)
        pred_return = rf.predict(X_scaled)[0]
        last_price = float(df["Close"].iloc[-1])
        next_price = last_price * (1 + float(pred_return))
        return float(next_price)
    except Exception as e:
        print("RF prediction error:", e)
        return None


def _load_lstm_model():
    """Load LSTM model and scaler, returns (model, scaler)."""
    try:
        if os.path.exists(LSTM_MODEL_KERAS):
            model = load_model(LSTM_MODEL_KERAS, compile=False)
        elif os.path.exists(LSTM_MODEL_H5):
            model = load_model(LSTM_MODEL_H5, compile=False)
        else:
            raise FileNotFoundError("No LSTM model found. Train LSTM and save to models/lstm_model.keras or .h5")

        if not os.path.exists(LSTM_SCALER):
            raise FileNotFoundError("LSTM scaler not found (models/lstm_scaler.pkl). Train LSTM first.")
        scaler = load(LSTM_SCALER)
        return model, scaler
    except Exception as e:
        print("Error loading LSTM model/scaler:", e)
        return None, None


def predict_lstm():
    """Predict next-day price using LSTM. Returns float or None if error."""
    try:
        model, scaler = _load_lstm_model()
        if model is None or scaler is None:
            return None

        df = _load_df()
        close = df["Close"].values.reshape(-1, 1)
        scaled = scaler.transform(close)
        seq_len = model.input_shape[1] if hasattr(model, "input_shape") else 60

        if len(scaled) < seq_len:
            raise ValueError("Not enough data to form LSTM input sequence.")

        last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
        pred_scaled = model.predict(last_seq)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        return float(pred_price)
    except Exception as e:
        print("LSTM prediction error:", e)
        return None


# Debugging when called directly
if __name__ == "__main__":
    rf_pred = predict_rf()
    lstm_pred = predict_lstm()
    print("RF next-day price:", rf_pred if rf_pred is not None else "Error")
    print("LSTM next-day price:", lstm_pred if lstm_pred is not None else "Error")
