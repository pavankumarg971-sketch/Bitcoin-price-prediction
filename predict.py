# predict.py - Enhanced with multi-day predictions
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
FEATURES_CSV = os.path.join("data", "btc_features.csv")


def _load_df():
    """Load the features dataframe."""
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(f"{FEATURES_CSV} not found. Run features.py first.")
    df = pd.read_csv(FEATURES_CSV, index_col=0, parse_dates=True)
    return df


def _calculate_features_for_row(df_history, last_close, last_volume):
    """Calculate features for a new predicted day."""
    # Use recent history for moving averages
    closes = np.append(df_history["Close"].values[-200:], last_close)
    
    features = {
        "Close": last_close,
        "Volume": last_volume,
        "HL_PCT": 0.02,  # approximate
        "PCT_change": (last_close - closes[-2]) / closes[-2] if len(closes) > 1 else 0,
        "MA7": np.mean(closes[-7:]) if len(closes) >= 7 else last_close,
        "MA21": np.mean(closes[-21:]) if len(closes) >= 21 else last_close,
        "MA50": np.mean(closes[-50:]) if len(closes) >= 50 else last_close,
        "MA200": np.mean(closes[-200:]) if len(closes) >= 200 else last_close,
        "STD21": np.std(closes[-21:]) if len(closes) >= 21 else 0,
    }
    return features


def predict_rf(days=1):
    """
    Predict next N days using RandomForest.
    Returns: dict with 'predictions' (list of prices) and 'dates' (list of date strings)
    """
    try:
        if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(RF_SCALER_PATH):
            raise FileNotFoundError("RF model or scaler not found. Run train_rf.py first.")
        
        rf = load(RF_MODEL_PATH)
        scaler = load(RF_SCALER_PATH)
        df = _load_df()
        
        predictions = []
        last_price = float(df["Close"].iloc[-1])
        last_volume = float(df["Volume"].iloc[-1])
        last_date = pd.to_datetime(df.index[-1])
        
        # Create working copy of recent history
        df_working = df.copy()
        dates = []
        
        for day in range(days):
            # Get features for current state
            last_row = df_working[FEATURES].iloc[-1:].copy()
            X_scaled = scaler.transform(last_row)
            
            # Predict return
            pred_return = rf.predict(X_scaled)[0]
            next_price = last_price * (1 + float(pred_return))
            predictions.append(float(next_price))
            
            # Calculate next date (skip weekends for crypto - though BTC trades 24/7)
            next_date = last_date + pd.Timedelta(days=1)
            dates.append(next_date.strftime("%Y-%m-%d"))
            
            # Update for next iteration
            new_features = _calculate_features_for_row(df_working, next_price, last_volume * 0.95)
            new_row = pd.DataFrame([new_features], index=[next_date])
            df_working = pd.concat([df_working, new_row])
            
            last_price = next_price
            last_date = next_date
        
        return {
            "success": True,
            "predictions": predictions,
            "dates": dates,
            "model": "Random Forest"
        }
    except Exception as e:
        print("RF prediction error:", e)
        return {"success": False, "error": str(e)}


def _load_lstm_model():
    """Load LSTM model and scaler."""
    try:
        if os.path.exists(LSTM_MODEL_KERAS):
            model = load_model(LSTM_MODEL_KERAS, compile=False)
        elif os.path.exists(LSTM_MODEL_H5):
            model = load_model(LSTM_MODEL_H5, compile=False)
        else:
            raise FileNotFoundError("No LSTM model found.")
        
        if not os.path.exists(LSTM_SCALER):
            raise FileNotFoundError("LSTM scaler not found.")
        scaler = load(LSTM_SCALER)
        return model, scaler
    except Exception as e:
        print("Error loading LSTM:", e)
        return None, None


def predict_lstm(days=1):
    """
    Predict next N days using LSTM.
    Returns: dict with 'predictions' (list) and 'dates' (list)
    """
    try:
        model, scaler = _load_lstm_model()
        if model is None or scaler is None:
            return {"success": False, "error": "LSTM model not loaded"}
        
        df = _load_df()
        close = df["Close"].values.reshape(-1, 1)
        scaled = scaler.transform(close)
        seq_len = model.input_shape[1] if hasattr(model, "input_shape") else 60
        
        if len(scaled) < seq_len:
            raise ValueError("Not enough data for LSTM sequence.")
        
        predictions = []
        dates = []
        last_date = pd.to_datetime(df.index[-1])
        
        # Start with last sequence
        current_seq = scaled[-seq_len:].copy()
        
        for day in range(days):
            # Predict next value
            input_seq = current_seq.reshape(1, seq_len, 1)
            pred_scaled = model.predict(input_seq, verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(float(pred_price))
            
            # Update date
            next_date = last_date + pd.Timedelta(days=day+1)
            dates.append(next_date.strftime("%Y-%m-%d"))
            
            # Slide window forward
            current_seq = np.append(current_seq[1:], pred_scaled[0])
        
        return {
            "success": True,
            "predictions": predictions,
            "dates": dates,
            "model": "LSTM"
        }
    except Exception as e:
        print("LSTM prediction error:", e)
        return {"success": False, "error": str(e)}


# Convenience functions for backward compatibility
def predict_rf_single():
    """Returns single next-day price (float or None)."""
    result = predict_rf(days=1)
    return result["predictions"][0] if result.get("success") else None


def predict_lstm_single():
    """Returns single next-day price (float or None)."""
    result = predict_lstm(days=1)
    return result["predictions"][0] if result.get("success") else None


if __name__ == "__main__":
    print("=== Single Day Predictions ===")
    rf_pred = predict_rf_single()
    lstm_pred = predict_lstm_single()
    print("RF next-day price:", rf_pred if rf_pred is not None else "Error")
    print("LSTM next-day price:", lstm_pred if lstm_pred is not None else "Error")
    
    print("\n=== 7-Day Predictions ===")
    rf_week = predict_rf(days=7)
    if rf_week.get("success"):
        print("RF 7-day forecast:")
        for date, price in zip(rf_week["dates"], rf_week["predictions"]):
            print(f"  {date}: ${price:.2f}")
    
    lstm_week = predict_lstm(days=7)
    if lstm_week.get("success"):
        print("\nLSTM 7-day forecast:")
        for date, price in zip(lstm_week["dates"], lstm_week["predictions"]):
            print(f"  {date}: ${price:.2f}")