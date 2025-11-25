import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

df = pd.read_csv("data/btc_features.csv")

data = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

seq = 60
X, y = [], []
for i in range(seq, len(data_scaled)):
    X.append(data_scaled[i-seq:i])
    y.append(data_scaled[i])

X, y = np.array(X), np.array(y)

split = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq, 1)),
    Dropout(0.2),
    LSTM(64),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
dump(scaler, "models/lstm_scaler.pkl")

print("LSTM Model Trained and Saved!")
