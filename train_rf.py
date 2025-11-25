import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

df = pd.read_csv("data/btc_features.csv")
features = ["Close", "Volume", "HL_PCT", "PCT_change", "MA7", "MA21", "MA50", "MA200", "STD21"]

X = df[features]
y = df["Target"]

split = int(len(X) * 0.85)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(
    n_estimators=250,
    max_depth=12,
    min_samples_leaf=12,
    random_state=42
)
rf.fit(X_train_scaled, y_train)

os.makedirs("models", exist_ok=True)
dump(rf, "models/rf_model.pkl")
dump(scaler, "models/scaler.pkl")

print("Random Forest Model Trained and Saved!")
