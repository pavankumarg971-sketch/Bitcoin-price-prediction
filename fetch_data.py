import yfinance as yf
import pandas as pd
import os

def fetch_btc_data():
    print("Downloading Bitcoin data (BTC-USD)...")
    df = yf.download("BTC-USD", start="2015-01-01")
    df.to_csv("data/btc_data.csv")
    print("Data saved to data/btc_data.csv")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    fetch_btc_data()
