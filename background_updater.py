# background_updater.py - Runs continuously to update Bitcoin data
"""
Run this script in the background to automatically update Bitcoin data every hour
Usage: python background_updater.py
"""

import time
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import subprocess
import sys

def log(message):
    """Print timestamped log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_live_bitcoin_data():
    """Fetch LIVE Bitcoin data including TODAY"""
    try:
        btc = yf.Ticker("BTC-USD")
        
        # Get real-time info
        info = btc.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # Get recent history
        df_recent = btc.history(period="5d", interval="1d")
        df_hist = btc.history(period="1y", interval="1d")
        
        # Combine
        df = pd.concat([df_hist, df_recent])
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        # Check if we have today's data
        today = datetime.now().date()
        latest_date = df.index[-1].date()
        
        # If no today's data, create it with live price
        if latest_date < today and current_price:
            today_ts = pd.Timestamp(datetime.now().replace(hour=16, minute=0, second=0))
            today_row = pd.DataFrame({
                'Open': [current_price * 0.999],
                'High': [current_price * 1.002],
                'Low': [current_price * 0.998],
                'Close': [current_price],
                'Volume': [df['Volume'].iloc[-1]]
            }, index=[today_ts])
            
            df = pd.concat([df, today_row])
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()
        
        return df, current_price
        
    except Exception as e:
        log(f"Error fetching data: {e}")
        return None, None

def update_data_files():
    """Update data files with latest Bitcoin data"""
    try:
        log("Fetching latest Bitcoin data...")
        df, live_price = get_live_bitcoin_data()
        
        if df is None:
            log("Failed to fetch data")
            return False
        
        # Clean data
        df = df.reset_index()
        df.columns = ['date'] + list(df.columns[1:])
        df = df.set_index('date')
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols]
        
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        # Save raw data
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/btc_data.csv")
        log(f"✓ Saved {len(df)} records (latest: ${df['Close'].iloc[-1]:,.2f})")
        
        # Generate features
        feat = df.copy()
        feat["HL_PCT"] = (feat["High"] - feat["Low"]) / feat["Close"]
        feat["PCT_change"] = (feat["Close"] - feat["Open"]) / feat["Open"]
        feat["MA7"] = feat["Close"].rolling(7, min_periods=7).mean()
        feat["MA21"] = feat["Close"].rolling(21, min_periods=21).mean()
        feat["MA50"] = feat["Close"].rolling(50, min_periods=50).mean()
        feat["MA200"] = feat["Close"].rolling(200, min_periods=200).mean()
        feat["STD21"] = feat["Close"].rolling(21, min_periods=21).std()
        feat["Return"] = feat["Close"].pct_change()
        feat["Target"] = feat["Return"].shift(-1)
        feat = feat.dropna()
        
        feat.to_csv("data/btc_features.csv")
        log(f"✓ Generated {len(feat)} features")
        
        # Get data age
        latest_date = df.index[-1].date()
        today = datetime.now().date()
        age = (today - latest_date).days
        
        if age == 0:
            log(f"✓ Data is CURRENT (TODAY)")
        else:
            log(f"⚠ Data is {age} day(s) old")
        
        if live_price:
            log(f"✓ Live price: ${live_price:,.2f}")
        
        return True
        
    except Exception as e:
        log(f"Error updating files: {e}")
        return False

def retrain_models():
    """Retrain ML models"""
    try:
        log("Retraining Random Forest...")
        result_rf = subprocess.run(
            [sys.executable, "train_rf.py"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result_rf.returncode == 0:
            log("✓ RF model retrained")
        else:
            log(f"⚠ RF training issue: {result_rf.stderr[:100]}")
        
        log("Retraining LSTM...")
        result_lstm = subprocess.run(
            [sys.executable, "train_lstm.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result_lstm.returncode == 0:
            log("✓ LSTM model retrained")
        else:
            log(f"⚠ LSTM training issue: {result_lstm.stderr[:100]}")
        
        return True
        
    except Exception as e:
        log(f"Error retraining: {e}")
        return False

def main():
    """Main loop - updates every hour"""
    log("=" * 60)
    log("Bitcoin Auto-Updater Started")
    log("Updates every 1 hour")
    log("Press Ctrl+C to stop")
    log("=" * 60)
    
    # Initial update
    update_data_files()
    retrain_models()
    
    # Loop forever
    update_count = 1
    while True:
        try:
            # Wait 1 hour
            log(f"Waiting 1 hour for next update...")
            time.sleep(3600)  # 3600 seconds = 1 hour
            
            # Update
            log("=" * 60)
            log(f"Starting update #{update_count + 1}")
            log("=" * 60)
            
            success = update_data_files()
            
            if success:
                # Only retrain if data updated successfully
                retrain_models()
            
            update_count += 1
            
        except KeyboardInterrupt:
            log("=" * 60)
            log("Auto-updater stopped by user")
            log("=" * 60)
            break
        except Exception as e:
            log(f"Error in main loop: {e}")
            log("Continuing...")
            time.sleep(60)

if __name__ == "__main__":
    main()