# auto_update.py - Automatic data update script
"""
This script automatically updates Bitcoin data and regenerates features.
Run it periodically using cron (Linux/Mac) or Task Scheduler (Windows).

Usage:
    python auto_update.py

For automatic updates every hour:
    Linux/Mac: Add to crontab
        0 * * * * cd /path/to/project && python auto_update.py
    
    Windows: Use Task Scheduler to run this script every hour
"""

import os
import sys
from datetime import datetime
import yfinance as yf
import pandas as pd
import subprocess

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def fetch_latest_data():
    """Fetch latest Bitcoin data from Yahoo Finance"""
    try:
        log("Fetching latest Bitcoin data...")
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period="30d")
        
        if df.empty:
            log("ERROR: No data received from Yahoo Finance")
            return False
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Load existing data if available
        if os.path.exists("data/btc_data.csv"):
            existing_df = pd.read_csv("data/btc_data.csv", index_col=0, parse_dates=True)
            # Combine and remove duplicates
            combined = pd.concat([existing_df, df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            combined.to_csv("data/btc_data.csv")
            log(f"Updated data with {len(df)} new rows. Total rows: {len(combined)}")
        else:
            df.to_csv("data/btc_data.csv")
            log(f"Created new data file with {len(df)} rows")
        
        return True
    except Exception as e:
        log(f"ERROR fetching data: {str(e)}")
        return False

def regenerate_features():
    """Regenerate feature dataset"""
    try:
        log("Regenerating features...")
        result = subprocess.run(
            [sys.executable, "features.py"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            log("Features regenerated successfully")
            return True
        else:
            log(f"ERROR regenerating features: {result.stderr}")
            return False
    except Exception as e:
        log(f"ERROR: {str(e)}")
        return False

def check_models_exist():
    """Check if trained models exist"""
    rf_model = os.path.exists("models/rf_model.pkl")
    lstm_model = os.path.exists("models/lstm_model.keras") or os.path.exists("models/lstm_model.h5")
    
    if not rf_model:
        log("WARNING: RF model not found. Run: python train_rf.py")
    if not lstm_model:
        log("WARNING: LSTM model not found. Run: python train_lstm.py")
    
    return rf_model and lstm_model

def main():
    log("=" * 60)
    log("Bitcoin Data Auto-Update Script")
    log("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("features.py"):
        log("ERROR: Run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Fetch latest data
    if not fetch_latest_data():
        log("Failed to fetch data. Exiting.")
        sys.exit(1)
    
    # Step 2: Regenerate features
    if not regenerate_features():
        log("Failed to regenerate features. Exiting.")
        sys.exit(1)
    
    # Step 3: Check models
    if check_models_exist():
        log("All models present - predictions will use latest data")
    else:
        log("Some models missing - train them for predictions")
    
    log("=" * 60)
    log("Update completed successfully!")
    log("=" * 60)

if __name__ == "__main__":
    main()