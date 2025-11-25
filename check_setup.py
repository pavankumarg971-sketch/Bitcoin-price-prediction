"""
Diagnostic script to check Bitcoin prediction project setup
Run: python check_setup.py
"""

import os
import sys

print("=" * 60)
print("🔍 BITCOIN PREDICTION PROJECT - DIAGNOSTIC CHECK")
print("=" * 60)

errors = []
warnings = []
success = []

# Check 1: Data files
print("\n📊 Checking Data Files...")
if os.path.exists("data/btc_data.csv"):
    success.append("✓ data/btc_data.csv exists")
else:
    errors.append("✗ data/btc_data.csv missing - Run: python fetch_data.py")

if os.path.exists("data/btc_features.csv"):
    success.append("✓ data/btc_features.csv exists")
else:
    errors.append("✗ data/btc_features.csv missing - Run: python features.py")

# Check 2: Model files
print("\n🤖 Checking Model Files...")
if os.path.exists("models/rf_model.pkl"):
    success.append("✓ RF model exists")
else:
    errors.append("✗ models/rf_model.pkl missing - Run: python train_rf.py")

if os.path.exists("models/scaler.pkl"):
    success.append("✓ RF scaler exists")
else:
    errors.append("✗ models/scaler.pkl missing - Run: python train_rf.py")

lstm_exists = False
if os.path.exists("models/lstm_model.keras"):
    success.append("✓ LSTM model exists (.keras)")
    lstm_exists = True
elif os.path.exists("models/lstm_model.h5"):
    success.append("✓ LSTM model exists (.h5)")
    lstm_exists = True
else:
    errors.append("✗ LSTM model missing - Run: python train_lstm.py")

if os.path.exists("models/lstm_scaler.pkl"):
    success.append("✓ LSTM scaler exists")
else:
    errors.append("✗ models/lstm_scaler.pkl missing - Run: python train_lstm.py")

# Check 3: Required packages
print("\n📦 Checking Required Packages...")
required_packages = [
    'fastapi', 'uvicorn', 'streamlit', 'pandas', 
    'numpy', 'sklearn', 'tensorflow', 'yfinance'
]

for package in required_packages:
    try:
        if package == 'sklearn':
            __import__('sklearn')
        else:
            __import__(package)
        success.append(f"✓ {package} installed")
    except ImportError:
        errors.append(f"✗ {package} not installed - Run: pip install {package}")

# Check 4: Test predictions
print("\n🔮 Testing Predictions...")
try:
    from predict import predict_rf_single, predict_lstm_single
    
    rf_pred = predict_rf_single()
    if rf_pred is not None:
        success.append(f"✓ RF prediction works: ${rf_pred:,.2f}")
    else:
        errors.append("✗ RF prediction failed")
    
    lstm_pred = predict_lstm_single()
    if lstm_pred is not None:
        success.append(f"✓ LSTM prediction works: ${lstm_pred:,.2f}")
    else:
        errors.append("✗ LSTM prediction failed")
except Exception as e:
    errors.append(f"✗ Prediction test error: {str(e)}")

# Check 5: API connectivity
print("\n🌐 Checking FastAPI...")
try:
    import requests
    resp = requests.get("http://127.0.0.1:8000/health", timeout=2)
    if resp.ok:
        success.append("✓ FastAPI is running")
    else:
        warnings.append("⚠ FastAPI responded but not healthy")
except:
    errors.append("✗ FastAPI not running - Start: uvicorn app_fastapi:app --reload")

# Print results
print("\n" + "=" * 60)
print("📋 DIAGNOSTIC RESULTS")
print("=" * 60)

if success:
    print(f"\n✅ SUCCESS ({len(success)} items):")
    for item in success:
        print(f"  {item}")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)} items):")
    for item in warnings:
        print(f"  {item}")

if errors:
    print(f"\n❌ ERRORS ({len(errors)} items):")
    for item in errors:
        print(f"  {item}")

# Summary
print("\n" + "=" * 60)
if not errors:
    print("✅ ALL CHECKS PASSED! System ready to use.")
    print("\nTo start the application:")
    print("  1. Terminal 1: uvicorn app_fastapi:app --reload")
    print("  2. Terminal 2: streamlit run app_streamlit.py")
else:
    print(f"❌ FOUND {len(errors)} ERROR(S) - Fix them before running")
    print("\n🔧 QUICK FIX COMMANDS:")
    print("  # Create folders")
    print("  mkdir data")
    print("  mkdir models")
    print("\n  # Get data")
    print("  python fetch_data.py")
    print("  python features.py")
    print("\n  # Train models")
    print("  python train_rf.py")
    print("  python train_lstm.py")
    print("\n  # Start services")
    print("  uvicorn app_fastapi:app --reload")
    print("  streamlit run app_streamlit.py")

print("=" * 60)