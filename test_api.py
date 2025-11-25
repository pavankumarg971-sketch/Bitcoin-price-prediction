"""
Test script to check if API endpoints are working
Run: python test_api.py
"""

import requests
import json

API_BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("🧪 TESTING FASTAPI ENDPOINTS")
print("=" * 60)

# Test 1: Root endpoint
print("\n1️⃣ Testing root endpoint (/)...")
try:
    resp = requests.get(f"{API_BASE}/", timeout=3)
    if resp.ok:
        print(f"✅ Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
    else:
        print(f"❌ Status: {resp.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Health endpoint
print("\n2️⃣ Testing health endpoint (/health)...")
try:
    resp = requests.get(f"{API_BASE}/health", timeout=3)
    if resp.ok:
        print(f"✅ Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
    else:
        print(f"❌ Status: {resp.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Single RF prediction
print("\n3️⃣ Testing RF single prediction (/predict/rf)...")
try:
    resp = requests.get(f"{API_BASE}/predict/rf", timeout=10)
    if resp.ok:
        print(f"✅ Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
    else:
        print(f"❌ Status: {resp.status_code}")
        print(f"Error: {resp.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Single LSTM prediction
print("\n4️⃣ Testing LSTM single prediction (/predict/lstm)...")
try:
    resp = requests.get(f"{API_BASE}/predict/lstm", timeout=10)
    if resp.ok:
        print(f"✅ Status: {resp.status_code}")
        print(f"Response: {json.dumps(resp.json(), indent=2)}")
    else:
        print(f"❌ Status: {resp.status_code}")
        print(f"Error: {resp.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Multi-day RF prediction
print("\n5️⃣ Testing RF multi-day prediction (/predict/rf/multi?days=7)...")
try:
    resp = requests.get(f"{API_BASE}/predict/rf/multi?days=7", timeout=15)
    if resp.ok:
        data = resp.json()
        print(f"✅ Status: {resp.status_code}")
        print(f"Success: {data.get('success')}")
        if data.get('success'):
            print(f"Days predicted: {len(data.get('predictions', []))}")
            print(f"First prediction: ${data['predictions'][0]:,.2f}")
            print(f"Last prediction: ${data['predictions'][-1]:,.2f}")
        else:
            print(f"Error: {data.get('error')}")
    else:
        print(f"❌ Status: {resp.status_code}")
        print(f"Error: {resp.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 6: Multi-day LSTM prediction
print("\n6️⃣ Testing LSTM multi-day prediction (/predict/lstm/multi?days=7)...")
try:
    resp = requests.get(f"{API_BASE}/predict/lstm/multi?days=7", timeout=15)
    if resp.ok:
        data = resp.json()
        print(f"✅ Status: {resp.status_code}")
        print(f"Success: {data.get('success')}")
        if data.get('success'):
            print(f"Days predicted: {len(data.get('predictions', []))}")
            print(f"First prediction: ${data['predictions'][0]:,.2f}")
            print(f"Last prediction: ${data['predictions'][-1]:,.2f}")
        else:
            print(f"Error: {data.get('error')}")
    else:
        print(f"❌ Status: {resp.status_code}")
        print(f"Error: {resp.text[:200]}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("🏁 TEST COMPLETE")
print("=" * 60)
print("\n💡 If tests 5 & 6 fail with 404:")
print("   Your app_fastapi.py needs the multi-day endpoints!")
print("   Update it with the enhanced version from the artifacts.")