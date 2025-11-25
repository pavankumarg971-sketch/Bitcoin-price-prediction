# app_fastapi.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from predict import predict_rf, predict_lstm

app = FastAPI(title="Bitcoin Prediction API")

@app.get("/")
def home():
    return {"message": "Bitcoin Prediction API", "routes": ["/predict/rf", "/predict/lstm"]}

@app.get("/predict/rf")
def rf_predict():
    try:
        pred = predict_rf()
        return JSONResponse(status_code=200, content={"model": "rf", "next_day_price": pred})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/predict/lstm")
def lstm_predict():
    try:
        pred = predict_lstm()
        return JSONResponse(status_code=200, content={"model": "lstm", "next_day_price": pred})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
