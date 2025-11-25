# app_fastapi.py - Enhanced with multi-day prediction endpoints
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_rf, predict_lstm, predict_rf_single, predict_lstm_single

app = FastAPI(
    title="Bitcoin Prediction API",
    description="Multi-day Bitcoin price prediction using RF and LSTM models",
    version="2.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "Bitcoin Prediction API v2.0",
        "routes": {
            "single_day": ["/predict/rf", "/predict/lstm"],
            "multi_day": ["/predict/rf/multi", "/predict/lstm/multi"],
            "example": "/predict/rf/multi?days=7"
        }
    }

@app.get("/predict/rf")
def rf_predict():
    """Single next-day prediction using Random Forest."""
    try:
        pred = predict_rf_single()
        if pred is None:
            return JSONResponse(
                status_code=500, 
                content={"error": "RF prediction failed. Check model files."}
            )
        return JSONResponse(
            status_code=200, 
            content={"model": "rf", "next_day_price": float(pred)}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/predict/lstm")
def lstm_predict():
    """Single next-day prediction using LSTM."""
    try:
        pred = predict_lstm_single()
        if pred is None:
            return JSONResponse(
                status_code=500,
                content={"error": "LSTM prediction failed. Check model files."}
            )
        return JSONResponse(
            status_code=200,
            content={"model": "lstm", "next_day_price": float(pred)}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/predict/rf/multi")
def rf_predict_multi(days: int = Query(default=7, ge=1, le=30)):
    """
    Multi-day prediction using Random Forest.
    
    Parameters:
    - days: Number of days to predict (1-30, default 7)
    """
    try:
        result = predict_rf(days=days)
        if not result.get("success"):
            return JSONResponse(
                status_code=500,
                content={"error": result.get("error", "Unknown error")}
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "model": "Random Forest",
                "days_predicted": days,
                "predictions": result["predictions"],
                "dates": result["dates"],
                "current_price": result["predictions"][0] if result["predictions"] else None
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/predict/lstm/multi")
def lstm_predict_multi(days: int = Query(default=7, ge=1, le=30)):
    """
    Multi-day prediction using LSTM.
    
    Parameters:
    - days: Number of days to predict (1-30, default 7)
    """
    try:
        result = predict_lstm(days=days)
        if not result.get("success"):
            return JSONResponse(
                status_code=500,
                content={"error": result.get("error", "Unknown error")}
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "model": "LSTM",
                "days_predicted": days,
                "predictions": result["predictions"],
                "dates": result["dates"],
                "current_price": result["predictions"][0] if result["predictions"] else None
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/predict/compare")
def compare_models(days: int = Query(default=7, ge=1, le=30)):
    """
    Compare RF and LSTM predictions side by side.
    
    Parameters:
    - days: Number of days to predict (1-30, default 7)
    """
    try:
        rf_result = predict_rf(days=days)
        lstm_result = predict_lstm(days=days)
        
        comparison = {
            "days_predicted": days,
            "models": {
                "random_forest": {
                    "success": rf_result.get("success", False),
                    "predictions": rf_result.get("predictions", []),
                    "dates": rf_result.get("dates", [])
                },
                "lstm": {
                    "success": lstm_result.get("success", False),
                    "predictions": lstm_result.get("predictions", []),
                    "dates": lstm_result.get("dates", [])
                }
            }
        }
        
        return JSONResponse(status_code=200, content=comparison)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
def health_check():
    """Health check endpoint."""
    import os
    return {
        "status": "healthy",
        "service": "Bitcoin Prediction API",
        "models": {
            "rf": os.path.exists("models/rf_model.pkl"),
            "lstm": os.path.exists("models/lstm_model.h5") or os.path.exists("models/lstm_model.keras")
        }
    }