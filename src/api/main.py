from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import pickle
import pandas as pd
import time
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))

from src.api.schemas import RefundRequest, PredictionResponse

# Initialize FastAPI
app = FastAPI(
    title="Refund Fraud Detection API",
    description="Real-time fraud detection for refund requests",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'fraud_predictions_total',
    'Total number of predictions',
    ['result']
)
PREDICTION_LATENCY = Histogram(
    'fraud_prediction_latency_seconds',
    'Prediction latency in seconds'
)

# Global model cache
model_cache = {
    'model': None,
    'artifacts': None,
    'loaded_at': None
}

def load_model():
    """Load model and artifacts"""
    if model_cache['model'] is None:
        try:
            with open('models/fraud_detector.pkl', 'rb') as f:
                artifacts = pickle.load(f)
            
            model_cache['artifacts'] = artifacts
            model_cache['model'] = artifacts['model']
            model_cache['loaded_at'] = datetime.now()
            
            print(f"✅ Model loaded at {model_cache['loaded_at']}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    return model_cache['artifacts']

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()
    print("🚀 Fraud Detection API started")

@app.get("/")
def root():
    return {
        "service": "Refund Fraud Detection API",
        "version": "1.0.0",
        "status": "healthy",
        "model_loaded": model_cache['model'] is not None,
        "model_loaded_at": model_cache['loaded_at'].isoformat() if model_cache['loaded_at'] else None
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_cache['model'] is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(request: RefundRequest):
    """Predict if refund request is fraudulent"""
    
    start_time = time.time()
    
    try:
        # Load model
        artifacts = load_model()
        model = artifacts['model']
        feature_engineer = artifacts['feature_engineer']
        threshold = artifacts['threshold']
        
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Apply feature engineering
        X = feature_engineer.create_features(input_data, is_training=False)
        
        # Make prediction
        fraud_proba = model.predict_proba(X)[0, 1]
        is_fraud = bool(fraud_proba >= threshold)
        
        # Calculate risk level
        if fraud_proba >= 0.8:
            risk_level = "HIGH"
        elif fraud_proba >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Update metrics
        PREDICTION_COUNTER.labels(result='fraud' if is_fraud else 'legitimate').inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return PredictionResponse(
            transaction_id=request.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=round(float(fraud_proba), 4),
            risk_level=risk_level,
            confidence=round(float(abs(fraud_proba - 0.5) * 2), 4),
            inference_time_ms=round(inference_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model/info")
def model_info():
    """Get model information"""
    artifacts = load_model()
    
    return {
        "model_type": "XGBoost",
        "threshold": artifacts['threshold'],
        "features_count": len(artifacts['feature_names']),
        "metadata": artifacts['metadata'],
        "loaded_at": model_cache['loaded_at'].isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)