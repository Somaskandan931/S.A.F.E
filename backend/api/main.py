"""
S.A.F.E Backend API - Optimized for Streamlit Integration
Fast, lightweight API for video-based anomaly detection
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import cv2
import io

app = FastAPI(
    title="S.A.F.E API",
    description="Lightweight API for real-time anomaly detection",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODELS_PATH = Path("C:/Users/somas/PycharmProjects/S.A.F.E/scripts/models")

AVAILABLE_MODELS = {
    'isolation_forest': 'Isolation Forest',
    'lstm_autoencoder': 'LSTM Autoencoder',
    'mad': 'Moving Average Detector',
    'oneclass_svm': 'One-Class SVM',
    'zscore': 'Z-Score Detector'
}

# Global state
class AppState:
    def __init__(self):
        self.loaded_models = {}
        self.current_model = None

app.state.safe = AppState()

# Models
class ModelInfo(BaseModel):
    model_id: str
    name: str
    loaded: bool
    available: bool

class FeatureVector(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    anomaly: bool
    features_used: int

# Helper functions
def load_model_from_disk(model_id: str):
    """Load model from disk"""
    model_path = MODELS_PATH / f"{model_id}.pkl"

    if not model_path.exists():
        return None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def extract_features_from_image(image_bytes):
    """Extract features from image data"""
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract features
    features = {
        'mean_intensity': float(np.mean(gray)),
        'std_intensity': float(np.std(gray)),
        'edge_density': float(np.sum(cv2.Canny(gray, 100, 200)) / gray.size),
        'density': float(np.sum(cv2.Canny(gray, 100, 200)) / gray.size * 100),
        'speed_mean': float(np.std(gray) / 10),  # Placeholder
        'direction_variance': float(np.var(gray) / 100)  # Placeholder
    }

    return features

def predict_anomaly(model, features_dict):
    """Make prediction using loaded model"""
    # Convert features to array
    feature_vector = np.array([[
        features_dict['density'],
        features_dict['speed_mean'],
        features_dict['direction_variance'],
        features_dict.get('mean_intensity', 0),
        features_dict.get('std_intensity', 0)
    ]])

    # Get prediction
    try:
        if hasattr(model, 'predict_score'):
            score = model.predict_score(feature_vector)[0]
        elif hasattr(model, 'decision_function'):
            raw_score = -model.decision_function(feature_vector)[0]
            # Normalize
            score = 1 / (1 + np.exp(-raw_score))
        else:
            prediction = model.predict(feature_vector)[0]
            score = 1.0 if prediction == -1 else 0.0

        # Normalize to 0-1
        score = float(np.clip(score, 0, 1))

        # Determine risk level
        if score >= 0.7:
            risk_level = "CRITICAL"
            anomaly = True
        elif score >= 0.4:
            risk_level = "WARNING"
            anomaly = True
        else:
            risk_level = "NORMAL"
            anomaly = False

        return score, risk_level, anomaly

    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0, "NORMAL", False

# Endpoints
@app.get("/")
async def root():
    return {
        "name": "S.A.F.E API",
        "version": "3.0.0",
        "status": "running",
        "models_available": len(AVAILABLE_MODELS),
        "models_loaded": len(app.state.safe.loaded_models)
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": list(AVAILABLE_MODELS.keys()),
        "models_loaded": list(app.state.safe.loaded_models.keys())
    }

@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    models = []

    for model_id, name in AVAILABLE_MODELS.items():
        model_path = MODELS_PATH / f"{model_id}.pkl"
        models.append(ModelInfo(
            model_id=model_id,
            name=name,
            loaded=model_id in app.state.safe.loaded_models,
            available=model_path.exists()
        ))

    return models

@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str):
    """Load a specific model"""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if already loaded
    if model_id in app.state.safe.loaded_models:
        return {
            "status": "success",
            "message": "Model already loaded",
            "model_id": model_id,
            "model_name": AVAILABLE_MODELS[model_id]
        }

    # Load model
    model = load_model_from_disk(model_id)

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model from {MODELS_PATH / f'{model_id}.pkl'}"
        )

    # Store model
    app.state.safe.loaded_models[model_id] = model
    app.state.safe.current_model = model

    return {
        "status": "success",
        "message": f"Model {AVAILABLE_MODELS[model_id]} loaded successfully",
        "model_id": model_id,
        "model_name": AVAILABLE_MODELS[model_id],
        "model_type": type(model).__name__
    }

@app.post("/api/predict/image", response_model=PredictionResponse)
async def predict_from_image(file: UploadFile = File(...), model_id: Optional[str] = None):
    """Predict anomaly from uploaded image"""
    # Get model
    if model_id and model_id in app.state.safe.loaded_models:
        model = app.state.safe.loaded_models[model_id]
    elif app.state.safe.current_model:
        model = app.state.safe.current_model
    else:
        raise HTTPException(status_code=400, detail="No model loaded")

    # Read image
    image_bytes = await file.read()

    # Extract features
    features = extract_features_from_image(image_bytes)

    # Predict
    risk_score, risk_level, anomaly = predict_anomaly(model, features)

    return PredictionResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        anomaly=anomaly,
        features_used=len(features)
    )

@app.post("/api/predict/features", response_model=PredictionResponse)
async def predict_from_features(request: FeatureVector, model_id: Optional[str] = None):
    """Predict anomaly from feature vector"""
    # Get model
    if model_id and model_id in app.state.safe.loaded_models:
        model = app.state.safe.loaded_models[model_id]
    elif app.state.safe.current_model:
        model = app.state.safe.current_model
    else:
        raise HTTPException(status_code=400, detail="No model loaded")

    # Convert to dict
    features_dict = {
        'density': request.features[0] if len(request.features) > 0 else 0,
        'speed_mean': request.features[1] if len(request.features) > 1 else 0,
        'direction_variance': request.features[2] if len(request.features) > 2 else 0,
        'mean_intensity': request.features[3] if len(request.features) > 3 else 0,
        'std_intensity': request.features[4] if len(request.features) > 4 else 0
    }

    # Predict
    risk_score, risk_level, anomaly = predict_anomaly(model, features_dict)

    return PredictionResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        anomaly=anomaly,
        features_used=len(request.features)
    )

@app.post("/api/predict/batch")
async def predict_batch(features: List[FeatureVector], model_id: Optional[str] = None):
    """Batch prediction"""
    # Get model
    if model_id and model_id in app.state.safe.loaded_models:
        model = app.state.safe.loaded_models[model_id]
    elif app.state.safe.current_model:
        model = app.state.safe.current_model
    else:
        raise HTTPException(status_code=400, detail="No model loaded")

    results = []

    for feature_vec in features:
        features_dict = {
            'density': feature_vec.features[0] if len(feature_vec.features) > 0 else 0,
            'speed_mean': feature_vec.features[1] if len(feature_vec.features) > 1 else 0,
            'direction_variance': feature_vec.features[2] if len(feature_vec.features) > 2 else 0,
            'mean_intensity': feature_vec.features[3] if len(feature_vec.features) > 3 else 0,
            'std_intensity': feature_vec.features[4] if len(feature_vec.features) > 4 else 0
        }

        risk_score, risk_level, anomaly = predict_anomaly(model, features_dict)

        results.append({
            'risk_score': risk_score,
            'risk_level': risk_level,
            'anomaly': anomaly
        })

    return {
        "status": "success",
        "predictions": results,
        "total": len(results),
        "anomalies_detected": sum(1 for r in results if r['anomaly'])
    }

@app.get("/api/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get detailed model information"""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")

    model_path = MODELS_PATH / f"{model_id}.pkl"

    info = {
        "model_id": model_id,
        "name": AVAILABLE_MODELS[model_id],
        "available": model_path.exists(),
        "loaded": model_id in app.state.safe.loaded_models,
        "path": str(model_path)
    }

    if model_id in app.state.safe.loaded_models:
        model = app.state.safe.loaded_models[model_id]
        info["model_type"] = type(model).__name__
        info["has_feature_names"] = hasattr(model, 'feature_names')

        if hasattr(model, 'feature_names'):
            info["feature_names"] = model.feature_names

    return info

@app.delete("/api/models/{model_id}")
async def unload_model(model_id: str):
    """Unload a model from memory"""
    if model_id not in app.state.safe.loaded_models:
        raise HTTPException(status_code=404, detail="Model not loaded")

    del app.state.safe.loaded_models[model_id]

    if app.state.safe.current_model and model_id == list(app.state.safe.loaded_models.keys())[-1]:
        app.state.safe.current_model = None

    return {
        "status": "success",
        "message": f"Model {model_id} unloaded",
        "models_remaining": len(app.state.safe.loaded_models)
    }

@app.get("/api/stats")
async def get_statistics():
    """Get API statistics"""
    return {
        "models_available": len(AVAILABLE_MODELS),
        "models_loaded": len(app.state.safe.loaded_models),
        "current_model": type(app.state.safe.current_model).__name__ if app.state.safe.current_model else None,
        "models_path": str(MODELS_PATH),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🛡️  S.A.F.E API Server")
    print("="*60)
    print(f"\n📁 Models Directory: {MODELS_PATH}")
    print(f"🤖 Available Models: {len(AVAILABLE_MODELS)}")

    # Check model files
    available_count = 0
    for model_id in AVAILABLE_MODELS:
        if (MODELS_PATH / f"{model_id}.pkl").exists():
            available_count += 1
            print(f"   ✅ {AVAILABLE_MODELS[model_id]}")
        else:
            print(f"   ❌ {AVAILABLE_MODELS[model_id]} (file not found)")

    print(f"\n📊 {available_count}/{len(AVAILABLE_MODELS)} models ready")
    print(f"\n🌐 Starting server at http://localhost:8000")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print("\n" + "="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)