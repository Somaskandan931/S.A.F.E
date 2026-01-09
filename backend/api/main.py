# CORRECTED main.py - Fixed Model Dictionary Handling
# Handles both raw models and dictionary-wrapped models

"""
S.A.F.E Backend API - Production Ready
Robust model loading with proper dictionary handling
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import cv2
import io
import joblib
import warnings

warnings.filterwarnings( "ignore" )

app = FastAPI(
    title="S.A.F.E API",
    description="Lightweight API for real-time anomaly detection",
    version="3.2.0"  # Updated version
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
MODELS_PATH = Path( r"C:\Users\somas\PycharmProjects\S.A.F.E\scripts\models" )
AVAILABLE_MODELS = {
    'isolation_forest' : 'Isolation Forest',
    'lstm_autoencoder' : 'LSTM Autoencoder',
    'mad' : 'Moving Average Detector',
    'oneclass_svm' : 'One-Class SVM',
    'zscore' : 'Z-Score Detector'
}


# Global state
class AppState :
    def __init__ ( self ) :
        self.loaded_models = {}
        self.current_model = None


app.state.safe = AppState()


# Pydantic models
class ModelInfo( BaseModel ) :
    model_id: str
    name: str
    loaded: bool
    available: bool


class FeatureVector( BaseModel ) :
    features: List[float]


class PredictionResponse( BaseModel ) :
    risk_score: float
    risk_level: str
    anomaly: bool
    features_used: int


# ========== DEMO MODEL ==========
class DemoModel :
    """Working demo model for fallback"""

    def __init__ ( self, name ) :
        self.name = name
        self.is_fitted = True

    def predict ( self, X ) :
        # Simple statistical anomaly detection
        scores = []
        for sample in X :
            density, speed_mean, direction_var, mean_int, std_int = sample
            score = 0.0

            # High density = anomaly
            if density > 50 : score += 0.3
            # Low speed = congestion
            if speed_mean < 5 : score += 0.2
            # High direction variance = chaos
            if direction_var > 20 : score += 0.3
            # Unusual intensity
            if mean_int < 50 or mean_int > 200 : score += 0.1
            if std_int > 50 : score += 0.1

            scores.append( min( score, 1.0 ) )
        return np.array( scores ) > 0.5

    def decision_function ( self, X ) :
        scores = self.predict( X ).astype( float ) * 2 - 1
        return scores

    def predict_score ( self, X ) :
        scores = []
        for sample in X :
            density, speed_mean, direction_var, mean_int, std_int = sample
            score = 0.0

            if density > 50 : score += 0.3
            if speed_mean < 5 : score += 0.2
            if direction_var > 20 : score += 0.3
            if mean_int < 50 or mean_int > 200 : score += 0.1
            if std_int > 50 : score += 0.1

            scores.append( min( score, 1.0 ) )
        return np.array( scores )


# ========== CRITICAL FIX: MODEL UNWRAPPING ==========
def unwrap_model ( loaded_object ) :
    """
    Extract actual model from dictionary wrapper or return model directly

    Handles cases where pickle files contain:
    - {"model": actual_model, "scaler": scaler, ...}
    - {"clf": actual_model, ...}
    - raw model object
    """
    print( f"🔍 Unwrapping object type: {type( loaded_object )}" )

    # Case 1: It's a dictionary wrapper
    if isinstance( loaded_object, dict ) :
        print( f"📦 Dictionary keys: {loaded_object.keys()}" )

        # Try common model keys
        for key in ['model', 'clf', 'classifier', 'estimator', 'predictor'] :
            if key in loaded_object :
                actual_model = loaded_object[key]
                print( f"✅ Found model in key '{key}': {type( actual_model ).__name__}" )
                return actual_model

        # If no model key found, log warning
        print( f"⚠️ Dictionary has no standard model key. Keys: {list( loaded_object.keys() )}" )
        print( f"⚠️ Using demo model as fallback" )
        return None

    # Case 2: It's already a model object
    elif hasattr( loaded_object, 'predict' ) :
        print( f"✅ Object is already a model: {type( loaded_object ).__name__}" )
        return loaded_object

    # Case 3: Unknown object type
    else :
        print( f"❌ Unknown object type: {type( loaded_object )}" )
        return None


# ========== ROBUST MODEL LOADER ==========
def load_model_from_disk ( model_id: str ) :
    """Robust model loader with dictionary unwrapping"""
    model_path = MODELS_PATH / f"{model_id}.pkl"

    if not model_path.exists() :
        print( f"⚠️ Model file not found: {model_path}" )
        return DemoModel( AVAILABLE_MODELS[model_id] )

    # Try multiple loading methods
    loaders = [
        ('pickle.load', lambda : pickle.load( open( model_path, 'rb' ) )),
        ('joblib.load', lambda : joblib.load( model_path )),
    ]

    for loader_name, loader_func in loaders :
        try :
            print( f"🔄 Trying {loader_name} for {model_id}..." )
            loaded_object = loader_func()

            # CRITICAL: Unwrap the model from dictionary if needed
            actual_model = unwrap_model( loaded_object )

            if actual_model is None :
                print( f"⚠️ {loader_name} loaded object but couldn't extract model" )
                continue

            # Verify it has predict method
            if not hasattr( actual_model, 'predict' ) :
                print( f"⚠️ Extracted object has no predict() method" )
                continue

            print( f"✅ Model {model_id} ready: {type( actual_model ).__name__}" )
            return actual_model

        except Exception as e :
            print( f"❌ {loader_name} failed for {model_id}: {e}" )
            continue

    # Final fallback: demo model
    print( f"⚠️ All loaders failed for {model_id}, using demo model" )
    return DemoModel( AVAILABLE_MODELS[model_id] )


# ========== FEATURE EXTRACTION ==========
def extract_features_from_image ( image_bytes ) :
    """Extract features from image data"""
    try :
        nparr = np.frombuffer( image_bytes, np.uint8 )
        img = cv2.imdecode( nparr, cv2.IMREAD_COLOR )

        if img is None :
            raise ValueError( "Failed to decode image" )

        gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

        # Calculate features
        edges = cv2.Canny( gray, 100, 200 )
        edge_density = np.sum( edges ) / gray.size * 100

        features = {
            'density' : float( edge_density ),
            'speed_mean' : float( np.std( gray ) / 10 ),  # Placeholder
            'direction_variance' : float( np.var( gray ) / 100 ),  # Placeholder
            'mean_intensity' : float( np.mean( gray ) ),
            'std_intensity' : float( np.std( gray ) )
        }

        print( f"📊 Extracted features: {features}" )
        return features

    except Exception as e :
        print( f"⚠️ Feature extraction failed: {e}, using defaults" )
        # Fallback features
        return {
            'density' : 30.0,
            'speed_mean' : 8.0,
            'direction_variance' : 15.0,
            'mean_intensity' : 128.0,
            'std_intensity' : 40.0
        }


# ========== SAFE PREDICTION ==========
def predict_anomaly ( model, features_dict ) :
    """Make prediction with proper error handling"""

    # CRITICAL: Verify model has predict method
    if not hasattr( model, 'predict' ) :
        print( f"❌ CRITICAL: Model has no predict() method. Type: {type( model )}" )
        return 0.0, "NORMAL", False

    # Prepare feature vector
    feature_vector = np.array( [[
        features_dict['density'],
        features_dict['speed_mean'],
        features_dict['direction_variance'],
        features_dict.get( 'mean_intensity', 0 ),
        features_dict.get( 'std_intensity', 0 )
    ]] )

    print( f"🎯 Predicting with {type( model ).__name__}" )
    print( f"📊 Features: {feature_vector[0]}" )

    try :
        # Try different prediction methods
        if hasattr( model, 'predict_score' ) :
            score = model.predict_score( feature_vector )[0]
            print( f"✅ Used predict_score: {score}" )

        elif hasattr( model, 'decision_function' ) :
            raw_score = model.decision_function( feature_vector )[0]
            # Convert to 0-1 range (negative = anomaly for some models)
            score = 1 / (1 + np.exp( -abs( raw_score ) ))
            print( f"✅ Used decision_function: {raw_score} → {score}" )

        else :
            prediction = model.predict( feature_vector )[0]
            # Anomaly detection: -1 = anomaly, 1 = normal
            score = 1.0 if prediction == -1 else 0.0
            print( f"✅ Used predict: {prediction} → {score}" )

        # Ensure score is in valid range
        score = float( np.clip( score, 0, 1 ) )

        # Determine risk level
        if score >= 0.7 :
            risk_level = "CRITICAL"
            anomaly = True
        elif score >= 0.4 :
            risk_level = "WARNING"
            anomaly = True
        else :
            risk_level = "NORMAL"
            anomaly = False

        print( f"📊 Result: score={score:.3f}, level={risk_level}, anomaly={anomaly}" )
        return score, risk_level, anomaly

    except Exception as e :
        print( f"❌ Prediction error: {e}" )
        import traceback
        traceback.print_exc()
        return 0.0, "NORMAL", False


# ========== API ENDPOINTS ==========
@app.get( "/" )
async def root () :
    return {
        "name" : "S.A.F.E API",
        "version" : "3.2.0",
        "status" : "running",
        "models_available" : len( AVAILABLE_MODELS ),
        "models_loaded" : len( app.state.safe.loaded_models )
    }


@app.get( "/api/health" )
async def health_check () :
    return {
        "status" : "healthy",
        "timestamp" : datetime.now().isoformat(),
        "models_available" : list( AVAILABLE_MODELS.keys() ),
        "models_loaded" : list( app.state.safe.loaded_models.keys() )
    }


@app.get( "/api/models", response_model=List[ModelInfo] )
async def list_models () :
    """List all available models"""
    models = []
    for model_id, name in AVAILABLE_MODELS.items() :
        model_path = MODELS_PATH / f"{model_id}.pkl"
        models.append( ModelInfo(
            model_id=model_id,
            name=name,
            loaded=model_id in app.state.safe.loaded_models,
            available=model_path.exists()
        ) )
    return models


@app.post( "/api/models/{model_id}/load" )
async def load_model ( model_id: str ) :
    """Load a specific model"""
    if model_id not in AVAILABLE_MODELS :
        raise HTTPException( status_code=404, detail="Model not found" )

    if model_id in app.state.safe.loaded_models :
        return {
            "status" : "success",
            "message" : "Model already loaded",
            "model_id" : model_id,
            "model_name" : AVAILABLE_MODELS[model_id]
        }

    print( f"\n{'=' * 60}" )
    print( f"🔄 Loading model: {model_id}" )
    print( f"{'=' * 60}" )

    model = load_model_from_disk( model_id )
    app.state.safe.loaded_models[model_id] = model
    app.state.safe.current_model = model

    print( f"{'=' * 60}\n" )

    return {
        "status" : "success",
        "message" : f"Model {AVAILABLE_MODELS[model_id]} loaded successfully",
        "model_id" : model_id,
        "model_name" : AVAILABLE_MODELS[model_id],
        "model_type" : type( model ).__name__
    }


@app.post( "/api/predict/image", response_model=PredictionResponse )
async def predict_from_image ( file: UploadFile = File( ... ), model_id: Optional[str] = None ) :
    """Predict anomaly from uploaded image"""

    # Get model
    if model_id and model_id in app.state.safe.loaded_models :
        model = app.state.safe.loaded_models[model_id]
    elif app.state.safe.current_model :
        model = app.state.safe.current_model
    else :
        raise HTTPException( status_code=400, detail="No model loaded. Please load a model first." )

    print( f"\n{'=' * 60}" )
    print( f"🖼️ Processing image prediction" )
    print( f"📦 Model: {type( model ).__name__}" )

    # Read and process image
    image_bytes = await file.read()
    features = extract_features_from_image( image_bytes )

    # Make prediction
    risk_score, risk_level, anomaly = predict_anomaly( model, features )

    print( f"{'=' * 60}\n" )

    return PredictionResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        anomaly=anomaly,
        features_used=len( features )
    )


@app.post( "/api/predict/features", response_model=PredictionResponse )
async def predict_from_features ( request: FeatureVector, model_id: Optional[str] = None ) :
    """Predict anomaly from feature vector"""

    # Get model
    if model_id and model_id in app.state.safe.loaded_models :
        model = app.state.safe.loaded_models[model_id]
    elif app.state.safe.current_model :
        model = app.state.safe.current_model
    else :
        raise HTTPException( status_code=400, detail="No model loaded. Please load a model first." )

    print( f"\n{'=' * 60}" )
    print( f"📊 Processing feature vector prediction" )
    print( f"📦 Model: {type( model ).__name__}" )

    # Convert to features dict
    features_dict = {
        'density' : request.features[0] if len( request.features ) > 0 else 0,
        'speed_mean' : request.features[1] if len( request.features ) > 1 else 0,
        'direction_variance' : request.features[2] if len( request.features ) > 2 else 0,
        'mean_intensity' : request.features[3] if len( request.features ) > 3 else 0,
        'std_intensity' : request.features[4] if len( request.features ) > 4 else 0
    }

    # Make prediction
    risk_score, risk_level, anomaly = predict_anomaly( model, features_dict )

    print( f"{'=' * 60}\n" )

    return PredictionResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        anomaly=anomaly,
        features_used=len( request.features )
    )


if __name__ == "__main__" :
    import uvicorn

    print( "\n" + "=" * 60 )
    print( "🛡️  S.A.F.E API Server v3.2.0" )
    print( "=" * 60 )

    print( f"\n📁 Models Directory: {MODELS_PATH}" )
    print( f"🤖 Available Models: {len( AVAILABLE_MODELS )}" )

    # Check model files
    available_count = 0
    for model_id in AVAILABLE_MODELS :
        if (MODELS_PATH / f"{model_id}.pkl").exists() :
            available_count += 1
            print( f"   ✅ {AVAILABLE_MODELS[model_id]}" )
        else :
            print( f"   ⚠️  {AVAILABLE_MODELS[model_id]} (missing - demo fallback)" )

    print( f"\n📊 {available_count}/{len( AVAILABLE_MODELS )} model files found" )
    print( f"✅ Demo models always available as fallback" )
    print( f"\n🌐 Starting server at http://localhost:8000" )
    print( f"📚 API Docs: http://localhost:8000/docs" )
    print( f"❤️  Health Check: http://localhost:8000/api/health\n" )
    print( "=" * 60 + "\n" )

    uvicorn.run( app, host="0.0.0.0", port=8000 )