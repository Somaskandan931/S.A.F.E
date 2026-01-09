"""Model Training and Prediction Routes - Fixed Version"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
import pickle
import joblib
from pathlib import Path
import pandas as pd
import numpy as np

router = APIRouter(prefix="/api/models", tags=["models"])

MODELS_PATH = Path("models")


class PredictionRequest(BaseModel):
    data: List[dict]
    model_name: str


def unwrap_model(loaded_object):
    """
    Extract actual model from dictionary wrapper or return model directly
    """
    if isinstance(loaded_object, dict):
        # Try common model keys
        for key in ['model', 'clf', 'classifier', 'estimator', 'predictor']:
            if key in loaded_object:
                return loaded_object[key]

        # If no standard key found, raise error
        raise ValueError(f"Dictionary has no standard model key. Available keys: {list(loaded_object.keys())}")

    # Already a model object
    elif hasattr(loaded_object, 'predict'):
        return loaded_object

    else:
        raise TypeError(f"Unknown object type: {type(loaded_object)}")


def load_model_safe(model_path: Path):
    """Load model with dictionary unwrapping"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Try multiple loaders
    loaders = [
        ('pickle', lambda: pickle.load(open(model_path, 'rb'))),
        ('joblib', lambda: joblib.load(model_path)),
    ]

    last_error = None
    for loader_name, loader_func in loaders:
        try:
            loaded_object = loader_func()
            actual_model = unwrap_model(loaded_object)

            if not hasattr(actual_model, 'predict'):
                raise ValueError("Extracted object has no predict() method")

            return actual_model

        except Exception as e:
            last_error = e
            continue

    raise Exception(f"All loaders failed. Last error: {last_error}")


@router.get("/list")
async def list_models():
    """List all available trained models"""
    try:
        models = []
        for model_file in MODELS_PATH.glob("*.pkl"):
            size = model_file.stat().st_size / 1024  # KB
            models.append({
                "name": model_file.stem,
                "filename": model_file.name,
                "size_kb": round(size, 2)
            })

        return {"success": True, "models": models, "total": len(models)}
    except Exception as e:
        raise HTTPException(500, f"Error listing models: {str(e)}")


@router.get("/load/{model_name}")
async def load_model(model_name: str):
    """Load a trained model with proper unwrapping"""
    try:
        model_path = MODELS_PATH / f"{model_name}.pkl"

        # Load with unwrapping
        model = load_model_safe(model_path)

        return {
            "success": True,
            "model_name": model_name,
            "model_type": type(model).__name__,
            "loaded": True,
            "has_predict": hasattr(model, 'predict'),
            "has_decision_function": hasattr(model, 'decision_function')
        }
    except FileNotFoundError:
        raise HTTPException(404, f"Model not found: {model_name}")
    except Exception as e:
        raise HTTPException(500, f"Error loading model: {str(e)}")


@router.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using a trained model"""
    try:
        model_path = MODELS_PATH / f"{request.model_name}.pkl"

        # Load model with unwrapping
        model = load_model_safe(model_path)

        # Convert data to DataFrame
        df = pd.DataFrame(request.data)

        # Make predictions
        predictions = model.predict(df)

        # Convert to list (handle different return types)
        if isinstance(predictions, np.ndarray):
            predictions_list = predictions.tolist()
        else:
            predictions_list = list(predictions)

        return {
            "success": True,
            "model_name": request.model_name,
            "predictions": predictions_list,
            "count": len(predictions_list)
        }
    except FileNotFoundError:
        raise HTTPException(404, f"Model not found: {request.model_name}")
    except Exception as e:
        raise HTTPException(500, f"Error making predictions: {str(e)}")


@router.get("/info/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a model"""
    try:
        model_path = MODELS_PATH / f"{model_name}.pkl"

        if not model_path.exists():
            raise HTTPException(404, f"Model not found: {model_name}")

        # Load and inspect
        model = load_model_safe(model_path)

        info = {
            "name": model_name,
            "type": type(model).__name__,
            "methods": [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))],
            "has_predict": hasattr(model, 'predict'),
            "has_predict_proba": hasattr(model, 'predict_proba'),
            "has_decision_function": hasattr(model, 'decision_function'),
            "has_score": hasattr(model, 'score'),
        }

        # Try to get additional attributes
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_

        if hasattr(model, 'classes_'):
            info['classes'] = model.classes_.tolist()

        return {"success": True, "info": info}

    except Exception as e:
        raise HTTPException(500, f"Error getting model info: {str(e)}")