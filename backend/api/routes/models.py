"""Model Training and Prediction Routes"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
import pickle
from pathlib import Path
import pandas as pd

router = APIRouter( prefix="/api/models", tags=["models"] )

MODELS_PATH = Path( "models" )


class PredictionRequest( BaseModel ) :
    data: List[dict]
    model_name: str


@router.get( "/list" )
async def list_models () :
    """List all available trained models"""
    try :
        models = []
        for model_file in MODELS_PATH.glob( "*.pkl" ) :
            size = model_file.stat().st_size / 1024  # KB
            models.append( {
                "name" : model_file.stem,
                "filename" : model_file.name,
                "size_kb" : round( size, 2 )
            } )

        return {"success" : True, "models" : models, "total" : len( models )}
    except Exception as e :
        raise HTTPException( 500, f"Error listing models: {str( e )}" )


@router.get( "/load/{model_name}" )
async def load_model ( model_name: str ) :
    """Load a trained model"""
    try :
        model_path = MODELS_PATH / f"{model_name}.pkl"

        if not model_path.exists() :
            raise HTTPException( 404, f"Model not found: {model_name}" )

        with open( model_path, 'rb' ) as f :
            model = pickle.load( f )

        return {
            "success" : True,
            "model_name" : model_name,
            "model_type" : type( model ).__name__,
            "loaded" : True
        }
    except HTTPException :
        raise
    except Exception as e :
        raise HTTPException( 500, f"Error loading model: {str( e )}" )


@router.post( "/predict" )
async def predict ( request: PredictionRequest ) :
    """Make predictions using a trained model"""
    try :
        model_path = MODELS_PATH / f"{request.model_name}.pkl"

        if not model_path.exists() :
            raise HTTPException( 404, f"Model not found: {request.model_name}" )

        with open( model_path, 'rb' ) as f :
            model = pickle.load( f )

        df = pd.DataFrame( request.data )
        predictions = model.predict( df )

        return {
            "success" : True,
            "model_name" : request.model_name,
            "predictions" : predictions.tolist(),
            "count" : len( predictions )
        }
    except HTTPException :
        raise
    except Exception as e :
        raise HTTPException( 500, f"Error making predictions: {str( e )}" )
