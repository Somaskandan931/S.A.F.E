"""Data Management Routes"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import pandas as pd
from pathlib import Path

router = APIRouter( prefix="/api/data", tags=["data"] )

DATA_PATH = Path( "data/processed" )


@router.get( "/load" )
async def load_data (
        file: str = Query( ..., description="Data file to load" ),
        limit: Optional[int] = Query( None, description="Limit rows" )
) :
    """Load processed data file"""
    try :
        file_path = DATA_PATH / file
        if not file_path.exists() :
            raise HTTPException( 404, f"File not found: {file}" )

        df = pd.read_csv( file_path )
        if limit :
            df = df.head( limit )

        return {
            "success" : True,
            "data" : df.to_dict( orient="records" ),
            "shape" : df.shape,
            "columns" : list( df.columns )
        }
    except Exception as e :
        raise HTTPException( 500, f"Error loading data: {str( e )}" )


@router.get( "/files" )
async def list_data_files () :
    """List available data files"""
    try :
        files = []
        for path in DATA_PATH.glob( "*.csv" ) :
            size = path.stat().st_size / (1024 * 1024)  # MB
            files.append( {
                "name" : path.name,
                "size_mb" : round( size, 2 ),
                "path" : str( path.relative_to( Path.cwd() ) )
            } )
        return {"success" : True, "files" : files}
    except Exception as e :
        raise HTTPException( 500, f"Error listing files: {str( e )}" )


@router.get( "/summary" )
async def data_summary ( file: str = Query( ... ) ) :
    """Get data summary statistics"""
    try :
        df = pd.read_csv( DATA_PATH / file )

        summary = {
            "shape" : df.shape,
            "columns" : list( df.columns ),
            "dtypes" : df.dtypes.astype( str ).to_dict(),
            "missing" : df.isnull().sum().to_dict(),
            "numeric_summary" : df.describe().to_dict() if len(
                df.select_dtypes( include='number' ).columns ) > 0 else {}
        }

        return {"success" : True, "summary" : summary}
    except Exception as e :
        raise HTTPException( 500, f"Error generating summary: {str( e )}" )