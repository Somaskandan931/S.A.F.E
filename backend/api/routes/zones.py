"""Zone Management Routes"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
from pathlib import Path

router = APIRouter( prefix="/api/zones", tags=["zones"] )


@router.get( "/list" )
async def list_zones (
        file: str = Query( "full_processed_data.csv" )
) :
    """List all unique zones in dataset"""
    try :
        df = pd.read_csv( Path( "data/processed" ) / file )

        if 'zone_id' not in df.columns :
            return {"success" : True, "zones" : [], "message" : "No zone_id column found"}

        zones = df['zone_id'].unique().tolist()
        zone_data = []

        for zone in zones :
            zone_df = df[df['zone_id'] == zone]
            zone_data.append( {
                "zone_id" : zone,
                "count" : len( zone_df ),
                "avg_footfall" : zone_df['footfall_count'].mean() if 'footfall_count' in df.columns else None,
                "max_risk" : zone_df['risk_score'].max() if 'risk_score' in df.columns else None
            } )

        return {"success" : True, "zones" : zone_data, "total" : len( zones )}
    except Exception as e :
        raise HTTPException( 500, f"Error listing zones: {str( e )}" )


@router.get( "/{zone_id}" )
async def get_zone_data (
        zone_id: str,
        file: str = Query( "full_processed_data.csv" ),
        limit: Optional[int] = None
) :
    """Get data for specific zone"""
    try :
        df = pd.read_csv( Path( "data/processed" ) / file )

        if 'zone_id' not in df.columns :
            raise HTTPException( 404, "No zone_id column in dataset" )

        zone_df = df[df['zone_id'] == zone_id]

        if len( zone_df ) == 0 :
            raise HTTPException( 404, f"Zone {zone_id} not found" )

        if limit :
            zone_df = zone_df.head( limit )

        return {
            "success" : True,
            "zone_id" : zone_id,
            "data" : zone_df.to_dict( orient="records" ),
            "count" : len( zone_df )
        }
    except HTTPException :
        raise
    except Exception as e :
        raise HTTPException( 500, f"Error fetching zone data: {str( e )}" )


@router.get( "/{zone_id}/stats" )
async def get_zone_statistics (
        zone_id: str,
        file: str = Query( "full_processed_data.csv" )
) :
    """Get statistics for specific zone"""
    try :
        df = pd.read_csv( Path( "data/processed" ) / file )
        zone_df = df[df['zone_id'] == zone_id]

        if len( zone_df ) == 0 :
            raise HTTPException( 404, f"Zone {zone_id} not found" )

        stats = {
            "zone_id" : zone_id,
            "total_records" : len( zone_df ),
            "time_range" : {
                "start" : zone_df['timestamp'].min() if 'timestamp' in zone_df else None,
                "end" : zone_df['timestamp'].max() if 'timestamp' in zone_df else None
            },
            "footfall" : {
                "mean" : zone_df['footfall_count'].mean() if 'footfall_count' in zone_df else None,
                "max" : zone_df['footfall_count'].max() if 'footfall_count' in zone_df else None,
                "min" : zone_df['footfall_count'].min() if 'footfall_count' in zone_df else None
            },
            "risk" : {
                "mean" : zone_df['risk_score'].mean() if 'risk_score' in zone_df else None,
                "max" : zone_df['risk_score'].max() if 'risk_score' in zone_df else None,
                "high_risk_count" : (zone_df['risk_level'] == 'High').sum() if 'risk_level' in zone_df else None
            }
        }

        return {"success" : True, "statistics" : stats}
    except HTTPException :
        raise
    except Exception as e :
        raise HTTPException( 500, f"Error calculating statistics: {str( e )}" )
