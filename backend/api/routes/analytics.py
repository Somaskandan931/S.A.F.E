"""Analytics and Reporting Routes"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

router = APIRouter( prefix="/api/analytics", tags=["analytics"] )


@router.get( "/risk-distribution" )
async def get_risk_distribution (
        file: str = Query( "full_processed_data.csv" ),
        zone_id: Optional[str] = None
) :
    """Get risk level distribution"""
    try :
        df = pd.read_csv( Path( "data/processed" ) / file )

        if zone_id and 'zone_id' in df.columns :
            df = df[df['zone_id'] == zone_id]

        if 'risk_level' not in df.columns :
            return {"success" : True, "distribution" : {}, "message" : "No risk_level column"}

        distribution = df['risk_level'].value_counts().to_dict()

        return {
            "success" : True,
            "distribution" : distribution,
            "total" : len( df ),
            "zone_id" : zone_id
        }
    except Exception as e :
        raise HTTPException( 500, f"Error calculating distribution: {str( e )}" )


@router.get( "/time-series" )
async def get_time_series (
        file: str = Query( "full_processed_data.csv" ),
        metric: str = Query( "risk_score", description="Metric to plot" ),
        zone_id: Optional[str] = None,
        limit: Optional[int] = 1000
) :
    """Get time series data for visualization"""
    try :
        df = pd.read_csv( Path( "data/processed" ) / file )

        if zone_id and 'zone_id' in df.columns :
            df = df[df['zone_id'] == zone_id]

        if 'timestamp' in df.columns :
            df = df.sort_values( 'timestamp' )

        if limit :
            df = df.tail( limit )

        if metric not in df.columns :
            raise HTTPException( 404, f"Metric not found: {metric}" )

        data = df[['timestamp', metric]].to_dict( orient="records" ) if 'timestamp' in df else df[[metric]].to_dict(
            orient="records" )

        return {
            "success" : True,
            "metric" : metric,
            "data" : data,
            "count" : len( data )
        }
    except HTTPException :
        raise
    except Exception as e :
        raise HTTPException( 500, f"Error generating time series: {str( e )}" )


@router.get( "/alerts" )
async def get_alerts (
        file: str = Query( "full_processed_data.csv" ),
        threshold: float = Query( 0.7, description="Risk threshold" ),
        limit: Optional[int] = 100
) :
    """Get high-risk alerts"""
    try :
        df = pd.read_csv( Path( "data/processed" ) / file )

        if 'risk_score' not in df.columns :
            return {"success" : True, "alerts" : [], "message" : "No risk_score column"}

        alerts_df = df[df['risk_score'] >= threshold]

        if 'timestamp' in alerts_df.columns :
            alerts_df = alerts_df.sort_values( 'timestamp', ascending=False )

        if limit :
            alerts_df = alerts_df.head( limit )

        alerts = alerts_df.to_dict( orient="records" )

        return {
            "success" : True,
            "alerts" : alerts,
            "count" : len( alerts ),
            "threshold" : threshold
        }
    except Exception as e :
        raise HTTPException( 500, f"Error fetching alerts: {str( e )}" )


@router.get( "/summary-report" )
async def get_summary_report ( file: str = Query( "full_processed_data.csv" ) ) :
    """Generate comprehensive summary report"""
    try :
        df = pd.read_csv( Path( "data/processed" ) / file )

        report = {
            "dataset" : {
                "total_records" : len( df ),
                "columns" : list( df.columns ),
                "time_range" : {
                    "start" : df['timestamp'].min() if 'timestamp' in df else None,
                    "end" : df['timestamp'].max() if 'timestamp' in df else None
                }
            },
            "zones" : {},
            "risk_summary" : {},
            "footfall_summary" : {}
        }

        if 'zone_id' in df.columns :
            report["zones"] = {
                "unique_zones" : df['zone_id'].nunique(),
                "zone_list" : df['zone_id'].unique().tolist()
            }

        if 'risk_score' in df.columns :
            report["risk_summary"] = {
                "mean" : float( df['risk_score'].mean() ),
                "max" : float( df['risk_score'].max() ),
                "high_risk_count" : int( (df['risk_score'] > 0.7).sum() )
            }

        if 'footfall_count' in df.columns :
            report["footfall_summary"] = {
                "mean" : float( df['footfall_count'].mean() ),
                "max" : int( df['footfall_count'].max() ),
                "total" : int( df['footfall_count'].sum() )
            }

        return {"success" : True, "report" : report}
    except Exception as e :
        raise HTTPException( 500, f"Error generating report: {str( e )}" )