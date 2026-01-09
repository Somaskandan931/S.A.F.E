from fastapi import APIRouter, HTTPException
import pandas as pd
from pathlib import Path

router = APIRouter()

# Path to processed data
DATA_PATH = Path("data/processed")


def load_dataset(file_name: str) -> pd.DataFrame:
    """Safely load dataset CSV"""
    file_path = DATA_PATH / file_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{file_name}' not found"
        )

    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dataset: {e}"
        )


@router.get("/zones")
def list_zones(file: str):
    """
    List all available zones in a dataset
    """
    df = load_dataset(file)

    if "zone_id" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="zone_id column missing in dataset"
        )

    zones = sorted(df["zone_id"].unique().tolist())

    return {
        "file": file,
        "total_zones": len(zones),
        "zones": zones
    }


@router.get("/zones/{zone_id}")
def get_zone_details(zone_id: int, file: str):
    """
    Get statistics for a specific zone
    """
    df = load_dataset(file)

    if "zone_id" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="zone_id column missing in dataset"
        )

    # ✅ FIX: zone_id is INTEGER
    zone_df = df[df["zone_id"] == zone_id]

    if zone_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for zone_id={zone_id}"
        )

    summary = {
        "zone_id": zone_id,
        "records": len(zone_df),
        "avg_footfall": float(zone_df["footfall_count"].mean())
        if "footfall_count" in zone_df.columns else None,
        "avg_density": float(zone_df["density"].mean())
        if "density" in zone_df.columns else None,
        "avg_speed": float(zone_df["velocity_mean"].mean())
        if "velocity_mean" in zone_df.columns else None
    }

    return summary
