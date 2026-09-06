"""
pipeline.py
End-to-end integration pipeline for the OilTrace-AI Drift Engine (M3).

Coordinates backward hindcasting and forward forecasting by integrating inputs
from Satellite AI/ML (M1, M2) and generating downstream contracts for
AIS Attribution (M4), FastAPI Backend (M5), and Leaflet Dashboard (M6).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import math

from src.drift.drift_result import DriftResult
from src.drift.simulation.backward import run_backward_drift
from src.drift.simulation.forward import run_forward_drift


def run_drift_pipeline(
    spill_id: str = "spill_001",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    detection_time: Optional[datetime] = None,
    spill_geometry: Optional[Union[Dict[str, Any], Any]] = None,
    detection_result: Optional[Union[Dict[str, Any], Any]] = None,
    duration_backward_hours: Optional[int] = None,
    duration_forward_hours: int = 24,
    current_data_path: Optional[str | Path] = "data/currents/test_currents.nc",
    wind_data_path: Optional[str | Path] = None,
    num_particles: int = 30,
) -> DriftResult:
    """
    Run full bidirectional (hindcast + forecast) drift simulation pipeline.

    Seamlessly integrates inputs from:
    - M2 (Neha): SpillGeometry (centroid, area_km2, geometry/polygon)
    - M1 (Nidhi): DetectionResult (estimated age proxy, confidence)

    And outputs:
    - DriftResult: Tailored for M4 (Priya AIS), M5 (Prachi API), M6 (Mansi UI).
    """
    # 1. Resolve Detection Coordinates and Timestamp
    detected_lat = latitude
    detected_lon = longitude
    spill_area_km2 = None
    polygon_coords = None

    # Ingest M2 (Neha - Satellite & Geospatial) SpillGeometry
    if spill_geometry is not None:
        if isinstance(spill_geometry, dict):
            centroid = spill_geometry.get("centroid")
            if centroid:
                if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
                    detected_lat, detected_lon = centroid[0], centroid[1]
                elif isinstance(centroid, dict):
                    detected_lat = centroid.get("lat", detected_lat)
                    detected_lon = centroid.get("lon", detected_lon)
            spill_area_km2 = spill_geometry.get("area_km2", spill_area_km2)
            polygon_coords = spill_geometry.get("coordinates")
        else:
            # Pydantic model or object attribute access
            if hasattr(spill_geometry, "centroid") and spill_geometry.centroid:
                detected_lat = getattr(spill_geometry.centroid, "lat", detected_lat)
                detected_lon = getattr(spill_geometry.centroid, "lon", detected_lon)
            spill_area_km2 = getattr(spill_geometry, "area_km2", spill_area_km2)

    if detected_lat is None or detected_lon is None:
        raise ValueError("Latitude and Longitude must be provided directly or via spill_geometry.")

    if detection_time is None:
        detection_time = datetime(2025, 1, 2, 0, 0)

    # 2. Ingest M1 (Nidhi - AI/ML) Detection Result (Age proxy)
    if duration_backward_hours is None:
        if detection_result is not None:
            if isinstance(detection_result, dict):
                duration_backward_hours = detection_result.get("age_proxy_hours", 24)
            else:
                duration_backward_hours = getattr(detection_result, "age_proxy_hours", 24)
        else:
            duration_backward_hours = 24

    # Calculate initial seeding radius based on detected slick area
    radius_seed_m = 1000.0
    if spill_area_km2 and spill_area_km2 > 0:
        # Area = pi * r^2  =>  r = sqrt(area / pi) * 1000 meters
        radius_seed_m = max(500.0, float(math.sqrt(spill_area_km2 / math.pi) * 1000.0))

    # 3. Execute Backward Hindcasting (Origin Reconstruction)
    backward_output = run_backward_drift(
        longitude=detected_lon,
        latitude=detected_lat,
        detection_time=detection_time,
        duration_hours=duration_backward_hours,
        data_path=current_data_path,
        wind_path=wind_data_path,
        num_particles=num_particles,
        radius_seed_m=radius_seed_m,
        polygon_coords=polygon_coords,
    )

    # 4. Execute Forward Forecasting (Disaster Threat & Shoreline Impact)
    forward_output = run_forward_drift(
        longitude=detected_lon,
        latitude=detected_lat,
        start_time=detection_time,
        duration_hours=duration_forward_hours,
        data_path=current_data_path,
        wind_path=wind_data_path,
        num_particles=num_particles,
        radius_seed_m=radius_seed_m,
    )

    # 5. Assemble and Return Unified DriftResult
    result = DriftResult(
        spill_id=spill_id,
        detection_time=detection_time,
        detection_lat=detected_lat,
        detection_lon=detected_lon,
        spill_area_km2=spill_area_km2,
        duration_backward_hours=duration_backward_hours,
        duration_forward_hours=duration_forward_hours,
        backward_trajectory=backward_output["trajectory"],
        forward_trajectory=forward_output["trajectory"],
        origin_search_windows=backward_output["origin_windows"],
        ensemble_snapshots=backward_output["snapshots"],
        stranding_risk=forward_output["stranding_risk"],
        first_stranding_time=forward_output["first_stranding_time"],
        current_dataset_info=str(current_data_path) if current_data_path else "Fallback Constant Field",
        wind_dataset_info=str(wind_data_path) if wind_data_path else "Fallback Wind",
        metadata={
            "particles": num_particles,
            "seed_radius_m": radius_seed_m,
            "pipeline_executed_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    return result