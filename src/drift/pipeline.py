"""
pipeline.py
End-to-end integration pipeline for the OilTrace-AI Drift Engine (M3).

Coordinates backward hindcasting and forward forecasting by integrating inputs
from Satellite AI/ML (M1, M2) and generating downstream contracts for
AIS Attribution (M4), FastAPI Backend (M5), and Leaflet Dashboard (M6).
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.drift.drift_result import DriftResult
from src.drift.simulation.backward import run_backward_drift
from src.drift.simulation.forward import run_forward_drift


def infer_weathering_age(
    features: Dict[str, Any],
    default_age_hours: float = 24.0,
) -> float:
    """
    Derive estimated physical spill age (hours) from slick morphology features (M2 Neha).

    Physical basis:
    - Fresh spills (0–8h): Compact, high circularity/solidity, low elongation.
    - Intermediate spills (8–18h): Progressive wind and current stretching.
    - Aged/Weathered spills (18–48h): High elongation (windrows), fragmented boundary, low solidity.
    """
    if not features:
        return default_age_hours

    aspect_ratio = float(features.get("aspect_ratio", features.get("elongation", 1.0)))
    solidity = float(features.get("solidity", 1.0))
    compactness = float(features.get("compactness", features.get("circularity", 1.0)))

    # Highly elongated and ragged boundary -> weathered slick
    if aspect_ratio >= 3.0 or solidity <= 0.65:
        return 30.0
    elif aspect_ratio >= 2.0 or solidity <= 0.75:
        return 18.0
    elif aspect_ratio <= 1.5 and (solidity >= 0.85 or compactness >= 0.6):
        # Fresh release
        return 8.0

    return default_age_hours


def run_drift_pipeline(
    spill_id: str = "spill_001",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    detection_time: Optional[datetime] = None,
    spill_geometry: Optional[Union[Dict[str, Any], str, Path, Any]] = None,
    detection_result: Optional[Union[Dict[str, Any], Any]] = None,
    duration_backward_hours: Optional[int] = None,
    duration_forward_hours: int = 24,
    scene_center: Tuple[float, float] = (7.5, 72.5),
    pixel_size_m: float = 10.0,
    current_data_path: Optional[str | Path] = "data/currents/test_currents.nc",
    wind_data_path: Optional[str | Path] = None,
    num_particles: int = 30,
) -> DriftResult:
    """
    Run full bidirectional (hindcast + forecast) drift simulation pipeline.

    Seamlessly integrates inputs from:
    - M2 (Neha): SpillGeometry or create_spill_output dictionary / JSON
    - M1 (Nidhi): Look-alike classification (predict_lookalike) & age proxy

    And outputs:
    - DriftResult: Tailored for M4 (Priya AIS), M5 (Prachi API), M6 (Mansi UI).
    """
    # 1. Ingest M2 (Neha - Satellite & Geospatial) Output or SpillGeometry
    detected_lat = latitude
    detected_lon = longitude
    spill_area_km2 = None
    polygon_coords = None
    image_id = None
    shape_features: Dict[str, Any] = {}

    if spill_geometry is not None:
        if isinstance(spill_geometry, (str, Path)):
            geom_path = Path(spill_geometry)
            if geom_path.exists():
                with open(geom_path, "r", encoding="utf-8") as f:
                    spill_geometry = json.load(f)

        if isinstance(spill_geometry, dict):
            # Check for M2 Neha's create_spill_output structure
            image_id = spill_geometry.get("image_id", image_id)
            if image_id and spill_id == "spill_001":
                spill_id = f"spill_{image_id}"

            shape_features = spill_geometry.get("features", {})

            # Extract from geometry_summary (Neha's standard structure)
            geom_summary = spill_geometry.get("geometry_summary")
            if geom_summary and isinstance(geom_summary, dict):
                area_pixels = geom_summary.get("area_pixels")
                if area_pixels and area_pixels > 0:
                    # Sentinel-1 10m pixel = 100 m^2 = 1e-4 km^2
                    spill_area_km2 = float(area_pixels) * (pixel_size_m ** 2) / 1e6

                centroid_info = geom_summary.get("centroid")
                if centroid_info and isinstance(centroid_info, dict):
                    # Check if pixel coordinates (x, y) from 256x256 SAR patch
                    if "x" in centroid_info and "y" in centroid_info:
                        px_x = float(centroid_info["x"])
                        px_y = float(centroid_info["y"])
                        center_lat, center_lon = scene_center
                        dx_m = (px_x - 128.0) * pixel_size_m
                        dy_m = (128.0 - px_y) * pixel_size_m
                        detected_lon = center_lon + (dx_m / (111320.0 * math.cos(math.radians(center_lat))))
                        detected_lat = center_lat + (dy_m / 110540.0)
                    elif "lat" in centroid_info and "lon" in centroid_info:
                        detected_lat = float(centroid_info["lat"])
                        detected_lon = float(centroid_info["lon"])

            # Direct centroid fallback
            if detected_lat is None or detected_lon is None:
                direct_centroid = spill_geometry.get("centroid")
                if direct_centroid:
                    if isinstance(direct_centroid, (list, tuple)) and len(direct_centroid) >= 2:
                        detected_lat, detected_lon = direct_centroid[0], direct_centroid[1]
                    elif isinstance(direct_centroid, dict):
                        detected_lat = direct_centroid.get("lat", detected_lat)
                        detected_lon = direct_centroid.get("lon", detected_lon)

            if spill_area_km2 is None:
                spill_area_km2 = spill_geometry.get("area_km2", spill_area_km2)

            # Ingest polygon if available in GeoJSON format
            geo_feat = spill_geometry.get("geometry")
            if geo_feat and isinstance(geo_feat, dict):
                geom_type = geo_feat.get("type")
                if geom_type == "Feature":
                    sub_geom = geo_feat.get("geometry", {})
                    if sub_geom.get("type") == "Polygon":
                        polygon_coords = sub_geom.get("coordinates", [[]])[0]
                elif geom_type == "Polygon":
                    polygon_coords = geo_feat.get("coordinates", [[]])[0]

            if polygon_coords:
                try:
                    import numpy as _np
                    poly_arr = _np.array(polygon_coords, dtype=float)
                    if poly_arr.ndim == 2 and poly_arr.shape[1] >= 2:
                        if _np.any(_np.abs(poly_arr[:, 1]) > 90.0) or _np.any(_np.abs(poly_arr[:, 0]) > 180.0):
                            c_lat, c_lon = scene_center
                            dx_m = (poly_arr[:, 0] - 128.0) * pixel_size_m
                            dy_m = (128.0 - poly_arr[:, 1]) * pixel_size_m
                            p_lons = c_lon + (dx_m / (111320.0 * math.cos(math.radians(c_lat))))
                            p_lats = c_lat + (dy_m / 110540.0)
                            polygon_coords = [[float(lo), float(la)] for lo, la in zip(p_lons, p_lats)]
                except Exception:
                    polygon_coords = None

        else:
            # Pydantic or class object attribute access
            if hasattr(spill_geometry, "centroid") and spill_geometry.centroid:
                detected_lat = getattr(spill_geometry.centroid, "lat", detected_lat)
                detected_lon = getattr(spill_geometry.centroid, "lon", detected_lon)
            spill_area_km2 = getattr(spill_geometry, "area_km2", spill_area_km2)

    # If coordinates are still not resolved, fall back to scene center
    if detected_lat is None or detected_lon is None:
        if scene_center:
            detected_lat, detected_lon = scene_center
        else:
            raise ValueError("Latitude and Longitude must be provided directly or via spill_geometry.")

    if detection_time is None:
        detection_time = datetime(2025, 1, 2, 0, 0)

    # 2. Ingest M1 (Nidhi - AI/ML) Look-alike Classification & Age Proxy
    is_oil_spill = True
    classification_confidence: Optional[float] = None
    age_proxy_hours: Optional[float] = None

    if detection_result is not None:
        if isinstance(detection_result, dict):
            cls_name = detection_result.get("class", "oil")
            is_oil_spill = (str(cls_name).lower() == "oil")
            classification_confidence = detection_result.get("confidence")
            age_proxy_hours = detection_result.get("age_proxy_hours")
        else:
            cls_name = getattr(detection_result, "class_name", "oil")
            is_oil_spill = (str(cls_name).lower() == "oil")
            classification_confidence = getattr(detection_result, "confidence", None)
            age_proxy_hours = getattr(detection_result, "age_proxy_hours", None)

    # Auto-infer weathering age if not explicitly provided
    if duration_backward_hours is None:
        if age_proxy_hours is not None:
            duration_backward_hours = int(round(age_proxy_hours))
        else:
            inferred_age = infer_weathering_age(shape_features, default_age_hours=24.0)
            duration_backward_hours = int(round(inferred_age))
            age_proxy_hours = inferred_age
    else:
        if age_proxy_hours is None:
            age_proxy_hours = float(duration_backward_hours)

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
        image_id=image_id,
        detection_time=detection_time,
        detection_lat=detected_lat,
        detection_lon=detected_lon,
        spill_area_km2=round(spill_area_km2, 4) if spill_area_km2 else None,
        duration_backward_hours=duration_backward_hours,
        duration_forward_hours=duration_forward_hours,
        is_oil_spill=is_oil_spill,
        classification_confidence=classification_confidence,
        estimated_age_hours=age_proxy_hours,
        shape_features=shape_features,
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


def run_pipeline_from_satellite_output(
    spill_output: Union[Dict[str, Any], str, Path],
    lookalike_result: Optional[Dict[str, Any]] = None,
    scene_center: Tuple[float, float] = (7.5, 72.5),
    detection_time: Optional[datetime] = None,
    current_data_path: Optional[str | Path] = "data/currents/test_currents.nc",
    **kwargs,
) -> DriftResult:
    """
    Convenience entrypoint to execute M3 simulation directly from M1 & M2 artifacts.

    Parameters
    ----------
    spill_output : dict or str or Path
        Output from M2 (Neha - create_spill_output) or path to 'spill_output.json'.
    lookalike_result : dict, optional
        Output from M1 (Nidhi - predict_lookalike: {'class': 'oil', 'confidence': ...}).
    scene_center : (lat, lon)
        Geographic reference anchor for pixel-to-geographic projection.
    """
    return run_drift_pipeline(
        spill_geometry=spill_output,
        detection_result=lookalike_result,
        scene_center=scene_center,
        detection_time=detection_time,
        current_data_path=current_data_path,
        **kwargs,
    )