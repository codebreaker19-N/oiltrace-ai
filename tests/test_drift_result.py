"""
test_drift_result.py
Tests for DriftResult schemas, serialization, GeoJSON formatting, and pipeline integration.
"""

from datetime import datetime, timedelta
import json
import pytest

from src.drift.drift_result import (
    DriftResult,
    TrajectoryPoint,
    OriginSearchWindow,
)
from src.drift.pipeline import run_drift_pipeline


def test_drift_result_schema_and_geojson():
    """Verify DriftResult Pydantic schema validation and GeoJSON FeatureCollection generation."""
    det_time = datetime(2025, 1, 2, 12, 0)

    # Mock trajectory points
    backward_pts = [
        TrajectoryPoint(
            time=det_time - timedelta(hours=i),
            lat=7.5 + i * 0.01,
            lon=72.5 + i * 0.01,
            uncertainty_radius_m=500.0 + i * 100.0,
        )
        for i in range(5)
    ]

    forward_pts = [
        TrajectoryPoint(
            time=det_time + timedelta(hours=i),
            lat=7.5 - i * 0.01,
            lon=72.5 - i * 0.01,
            uncertainty_radius_m=500.0 + i * 80.0,
        )
        for i in range(5)
    ]

    windows = [
        OriginSearchWindow(
            window_id=1,
            start_time=det_time - timedelta(hours=4),
            end_time=det_time,
            min_lat=7.48,
            max_lat=7.56,
            min_lon=72.48,
            max_lon=72.56,
            centroid_lat=7.52,
            centroid_lon=72.52,
            radius_km=3.5,
        )
    ]

    result = DriftResult(
        spill_id="TEST_SPILL_001",
        detection_time=det_time,
        detection_lat=7.5,
        detection_lon=72.5,
        backward_trajectory=backward_pts,
        forward_trajectory=forward_pts,
        origin_search_windows=windows,
    )

    # Test GeoJSON export for M6 Mansi (React Leaflet)
    geojson = result.to_geojson()
    assert geojson["type"] == "FeatureCollection"
    assert geojson["spill_id"] == "TEST_SPILL_001"
    assert len(geojson["features"]) >= 4

    layers = [f["properties"]["layer"] for f in geojson["features"]]
    assert "detection_point" in layers
    assert "backward_trajectory" in layers
    assert "origin_search_window" in layers
    assert "forward_trajectory" in layers

    # Test AIS Query parameters for M4 Priya
    ais_filters = result.to_ais_query_filters()
    assert len(ais_filters) == 1
    assert ais_filters[0]["window_id"] == 1
    assert ais_filters[0]["radius_km"] == 3.5

    # Test SQL filter snippet
    sql = windows[0].to_sql_filter("ais_vessels")
    assert "SELECT DISTINCT" in sql
    assert "ais_vessels" in sql


def test_full_pipeline_with_teammates_inputs():
    """
    Test pipeline handoffs:
    - Input from M2 (Neha): SpillGeometry
    - Input from M1 (Nidhi): DetectionResult
    - Output to M4 (Priya): OriginSearchWindow
    - Output to M5 (Prachi) & M6 (Mansi): DriftResult + GeoJSON
    """
    # Simulated M2 (Neha) SpillGeometry
    mock_spill_geometry = {
        "centroid": [7.5, 72.5],
        "area_km2": 4.5,
        "coordinates": [[72.49, 7.49], [72.51, 7.49], [72.51, 7.51], [72.49, 7.51]],
    }

    # Simulated M1 (Nidhi) DetectionResult
    mock_detection_result = {
        "confidence": 0.94,
        "age_proxy_hours": 6,
        "label": "crude_oil_spill",
    }

    result = run_drift_pipeline(
        spill_id="SPILL_HANDOFF_TEST",
        detection_time=datetime(2025, 1, 2, 0, 0),
        spill_geometry=mock_spill_geometry,
        detection_result=mock_detection_result,
        duration_forward_hours=6,
        current_data_path="data/currents/test_currents.nc",
        num_particles=10,
    )

    assert result.spill_id == "SPILL_HANDOFF_TEST"
    assert result.detection_lat == 7.5
    assert result.detection_lon == 72.5
    assert result.duration_backward_hours == 6  # Inherited from M1 age proxy!
    assert len(result.backward_trajectory) > 0
    assert len(result.forward_trajectory) > 0
    assert len(result.origin_search_windows) > 0