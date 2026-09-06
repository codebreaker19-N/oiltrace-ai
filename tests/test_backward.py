"""
test_backward.py
Unit and integration tests for backward (hindcast) drift simulation.
"""

from datetime import datetime, timedelta
import pytest

from src.drift.simulation.backward import run_backward_drift


def test_run_backward_drift_with_dataset():
    """Test backward drift using local Copernicus currents dataset."""
    detection_t = datetime(2025, 1, 2, 0, 0)
    duration_h = 12

    result = run_backward_drift(
        longitude=72.5,
        latitude=7.5,
        detection_time=detection_t,
        duration_hours=duration_h,
        data_path="data/currents/test_currents.nc",
        num_particles=15,
        radius_seed_m=500.0,
    )

    trajectory = result["trajectory"]
    origin_windows = result["origin_windows"]

    assert len(trajectory) > 0, "Trajectory points should not be empty"
    # First point should be at or near detection time
    assert trajectory[0].time == detection_t
    # Trajectory should step backward into the past
    assert trajectory[-1].time < trajectory[0].time
    # Expected duration check
    expected_end = detection_t - timedelta(hours=duration_h)
    assert trajectory[-1].time == expected_end

    # Origin windows check
    assert len(origin_windows) > 0, "Should generate origin search windows for M4"
    for w in origin_windows:
        assert w.min_lat <= w.centroid_lat <= w.max_lat
        assert w.min_lon <= w.centroid_lon <= w.max_lon
        assert w.radius_km > 0.0


def test_run_backward_drift_fallback():
    """Test backward drift gracefully falls back to synthetic/constant reader if file missing."""
    detection_t = datetime(2025, 1, 2, 0, 0)

    result = run_backward_drift(
        longitude=72.5,
        latitude=7.5,
        detection_time=detection_t,
        duration_hours=6,
        data_path="non_existent_file.nc",
        num_particles=10,
    )

    assert len(result["trajectory"]) > 0
    assert len(result["origin_windows"]) > 0