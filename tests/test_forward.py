"""
test_forward.py
Unit and integration tests for forward (forecast) drift simulation.
"""

from datetime import datetime, timedelta
import pytest

from src.drift.simulation.forward import run_forward_drift


def test_run_forward_drift_with_dataset():
    """Test forward drift forecast using local Copernicus currents dataset."""
    start_t = datetime(2025, 1, 1, 0, 0)
    duration_h = 12

    result = run_forward_drift(
        longitude=72.5,
        latitude=7.5,
        start_time=start_t,
        duration_hours=duration_h,
        data_path="data/currents/test_currents.nc",
        num_particles=15,
        radius_seed_m=500.0,
    )

    trajectory = result["trajectory"]

    assert len(trajectory) > 0, "Trajectory points should not be empty"
    # First point should match start time
    assert trajectory[0].time == start_t
    # Trajectory should step forward into future
    assert trajectory[-1].time > trajectory[0].time
    expected_end = start_t + timedelta(hours=duration_h)
    assert trajectory[-1].time == expected_end
    assert isinstance(result["stranding_risk"], bool)


def test_run_forward_drift_fallback():
    """Test forward drift works cleanly with fallback reader."""
    start_t = datetime(2025, 1, 1, 0, 0)

    result = run_forward_drift(
        longitude=72.5,
        latitude=7.5,
        start_time=start_t,
        duration_hours=6,
        data_path="missing_currents.nc",
        num_particles=10,
    )

    assert len(result["trajectory"]) > 0