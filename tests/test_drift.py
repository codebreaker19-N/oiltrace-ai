"""
Tests for the oil-spill drift module.
"""

from drift.source_region import (
    estimate_source_region,
    haversine_distance_km,
)
from drift.opendrift_wrapper import DriftModel


def test_haversine_distance():
    """Distance between identical points should be zero."""

    distance = haversine_distance_km(
        19.245,
        72.821,
        19.245,
        72.821,
    )

    assert distance == 0.0


def test_source_region_estimation():
    """Source region should contain center, radius and confidence."""

    positions = [
        (19.10, 72.70),
        (19.12, 72.72),
        (19.14, 72.74),
    ]

    result = estimate_source_region(positions)

    assert "center" in result
    assert "radius_km" in result
    assert "particle_count" in result
    assert "confidence" in result

    assert result["particle_count"] == 3
    assert result["radius_km"] >= 0
    assert 0 < result["confidence"] <= 1


def test_drift_model_demo():
    """Demo drift model should generate the requested particles."""

    model = DriftModel()

    result = model.run_backward(
        latitude=19.245,
        longitude=72.821,
        start_time="2026-08-20T10:00:00Z",
        duration_hours=72,
        particles=10,
    )

    assert result["status"] == "demo"
    assert result["particles"] == 10
    assert len(result["trajectories"]) == 10
