"""
Ensemble backtracking for oil-spill source attribution.

This module manages multiple backward trajectory simulations
and prepares their results for source-region estimation.
"""

from datetime import datetime

from drift.source_region import estimate_source_region


class BacktrackEnsemble:
    """Run and manage an ensemble of backward drift simulations."""

    def __init__(self, drift_model, config=None):
        self.drift_model = drift_model
        self.config = config or {}

    def run(
        self,
        event_id,
        latitude,
        longitude,
        detection_time,
        duration_hours=72,
        particles=100,
    ):
        """
        Run a backward trajectory ensemble.
        """

        if not event_id:
            raise ValueError("event_id is required.")

        if isinstance(detection_time, str):
            detection_time = datetime.fromisoformat(
                detection_time.replace("Z", "+00:00")
            )

        result = self.drift_model.run_backward(
            latitude=latitude,
            longitude=longitude,
            start_time=detection_time,
            duration_hours=duration_hours,
            particles=particles,
        )

        trajectories = result.get("trajectories", [])

        source_region = None

        if trajectories:
            positions = [
                (point["latitude"], point["longitude"])
                for point in trajectories
                if "latitude" in point and "longitude" in point
            ]

            if positions:
                source_region = estimate_source_region(positions)

        return {
            "event_id": event_id,
            "detection": {
                "latitude": latitude,
                "longitude": longitude,
                "time": detection_time.isoformat(),
            },
            "simulation": result,
            "source_region": source_region,
        }
