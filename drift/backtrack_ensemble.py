"""
Ensemble backtracking for oil-spill source attribution.

This module manages multiple backward trajectory simulations
and prepares their results for source-region estimation.
"""

from datetime import datetime


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

        Parameters
        ----------
        event_id : str
            Unique identifier for the spill event.
        latitude : float
            Spill detection latitude.
        longitude : float
            Spill detection longitude.
        detection_time : datetime or str
            Time when the spill was detected.
        duration_hours : int
            Number of hours to backtrack.
        particles : int
            Number of particles in the ensemble.

        Returns
        -------
        dict
            Ensemble simulation results.
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

        return {
            "event_id": event_id,
            "detection": {
                "latitude": latitude,
                "longitude": longitude,
                "time": detection_time.isoformat(),
            },
            "simulation": result,
        }
