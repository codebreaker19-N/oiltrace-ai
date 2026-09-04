"""
Wrapper around OpenDrift for oil-spill trajectory simulations.

This module keeps OpenDrift-specific implementation separate from
the rest of the drift pipeline.
"""

from datetime import datetime, timedelta


class DriftModel:
    """Interface for running oil-spill drift simulations."""

    def __init__(self, config=None):
        self.config = config or {}

    def run_backward(
        self,
        latitude,
        longitude,
        start_time,
        duration_hours=72,
        particles=100,
    ):
        """
        Run a backward oil-spill trajectory simulation.

        Parameters
        ----------
        latitude : float
            Spill detection latitude.
        longitude : float
            Spill detection longitude.
        start_time : datetime
            Time when the spill was detected.
        duration_hours : int
            Number of hours to backtrack.
        particles : int
            Number of particles in the ensemble.

        Returns
        -------
        dict
            Simulation metadata and trajectory results.
        """

        if not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90.")

        if not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180.")

        if particles <= 0:
            raise ValueError("Particles must be greater than zero.")

        if duration_hours <= 0:
            raise ValueError("Duration must be greater than zero.")

        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(
                start_time.replace("Z", "+00:00")
            )

        end_time = start_time - timedelta(hours=duration_hours)

        return {
            "start_location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "start_time": start_time.isoformat(),
            "backtrack_start_time": end_time.isoformat(),
            "duration_hours": duration_hours,
            "particles": particles,
            "status": "initialized",
            "trajectories": [],
        }
