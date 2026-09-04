"""
Wrapper around OpenDrift for oil-spill trajectory simulations.

This module currently provides a demo trajectory generator.
The demo implementation will later be replaced with a real
OpenDrift simulation using environmental forcing data.
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
        """Run a backward oil-spill trajectory simulation."""

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

        trajectories = []

        for particle_id in range(particles):
            offset = (particle_id - particles / 2) * 0.0005

            trajectories.append(
                {
                    "particle_id": particle_id + 1,
                    "latitude": latitude - 0.1 + offset,
                    "longitude": longitude - 0.1 + offset,
                }
            )

        return {
            "start_location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "start_time": start_time.isoformat(),
            "backtrack_start_time": end_time.isoformat(),
            "duration_hours": duration_hours,
            "particles": particles,
            "status": "demo",
            "trajectories": trajectories,
        }
