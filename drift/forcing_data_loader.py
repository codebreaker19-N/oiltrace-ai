"""
Environmental forcing data loader for the oil-spill drift module.

This module provides a common interface for loading:
- Ocean current data
- Wind data

The actual data provider can be replaced later (Copernicus, NOAA, etc.)
without changing the rest of the drift pipeline.
"""


class ForcingDataLoader:
    """Base interface for environmental forcing data."""

    def __init__(self, provider="demo"):
        self.provider = provider

    def load_currents(self, start_time, end_time, bbox):
        """
        Load ocean current data.

        Parameters
        ----------
        start_time : str
            Start of the time window.
        end_time : str
            End of the time window.
        bbox : tuple
            Bounding box as (min_lon, min_lat, max_lon, max_lat).

        Returns
        -------
        object
            Ocean current forcing data.
        """
        raise NotImplementedError(
            "Ocean current loading is not implemented yet."
        )

    def load_wind(self, start_time, end_time, bbox):
        """
        Load wind data.

        Parameters
        ----------
        start_time : str
            Start of the time window.
        end_time : str
            End of the time window.
        bbox : tuple
            Bounding box as (min_lon, min_lat, max_lon, max_lat).

        Returns
        -------
        object
            Wind forcing data.
        """
        raise NotImplementedError(
            "Wind data loading is not implemented yet."
        )
