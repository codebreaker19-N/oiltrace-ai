"""
currents.py
Ocean current data reader module for OilTrace-AI (M3).

Loads Copernicus Marine (CMEMS) hydrodynamic reanalysis/forecast data
(variables uo, vo) into OpenDrift. Includes domain coverage inspection
and fallback current generation for offline / testing scenarios.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_constant


def load_current_reader(data_path: str | Path) -> reader_netCDF_CF_generic.Reader:
    """
    Load Copernicus Marine NetCDF current data into an OpenDrift generic CF reader.

    Parameters
    ----------
    data_path : str or Path
        Path to Copernicus NetCDF current file containing 'uo' and 'vo'.

    Returns
    -------
    reader : reader_netCDF_CF_generic.Reader
        OpenDrift CF reader instance.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Ocean current data file not found at: {data_path}"
        )

    reader = reader_netCDF_CF_generic.Reader(str(data_path))
    return reader


def get_current_coverage(reader: Any) -> Dict[str, Any]:
    """
    Extract spatial and temporal coverage metadata from a current reader.
    """
    return {
        "start_time": getattr(reader, "start_time", None),
        "end_time": getattr(reader, "end_time", None),
        "xmin": getattr(reader, "xmin", None),
        "xmax": getattr(reader, "xmax", None),
        "ymin": getattr(reader, "ymin", None),
        "ymax": getattr(reader, "ymax", None),
        "variables": getattr(reader, "variables", []),
    }


def is_in_coverage(
    reader: Any,
    lat: float,
    lon: float,
    check_time: Optional[datetime] = None,
) -> bool:
    """
    Check if a geographic coordinate and timestamp fall within reader bounds.
    """
    xmin = getattr(reader, "xmin", -180.0)
    xmax = getattr(reader, "xmax", 180.0)
    ymin = getattr(reader, "ymin", -90.0)
    ymax = getattr(reader, "ymax", 90.0)

    if not (xmin <= lon <= xmax and ymin <= lat <= ymax):
        return False

    if check_time is not None:
        start = getattr(reader, "start_time", None)
        end = getattr(reader, "end_time", None)
        if start and end and not (start <= check_time <= end):
            return False

    return True


def create_fallback_current_reader(
    u_velocity: float = 0.2,
    v_velocity: float = 0.1,
) -> reader_constant.Reader:
    """
    Create a constant velocity ocean current reader for offline testing or fallback.

    Parameters
    ----------
    u_velocity : float
        Eastward sea water velocity (m/s). Default 0.2 m/s.
    v_velocity : float
        Northward sea water velocity (m/s). Default 0.1 m/s.
    """
    reader = reader_constant.Reader({
        "x_sea_water_velocity": u_velocity,
        "y_sea_water_velocity": v_velocity,
    })
    return reader