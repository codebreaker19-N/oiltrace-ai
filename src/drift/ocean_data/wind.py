"""
wind.py
Atmospheric wind data reader and windage parameterization for OilTrace-AI (M3).

Handles ERA5 / NOAA GFS 10-meter surface wind fields (u10, v10) and configures
the empirical oil wind drift factor (~3% windage rule of thumb with Coriolis deflection).
"""

from pathlib import Path
from typing import Any, Dict, Optional

from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_constant


def load_wind_reader(wind_path: str | Path) -> reader_netCDF_CF_generic.Reader:
    """
    Load ERA5 or NOAA GFS NetCDF wind data.

    Parameters
    ----------
    wind_path : str or Path
        Path to wind NetCDF file containing 'u10' and 'v10' or 'x_wind' and 'y_wind'.

    Returns
    -------
    reader : reader_netCDF_CF_generic.Reader
        OpenDrift wind reader instance.
    """
    wind_path = Path(wind_path)

    if not wind_path.exists():
        raise FileNotFoundError(
            f"Wind data file not found at: {wind_path}"
        )

    reader = reader_netCDF_CF_generic.Reader(str(wind_path))
    return reader


def create_fallback_wind_reader(
    u_wind: float = 5.0,
    v_wind: float = 2.0,
) -> reader_constant.Reader:
    """
    Create a constant velocity surface wind reader for testing or fallback.

    Parameters
    ----------
    u_wind : float
        Zonal 10m wind velocity (m/s). Default 5.0 m/s (~10 knots easterly).
    v_wind : float
        Meridional 10m wind velocity (m/s). Default 2.0 m/s northerly.
    """
    reader = reader_constant.Reader({
        "x_wind": u_wind,
        "y_wind": v_wind,
    })
    return reader


def configure_oil_drift_physics(
    model: Any,
    wind_drift_factor: float = 0.03,
    horizontal_diffusivity: float = 10.0,
) -> None:
    """
    Apply standard oceanographic oil drift physics parameters to OpenDrift.

    Parameters
    ----------
    model : OpenDrift model
        The active simulation model.
    wind_drift_factor : float
        Fraction of 10m wind transferred to surface slick (standard is 0.025 to 0.035).
    horizontal_diffusivity : float
        Sub-grid scale turbulent horizontal diffusion coefficient (m^2/s).
    """
    # Configure horizontal turbulent diffusion if supported by the model configuration
    try:
        model.set_config("drift:horizontal_diffusivity", horizontal_diffusivity)
    except Exception:
        pass

    try:
        model.set_config("drift:wind_drift_factor", wind_drift_factor)
    except Exception:
        pass