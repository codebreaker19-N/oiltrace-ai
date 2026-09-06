"""
forward.py
Forward (forecast) oil drift simulation module for OilTrace-AI (M3).

Projects future trajectory, spread expansion, and coastal stranding risk of
detected oil slicks to support maritime containment and disaster management.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from opendrift.models.oceandrift import OceanDrift

from src.drift.drift_result import (
    ParticleCoordinate,
    EnsembleTimeSnapshot,
    TrajectoryPoint,
)
from src.drift.ocean_data.currents import (
    load_current_reader,
    create_fallback_current_reader,
)
from src.drift.ocean_data.wind import (
    load_wind_reader,
    create_fallback_wind_reader,
)


def run_forward_drift(
    longitude: float,
    latitude: float,
    start_time: datetime,
    duration_hours: int = 24,
    data_path: Optional[str | Path] = "data/currents/test_currents.nc",
    wind_path: Optional[str | Path] = None,
    num_particles: int = 30,
    radius_seed_m: float = 1000.0,
    time_step_hours: int = 1,
    current_uncertainty: float = 0.05,
    wind_uncertainty: float = 0.5,
) -> Dict[str, Any]:
    """
    Execute a forward (forecast) Lagrangian ensemble simulation.

    Parameters
    ----------
    longitude : float
        Current slick centroid longitude.
    latitude : float
        Current slick centroid latitude.
    start_time : datetime
        UTC timestamp from which to forecast into the future.
    duration_hours : int
        Forecast horizon (hours).
    data_path : str or Path, optional
        Path to Copernicus ocean current NetCDF. Uses fallback if missing.
    wind_path : str or Path, optional
        Path to wind NetCDF. Uses fallback if missing.
    num_particles : int
        Size of Monte Carlo particle ensemble.
    radius_seed_m : float
        Current slick radius in meters.
    time_step_hours : int
        Simulation step resolution in hours.
    current_uncertainty : float
        Current turbulence uncertainty (m/s).
    wind_uncertainty : float
        Wind turbulence uncertainty (m/s).

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': Completed OpenDrift simulation instance
        - 'trajectory': List of TrajectoryPoint from detection time forward
        - 'stranding_risk': Boolean indicating if slick threatens shoreline
        - 'first_stranding_time': Earliest timestamp of coastal impact (if any)
        - 'snapshots': List of EnsembleTimeSnapshot for visualization
    """
    # 1. Initialize OpenDrift OceanDrift Model
    model = OceanDrift(loglevel=30)

    # 2. Attach Ocean Current Reader
    reader_loaded = False
    if data_path:
        data_path = Path(data_path)
        if data_path.exists():
            try:
                current_reader = load_current_reader(data_path)
                model.add_reader(current_reader)
                reader_loaded = True
            except Exception:
                reader_loaded = False

    if not reader_loaded:
        fallback_current = create_fallback_current_reader(u_velocity=0.15, v_velocity=0.08)
        model.add_reader(fallback_current)

    # 3. Attach Wind Reader (if available or fallback)
    if wind_path and Path(wind_path).exists():
        try:
            wind_reader = load_wind_reader(wind_path)
            model.add_reader(wind_reader)
        except Exception:
            pass

    # 4. Configure Coastline Action and Turbulence
    try:
        model.set_config("general:coastline_action", "stranding")
    except Exception:
        pass
    try:
        model.set_config("drift:current_uncertainty", current_uncertainty)
    except Exception:
        pass
    try:
        model.set_config("drift:wind_uncertainty", wind_uncertainty)
    except Exception:
        pass

    # 5. Seed Ensemble Elements
    model.seed_elements(
        lon=longitude,
        lat=latitude,
        time=start_time,
        number=num_particles,
        radius=radius_seed_m,
    )

    # 6. Run Forward Simulation
    step_delta = timedelta(hours=time_step_hours)
    model.run(
        duration=timedelta(hours=duration_hours),
        time_step=step_delta,
        time_step_output=step_delta,
    )

    # 7. Post-process Forward Forecast
    times = model.result.time.values
    num_steps = len(times)

    trajectory_points: List[TrajectoryPoint] = []
    snapshots: List[EnsembleTimeSnapshot] = []

    stranding_risk = False
    first_stranding_time: Optional[datetime] = None

    for step_idx in range(num_steps):
        dt_val = times[step_idx]
        if isinstance(dt_val, np.datetime64):
            ts = (dt_val - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
            step_time = datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None)
        else:
            step_time = dt_val

        lons = np.array(model.result.lon[:, step_idx], dtype=float)
        lats = np.array(model.result.lat[:, step_idx], dtype=float)
        status = np.array(model.result.status[:, step_idx])

        # Check stranding (status != 0 typically indicates stranded/deactivated)
        is_stranded_now = bool(np.any(status != 0))
        if is_stranded_now and not stranding_risk:
            stranding_risk = True
            first_stranding_time = step_time

        valid_mask = ~(np.isnan(lons) | np.isnan(lats))
        valid_lons = lons[valid_mask]
        valid_lats = lats[valid_mask]

        if len(valid_lons) == 0:
            continue

        c_lon = float(np.mean(valid_lons))
        c_lat = float(np.mean(valid_lats))

        rad_lat = np.radians(c_lat)
        dx_m = (valid_lons - c_lon) * 111320.0 * np.cos(rad_lat)
        dy_m = (valid_lats - c_lat) * 110540.0
        sigma_m = float(np.sqrt(np.mean(dx_m**2 + dy_m**2)))
        spread_radius_m = max(float(radius_seed_m), float(2.0 * sigma_m))

        trajectory_points.append(
            TrajectoryPoint(
                time=step_time,
                lat=round(c_lat, 6),
                lon=round(c_lon, 6),
                uncertainty_radius_m=round(spread_radius_m, 1),
                active_particles=int(np.sum(status == 0)),
                stranded=is_stranded_now,
            )
        )

        particle_coords = [
            ParticleCoordinate(
                particle_id=pid,
                lat=round(float(valid_lats[pid]), 6),
                lon=round(float(valid_lons[pid]), 6),
            )
            for pid in range(len(valid_lons))
        ]
        snapshots.append(
            EnsembleTimeSnapshot(time=step_time, particles=particle_coords)
        )

    return {
        "model": model,
        "trajectory": trajectory_points,
        "stranding_risk": stranding_risk,
        "first_stranding_time": first_stranding_time,
        "snapshots": snapshots,
    }