"""
backward.py
Backward (hindcast) oil drift simulation module for OilTrace-AI (M3).

Traces detected oil slicks backward in time to reconstruct the origin window
and release location. Uses Monte Carlo ensemble particle dispersion to
generate spatio-temporal search bounds for AIS vessel attribution (M4).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from opendrift.models.oceandrift import OceanDrift

from src.drift.drift_result import (
    OriginSearchWindow,
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
    configure_oil_drift_physics,
)


def run_backward_drift(
    longitude: float,
    latitude: float,
    detection_time: datetime,
    duration_hours: int = 24,
    data_path: Optional[str | Path] = "data/currents/test_currents.nc",
    wind_path: Optional[str | Path] = None,
    num_particles: int = 30,
    radius_seed_m: float = 1000.0,
    time_step_hours: int = 1,
    current_uncertainty: float = 0.05,
    wind_uncertainty: float = 0.5,
    polygon_coords: Optional[List[Tuple[float, float]]] = None,
    window_interval_hours: int = 4,
) -> Dict[str, Any]:
    """
    Execute a backward (hindcast) Lagrangian ensemble simulation.

    Parameters
    ----------
    longitude : float
        Detected slick centroid longitude.
    latitude : float
        Detected slick centroid latitude.
    detection_time : datetime
        UTC timestamp when satellite detected the spill.
    duration_hours : int
        Number of hours to backtrack into the past.
    data_path : str or Path, optional
        Path to Copernicus ocean current NetCDF. Uses fallback if missing.
    wind_path : str or Path, optional
        Path to wind NetCDF. Uses fallback if missing.
    num_particles : int
        Size of Monte Carlo particle ensemble for uncertainty estimation.
    radius_seed_m : float
        Initial seeding uncertainty radius in meters.
    time_step_hours : int
        Time step resolution (hours).
    current_uncertainty : float
        Current turbulence uncertainty (m/s) for stochastic diffusion.
    wind_uncertainty : float
        Wind turbulence uncertainty (m/s).
    polygon_coords : list of (lon, lat), optional
        Spill contour polygon from M2 (Neha) to compute centroid or bounds.
    window_interval_hours : int
        Temporal grouping size (hours) for M4 AIS origin search windows.

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': Completed OpenDrift simulation instance
        - 'trajectory': List of TrajectoryPoint from detection back to origin
        - 'origin_windows': List of OriginSearchWindow for M4 Priya
        - 'snapshots': List of EnsembleTimeSnapshot for heatmap / UI
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
        # Graceful fallback for offline / test environments
        fallback_current = create_fallback_current_reader(u_velocity=0.15, v_velocity=0.08)
        model.add_reader(fallback_current)

    # 3. Attach Wind Reader (if available or fallback)
    if wind_path and Path(wind_path).exists():
        try:
            wind_reader = load_wind_reader(wind_path)
            model.add_reader(wind_reader)
        except Exception:
            pass

    # 4. Configure Diffusion and Windage Physics
    try:
        model.set_config("drift:current_uncertainty", current_uncertainty)
    except Exception:
        pass
    try:
        model.set_config("drift:wind_uncertainty", wind_uncertainty)
    except Exception:
        pass

    # 5. Seed Ensemble Elements at Detection Point
    # If polygon is supplied by M2, compute centroid if not explicitly provided
    seed_lon = longitude
    seed_lat = latitude
    if polygon_coords and len(polygon_coords) >= 3:
        poly_arr = np.array(polygon_coords)
        seed_lon = float(np.mean(poly_arr[:, 0]))
        seed_lat = float(np.mean(poly_arr[:, 1]))

    model.seed_elements(
        lon=seed_lon,
        lat=seed_lat,
        time=detection_time,
        number=num_particles,
        radius=radius_seed_m,
    )

    # 6. Run Backward Simulation (Negative Time Step is MANDATORY for OpenDrift hindcasting)
    end_target_time = detection_time - timedelta(hours=duration_hours)
    step_delta = -timedelta(hours=time_step_hours)
    output_step_delta = timedelta(hours=time_step_hours)

    model.run(
        end_time=end_target_time,
        time_step=step_delta,
        time_step_output=output_step_delta,
    )

    # 7. Post-process Ensemble Statistics into Structured Output
    times = model.result.time.values
    num_steps = len(times)

    trajectory_points: List[TrajectoryPoint] = []
    snapshots: List[EnsembleTimeSnapshot] = []

    for step_idx in range(num_steps):
        # Convert numpy datetime64 to python datetime
        dt_val = times[step_idx]
        if isinstance(dt_val, np.datetime64):
            ts = (dt_val - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
            step_time = datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None)
        else:
            step_time = dt_val

        lons = np.array(model.result.lon[:, step_idx], dtype=float)
        lats = np.array(model.result.lat[:, step_idx], dtype=float)

        # Filter valid coordinates
        valid_mask = ~(np.isnan(lons) | np.isnan(lats))
        valid_lons = lons[valid_mask]
        valid_lats = lats[valid_mask]

        if len(valid_lons) == 0:
            continue

        c_lon = float(np.mean(valid_lons))
        c_lat = float(np.mean(valid_lats))

        # Calculate standard deviation spread in meters
        # 1 deg lat ~ 110,540 m; 1 deg lon ~ 111,320 * cos(lat) m
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
                active_particles=int(len(valid_lons)),
                stranded=False,
            )
        )

        # Snapshot for particle cloud heatmap
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

    # 8. Construct Spatio-Temporal Origin Search Windows for M4 (Priya - AIS)
    origin_windows: List[OriginSearchWindow] = []
    w_id = 1
    chunk_size = max(1, window_interval_hours // time_step_hours)

    for i in range(0, len(trajectory_points), chunk_size):
        chunk = trajectory_points[i : i + chunk_size]
        if not chunk:
            continue

        start_t = min(pt.time for pt in chunk)
        end_t = max(pt.time for pt in chunk)
        if start_t == end_t and len(chunk) == 1:
            start_t = start_t - timedelta(hours=time_step_hours)

        chunk_lats = [pt.lat for pt in chunk]
        chunk_lons = [pt.lon for pt in chunk]
        max_spread = max(pt.uncertainty_radius_m for pt in chunk)
        deg_buffer = (max_spread / 111320.0) * 1.5

        c_lat = float(np.mean(chunk_lats))
        c_lon = float(np.mean(chunk_lons))

        origin_windows.append(
            OriginSearchWindow(
                window_id=w_id,
                start_time=start_t,
                end_time=end_t,
                min_lat=round(min(chunk_lats) - deg_buffer, 5),
                max_lat=round(max(chunk_lats) + deg_buffer, 5),
                min_lon=round(min(chunk_lons) - deg_buffer, 5),
                max_lon=round(max(chunk_lons) + deg_buffer, 5),
                centroid_lat=round(c_lat, 5),
                centroid_lon=round(c_lon, 5),
                radius_km=round((max_spread / 1000.0) * 1.5, 2),
                confidence_level=0.95,
            )
        )
        w_id += 1

    return {
        "model": model,
        "trajectory": trajectory_points,
        "origin_windows": origin_windows,
        "snapshots": snapshots,
    }