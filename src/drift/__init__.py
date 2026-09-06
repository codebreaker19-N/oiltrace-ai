"""
OilTrace-AI Ocean & Drift Engine (M3).
"""

from src.drift.drift_result import (
    DriftResult,
    TrajectoryPoint,
    OriginSearchWindow,
    EnsembleTimeSnapshot,
    ParticleCoordinate,
)
from src.drift.ocean_data.currents import (
    load_current_reader,
    create_fallback_current_reader,
    is_in_coverage,
)
from src.drift.ocean_data.wind import (
    load_wind_reader,
    create_fallback_wind_reader,
)
from src.drift.simulation.backward import run_backward_drift
from src.drift.simulation.forward import run_forward_drift
from src.drift.pipeline import run_drift_pipeline

__all__ = [
    "DriftResult",
    "TrajectoryPoint",
    "OriginSearchWindow",
    "EnsembleTimeSnapshot",
    "ParticleCoordinate",
    "load_current_reader",
    "create_fallback_current_reader",
    "is_in_coverage",
    "load_wind_reader",
    "create_fallback_wind_reader",
    "run_backward_drift",
    "run_forward_drift",
    "run_drift_pipeline",
]