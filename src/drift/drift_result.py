"""
drift_result.py
Data models and serialization contracts for OilTrace-AI Ocean & Drift Engine (M3).

Defines standard Pydantic models for backward (hindcast) and forward (forecast)
simulations, origin search windows for AIS correlation (M4), and GeoJSON outputs
for the FastAPI backend (M5) and React/Leaflet dashboard (M6).
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrajectoryPoint(BaseModel):
    """Represents the slick centroid and dispersion radius at a specific timestamp."""
    time: datetime
    lat: float
    lon: float
    uncertainty_radius_m: float = Field(default=500.0, description="2-sigma ensemble spread radius in meters")
    stranded: bool = Field(default=False, description="True if slick elements grounded on coastline")
    active_particles: int = Field(default=1, description="Count of active floating elements")


class OriginSearchWindow(BaseModel):
    """
    Spatio-temporal search bounding box generated for M4 (Priya - AIS Attribution).
    Used to filter candidate AIS vessels active in the spill origin window.
    """
    window_id: int
    start_time: datetime
    end_time: datetime
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    centroid_lat: float
    centroid_lon: float
    radius_km: float
    confidence_level: float = Field(default=0.95, description="Confidence probability for origin window")

    def to_postgis_bbox(self) -> str:
        """Return PostGIS ST_MakeEnvelope polygon string."""
        return f"ST_MakeEnvelope({self.min_lon}, {self.min_lat}, {self.max_lon}, {self.max_lat}, 4326)"

    def to_sql_filter(self, table_name: str = 'ais_messages') -> str:
        """Helper SQL snippet for M4 to query candidate vessels."""
        return (
            f"SELECT DISTINCT mmsi, vessel_name, lat, lon, timestamp FROM {table_name} "
            f"WHERE timestamp BETWEEN '{self.start_time.isoformat()}' AND '{self.end_time.isoformat()}' "
            f"AND lat BETWEEN {self.min_lat:.5f} AND {self.max_lat:.5f} "
            f"AND lon BETWEEN {self.min_lon:.5f} AND {self.max_lon:.5f};"
        )


class ParticleCoordinate(BaseModel):
    particle_id: int
    lat: float
    lon: float


class EnsembleTimeSnapshot(BaseModel):
    """Snapshot of individual ensemble particles for heatmap / particle cloud rendering."""
    time: datetime
    particles: List[ParticleCoordinate] = Field(default_factory=list)


class DriftResult(BaseModel):
    """
    Comprehensive container for all drift simulation outputs.
    Serves as the standardized handoff artifact between M3 and teammates M4, M5, M6.
    """
    spill_id: str = Field(default="spill_001", description="Unique spill identifier")
    detection_time: datetime
    detection_lat: float
    detection_lon: float
    spill_area_km2: Optional[float] = None
    duration_backward_hours: int = 24
    duration_forward_hours: int = 24

    backward_trajectory: List[TrajectoryPoint] = Field(default_factory=list)
    forward_trajectory: List[TrajectoryPoint] = Field(default_factory=list)
    origin_search_windows: List[OriginSearchWindow] = Field(default_factory=list)
    ensemble_snapshots: Optional[List[EnsembleTimeSnapshot]] = Field(default=None)

    stranding_risk: bool = Field(default=False, description="Whether forward forecast hits shoreline")
    first_stranding_time: Optional[datetime] = None
    current_dataset_info: Optional[str] = None
    wind_dataset_info: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_geojson(self) -> Dict[str, Any]:
        """
        Export trajectory and origin search bounds into standard GeoJSON FeatureCollection
        for M6 (Mansi - React Leaflet dashboard) and M5 (Prachi - FastAPI).
        """
        features: List[Dict[str, Any]] = []

        # 1. Detection Point Feature
        features.append({
            "type": "Feature",
            "properties": {
                "layer": "detection_point",
                "spill_id": self.spill_id,
                "timestamp": self.detection_time.isoformat(),
                "label": "Spill Detection Centroid",
            },
            "geometry": {
                "type": "Point",
                "coordinates": [self.detection_lon, self.detection_lat],
            },
        })

        # 2. Backward Trajectory Line
        if self.backward_trajectory:
            backward_coords = [[pt.lon, pt.lat] for pt in self.backward_trajectory]
            features.append({
                "type": "Feature",
                "properties": {
                    "layer": "backward_trajectory",
                    "color": "#FF4500",
                    "description": "Hindcast Origin Track (Backward Drift)",
                    "num_steps": len(self.backward_trajectory),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": backward_coords,
                },
            })

        # 3. Origin Search Windows (Polygons for M4 and M6)
        for window in self.origin_search_windows:
            bbox_poly = [
                [window.min_lon, window.min_lat],
                [window.max_lon, window.min_lat],
                [window.max_lon, window.max_lat],
                [window.min_lon, window.max_lat],
                [window.min_lon, window.min_lat],
            ]
            features.append({
                "type": "Feature",
                "properties": {
                    "layer": "origin_search_window",
                    "window_id": window.window_id,
                    "start_time": window.start_time.isoformat(),
                    "end_time": window.end_time.isoformat(),
                    "radius_km": round(window.radius_km, 2),
                    "confidence": window.confidence_level,
                    "color": "#DC143C",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [bbox_poly],
                },
            })

        # 4. Forward Trajectory Line
        if self.forward_trajectory:
            forward_coords = [[pt.lon, pt.lat] for pt in self.forward_trajectory]
            features.append({
                "type": "Feature",
                "properties": {
                    "layer": "forward_trajectory",
                    "color": "#00BFFF",
                    "description": "Forecast Drift Track (Forward Drift)",
                    "stranding_risk": self.stranding_risk,
                    "num_steps": len(self.forward_trajectory),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": forward_coords,
                },
            })

        return {
            "type": "FeatureCollection",
            "spill_id": self.spill_id,
            "features": features,
        }

    def to_ais_query_filters(self) -> List[Dict[str, Any]]:
        """
        Pre-formatted parameters specifically for M4 (Priya - AIS & Attribution)
        to filter candidate vessel positions.
        """
        filters = []
        for w in self.origin_search_windows:
            filters.append({
                "window_id": w.window_id,
                "start_time": w.start_time,
                "end_time": w.end_time,
                "centroid": (w.centroid_lat, w.centroid_lon),
                "radius_km": w.radius_km,
                "bbox": {
                    "min_lat": w.min_lat,
                    "max_lat": w.max_lat,
                    "min_lon": w.min_lon,
                    "max_lon": w.max_lon,
                },
            })
        return filters

    def save_json(self, path: str) -> None:
        """Save DriftResult as JSON file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, path: str) -> "DriftResult":
        """Load DriftResult from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)