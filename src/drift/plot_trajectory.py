"""
plot_trajectory.py
Visualization utility for OilTrace-AI (M3).

Generates presentation-grade and dashboard-ready map plots displaying:
- Detected spill centroid
- Backward hindcasting trajectory (origin reconstruction with uncertainty cone)
- Forward forecasting trajectory (slick spread and disaster projection)
- Origin search bounding boxes for AIS attribution (M4)
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.drift.drift_result import DriftResult


def plot_drift_result(
    drift_result: DriftResult,
    output_path: Union[str, Path] = "data/currents/drift_trajectory_map.png",
    title: Optional[str] = None,
    dpi: int = 200,
) -> Path:
    """
    Generate a detailed trajectory map plot from a DriftResult.

    Parameters
    ----------
    drift_result : DriftResult
        The completed simulation result container.
    output_path : str or Path
        Destination path to save the output figure (PNG).
    title : str, optional
        Custom title for the figure.
    dpi : int
        Image resolution. Default 200 dpi.

    Returns
    -------
    Path
        Absolute path to the saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#FAFAFA")
    ax.set_facecolor("#EBF4F6")  # Light oceanic background

    # 1. Plot Detection Point (Starting seed for backward/forward)
    det_lon = drift_result.detection_lon
    det_lat = drift_result.detection_lat
    ax.scatter(
        [det_lon],
        [det_lat],
        color="#D90429",
        s=180,
        marker="*",
        edgecolors="black",
        linewidth=1.2,
        zorder=10,
        label=f"Spill Detection Centroid\n({drift_result.detection_time.strftime('%Y-%m-%d %H:%M UTC')})",
    )

    # 2. Plot Backward Trajectory (Hindcast -> Origin Reconstruction)
    if drift_result.backward_trajectory:
        b_lons = [p.lon for p in drift_result.backward_trajectory]
        b_lats = [p.lat for p in drift_result.backward_trajectory]
        b_radii_deg = [(p.uncertainty_radius_m / 111320.0) for p in drift_result.backward_trajectory]

        # Draw uncertainty cone envelope
        upper_lat = np.array(b_lats) + np.array(b_radii_deg)
        lower_lat = np.array(b_lats) - np.array(b_radii_deg)
        ax.fill_betweenx(
            b_lats,
            np.array(b_lons) - np.array(b_radii_deg),
            np.array(b_lons) + np.array(b_radii_deg),
            color="#FF9F1C",
            alpha=0.25,
            label="Origin Uncertainty Corridor (2σ)",
            zorder=3,
        )

        ax.plot(
            b_lons,
            b_lats,
            color="#F77F00",
            linestyle="--",
            linewidth=2.2,
            marker="o",
            markersize=5,
            zorder=5,
            label=f"Backward Hindcast ({drift_result.duration_backward_hours}h back)",
        )

        # Mark estimated origin point (oldest point in backward trajectory)
        orig_pt = drift_result.backward_trajectory[-1]
        ax.scatter(
            [orig_pt.lon],
            [orig_pt.lat],
            color="#9D0208",
            s=120,
            marker="X",
            edgecolors="black",
            zorder=9,
            label=f"Probable Release Window ({orig_pt.time.strftime('%m-%d %H:%M')})",
        )

    # 3. Plot Forward Trajectory (Forecast -> Slick Movement)
    if drift_result.forward_trajectory:
        f_lons = [p.lon for p in drift_result.forward_trajectory]
        f_lats = [p.lat for p in drift_result.forward_trajectory]
        f_radii_deg = [(p.uncertainty_radius_m / 111320.0) for p in drift_result.forward_trajectory]

        ax.fill_betweenx(
            f_lats,
            np.array(f_lons) - np.array(f_radii_deg),
            np.array(f_lons) + np.array(f_radii_deg),
            color="#0077B6",
            alpha=0.2,
            label="Forecast Dispersion Corridor",
            zorder=2,
        )

        ax.plot(
            f_lons,
            f_lats,
            color="#0096C7",
            linestyle="-",
            linewidth=2.2,
            marker="s",
            markersize=4,
            zorder=6,
            label=f"Forward Forecast ({drift_result.duration_forward_hours}h ahead)",
        )

    # 4. Plot Origin Search Bounding Boxes for M4 AIS Correlation
    for w in drift_result.origin_search_windows:
        width = w.max_lon - w.min_lon
        height = w.max_lat - w.min_lat
        rect = patches.Rectangle(
            (w.min_lon, w.min_lat),
            width,
            height,
            linewidth=1.2,
            edgecolor="#C1121F",
            facecolor="none",
            linestyle=":",
            zorder=7,
        )
        ax.add_patch(rect)

    # Annotate M4 AIS bounding box
    if drift_result.origin_search_windows:
        first_box = drift_result.origin_search_windows[-1]
        ax.annotate(
            f"AIS Query Box (M4)\nRadius: {first_box.radius_km:.1f} km",
            xy=(first_box.centroid_lon, first_box.centroid_lat),
            xytext=(first_box.centroid_lon + 0.02, first_box.centroid_lat + 0.02),
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3B0", ec="#E09F3E", lw=1),
            arrowprops=dict(arrowstyle="->", color="#333333"),
            fontsize=8,
            fontweight="bold",
            zorder=12,
        )

    # 5. Styling and Labels
    plot_title = title or (
        f"OilTrace-AI (SIH26143) - Hydrodynamic Drift Simulation\n"
        f"Spill ID: {drift_result.spill_id} | CMEMS Hydrodynamic & Wind Forcing"
    )
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude (°E)", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Latitude (°N)", fontsize=10, fontweight="semibold")
    ax.grid(True, linestyle="--", alpha=0.5, color="#A0AAB2")

    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#B0B0B0",
    )

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=dpi)
    plt.close(fig)

    print(f"Trajectory map plot successfully saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    from datetime import datetime
    from src.drift.pipeline import run_drift_pipeline

    print("Running sample drift pipeline to generate plot...")
    result = run_drift_pipeline(
        spill_id="SIH26143_SAMPLE_01",
        longitude=72.5,
        latitude=7.5,
        detection_time=datetime(2025, 1, 2, 0, 0),
        duration_backward_hours=24,
        duration_forward_hours=24,
        num_particles=25,
    )
    plot_drift_result(result, "data/currents/test_drift_trajectory.png")