"""
Demo runner for the oil-spill drift attribution pipeline.

This script connects:
1. Demo spill detection event
2. Event validation
3. Backward drift ensemble
4. Source-region estimation
5. Result export
"""

from drift.io_utils import load_event, validate_event, save_result
from drift.opendrift_wrapper import DriftModel
from drift.backtrack_ensemble import BacktrackEnsemble


def run_demo(event_path):
    """Run the complete demo drift pipeline."""

    event = load_event(event_path)
    validate_event(event)

    model = DriftModel()

    ensemble = BacktrackEnsemble(
        drift_model=model
    )

    result = ensemble.run(
        event_id=event["event_id"],
        latitude=event["centroid"]["lat"],
        longitude=event["centroid"]["lon"],
        detection_time=event["detection_time"],
        duration_hours=72,
        particles=100,
    )

    return result


if __name__ == "__main__":
    event_path = "data/processed/demo_spill_event.json"
    output_path = "data/processed/demo_drift_result.json"

    result = run_demo(event_path)

    save_result(
        result,
        output_path,
    )

    print("Oil Spill Drift Demo")
    print("--------------------")
    print(f"Event ID: {result['event_id']}")
    print(
        f"Detection location: "
        f"{result['detection']['latitude']}, "
        f"{result['detection']['longitude']}"
    )
    print(
        f"Backtracking duration: "
        f"{result['simulation']['duration_hours']} hours"
    )
    print(
        f"Particles: "
        f"{result['simulation']['particles']}"
    )
    print(
        f"Status: "
        f"{result['simulation']['status']}"
    )

    if result["source_region"]:
        center = result["source_region"]["center"]

        print(
            f"Estimated source region: "
            f"{center['latitude']}, "
            f"{center['longitude']}"
        )

        print(
            f"Source region radius: "
            f"{result['source_region']['radius_km']:.2f} km"
        )
