"""
Input/output utilities for the drift module.
"""

import json
from pathlib import Path


def load_event(path):
    """
    Load an oil-spill detection event from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the event JSON file.

    Returns
    -------
    dict
        Detection event data.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_result(result, path):
    """
    Save drift results as JSON.

    Parameters
    ----------
    result : dict
        Drift simulation result.
    path : str or Path
        Output path.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)


def validate_event(event):
    """
    Validate the minimum fields required by the drift module.

    Required fields:
    - event_id
    - detection_time
    - centroid.latitude
    - centroid.longitude
    """

    required_fields = [
        "event_id",
        "detection_time",
        "centroid",
    ]

    for field in required_fields:
        if field not in event:
            raise ValueError(f"Missing required field: {field}")

    centroid = event["centroid"]

    if "lat" not in centroid:
        raise ValueError("Missing centroid latitude.")

    if "lon" not in centroid:
        raise ValueError("Missing centroid longitude.")

    latitude = centroid["lat"]
    longitude = centroid["lon"]

    if not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude.")

    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude.")

    return True
