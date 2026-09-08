import json
from pathlib import Path

from src.satellite.geometry import (
    extract_spill_geometry,
    geometry_to_geojson,
)

from src.satellite.features import (
    extract_shape_features,
)


def create_spill_output(
    image_id,
    binary_mask,
):
    """
    Create a standardized spill output containing
    geometry and shape features.

    Parameters
    ----------
    image_id : str
        Unique image identifier.

    binary_mask : numpy.ndarray
        Binary oil-spill mask with values 0 and 1.

    Returns
    -------
    dict
        Combined spill information.
    """

    # Extract geometry
    geometry = extract_spill_geometry(
        binary_mask
    )

    # Extract shape features
    features = extract_shape_features(
        binary_mask
    )

    # Convert geometry to GeoJSON
    geojson = geometry_to_geojson(
        binary_mask
    )

    # Create standardized output
    output = {
        "image_id": image_id,

        "geometry": geojson,

        "geometry_summary": geometry,

        "features": features,
    }

    return output


def save_spill_output(
    output,
    output_path,
):
    """
    Save standardized spill output as JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    return True