import json

import cv2
import numpy as np
from shapely.geometry import Polygon


def extract_contours(binary_mask):
    """
    Extract oil-spill contours from a binary mask.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        Binary mask containing 0 (background) and 1 (oil spill).

    Returns
    -------
    contours : list
        List of detected contours.
    """

    mask = (binary_mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def contour_to_polygon(contour):
    """
    Convert a contour into a Shapely polygon.

    Parameters
    ----------
    contour : numpy.ndarray
        OpenCV contour.

    Returns
    -------
    Polygon or None
        Shapely polygon if valid enough.
    """

    points = contour.reshape(-1, 2)

    if len(points) < 3:
        return None

    polygon = Polygon(points)

    # Fix invalid polygons if possible
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    return polygon


def extract_spill_geometry(binary_mask):
    """
    Extract geometry information from an oil-spill binary mask.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        Binary mask with values 0 and 1.

    Returns
    -------
    dict
        Spill geometry information.
    """

    contours = extract_contours(binary_mask)

    polygons = []

    for contour in contours:

        polygon = contour_to_polygon(contour)

        if polygon is not None and not polygon.is_empty:
            polygons.append(polygon)

    # No spill detected
    if not polygons:
        return {
            "polygon_count": 0,
            "area_pixels": 0,
            "centroid": None,
            "bounding_box": None,
        }

    # Select largest spill region
    largest_polygon = max(
        polygons,
        key=lambda polygon: polygon.area
    )

    # Bounding box
    min_x, min_y, max_x, max_y = largest_polygon.bounds

    # Centroid
    centroid = largest_polygon.centroid

    return {
        "polygon_count": len(polygons),

        "area_pixels": float(
            largest_polygon.area
        ),

        "centroid": {
            "x": float(centroid.x),
            "y": float(centroid.y),
        },

        "bounding_box": {
            "min_x": float(min_x),
            "min_y": float(min_y),
            "max_x": float(max_x),
            "max_y": float(max_y),
        },
    }


def geometry_to_geojson(binary_mask):
    """
    Convert the largest oil-spill region into GeoJSON.

    IMPORTANT:
    The current dataset is PNG-based and does not contain
    geographic georeferencing. Therefore, the coordinates
    generated here are PIXEL coordinates, not latitude/longitude.
    """

    contours = extract_contours(binary_mask)

    if not contours:
        return None

    # Find largest contour
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    # Convert contour to polygon
    polygon = contour_to_polygon(largest_contour)

    if polygon is None or polygon.is_empty:
        return None

    # Convert polygon coordinates
    coordinates = [
        [
            [
                float(x),
                float(y)
            ]
            for x, y in polygon.exterior.coords
        ]
    ]

    geojson = {
        "type": "Feature",

        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        },

        "properties": {
            "area_pixels": float(
                polygon.area
            ),

            "centroid_x": float(
                polygon.centroid.x
            ),

            "centroid_y": float(
                polygon.centroid.y
            )
        }
    }

    return geojson


def save_geometry_geojson(binary_mask, output_path):
    """
    Extract spill geometry and save it as a GeoJSON file.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        Binary oil-spill mask.

    output_path : str or Path
        Output GeoJSON file path.

    Returns
    -------
    bool
        True if file was successfully saved,
        False if no geometry was found.
    """

    geojson = geometry_to_geojson(binary_mask)

    if geojson is None:
        return False

    # Make sure output directory exists
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save GeoJSON
    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            geojson,
            f,
            indent=2
        )

    return True