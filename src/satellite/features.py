import cv2
import numpy as np
import json

def extract_shape_features(binary_mask):
    """
    Extract useful shape features from an oil-spill binary mask.

    Parameters
    ----------
    binary_mask : numpy.ndarray
        Binary mask containing:
        0 = background
        1 = oil spill

    Returns
    -------
    dict
        Shape and geometry features.
    """

    mask = (binary_mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # No spill detected
    if not contours:
        return {
            "area_pixels": 0.0,
            "perimeter_pixels": 0.0,
            "width_pixels": 0.0,
            "height_pixels": 0.0,
            "aspect_ratio": 0.0,
            "compactness": 0.0,
            "extent": 0.0,
            "solidity": 0.0,
        }

    # Largest spill region
    contour = max(
        contours,
        key=cv2.contourArea
    )

    # Area
    area = cv2.contourArea(contour)

    # Perimeter
    perimeter = cv2.arcLength(
        contour,
        True
    )

    # Bounding rectangle
    x, y, width, height = cv2.boundingRect(
        contour
    )

    # Aspect ratio
    if height > 0:
        aspect_ratio = width / height
    else:
        aspect_ratio = 0.0

    # Compactness
    if perimeter > 0:
        compactness = (
            4 * np.pi * area
        ) / (perimeter ** 2)
    else:
        compactness = 0.0

    # Extent
    bounding_area = width * height

    if bounding_area > 0:
        extent = area / bounding_area
    else:
        extent = 0.0

    # Solidity
    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(
        hull
    )

    if hull_area > 0:
        solidity = area / hull_area
    else:
        solidity = 0.0

    return {
        "area_pixels": float(area),
        "perimeter_pixels": float(perimeter),
        "width_pixels": float(width),
        "height_pixels": float(height),
        "aspect_ratio": float(aspect_ratio),
        "compactness": float(compactness),
        "extent": float(extent),
        "solidity": float(solidity),
    }
def save_features_json(features, output_path):
    """
    Save extracted spill features as a JSON file.
    """

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
            features,
            f,
            indent=2
        )

    return True