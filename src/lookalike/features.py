import cv2
import numpy as np


def extract_features(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Extract image and geometry features from a SAR image and binary spill mask.

    Parameters
    ----------
    image : np.ndarray
        Grayscale SAR image, shape (H, W).
    mask : np.ndarray
        Binary spill mask, shape (H, W).

    Returns
    -------
    dict
        Numerical features for the look-alike classifier.
    """

    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask)

    # Convert mask to binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Basic area
    area = float(np.sum(binary_mask))

    # If no spill pixels are present
    if area == 0:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "compactness": 0.0,
            "aspect_ratio": 0.0,
            "mean_intensity": 0.0,
            "intensity_std": 0.0,
            "intensity_contrast": 0.0,
        }

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        perimeter = float(cv2.arcLength(largest_contour, True))

        x, y, width, height = cv2.boundingRect(largest_contour)

        aspect_ratio = float(width / height) if height > 0 else 0.0
    else:
        perimeter = 0.0
        aspect_ratio = 0.0

    # Compactness
    if perimeter > 0:
        compactness = float((4 * np.pi * area) / (perimeter ** 2))
    else:
        compactness = 0.0

    # Pixel intensities inside predicted spill region
    spill_pixels = image[binary_mask == 1]

    mean_intensity = float(np.mean(spill_pixels))
    intensity_std = float(np.std(spill_pixels))

    # Contrast between spill region and surrounding background
    background_pixels = image[binary_mask == 0]

    if len(background_pixels) > 0:
        background_mean = float(np.mean(background_pixels))
        intensity_contrast = float(
            abs(mean_intensity - background_mean)
        )
    else:
        intensity_contrast = 0.0

    return {
        "area": area,
        "perimeter": perimeter,
        "compactness": compactness,
        "aspect_ratio": aspect_ratio,
        "mean_intensity": mean_intensity,
        "intensity_std": intensity_std,
        "intensity_contrast": intensity_contrast,
    }