from pathlib import Path

import numpy as np
from PIL import Image

from src.satellite.preprocessing import (
    load_png_image,
    normalize_image,
    convert_mask_to_binary,
)


DATASET_DIR = Path("data/raw/refined_deep_sar")

IMAGE_PATH = DATASET_DIR / "images" / "train" / "palsar_0.png"
MASK_PATH = DATASET_DIR / "masks" / "train" / "palsar_0.png"


def main():

    # Load image and mask
    image = load_png_image(IMAGE_PATH)
    mask = load_png_image(MASK_PATH)

    print("Original image:")
    print("Shape:", image.shape)
    print("Data type:", image.dtype)
    print("Minimum:", image.min())
    print("Maximum:", image.max())

    # Normalize image
    normalized_image = normalize_image(image)

    print("\nNormalized image:")
    print("Shape:", normalized_image.shape)
    print("Data type:", normalized_image.dtype)
    print("Minimum:", normalized_image.min())
    print("Maximum:", normalized_image.max())

    # Convert mask
    binary_mask = convert_mask_to_binary(mask)

    print("\nBinary mask:")
    print("Shape:", binary_mask.shape)
    print("Unique values:", np.unique(binary_mask))

    # Checks
    assert normalized_image.min() >= 0
    assert normalized_image.max() <= 1

    assert binary_mask.shape == (256, 256)
    assert set(np.unique(binary_mask)).issubset({0, 1})

    print("\n✅ Image normalization successful!")
    print("✅ Binary mask conversion successful!")


if __name__ == "__main__":
    main()