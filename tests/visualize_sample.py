from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.satellite.preprocessing import (
    load_png_image,
    convert_mask_to_binary,
)


DATASET_DIR = Path("data/raw/refined_deep_sar")

IMAGE_PATH = DATASET_DIR / "images" / "train" / "palsar_0.png"
MASK_PATH = DATASET_DIR / "masks" / "train" / "palsar_0.png"


def main():

    image = load_png_image(IMAGE_PATH)
    mask = load_png_image(MASK_PATH)

    binary_mask = convert_mask_to_binary(mask)

    # RGB image ko grayscale mein display karne ke liye
    if image.ndim == 3:
        image_display = image.mean(axis=2)
    else:
        image_display = image

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_display, cmap="gray")
    plt.title("SAR Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Oil Spill Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image_display, cmap="gray")
    plt.imshow(binary_mask, alpha=0.4)
    plt.title("Image + Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()