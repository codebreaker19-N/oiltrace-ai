import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

from PIL import Image


def load_png_image(path):
    """Load a PNG image as a NumPy array."""
    image = Image.open(path)
    return np.array(image)


def normalize_image(image):
    """Normalize an image to the range 0-1."""
    image = image.astype(np.float32)

    min_val = image.min()
    max_val = image.max()

    return (image - min_val) / (max_val - min_val + 1e-8)


def convert_mask_to_binary(mask):
    """
    Convert RGB mask into a single-channel binary mask.

    Background = 0
    Oil spill = 1
    """

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    binary_mask = (mask > 0).astype(np.uint8)

    return binary_mask