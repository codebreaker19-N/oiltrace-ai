import numpy as np
try:
    import rasterio
except ImportError:
    rasterio = None


def load_sar_image(path):
    """
    Load a SAR image using Rasterio.
    """
    with rasterio.open(path) as src:
        image = src.read()
        profile = src.profile

    return image, profile


def normalize_image(image):
    """
    Normalize SAR image values between 0 and 1.
    """
    image = image.astype(np.float32)

    min_val = np.nanmin(image)
    max_val = np.nanmax(image)

    image = (image - min_val) / (max_val - min_val + 1e-8)

    return image