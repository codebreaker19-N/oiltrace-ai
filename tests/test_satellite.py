import numpy as np

from src.satellite.preprocessing import normalize_image
from src.satellite.tiling import create_tiles


def test_normalize_image():
    image = np.array([[0, 10], [20, 30]], dtype=np.float32)

    normalized = normalize_image(image)

    assert normalized.min() == 0
    assert normalized.max() == 1


def test_create_tiles():
    image = np.random.rand(1, 512, 512)

    tiles = create_tiles(image, tile_size=256)

    assert tiles.shape[0] == 4
    assert tiles.shape[-2:] == (256, 256)