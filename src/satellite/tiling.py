import numpy as np


def create_tiles(image, tile_size=256):
    """
    Split image into fixed-size tiles.
    """
    tiles = []

    height, width = image.shape[-2:]

    for y in range(0, height - tile_size + 1, tile_size):
        for x in range(0, width - tile_size + 1, tile_size):

            tile = image[..., y:y + tile_size, x:x + tile_size]

            if tile.shape[-2:] == (tile_size, tile_size):
                tiles.append(tile)

    return np.array(tiles)