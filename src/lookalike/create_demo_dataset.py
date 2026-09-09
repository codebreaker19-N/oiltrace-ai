from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.lookalike.features import extract_features


IMAGE_DIR = Path("data/raw/deep_sar/images/images/train")
MASK_DIR = Path("data/raw/deep_sar/masks/masks/train")

OUTPUT_DIR = Path("data/processed/lookalike")
OUTPUT_FILE = OUTPUT_DIR / "demo_classifier_dataset.csv"

N_OIL = 300
N_LOOKALIKE = 300

RANDOM_SEED = 42


def load_files():
    images = sorted(
        p for p in IMAGE_DIR.rglob("*.png")
        if not p.name.startswith(".")
    )

    masks = [
        p for p in MASK_DIR.rglob("*.png")
        if not p.name.startswith(".")
    ]

    mask_map = {p.name: p for p in masks}

    pairs = []

    for image_path in images:
        mask_path = mask_map.get(image_path.name)

        if mask_path is not None:
            pairs.append((image_path, mask_path))

    return pairs


def create_lookalike_mask(shape, rng):
    """
    Create a synthetic irregular dark region
    to simulate a look-alike candidate for the demo.
    """

    height, width = shape

    mask = np.zeros((height, width), dtype=np.uint8)

    # Random center
    cx = rng.integers(width // 4, 3 * width // 4)
    cy = rng.integers(height // 4, 3 * height // 4)

    # Random ellipse dimensions
    rx = rng.integers(15, 45)
    ry = rng.integers(8, 35)

    yy, xx = np.ogrid[:height, :width]

    ellipse = (
        ((xx - cx) / rx) ** 2
        + ((yy - cy) / ry) ** 2
        <= 1
    )

    mask[ellipse] = 1

    return mask


def main():

    rng = np.random.default_rng(RANDOM_SEED)

    pairs = load_files()

    print(f"Available image-mask pairs : {len(pairs)}")

    # Use only a small subset for the demo
    oil_pairs = pairs[:N_OIL]

    rows = []

    # -------------------------
    # Real oil samples
    # -------------------------

    print("\nGenerating oil samples...")

    for i, (image_path, mask_path) in enumerate(oil_pairs):

        image = np.array(
            Image.open(image_path).convert("L"),
            dtype=np.float32
        )

        mask_rgb = np.array(
            Image.open(mask_path).convert("RGB"),
            dtype=np.uint8
        )

        mask = np.all(mask_rgb == 255, axis=2).astype(np.uint8)

        features = extract_features(image, mask)

        features["label"] = 1
        features["class_name"] = "oil"

        rows.append(features)

    # -------------------------
    # Synthetic look-alike
    # -------------------------

    print("Generating look-alike samples...")

    for i in range(N_LOOKALIKE):

        image_path, _ = pairs[
            rng.integers(0, len(pairs))
        ]

        image = np.array(
            Image.open(image_path).convert("L"),
            dtype=np.float32
        )

        mask = create_lookalike_mask(
            image.shape,
            rng
        )

        features = extract_features(image, mask)

        features["label"] = 0
        features["class_name"] = "lookalike"

        rows.append(features)

    # -------------------------
    # Save dataset
    # -------------------------

    df = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========================================")
    print("DEMO CLASSIFIER DATASET CREATED")
    print("========================================")
    print(f"Total samples : {len(df)}")
    print(f"Oil samples   : {(df['label'] == 1).sum()}")
    print(f"Look-alikes   : {(df['label'] == 0).sum()}")
    print(f"Saved to      : {OUTPUT_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()