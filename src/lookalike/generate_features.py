from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.lookalike.features import extract_features


IMAGE_DIR = Path("data/raw/deep_sar/images/images/train")
MASK_DIR = Path("data/raw/deep_sar/masks/masks/train")

OUTPUT_DIR = Path("data/processed/lookalike")
OUTPUT_FILE = OUTPUT_DIR / "train_features.csv"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in IMAGE_DIR.rglob("*.png")
        if not p.name.startswith(".")
    )

    mask_files = [
        p for p in MASK_DIR.rglob("*.png")
        if not p.name.startswith(".")
    ]

    mask_map = {p.name: p for p in mask_files}

    rows = []

    print(f"Images found : {len(image_files)}")

    for i, image_path in enumerate(image_files, start=1):

        mask_path = mask_map.get(image_path.name)

        if mask_path is None:
            continue

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

        features["image_name"] = image_path.name

        rows.append(features)

        if i % 500 == 0:
            print(f"Processed {i}/{len(image_files)}")

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n========================================")
    print("FEATURE GENERATION COMPLETED")
    print("========================================")
    print(f"Samples generated : {len(df)}")
    print(f"Features          : {len(df.columns) - 1}")
    print(f"Saved to          : {OUTPUT_FILE}")
    print("========================================")


if __name__ == "__main__":
    main()