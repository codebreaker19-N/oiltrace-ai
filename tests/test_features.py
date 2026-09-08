from pathlib import Path

import numpy as np

from src.satellite.preprocessing import (
    load_png_image,
    convert_mask_to_binary,
)

from src.satellite.features import (
    extract_shape_features,
    save_features_json,
)


# --------------------------------------------------
# Dataset paths
# --------------------------------------------------

DATASET_DIR = Path(
    "data/raw/refined_deep_sar"
)

MASK_PATH = (
    DATASET_DIR
    / "masks"
    / "train"
    / "palsar_0.png"
)


# --------------------------------------------------
# Main test
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # 1. Load mask
    # --------------------------------------------------

    mask = load_png_image(
        MASK_PATH
    )

    print("Original mask:")
    print("Shape:", mask.shape)
    print("Values:", np.unique(mask))


    # --------------------------------------------------
    # 2. Convert mask to binary
    # --------------------------------------------------

    binary_mask = convert_mask_to_binary(
        mask
    )

    print("\nBinary mask:")
    print("Shape:", binary_mask.shape)
    print("Values:", np.unique(binary_mask))


    # --------------------------------------------------
    # 3. Validate binary mask
    # --------------------------------------------------

    assert binary_mask.shape == (
        256,
        256
    )

    assert set(
        np.unique(binary_mask)
    ).issubset({0, 1})


    print(
        "\n✅ Mask converted to single channel!"
    )

    print(
        "✅ Mask values are correctly 0 and 1!"
    )


    # --------------------------------------------------
    # 4. Extract shape features
    # --------------------------------------------------

    features = extract_shape_features(
        binary_mask
    )


    # --------------------------------------------------
    # 5. Display extracted features
    # --------------------------------------------------

    print("\nOil Spill Shape Features:")

    for name, value in features.items():

        print(
            f"{name}: {value}"
        )


    # --------------------------------------------------
    # 6. Validate features
    # --------------------------------------------------

    assert "area_pixels" in features
    assert "perimeter_pixels" in features
    assert "width_pixels" in features
    assert "height_pixels" in features
    assert "aspect_ratio" in features
    assert "compactness" in features
    assert "extent" in features
    assert "solidity" in features


    assert features["area_pixels"] >= 0
    assert features["perimeter_pixels"] >= 0
    assert features["width_pixels"] >= 0
    assert features["height_pixels"] >= 0
    assert features["aspect_ratio"] >= 0
    assert features["compactness"] >= 0
    assert features["extent"] >= 0
    assert features["solidity"] >= 0


    print(
        "\n✅ Shape feature extraction successful!"
    )


    # --------------------------------------------------
    # 7. Save features as JSON
    # --------------------------------------------------

    output_path = Path(
        "data/processed/sample_spill_features.json"
    )


    saved = save_features_json(
        features,
        output_path
    )


    # --------------------------------------------------
    # 8. Validate saved file
    # --------------------------------------------------

    assert saved
    assert output_path.exists()


    print(
        "\n✅ Features JSON saved successfully!"
    )

    print(
        f"📁 File: {output_path}"
    )


# --------------------------------------------------
# Run test
# --------------------------------------------------

if __name__ == "__main__":

    main()