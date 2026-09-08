from pathlib import Path

import numpy as np

from src.satellite.preprocessing import (
    load_png_image,
    convert_mask_to_binary,
)

from src.satellite.output import (
    create_spill_output,
    save_spill_output,
)


DATASET_DIR = Path(
    "data/raw/refined_deep_sar"
)

MASK_PATH = (
    DATASET_DIR
    / "masks"
    / "train"
    / "palsar_0.png"
)


def main():

    # Load mask
    mask = load_png_image(MASK_PATH)

    # Convert to binary
    binary_mask = convert_mask_to_binary(mask)

    print("Binary mask:")
    print("Shape:", binary_mask.shape)
    print("Values:", np.unique(binary_mask))

    # Create combined output
    output = create_spill_output(
        image_id="palsar_0",
        binary_mask=binary_mask,
    )

    print("\nCombined Spill Output:")
    print("Image ID:", output["image_id"])
    print("Geometry:", output["geometry"] is not None)
    print("Geometry Summary:", output["geometry_summary"])
    print("Features:", output["features"])

    # Validation
    assert output["image_id"] == "palsar_0"
    assert "geometry" in output
    assert "geometry_summary" in output
    assert "features" in output

    assert "area_pixels" in output["features"]
    assert "aspect_ratio" in output["features"]
    assert "compactness" in output["features"]
    assert "extent" in output["features"]
    assert "solidity" in output["features"]

    print("\n✅ Combined output created successfully!")

    # Save output
    output_path = Path(
        "data/processed/spill_output.json"
    )

    save_spill_output(
        output,
        output_path,
    )

    assert output_path.exists()

    print("✅ Spill output JSON saved successfully!")
    print(f"📁 File: {output_path}")


if __name__ == "__main__":
    main()