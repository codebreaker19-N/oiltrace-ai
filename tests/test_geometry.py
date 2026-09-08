from pathlib import Path

import numpy as np

from src.satellite.preprocessing import (
    load_png_image,
    convert_mask_to_binary,
)

from src.satellite.geometry import (
    extract_spill_geometry,
    geometry_to_geojson,
    save_geometry_geojson,
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
    # 4. Extract spill geometry
    # --------------------------------------------------

    geometry = extract_spill_geometry(
        binary_mask
    )

    print("\nOil Spill Geometry:")

    print(
        "Polygon count:",
        geometry["polygon_count"]
    )

    print(
        "Area (pixels):",
        geometry["area_pixels"]
    )

    print(
        "Centroid:",
        geometry["centroid"]
    )

    print(
        "Bounding box:",
        geometry["bounding_box"]
    )


    assert geometry["polygon_count"] >= 0


    print(
        "\n✅ Geometry extraction successful!"
    )


    # --------------------------------------------------
    # 5. Convert geometry to GeoJSON
    # --------------------------------------------------

    geojson = geometry_to_geojson(
        binary_mask
    )

    print("\nGeoJSON:")
    print(geojson)


    # --------------------------------------------------
    # 6. Validate GeoJSON
    # --------------------------------------------------

    assert geojson is not None

    assert geojson["type"] == "Feature"

    assert (
        geojson["geometry"]["type"]
        == "Polygon"
    )


    print(
        "\n✅ GeoJSON conversion successful!"
    )


    # --------------------------------------------------
    # 7. Save GeoJSON file
    # --------------------------------------------------

    output_path = Path(
        "data/processed/sample_spill.geojson"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    saved = save_geometry_geojson(
        binary_mask,
        output_path
    )

    assert saved

    assert output_path.exists()


    print(
        "\n✅ GeoJSON saved successfully!"
    )

    print(
        f"📁 File: {output_path}"
    )


# ------------------------------------------------------
# Run test
# ------------------------------------------------------

if __name__ == "__main__":
    main()