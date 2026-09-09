import numpy as np

from src.lookalike.features import extract_features


def test_feature_extraction():

    # Dummy SAR image
    image = np.random.randint(
        0, 256, size=(256, 256)
    ).astype(np.float32)

    # Dummy square spill region
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[80:160, 80:160] = 1

    features = extract_features(image, mask)

    print("\nExtracted features:")

    for name, value in features.items():
        print(f"{name:20s}: {value}")

    # Basic checks
    assert isinstance(features, dict)

    assert features["area"] > 0
    assert features["perimeter"] > 0
    assert features["compactness"] > 0
    assert features["aspect_ratio"] > 0

    assert features["mean_intensity"] >= 0
    assert features["intensity_std"] >= 0
    assert features["intensity_contrast"] >= 0

    print("\nFeature extraction PASSED!")


if __name__ == "__main__":
    test_feature_extraction()