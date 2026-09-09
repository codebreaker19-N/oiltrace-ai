import numpy as np

from src.lookalike.predict import predict_lookalike


def main():

    # Demo SAR image
    image = np.random.randint(
        0, 256, size=(256, 256)
    ).astype(np.float32)

    # Demo candidate region
    mask = np.zeros(
        (256, 256),
        dtype=np.uint8
    )

    mask[80:160, 80:160] = 1

    result = predict_lookalike(
        image,
        mask
    )

    print("\n========================================")
    print("LOOK-ALIKE PREDICTION")
    print("========================================")
    print(f"Class      : {result['class']}")
    print(f"Confidence : {result['confidence']:.4f}")
    print("========================================")


if __name__ == "__main__":
    main()