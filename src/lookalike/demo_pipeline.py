from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.satellite.unet import UNet
from src.lookalike.predict import predict_lookalike


IMAGE_PATH = Path(
    "data/raw/deep_sar/images/images/val/palsar_0.png"
)

MODEL_PATH = Path(
    "models/segmentation/best_unet.pth"
)


def load_unet():

    model = UNet(
        in_channels=1,
        out_channels=1
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def main():

    print("\n========================================")
    print("OILTRACE-AI - END-TO-END DEMO")
    print("========================================")

    # -------------------------
    # Load image
    # -------------------------

    image = np.array(
        Image.open(IMAGE_PATH).convert("L"),
        dtype=np.float32
    )

    image_normalized = image / 255.0

    tensor = torch.from_numpy(
        image_normalized
    ).unsqueeze(0).unsqueeze(0)

    # -------------------------
    # U-Net prediction
    # -------------------------

    print("\nRunning U-Net...")

    model = load_unet()

    with torch.no_grad():

        logits = model(tensor)

        probability = torch.sigmoid(logits)

        predicted_mask = (
            probability >= 0.5
        ).float()

    mask = predicted_mask.squeeze().numpy().astype(
        np.uint8
    )

    spill_pixels = int(mask.sum())

    print(
        f"Predicted spill pixels : {spill_pixels}"
    )

    # -------------------------
    # Look-alike classification
    # -------------------------

    print("\nRunning XGBoost...")

    result = predict_lookalike(
        image,
        mask
    )

    # -------------------------
    # Final output
    # -------------------------

    print("\n========================================")
    print("FINAL RESULT")
    print("========================================")

    print(
        f"Classification : {result['class']}"
    )

    print(
        f"Confidence     : {result['confidence']:.4f}"
    )

    print("========================================")


if __name__ == "__main__":
    main()