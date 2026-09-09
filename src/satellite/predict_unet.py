from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from unet import UNet


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/segmentation/best_unet.pth"

IMAGE_PATH = (
    "data/raw/deep_sar/images/images/val/palsar_0.png"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")


# ============================================================
# LOAD MODEL
# ============================================================

model = UNet(
    in_channels=1,
    out_channels=1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Saved checkpoint se model weights load karo
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# ============================================================
# LOAD IMAGE
# ============================================================

image = np.array(
    Image.open(IMAGE_PATH).convert("L"),
    dtype=np.float32
)

# Normalize 0-255 → 0-1
image_normalized = image / 255.0

# [H,W] → [1,1,H,W]
input_tensor = torch.from_numpy(
    image_normalized
).unsqueeze(0).unsqueeze(0)

input_tensor = input_tensor.to(device)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    logits = model(input_tensor)

    probabilities = torch.sigmoid(logits)

    prediction = (
        probabilities > 0.5
    ).float()


# ============================================================
# CONVERT TO NUMPY
# ============================================================

probability_map = (
    probabilities[0, 0]
    .cpu()
    .numpy()
)

predicted_mask = (
    prediction[0, 0]
    .cpu()
    .numpy()
)


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original SAR Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(probability_map, cmap="hot")
plt.title("Oil Probability Map")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask, cmap="gray")
plt.title("Predicted Oil Mask")
plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# STATISTICS
# ============================================================

oil_pixels = predicted_mask.sum()
total_pixels = predicted_mask.size

oil_percentage = (
    oil_pixels / total_pixels
) * 100

print("\nPrediction Statistics")
print("=" * 40)
print(f"Image size       : {image.shape}")
print(f"Oil pixels       : {int(oil_pixels)}")
print(f"Total pixels     : {total_pixels}")
print(f"Predicted oil %  : {oil_percentage:.2f}%")
print(
    f"Max probability  : "
    f"{probability_map.max():.4f}"
)
print(
    f"Mean probability : "
    f"{probability_map.mean():.4f}"
)