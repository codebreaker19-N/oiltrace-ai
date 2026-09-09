from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

IMAGE_ROOT = Path("data/raw/deep_sar/images/images")
MASK_ROOT = Path("data/raw/deep_sar/masks/masks")


# ---------------------------------------------------------
# Find valid PNG files
# ---------------------------------------------------------

image_files = sorted(
    p for p in IMAGE_ROOT.rglob("*.png")
    if not p.name.startswith(".")
)

mask_files = sorted(
    p for p in MASK_ROOT.rglob("*.png")
    if not p.name.startswith(".")
)


# ---------------------------------------------------------
# Create filename → path mapping
# ---------------------------------------------------------

image_map = {p.name: p for p in image_files}
mask_map = {p.name: p for p in mask_files}


# ---------------------------------------------------------
# Select sample
# ---------------------------------------------------------

sample_name = sorted(image_map.keys())[0]

image_path = image_map[sample_name]
mask_path = mask_map[sample_name]


# ---------------------------------------------------------
# Load image and mask
# ---------------------------------------------------------

image = np.array(Image.open(image_path).convert("RGB"))
mask_rgb = np.array(Image.open(mask_path).convert("RGB"))


# ---------------------------------------------------------
# Convert RGB mask to binary
# ---------------------------------------------------------

mask = np.all(mask_rgb == 255, axis=2).astype(np.uint8)


# ---------------------------------------------------------
# Print information
# ---------------------------------------------------------

print("=" * 60)
print("        OILTRACE-AI - DATASET VISUALIZATION")
print("=" * 60)

print("\nSample")
print("-" * 60)
print("Filename :", sample_name)
print("Image    :", image_path)
print("Mask     :", mask_path)

print("\nImage")
print("-" * 60)
print("Shape :", image.shape)
print("Min   :", image.min())
print("Max   :", image.max())

print("\nBinary mask")
print("-" * 60)
print("Shape :", mask.shape)
print("Unique values :", np.unique(mask))

oil_pixels = np.sum(mask == 1)
background_pixels = np.sum(mask == 0)
total_pixels = mask.size

print("Oil pixels        :", oil_pixels)
print("Background pixels :", background_pixels)

print("Oil percentage    :", round(
    100 * oil_pixels / total_pixels, 2
), "%")


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Sentinel/PALSAR SAR Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Binary Oil Spill Mask")
plt.axis("off")

plt.tight_layout()
plt.show()