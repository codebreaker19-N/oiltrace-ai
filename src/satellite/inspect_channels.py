from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

IMAGE_ROOT = Path("data/raw/deep_sar/images/images")


# ---------------------------------------------------------
# Find valid PNG files
# ---------------------------------------------------------

image_files = sorted(
    p for p in IMAGE_ROOT.rglob("*.png")
    if not p.name.startswith(".")
)

print("=" * 60)
print("        OILTRACE-AI - CHANNEL INSPECTION")
print("=" * 60)

print("\nTotal images :", len(image_files))


# ---------------------------------------------------------
# Inspect first sample
# ---------------------------------------------------------

sample_path = image_files[0]

image = np.array(
    Image.open(sample_path).convert("RGB")
)

print("\nSample")
print("-" * 60)
print("Filename :", sample_path.name)
print("Path     :", sample_path)
print("Shape    :", image.shape)
print("Dtype    :", image.dtype)


# ---------------------------------------------------------
# Channel statistics
# ---------------------------------------------------------

print("\nChannel statistics")
print("-" * 60)

for channel in range(3):
    data = image[:, :, channel]

    print(f"\nChannel {channel}")
    print("  Min  :", data.min())
    print("  Max  :", data.max())
    print("  Mean :", round(float(data.mean()), 3))
    print("  Std  :", round(float(data.std()), 3))


# ---------------------------------------------------------
# Check whether channels are identical
# ---------------------------------------------------------

r = image[:, :, 0]
g = image[:, :, 1]
b = image[:, :, 2]

print("\nChannel equality")
print("-" * 60)

print("R == G :", np.array_equal(r, g))
print("G == B :", np.array_equal(g, b))
print("R == B :", np.array_equal(r, b))


# ---------------------------------------------------------
# Difference between channels
# ---------------------------------------------------------

print("\nMean absolute channel differences")
print("-" * 60)

print("R-G :", round(float(np.mean(np.abs(r.astype(float) - g.astype(float)))), 3))
print("G-B :", round(float(np.mean(np.abs(g.astype(float) - b.astype(float)))), 3))
print("R-B :", round(float(np.mean(np.abs(r.astype(float) - b.astype(float)))), 3))


# ---------------------------------------------------------
# Final conclusion
# ---------------------------------------------------------

if np.array_equal(r, g) and np.array_equal(g, b):
    print("\nConclusion")
    print("-" * 60)
    print("All three channels are identical.")
    print("The image is effectively grayscale stored as RGB.")
else:
    print("\nConclusion")
    print("-" * 60)
    print("The three channels contain different values.")
    print("Keep the 3-channel representation for now.")

print("\n" + "=" * 60)
print("Inspection completed.")
print("=" * 60)