import torch

from dataset import DeepSARDataset


IMAGE_DIR = "data/raw/deep_sar/images/images/train"
MASK_DIR = "data/raw/deep_sar/masks/masks/train"


dataset = DeepSARDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR
)


print("=" * 60)
print("        OILTRACE-AI - PYTORCH DATASET TEST")
print("=" * 60)

print("\nDataset size :", len(dataset))


# Get first sample
image, mask = dataset[0]

print("\nSample")
print("-" * 60)

print("Image shape :", image.shape)
print("Image dtype :", image.dtype)
print("Image min   :", image.min().item())
print("Image max   :", image.max().item())

print("\nMask shape  :", mask.shape)
print("Mask dtype  :", mask.dtype)
print("Mask values :", torch.unique(mask))


# Basic validation
assert image.shape == (1, 256, 256)
assert mask.shape == (1, 256, 256)

assert image.min() >= 0
assert image.max() <= 1

assert torch.all(
    (mask == 0) | (mask == 1)
)

print("\nAll Dataset checks PASSED!")

print("\n" + "=" * 60)