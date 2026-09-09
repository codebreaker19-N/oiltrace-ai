import torch
from torch.utils.data import DataLoader

from dataset import DeepSARDataset


IMAGE_DIR = "data/raw/deep_sar/images/images/train"
MASK_DIR = "data/raw/deep_sar/masks/masks/train"


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

dataset = DeepSARDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR
)


# ---------------------------------------------------------
# DataLoader
# ---------------------------------------------------------

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)


print("=" * 60)
print("        OILTRACE-AI - DATALOADER TEST")
print("=" * 60)

print("\nDataset size :", len(dataset))
print("Batch size   :", 4)


# ---------------------------------------------------------
# Get one batch
# ---------------------------------------------------------

images, masks = next(iter(loader))


print("\nBatch information")
print("-" * 60)

print("Images shape :", images.shape)
print("Images dtype :", images.dtype)

print("Masks shape  :", masks.shape)
print("Masks dtype  :", masks.dtype)


print("\nImage range")
print("-" * 60)

print("Min :", images.min().item())
print("Max :", images.max().item())


print("\nMask values")
print("-" * 60)

print(torch.unique(masks))


# ---------------------------------------------------------
# Assertions
# ---------------------------------------------------------

assert images.shape == (4, 1, 256, 256)
assert masks.shape == (4, 1, 256, 256)

assert images.dtype == torch.float32
assert masks.dtype == torch.float32

assert images.min() >= 0
assert images.max() <= 1

assert torch.all(
    (masks == 0) | (masks == 1)
)


print("\nAll DataLoader checks PASSED!")

print("\n" + "=" * 60)