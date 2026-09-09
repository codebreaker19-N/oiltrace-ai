from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import DeepSARDataset
from unet import UNet


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/segmentation/best_unet.pth"

IMAGE_VAL_DIR = "data/raw/deep_sar/images/images/val"
MASK_VAL_DIR = "data/raw/deep_sar/masks/masks/val"

BATCH_SIZE = 4

# Pehle small evaluation
QUICK_TEST = False
QUICK_TEST_SAMPLES = 100


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("OILTRACE-AI - U-Net Evaluation")
print("=" * 60)

print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {device}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading validation dataset...")

val_dataset = DeepSARDataset(
    IMAGE_VAL_DIR,
    MASK_VAL_DIR
)

print(f"Full validation dataset : {len(val_dataset)}")


if QUICK_TEST:

    count = min(
        QUICK_TEST_SAMPLES,
        len(val_dataset)
    )

    val_dataset = Subset(
        val_dataset,
        range(count)
    )

    print(f"Quick test samples      : {len(val_dataset)}")


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Validation batches      : {len(val_loader)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained U-Net...")

model = UNet(
    in_channels=1,
    out_channels=1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# ============================================================
# EVALUATION
# ============================================================

total_dice = 0.0
total_iou = 0.0

num_batches = 0


with torch.no_grad():

    for batch_idx, (images, masks) in enumerate(val_loader):

        images = images.to(device)
        masks = masks.to(device)

        # Model prediction
        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities > 0.5
        ).float()


        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        predictions_flat = predictions.view(
            predictions.size(0), -1
        )

        masks_flat = masks.view(
            masks.size(0), -1
        )


        # ----------------------------------------------------
        # Intersection
        # ----------------------------------------------------

        intersection = (
            predictions_flat * masks_flat
        ).sum(dim=1)


        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        dice = (
            (2 * intersection + 1.0)
            /
            (
                predictions_flat.sum(dim=1)
                + masks_flat.sum(dim=1)
                + 1.0
            )
        )


        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        union = (
            predictions_flat
            + masks_flat
            - predictions_flat * masks_flat
        ).sum(dim=1)

        iou = (
            (intersection + 1.0)
            /
            (union + 1.0)
        )


        total_dice += dice.mean().item()
        total_iou += iou.mean().item()

        num_batches += 1

        print(
            f"\rEvaluating batch "
            f"{batch_idx + 1}/{len(val_loader)}",
            end=""
        )


# ============================================================
# FINAL RESULTS
# ============================================================

average_dice = total_dice / num_batches
average_iou = total_iou / num_batches


print("\n")
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"Dice Score : {average_dice:.4f}")
print(f"IoU Score  : {average_iou:.4f}")

print("=" * 60)