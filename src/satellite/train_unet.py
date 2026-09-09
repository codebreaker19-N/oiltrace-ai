from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from dataset import DeepSARDataset
from unet import UNet
from losses import BCEDiceLoss


 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("OILTRACE-AI - U-Net Training")
print("=" * 60)

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"Device          : {device}")

if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
else:
    print("GPU             : CPU only")

print("=" * 60)


IMAGE_TRAIN_DIR = "data/raw/deep_sar/images/images/train"
MASK_TRAIN_DIR = "data/raw/deep_sar/masks/masks/train"

IMAGE_VAL_DIR = "data/raw/deep_sar/images/images/val"
MASK_VAL_DIR = "data/raw/deep_sar/masks/masks/val"



# CPU ke liye chhota batch size
BATCH_SIZE = 4

# Pehle sirf 1 epoch test karenge
EPOCHS = 3

LEARNING_RATE = 1e-4

# Windows + CPU par 0 safest hai
NUM_WORKERS = 0

# CPU par AMP use nahi karna
USE_AMP = False

# True = sirf limited images se test
# False = poora dataset
QUICK_TEST = False

# Quick test mein kitne samples
QUICK_TEST_SAMPLES = 100


# ============================================================
# 4. MODEL OUTPUT DIRECTORY
# ============================================================

MODEL_DIR = Path("models/segmentation")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_unet.pth"


# ============================================================
# 5. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = DeepSARDataset(
    IMAGE_TRAIN_DIR,
    MASK_TRAIN_DIR
)

val_dataset = DeepSARDataset(
    IMAGE_VAL_DIR,
    MASK_VAL_DIR
)

print(f"Full training dataset   : {len(train_dataset)}")
print(f"Full validation dataset : {len(val_dataset)}")


# ============================================================
# 6. QUICK TEST MODE
# ============================================================

if QUICK_TEST:

    train_count = min(QUICK_TEST_SAMPLES, len(train_dataset))
    val_count = min(
        QUICK_TEST_SAMPLES // 5,
        len(val_dataset)
    )

    train_dataset = torch.utils.data.Subset(
        train_dataset,
        range(train_count)
    )

    val_dataset = torch.utils.data.Subset(
        val_dataset,
        range(val_count)
    )

    print("\nQUICK TEST MODE")
    print(f"Training samples   : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")


# ============================================================
# 7. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

print(f"Training batches   : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")


# ============================================================
# 8. MODEL
# ============================================================

print("\nCreating U-Net...")

model = UNet(
    in_channels=1,
    out_channels=1
)

model = model.to(device)

print(
    f"Trainable parameters : "
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================
# 9. LOSS + OPTIMIZER
# ============================================================

criterion = BCEDiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 10. DICE SCORE
# ============================================================

def dice_score(logits, targets, threshold=0.5):

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities > threshold).float()

    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)

    dice = (
        (2 * intersection + 1.0)
        /
        (
            predictions.sum(dim=1)
            + targets.sum(dim=1)
            + 1.0
        )
    )

    return dice.mean().item()


# ============================================================
# 11. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_dice = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(outputs, masks)

        print(
            f"\rTrain batch "
            f"{batch_idx + 1}/{len(train_loader)} "
            f"| Loss: {loss.item():.4f}",
            end=""
        )

    print()

    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)

    return avg_loss, avg_dice


# ============================================================
# 12. VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)

    return avg_loss, avg_dice


# ============================================================
# 13. TRAINING LOOP
# ============================================================

best_val_loss = float("inf")

print("\nStarting training...")
print("=" * 60)

for epoch in range(EPOCHS):

    start_time = time.time()

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    train_loss, train_dice = train_one_epoch()

    val_loss, val_dice = validate()

    epoch_time = time.time() - start_time

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Dice : {train_dice:.4f}"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Dice   : {val_dice:.4f}"
    )

    print(
        f"Time       : {epoch_time:.1f} sec"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "learning_rate": LEARNING_RATE,
            },
            BEST_MODEL_PATH
        )

        print(
            f"Best model saved -> {BEST_MODEL_PATH}"
        )


# ============================================================
# 14. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Best model : {BEST_MODEL_PATH}")
print(f"Best Val Loss : {best_val_loss:.4f}")