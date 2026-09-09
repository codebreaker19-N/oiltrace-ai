from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DeepSARDataset(Dataset):
    """
    PyTorch Dataset for the Deep-SAR oil spill segmentation dataset.

    Input:
        Grayscale SAR image

    Target:
        Binary oil-spill mask
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        # Find valid PNG images
        self.image_files = sorted(
            p for p in self.image_dir.rglob("*.png")
            if not p.name.startswith(".")
        )

        # Find valid PNG masks
        mask_files = [
            p for p in self.mask_dir.rglob("*.png")
            if not p.name.startswith(".")
        ]

        self.mask_map = {p.name: p for p in mask_files}

        # Keep only images having a corresponding mask
        self.samples = []

        for image_path in self.image_files:
            mask_path = self.mask_map.get(image_path.name)

            if mask_path is not None:
                self.samples.append((image_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                "No image-mask pairs found. Check dataset paths."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        # -------------------------------------------------
        # Load SAR image
        # -------------------------------------------------

        image = np.array(
            Image.open(image_path).convert("L"),
            dtype=np.float32
        )

        # -------------------------------------------------
        # Normalize image
        # -------------------------------------------------

        image = image / 255.0

        # Add channel dimension:
        # (256, 256) → (1, 256, 256)
        image = np.expand_dims(image, axis=0)

        # -------------------------------------------------
        # Load mask
        # -------------------------------------------------

        mask_rgb = np.array(
            Image.open(mask_path).convert("RGB"),
            dtype=np.uint8
        )

        # White = oil
        # Black = background
        mask = np.all(mask_rgb == 255, axis=2)

        mask = mask.astype(np.float32)

        # Add channel dimension
        # (256, 256) → (1, 256, 256)
        mask = np.expand_dims(mask, axis=0)

        # -------------------------------------------------
        # Convert to tensors
        # -------------------------------------------------

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask