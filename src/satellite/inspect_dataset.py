from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_DIR = Path("data/raw/deep_sar/images")
MASK_DIR = Path("data/raw/deep_sar/masks")



def get_png_files(folder):
    """
    Recursively find PNG files and ignore macOS metadata files.
    Returns:
        {filename: full_path}
    """
    return {
        file.name: file
        for file in folder.rglob("*.png")
        if not file.name.startswith(".")
        and not file.name.startswith("._")
    }



def main():

    print("=" * 60)
    print("        OILTRACE-AI - DEEP-SAR DATASET INSPECTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check dataset directories
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():
        print(f"\nERROR: Image directory not found:")
        print(IMAGE_DIR)
        return

    if not MASK_DIR.exists():
        print(f"\nERROR: Mask directory not found:")
        print(MASK_DIR)
        return


    images = get_png_files(IMAGE_DIR)
    masks = get_png_files(MASK_DIR)

    image_names = set(images.keys())
    mask_names = set(masks.keys())

    print("\nDataset counts")
    print("-" * 60)
    print(f"Images : {len(images)}")
    print(f"Masks  : {len(masks)}")


    missing_masks = image_names - mask_names
    missing_images = mask_names - image_names

    print("\nImage-Mask pairing")
    print("-" * 60)
    print(f"Images without masks : {len(missing_masks)}")
    print(f"Masks without images : {len(missing_images)}")

    if missing_masks:
        print("\nExample images without masks:")
        for name in sorted(missing_masks)[:10]:
            print(" ", name)

    if missing_images:
        print("\nExample masks without images:")
        for name in sorted(missing_images)[:10]:
            print(" ", name)


    if not image_names:
        print("\nNo valid PNG images found.")
        return

    if missing_masks:
        print("\nCannot safely inspect a pair because some masks are missing.")
        return

    sample_name = sorted(image_names)[0]

    image_path = images[sample_name]
    mask_path = masks[sample_name]

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    print("\nSample pair")
    print("-" * 60)
    print(f"Filename   : {sample_name}")
    print(f"Image path : {image_path}")
    print(f"Mask path  : {mask_path}")

    print(f"\nImage size : {image.size}")
    print(f"Image mode : {image.mode}")

    print(f"Mask size  : {mask.size}")
    print(f"Mask mode  : {mask.mode}")


    image_array = np.array(image)
    mask_array = np.array(mask)

    print("\nImage pixel information")
    print("-" * 60)
    print(f"Shape : {image_array.shape}")
    print(f"Dtype : {image_array.dtype}")
    print(f"Min   : {image_array.min()}")
    print(f"Max   : {image_array.max()}")


    print("\nMask pixel information")
    print("-" * 60)
    print(f"Shape : {mask_array.shape}")
    print(f"Dtype : {mask_array.dtype}")
    print(f"Min   : {mask_array.min()}")
    print(f"Max   : {mask_array.max()}")


    if mask_array.ndim == 3:

        pixels = mask_array.reshape(-1, mask_array.shape[-1])

        unique_mask_colors = np.unique(
            pixels,
            axis=0
        )

        print(f"\nUnique mask colors : {len(unique_mask_colors)}")

        print("\nFirst mask colors:")
        for color in unique_mask_colors[:20]:
            print(" ", color)

    else:

        unique_mask_values = np.unique(mask_array)

        print(f"\nUnique mask values : {len(unique_mask_values)}")

        print("\nMask values:")
        print(unique_mask_values[:20])


    print("\nDimension check")
    print("-" * 60)

    if image.size == mask.size:
        print("Image and mask dimensions MATCH ")
    else:
        print("WARNING: Image and mask dimensions DO NOT MATCH ")


    print("\n" + "=" * 60)
    print("Inspection completed.")
    print("=" * 60)



if __name__ == "__main__":
    main()