from pathlib import Path


DATASET_DIR = Path("data/raw/refined_deep_sar")

IMAGE_TRAIN = DATASET_DIR / "images" / "train"
IMAGE_VAL = DATASET_DIR / "images" / "val"

MASK_TRAIN = DATASET_DIR / "masks" / "train"
MASK_VAL = DATASET_DIR / "masks" / "val"


def get_png_files(folder):
    return sorted(
        file for file in folder.glob("*.png")
        if not file.name.startswith("._")
    )


def check_split(image_folder, mask_folder, split_name):
    images = get_png_files(image_folder)
    masks = get_png_files(mask_folder)

    print(f"\n--- {split_name} ---")
    print("Images:", len(images))
    print("Masks :", len(masks))

    image_names = {file.name for file in images}
    mask_names = {file.name for file in masks}

    missing_masks = image_names - mask_names
    missing_images = mask_names - image_names

    print("Missing masks :", len(missing_masks))
    print("Missing images:", len(missing_images))

    if not missing_masks and not missing_images:
        print("✅ Images and masks match perfectly!")
    else:
        print("⚠️ Images and masks do not match.")


if __name__ == "__main__":
    check_split(IMAGE_TRAIN, MASK_TRAIN, "TRAIN")
    check_split(IMAGE_VAL, MASK_VAL, "VALIDATION")
    