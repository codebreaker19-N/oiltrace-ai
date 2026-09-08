from pathlib import Path
from PIL import Image


DATASET_DIR = Path("data/raw/refined_deep_sar")

IMAGE_DIRS = [
    DATASET_DIR / "images" / "train",
    DATASET_DIR / "images" / "val",
]

MASK_DIRS = [
    DATASET_DIR / "masks" / "train",
    DATASET_DIR / "masks" / "val",
]


def check_dimensions():

    image_sizes = {}
    mask_sizes = {}

    for directory in IMAGE_DIRS:
        for path in directory.glob("*.png"):
            with Image.open(path) as img:
                size = img.size
                image_sizes[size] = image_sizes.get(size, 0) + 1

    for directory in MASK_DIRS:
        for path in directory.glob("*.png"):
            with Image.open(path) as img:
                size = img.size
                mask_sizes[size] = mask_sizes.get(size, 0) + 1

    print("Image dimensions:")
    for size, count in image_sizes.items():
        print(size, ":", count)

    print("\nMask dimensions:")
    for size, count in mask_sizes.items():
        print(size, ":", count)

    assert len(image_sizes) == 1
    assert len(mask_sizes) == 1
    assert list(image_sizes.keys())[0] == (256, 256)
    assert list(mask_sizes.keys())[0] == (256, 256)

    print("\n✅ All images are 256x256!")
    print("✅ All masks are 256x256!")
    print("✅ No tiling required for this dataset!")


if __name__ == "__main__":
    check_dimensions()