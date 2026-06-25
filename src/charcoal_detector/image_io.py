from __future__ import annotations

from pathlib import Path

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_binary_mask(path: Path) -> Image.Image:
    return Image.open(path).convert("L")

