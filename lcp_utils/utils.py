from pathlib import Path
import rawpy
from PIL import Image

_RAW_SUFFIXES = {".arw", ".dng", ".nef", ".cr3"}
_IMAGE_SUFFIXES = _RAW_SUFFIXES | {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def list_images(path: Path) -> list[Path]:
    paths = [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not paths:
        raise FileNotFoundError(f"no calibration images found in {path}")
    return sorted(paths)


def load_image(path: Path) -> Image.Image:
    if path.suffix.lower() in _RAW_SUFFIXES:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        return Image.fromarray(rgb)

    with Image.open(path) as image:
        return image.convert("RGB")
