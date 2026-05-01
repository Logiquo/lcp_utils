from __future__ import annotations

import json
import re
from pathlib import Path

import rawpy
from PIL import Image

from lcp_utils.parser.lcp import Perspective
from lcp_utils.perspective.charuco import calibrate, new_board

__all__ = ["calibrate", "new_board", "perspective"]

_RAW_SUFFIXES = {".arw", ".dng", ".nef", ".cr3"}
_IMAGE_SUFFIXES = _RAW_SUFFIXES | {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def perspective(path: Path) -> Perspective:
    settings_path = path / "metadata.json"
    if settings_path.exists():
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
    else:
        settings = {}

    method = settings.get("method")
    if method is None:
        raise ValueError("metadata.json must define method, e.g. charuco(16x10)")

    match = re.fullmatch(r"charuco\((\d+)x(\d+)\)", method)
    if match is None:
        raise ValueError(f"unsupported perspective calibration method: {method}")

    width = int(match.group(1))
    height = int(match.group(2))
    images = [_load_image(image_path) for image_path in _image_paths(path)]
    return calibrate(width, height, images)


def _image_paths(path: Path) -> list[Path]:
    paths = [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not paths:
        raise FileNotFoundError(f"no calibration images found in {path}")
    return sorted(paths)


def _load_image(path: Path) -> Image.Image:
    if path.suffix.lower() in _RAW_SUFFIXES:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        return Image.fromarray(rgb)

    with Image.open(path) as image:
        return image.convert("RGB")
