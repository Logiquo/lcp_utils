from __future__ import annotations

import re
from pathlib import Path

from lcp_utils.parser.lcp import Perspective
from lcp_utils.perspective import charuco
from lcp_utils.utils import load_image, list_images

__all__ = ["calibrate", "charuco"]


def calibrate(path: Path, method: str) -> Perspective:
    match = re.fullmatch(r"charuco\((\d+)x(\d+)\)", method)
    if match is not None:
        width = int(match.group(1))
        height = int(match.group(2))
        images = [load_image(image_path) for image_path in list_images(path)]
        return charuco.calibrate(images, width, height)
    raise ValueError(f"unsupported perspective calibration method: {method}")
