from __future__ import annotations

import re
from pathlib import Path

from lcp_utils.parser.lcp import Vignette
from lcp_utils.vignette import uniform
from lcp_utils.utils import load_image, list_images

__all__ = ["calibrate"]


def calibrate(path: Path, method: str) -> Vignette:
    if method == "uniform":
        images = [load_image(image_path) for image_path in list_images(path)]
        return uniform.calibrate(images)
    raise ValueError(f"unsupported perspective calibration method: {method}")
