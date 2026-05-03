from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

from PIL import Image

from lcp_utils.parser.lcp import Perspective
from lcp_utils.perspective import charuco
from lcp_utils.utils import k_fold, list_images, load_image

__all__ = ["calibrate", "charuco"]

CalibrateFunc = Callable[[list[Image.Image], int], Perspective]
ValidateFunc = Callable[[Perspective, list[Image.Image]], float]


def parse_method(
    method: str,
) -> tuple[CalibrateFunc, ValidateFunc]:
    match = re.fullmatch(r"charuco\((\d+)x(\d+)\)", method)
    if match is not None:
        width = int(match.group(1))
        height = int(match.group(2))
        return (
            partial(charuco.calibrate, width=width, height=height),
            partial(charuco.validate, width=width, height=height),
        )
    raise ValueError(f"unsupported perspective calibration method: {method}")


def calibrate(path: Path, method: str) -> Perspective:
    calibrate_func, validate_func = parse_method(method=method)
    images = [load_image(image_path) for image_path in list_images(path)]

    scores = []
    for precision in range(4):
        fold_errors = []
        for train_images, validate_images in k_fold(images, 5):
            params = calibrate_func(train_images, precision)
            fold_errors.append(validate_func(params, validate_images))
        err = sum(fold_errors) / len(fold_errors)
        del fold_errors
        scores.append((err, precision))
        print(f"    Fitted with precision {precision}, error {err}")

    return calibrate_func(images, min(scores)[1])
