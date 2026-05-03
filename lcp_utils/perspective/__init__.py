from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from lcp_utils.parser.lcp import Perspective
from lcp_utils.utils import k_fold, list_images

__all__ = ["calibrate", "charuco"]


class PerspecitveCalibration(ABC):
    @abstractmethod
    def process_images(self, images: list[Path]) -> None:
        pass

    @abstractmethod
    def fit(self, indices: list[int], precision: int) -> Perspective:
        pass

    @abstractmethod
    def val(self, params: Perspective, indices: list[int]) -> float:
        pass


from lcp_utils.perspective import charuco  # noqa: E402


def parse_method(method: str) -> PerspecitveCalibration:
    match = re.fullmatch(r"charuco\((\d+)x(\d+)\)", method)
    if match is not None:
        width = int(match.group(1))
        height = int(match.group(2))
        return charuco.ChArUcoPerspecitveCalibration(width, height)
    raise ValueError(f"unsupported perspective calibration method: {method}")


def calibrate(path: Path, method: str) -> Perspective:
    calibration = parse_method(method)
    image_paths = list_images(path)
    calibration.process_images(image_paths)
    indices = list(range(len(image_paths)))

    scores = []
    for precision in range(4):
        fold_errors = []
        for train_indices, validate_indices in k_fold(indices, 5):
            params = calibration.fit(train_indices, precision)
            fold_errors.append(calibration.val(params, validate_indices))
        err = sum(fold_errors) / len(fold_errors)
        scores.append((err, precision))
        print(f"    Fitted with precision {precision}, error {err}")

    return calibration.fit(indices, min(scores)[1])
