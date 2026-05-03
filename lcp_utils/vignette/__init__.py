from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from lcp_utils.parser.lcp import Vignette
from lcp_utils.utils import k_fold, list_images

__all__ = ["calibrate", "uniform"]


class VignetteCalibration(ABC):
    @abstractmethod
    def process_images(self, images: list[Path]) -> None:
        pass

    @abstractmethod
    def fit(self, indices: list[int], precision: int) -> Vignette:
        pass

    @abstractmethod
    def val(
        self,
        params: Vignette,
        indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        pass


from lcp_utils.vignette import uniform  # noqa: E402


def parse_method(method: str) -> VignetteCalibration:
    if method == "uniform":
        return uniform.UniformVignetteCalibration()
    raise ValueError(f"unsupported vignette calibration method: {method}")


def calibrate(path: Path, method: str) -> Vignette:
    calibration = parse_method(method)
    image_paths = list_images(path)
    calibration.process_images(image_paths)
    indices = list(range(len(image_paths)))

    scores = []
    for precision in range(4):
        val_errors = []
        val_weights = []
        for train_indices, validate_indices in k_fold(indices, 5):
            params = calibration.fit(train_indices, precision)
            errors, weights = calibration.val(params, validate_indices)
            val_errors.append(errors)
            val_weights.append(weights)
        val_errors = np.concatenate(val_errors)
        val_weights = np.concatenate(val_weights)
        mu = float(np.average(val_errors, weights=val_weights))
        variance = float(np.average((val_errors - mu) ** 2, weights=val_weights))
        sigma = float(np.sqrt(variance))
        scores.append((mu, precision))
        print(f"    Fitted with precision {precision}, μ {mu}, σ {sigma}")

    precision = _prompt_precision(min(scores)[1])
    return calibration.fit(indices, precision)


def _prompt_precision(default: int) -> int:
    while True:
        response = input(f"Select vignette precision [0-3] (default {default}): ")
        if response.strip() == "":
            return default
        try:
            precision = int(response)
        except ValueError:
            print("Please enter an integer from 0 to 3.")
            continue
        if 0 <= precision <= 3:
            return precision
        print("Please enter an integer from 0 to 3.")
