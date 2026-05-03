from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from lcp_utils.parser.lcp import Vignette
from lcp_utils.utils import load_image
from lcp_utils.vignette import VignetteCalibration

MIN_BIN_COUNT = 32
MAX_BIN_COUNT = 768
PIXELS_PER_RADIAL_BIN = 16
CENTER_RADIUS_FRACTION = 0.15
BORDER_FRACTION = 0.01
BLACK_MARGIN = 0.01
WHITE_MARGIN = 0.01
MIN_BIN_SAMPLES = 1000


class UniformVignetteCalibration(VignetteCalibration):
    def __init__(self) -> None:
        self._image_size: tuple[int, int] | None = None
        self._image_x_center: float | None = None
        self._image_y_center: float | None = None
        self._frames: list[_FrameBins] = []

    def process_images(self, images: list[Path]) -> None:
        self._image_size = None
        self._image_x_center = None
        self._image_y_center = None
        self._frames = []

        for index, image_path in enumerate(
            tqdm(images, desc="Processing vignette images")
        ):
            image = load_image(image_path)
            if self._image_size is None:
                self._image_size = image.size
                width, height = image.size
                dmax = float(max(width, height))
                self._image_x_center = width / (2.0 * dmax)
                self._image_y_center = height / (2.0 * dmax)
            elif image.size != self._image_size:
                raise ValueError(
                    f"image {index} has size {image.size}, expected "
                    f"{self._image_size}"
                )

            assert self._image_x_center is not None
            assert self._image_y_center is not None
            self._frames.append(
                _frame_bins(
                    image,
                    index=index,
                    image_x_center=self._image_x_center,
                    image_y_center=self._image_y_center,
                )
            )

        if self._image_size is None:
            raise ValueError("at least 2 flat-field images are required")

    def fit(self, indices: list[int], precision: int) -> Vignette:
        if not isinstance(precision, int) or precision < 0:
            raise ValueError("precision must be an integer from 0 to 3")
        if precision > 3:
            raise ValueError("precision must be an integer from 0 to 3")

        if len(indices) < 2:
            raise ValueError("at least 2 flat-field images are required")

        frames = [self._frames[index] for index in indices]
        r2 = np.concatenate([frame.r2 for frame in frames])
        brightness = np.concatenate([frame.brightness for frame in frames])
        counts = np.concatenate([frame.counts for frame in frames])
        mad = np.concatenate([frame.mad for frame in frames])

        if len(r2) < 4:
            raise ValueError("not enough populated radial bins to fit vignette model")

        if precision == 0:
            residuals = brightness / np.median(brightness) - 1.0
            residual_mean = float(np.average(np.abs(residuals), weights=counts))
            return Vignette(param1=0.0, residual_mean_error=residual_mean)

        params = _fit_polynomial(r2, brightness, counts, mad, precision)
        modeled = _brightness_model(r2, params)
        if np.any(modeled <= 0):
            raise ValueError("fitted vignette model becomes non-positive")

        corrected = brightness / modeled
        corrected /= np.median(corrected)
        residuals = corrected - 1.0
        residual_mean = float(np.average(np.abs(residuals), weights=counts))

        return Vignette(
            param1=float(params[0]),
            param2=float(params[1]) if precision >= 2 else None,
            param3=float(params[2]) if precision >= 3 else None,
            residual_mean_error=residual_mean,
        )

    def val(
        self,
        params: Vignette,
        indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = [self._frames[index] for index in indices]
        r2 = np.concatenate([frame.r2 for frame in frames])
        brightness = np.concatenate([frame.brightness for frame in frames])
        counts = np.concatenate([frame.counts for frame in frames])

        modeled = _brightness_model(
            r2,
            np.asarray(
                [
                    params.param1,
                    params.param2 or 0.0,
                    params.param3 or 0.0,
                ],
                dtype=np.float64,
            ),
        )
        if np.any(modeled <= 0):
            raise ValueError("fitted vignette model becomes non-positive")

        corrected = brightness / modeled
        corrected /= np.median(corrected)
        return np.abs(corrected - 1.0), counts


@dataclass(frozen=True, kw_only=True)
class _FrameBins:
    r2: np.ndarray
    brightness: np.ndarray
    counts: np.ndarray
    mad: np.ndarray


def _frame_bins(
    image: Image.Image,
    *,
    index: int,
    image_x_center: float,
    image_y_center: float,
) -> _FrameBins:
    brightness = _image_brightness(image)
    height, width = brightness.shape

    valid = (brightness > BLACK_MARGIN) & (brightness < 1.0 - WHITE_MARGIN)
    valid &= _border_mask(width, height, BORDER_FRACTION)

    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError(f"image {index} has no valid flat-field pixels")

    ys, xs = np.nonzero(valid)
    brightness = brightness[ys, xs]

    dmax = float(max(width, height))
    u = xs.astype(np.float64) + 0.5
    v = ys.astype(np.float64) + 0.5
    center_u = image_x_center * dmax
    center_v = image_y_center * dmax
    radius_pixels = np.hypot(u - center_u, v - center_v)
    center_radius = CENTER_RADIUS_FRACTION * math.hypot(width, height)
    center_values = brightness[radius_pixels <= center_radius]
    if len(center_values) < MIN_BIN_SAMPLES:
        raise ValueError(f"image {index} has too few valid central pixels")

    center_brightness = float(np.median(center_values))
    if center_brightness <= 0:
        raise ValueError(f"image {index} has non-positive central brightness")
    brightness = brightness / center_brightness

    r2 = _r2_coordinates(
        u,
        v,
        dmax=dmax,
        image_x_center=image_x_center,
        image_y_center=image_y_center,
    )
    return _bin_samples(
        r2,
        brightness.astype(np.float64, copy=False),
        bins=_bin_count(width, height),
        min_bin_samples=MIN_BIN_SAMPLES,
    )


def _image_brightness(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"))
    if np.issubdtype(array.dtype, np.integer):
        scale = float(np.iinfo(array.dtype).max)
    else:
        scale = 1.0

    rgb = array.astype(np.float64, copy=False) / scale
    # White balance each calibration frame independently before using green.
    rgb = _white_balance(rgb)
    return rgb[:, :, 1]


def _white_balance(rgb: np.ndarray) -> np.ndarray:
    valid = np.all((rgb > BLACK_MARGIN) & (rgb < 1.0 - WHITE_MARGIN), axis=2)
    if not np.any(valid):
        return rgb

    channel_medians = np.median(rgb[valid], axis=0)
    if np.any(channel_medians <= 0):
        return rgb

    target = float(np.mean(channel_medians))
    balanced = rgb * (target / channel_medians)
    return np.clip(balanced, 0.0, 1.0)


def _bin_count(width: int, height: int) -> int:
    radial_pixels = math.hypot(width, height) / 2.0
    return max(
        MIN_BIN_COUNT,
        min(MAX_BIN_COUNT, int(round(radial_pixels / PIXELS_PER_RADIAL_BIN))),
    )


def _border_mask(width: int, height: int, border_fraction: float) -> np.ndarray:
    border = max(0, int(round(min(width, height) * border_fraction)))
    mask = np.ones((height, width), dtype=bool)
    if border:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False
    return mask


def _r2_coordinates(
    u: np.ndarray,
    v: np.ndarray,
    *,
    dmax: float,
    image_x_center: float,
    image_y_center: float,
) -> np.ndarray:
    x = u / dmax - image_x_center
    y = v / dmax - image_y_center
    return x * x + y * y


def _bin_samples(
    r2: np.ndarray,
    brightness: np.ndarray,
    *,
    bins: int,
    min_bin_samples: int,
) -> _FrameBins:
    edges = np.linspace(0.0, float(np.max(r2)), bins + 1)
    indexes = np.clip(np.searchsorted(edges, r2, side="right") - 1, 0, bins - 1)
    order = np.argsort(indexes)
    sorted_indexes = indexes[order]
    sorted_r2 = r2[order]
    sorted_brightness = brightness[order]

    unique, starts, counts = np.unique(
        sorted_indexes,
        return_index=True,
        return_counts=True,
    )
    del unique

    bin_r2 = []
    bin_brightness = []
    bin_counts = []
    bin_mad = []
    for start, count in zip(starts, counts, strict=True):
        if count < min_bin_samples:
            continue
        stop = start + count
        values = sorted_brightness[start:stop]
        median = float(np.median(values))
        bin_r2.append(float(np.median(sorted_r2[start:stop])))
        bin_brightness.append(median)
        bin_counts.append(float(count))
        bin_mad.append(float(np.median(np.abs(values - median))) + 1e-6)

    if not bin_r2:
        raise ValueError("no radial bins had enough valid samples")

    return _FrameBins(
        r2=np.asarray(bin_r2, dtype=np.float64),
        brightness=np.asarray(bin_brightness, dtype=np.float64),
        counts=np.asarray(bin_counts, dtype=np.float64),
        mad=np.asarray(bin_mad, dtype=np.float64),
    )


def _fit_polynomial(
    r2: np.ndarray,
    brightness: np.ndarray,
    counts: np.ndarray,
    mad: np.ndarray,
    precision: int,
) -> np.ndarray:
    a = np.column_stack([r2**order for order in range(1, precision + 1)])
    b = brightness - 1.0
    weights = np.sqrt(counts) / np.maximum(mad, 1e-4)
    params, _, _, _ = np.linalg.lstsq(a * weights[:, None], b * weights, rcond=None)
    padded = np.zeros(3, dtype=np.float64)
    padded[: len(params)] = params
    return padded


def _brightness_model(r2: np.ndarray, params: np.ndarray) -> np.ndarray:
    return 1.0 + params[0] * r2 + params[1] * r2**2 + params[2] * r2**3
