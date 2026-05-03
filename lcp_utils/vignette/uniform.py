from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image
import numpy as np

from lcp_utils.parser.lcp import Vignette

BIN_COUNT = 512
CENTER_RADIUS_FRACTION = 0.15
BORDER_FRACTION = 0.01
BLACK_MARGIN = 0.01
WHITE_MARGIN = 0.01
MIN_BIN_SAMPLES = 1000


def calibrate(images: list[Image.Image]) -> Vignette:
    """Fit an LCP vignette model from uniform flat-field captures.

    The fitter uses the green channel from RGB images as the flat-field signal.
    It uses the image center and unit focal length as the internal radial coordinate
    system, and emits only the polynomial parameters needed by the LCP vignette
    model.
    """

    if len(images) < 2:
        raise ValueError("at least 2 flat-field images are required")

    width, height = images[0].size
    dmax = float(max(width, height))
    image_x_center = width / (2.0 * dmax)
    image_y_center = height / (2.0 * dmax)

    all_r2: list[np.ndarray] = []
    all_brightness: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    all_mad: list[np.ndarray] = []

    for index, image in enumerate(images):
        if image.size != (width, height):
            raise ValueError(
                f"image {index} has size {image.size}, expected {(width, height)}"
            )
        frame = _frame_bins(
            image,
            index=index,
            image_x_center=image_x_center,
            image_y_center=image_y_center,
        )
        all_r2.append(frame.r2)
        all_brightness.append(frame.brightness)
        all_counts.append(frame.counts)
        all_mad.append(frame.mad)

    r2 = np.concatenate(all_r2)
    brightness = np.concatenate(all_brightness)
    counts = np.concatenate(all_counts)
    mad = np.concatenate(all_mad)

    if len(r2) < 4:
        raise ValueError("not enough populated radial bins to fit vignette model")

    params = _fit_polynomial(r2, brightness, counts, mad)
    modeled = _brightness_model(r2, params)
    if np.any(modeled <= 0):
        raise ValueError("fitted vignette model becomes non-positive")

    corrected = brightness / modeled
    corrected /= np.median(corrected)
    residuals = corrected - 1.0
    residual_mean = float(np.average(np.abs(residuals), weights=counts))

    return Vignette(
        param1=float(params[0]),
        residual_mean_error=residual_mean,
    )


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
        bins=BIN_COUNT,
        min_bin_samples=MIN_BIN_SAMPLES,
    )


def _image_brightness(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"))
    if np.issubdtype(array.dtype, np.integer):
        scale = float(np.iinfo(array.dtype).max)
    else:
        scale = 1.0

    rgb = array.astype(np.float64, copy=False) / scale
    return rgb[:, :, 1]


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
) -> np.ndarray:
    a = r2[:, None]
    b = brightness - 1.0
    weights = np.sqrt(counts) / np.maximum(mad, 1e-4)
    params, _, _, _ = np.linalg.lstsq(a * weights[:, None], b * weights, rcond=None)
    return np.asarray(params, dtype=np.float64)


def _brightness_model(r2: np.ndarray, params: np.ndarray) -> np.ndarray:
    return 1.0 + params[0] * r2
