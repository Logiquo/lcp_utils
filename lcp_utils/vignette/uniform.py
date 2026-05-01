from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lcp_utils.parser.lcp import Vignette

RAW_SUFFIXES = {".arw", ".dng", ".nef", ".cr2", ".cr3", ".raf", ".rw2"}
BIN_COUNT = 512
CENTER_RADIUS_FRACTION = 0.15
BORDER_FRACTION = 0.01
BLACK_MARGIN = 32.0
WHITE_MARGIN = 128.0
MIN_BIN_SAMPLES = 1000


def calibrate(path: str | Path) -> Vignette:
    """Fit an LCP vignette model from uniform raw flat-field captures.

    The fitter works on linear raw CFA data and uses green samples by default.
    It uses the image center and unit focal length as the internal radial
    coordinate system, and emits only the polynomial parameters needed by the
    LCP vignette model.
    """

    folder = Path(path)
    raw_paths = _raw_paths(folder)
    if not raw_paths:
        raise ValueError(f"no supported raw files found in {folder}")
    if len(raw_paths) < 2:
        raise ValueError("at least 2 flat-field raw files are required")

    first = _raw_info(raw_paths[0])
    width = first.width
    height = first.height
    dmax = float(max(width, height))
    image_x_center = width / (2.0 * dmax)
    image_y_center = height / (2.0 * dmax)

    all_r2: list[np.ndarray] = []
    all_brightness: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    all_mad: list[np.ndarray] = []

    for raw_path in raw_paths:
        frame = _frame_bins(
            raw_path,
            width=width,
            height=height,
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
class _RawInfo:
    width: int
    height: int


@dataclass(frozen=True, kw_only=True)
class _FrameBins:
    r2: np.ndarray
    brightness: np.ndarray
    counts: np.ndarray
    mad: np.ndarray


def _raw_paths(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in RAW_SUFFIXES
    )


def _raw_info(path: Path) -> _RawInfo:
    rawpy = _rawpy()
    with rawpy.imread(str(path)) as raw:
        array = raw.raw_image_visible
        return _RawInfo(width=int(array.shape[1]), height=int(array.shape[0]))


def _frame_bins(
    path: Path,
    *,
    width: int,
    height: int,
    image_x_center: float,
    image_y_center: float,
) -> _FrameBins:
    rawpy = _rawpy()
    with rawpy.imread(str(path)) as raw:
        raw_image = np.asarray(raw.raw_image_visible, dtype=np.float32)
        if raw_image.shape != (height, width):
            raise ValueError(
                f"{path.name} has size {raw_image.shape[::-1]}, "
                f"expected {(width, height)}"
            )

        colors = _raw_colors(raw)
        black_levels = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        white_level = float(raw.white_level)
        green_ids = _green_color_ids(raw)

    if colors.shape != raw_image.shape:
        raise ValueError(f"raw color map for {path.name} has unexpected shape")

    black = black_levels[np.clip(colors, 0, len(black_levels) - 1)]
    saturated = raw_image >= white_level - WHITE_MARGIN
    valid = (raw_image > black + BLACK_MARGIN) & ~saturated
    valid &= np.isin(colors, green_ids)
    valid &= _border_mask(width, height, BORDER_FRACTION)

    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError(f"{path.name} has no valid green flat-field pixels")

    ys, xs = np.nonzero(valid)
    channel_black = black[ys, xs]
    headroom = np.maximum(1.0, white_level - channel_black)
    brightness = (raw_image[ys, xs] - channel_black) / headroom

    dmax = float(max(width, height))
    u = xs.astype(np.float64) + 0.5
    v = ys.astype(np.float64) + 0.5
    center_u = image_x_center * dmax
    center_v = image_y_center * dmax
    radius_pixels = np.hypot(u - center_u, v - center_v)
    center_radius = CENTER_RADIUS_FRACTION * math.hypot(width, height)
    center_values = brightness[radius_pixels <= center_radius]
    if len(center_values) < MIN_BIN_SAMPLES:
        raise ValueError(f"{path.name} has too few valid central pixels")

    center_brightness = float(np.median(center_values))
    if center_brightness <= 0:
        raise ValueError(f"{path.name} has non-positive central brightness")
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


def _raw_colors(raw: Any) -> np.ndarray:
    colors = getattr(raw, "raw_colors_visible", None)
    if colors is not None:
        return np.asarray(colors, dtype=np.int16)

    pattern = np.asarray(raw.raw_pattern, dtype=np.int16)
    height, width = raw.raw_image_visible.shape
    tiled = np.tile(
        pattern,
        (math.ceil(height / pattern.shape[0]), math.ceil(width / pattern.shape[1])),
    )
    return tiled[:height, :width]


def _green_color_ids(raw: Any) -> np.ndarray:
    color_desc = bytes(raw.color_desc).decode("ascii", "ignore")
    green_ids = [index for index, name in enumerate(color_desc) if name == "G"]
    if not green_ids:
        raise ValueError("raw file does not expose green CFA channels")
    return np.asarray(green_ids, dtype=np.int16)


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


def _rawpy() -> Any:
    try:
        import rawpy
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "rawpy is required for uniform vignette calibration"
        ) from exc
    return rawpy
