from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from lcp_utils.parser.lcp import Perspective

LENGTH_SQUARE = 1
LENGTH_MARKER = 0.5
DICTIONARY = cv2.aruco.DICT_6X6_250


def new_board(width: int, height: int) -> Image.Image:
    PPS = 500

    board = _board(width, height)
    image = board.generateImage((width * PPS, height * PPS))
    return Image.fromarray(image)


def calibrate(images: list[Image.Image], width: int, height: int) -> Perspective:
    """Estimate an LCP perspective model from ChArUco board images."""

    if not images:
        raise ValueError("at least one calibration image is required")

    board = _board(width, height)
    image_size = images[0].size
    object_points = []
    image_points = []

    for index, image in enumerate(images):
        if image.size != image_size:
            raise ValueError(
                f"image {index} has size {image.size}, expected {image_size}"
            )

        charuco_corners, charuco_ids = _detect(board, image)
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 8:
            continue

        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)  # type: ignore
        if obj_points is None or img_points is None or len(obj_points) < 8:
            continue

        object_points.append(np.asarray(obj_points, dtype=np.float32))
        image_points.append(np.asarray(img_points, dtype=np.float32))

    if len(object_points) < 3:
        raise ValueError(
            "at least 3 images with 8 or more detected ChArUco corners are required; "
            f"found {len(object_points)}"
        )

    camera_matrix = cv2.initCameraMatrix2D(object_points, image_points, image_size)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    flags = (
        cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 | cv2.CALIB_USE_INTRINSIC_GUESS
    )

    (
        _,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        _,
        _,
        _,
    ) = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        dist_coeffs,
        flags=flags,
    )

    dmax = float(max(image_size))
    residuals = _relative_residuals(
        object_points,
        image_points,
        rvecs,  # type: ignore
        tvecs,  # type: ignore
        camera_matrix,
        dist_coeffs,
        dmax,
    )
    k1, k2, _, _, _ = _distortion_coefficients(dist_coeffs)

    return Perspective(
        radial_distort_param1=k1,
        radial_distort_param2=k2,
        residual_mean_error=float(np.mean(residuals)),
        residual_standard_deviation=float(np.std(residuals)),
    )


def _board(width: int, height: int) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return cv2.aruco.CharucoBoard(
        (width, height),
        LENGTH_SQUARE,
        LENGTH_MARKER,
        dictionary,
    )


def _detect(
    board: cv2.aruco.CharucoBoard,
    image: Image.Image,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    array = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        return charuco_corners, charuco_ids

    dictionary = board.getDictionary()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)  # type: ignore
    if marker_ids is None or len(marker_ids) == 0:
        return None, None

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(  # type: ignore
        marker_corners,
        marker_ids,
        gray,
        board,
    )
    return charuco_corners, charuco_ids


def _distortion_coefficients(dist_coeffs: np.ndarray) -> tuple[float, ...]:
    coefficients = np.asarray(dist_coeffs, dtype=np.float64).ravel()
    padded = np.zeros(5, dtype=np.float64)
    padded[: min(len(coefficients), len(padded))] = coefficients[: len(padded)]
    return tuple(float(value) for value in padded)


def _relative_residuals(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    dmax: float,
) -> np.ndarray:
    residuals = []
    for obj_points, img_points, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(
            obj_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        errors = np.linalg.norm(
            np.asarray(img_points).reshape(-1, 2) - projected.reshape(-1, 2),
            axis=1,
        )
        residuals.extend(errors / dmax)

    return np.asarray(residuals, dtype=np.float64)
