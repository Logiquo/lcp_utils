from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from lcp_utils.parser.lcp import Perspective
from lcp_utils.perspective import PerspecitveCalibration
from lcp_utils.utils import load_image

LENGTH_SQUARE = 1
LENGTH_MARKER = 0.5
DICTIONARY = cv2.aruco.DICT_6X6_250


class ChArUcoPerspecitveCalibration(PerspecitveCalibration):
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._board = _board(width, height)
        self._image_size: tuple[int, int] | None = None
        self._object_points: list[np.ndarray | None] = []
        self._image_points: list[np.ndarray | None] = []

    def process_images(self, images: list[Path]) -> None:
        self._image_size = None
        self._object_points = []
        self._image_points = []

        for index, image_path in enumerate(
            tqdm(images, desc="Processing images")
        ):
            image = load_image(image_path)
            if self._image_size is None:
                self._image_size = image.size
            elif image.size != self._image_size:
                raise ValueError(
                    f"image {index} has size {image.size}, expected "
                    f"{self._image_size}"
                )

            point_set = _detect_points(self._board, image)
            if point_set is None:
                self._object_points.append(None)
                self._image_points.append(None)
                continue

            obj_points, img_points = point_set
            self._object_points.append(obj_points)
            self._image_points.append(img_points)

        if self._image_size is None:
            raise ValueError("at least one calibration image is required")

    def fit(self, indices: list[int], precision: int) -> Perspective:
        if not isinstance(precision, int) or precision < 0:
            raise ValueError("precision must be an integer from 0 to 3")
        if precision > 3:
            raise ValueError("precision must be an integer from 0 to 3")

        if precision == 0:
            return Perspective(radial_distort_param1=0.0)

        image_size, object_points, image_points = self._points_for_indices(indices)
        if len(object_points) < 3:
            raise ValueError(
                "at least 3 images with 8 or more detected ChArUco corners are "
                f"required; found {len(object_points)}"
            )

        params, residuals = self._fit_points(
            object_points,
            image_points,
            image_size,
            precision,
        )
        k1, k2, _, _, k3 = _distortion_coefficients(params)

        return Perspective(
            radial_distort_param1=k1,
            radial_distort_param2=k2 if precision >= 2 else None,
            radial_distort_param3=k3 if precision >= 3 else None,
            residual_mean_error=float(np.mean(residuals)),
            residual_standard_deviation=float(np.std(residuals)),
        )

    def val(self, params: Perspective, indices: list[int]) -> float:
        image_size, object_points, image_points = self._points_for_indices(indices)
        if not object_points:
            raise ValueError(
                "at least one image with 8 or more ChArUco corners is required"
            )

        dist_coeffs = _distortion_array(params)
        camera_matrix = cv2.initCameraMatrix2D(
            object_points,
            image_points,
            image_size,
        )
        flags = (
            cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
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
        residuals = _relative_residuals(
            object_points,
            image_points,
            rvecs,  # type: ignore
            tvecs,  # type: ignore
            camera_matrix,
            dist_coeffs,
            float(max(image_size)),
        )
        return float(np.mean(residuals))

    def _points_for_indices(
        self,
        indices: list[int],
    ) -> tuple[tuple[int, int], list[np.ndarray], list[np.ndarray]]:
        if self._image_size is None:
            raise ValueError("process_images must be called before fit or val")

        object_points = []
        image_points = []
        for index in indices:
            obj_points = self._object_points[index]
            img_points = self._image_points[index]
            if obj_points is None or img_points is None:
                continue
            object_points.append(obj_points)
            image_points.append(img_points)

        return self._image_size, object_points, image_points

    def _fit_points(
        self,
        object_points: list[np.ndarray],
        image_points: list[np.ndarray],
        image_size: tuple[int, int],
        precision: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        camera_matrix = cv2.initCameraMatrix2D(
            object_points,
            image_points,
            image_size,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS
        if precision < 2:
            flags |= cv2.CALIB_FIX_K2
        if precision < 3:
            flags |= cv2.CALIB_FIX_K3

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

        residuals = _relative_residuals(
            object_points,
            image_points,
            rvecs,  # type: ignore
            tvecs,  # type: ignore
            camera_matrix,
            dist_coeffs,
            float(max(image_size)),
        )
        return dist_coeffs, residuals


def new_board(width: int, height: int) -> Image.Image:
    pps = 500

    board = _board(width, height)
    image = board.generateImage((width * pps, height * pps))
    return Image.fromarray(image)


def _board(width: int, height: int) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    return cv2.aruco.CharucoBoard(
        (width, height),
        LENGTH_SQUARE,
        LENGTH_MARKER,
        dictionary,
    )


def _detect_points(
    board: cv2.aruco.CharucoBoard,
    image: Image.Image,
) -> tuple[np.ndarray, np.ndarray] | None:
    charuco_corners, charuco_ids = _detect(board, image)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 8:
        return None

    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)  # type: ignore
    if obj_points is None or img_points is None or len(obj_points) < 8:
        return None

    return (
        np.asarray(obj_points, dtype=np.float32),
        np.asarray(img_points, dtype=np.float32),
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


def _distortion_array(params: Perspective) -> np.ndarray:
    return np.asarray(
        [
            params.radial_distort_param1,
            params.radial_distort_param2 or 0.0,
            0.0,
            0.0,
            params.radial_distort_param3 or 0.0,
        ],
        dtype=np.float64,
    ).reshape(5, 1)


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
