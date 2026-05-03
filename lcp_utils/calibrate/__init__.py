from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from lcp_utils.calibrate import exif
from lcp_utils.parser.lcp import Profile, Perspective, Vignette
from lcp_utils import perspective
from lcp_utils import vignette

__all__ = ["calibrate"]


def calibrate(path: Path) -> list[Profile]:
    with open(path / "calibration.json") as f:
        metadata: dict[str, Any] = json.load(f)

    if "Calibrations" not in metadata or len(metadata["Calibrations"]) == 0:
        raise Exception("There is nothing to calibrate")

    calibrations: list[dict[str, Any]] = metadata["Calibrations"]
    persp_tasks: set[str] = set()
    vigne_tasks: set[str] = set()
    for calibration in calibrations:
        if "Perspective" in calibration:
            persp_tasks.add(calibration["Perspective"])
        if "Vignette" in calibration:
            vigne_tasks.add(calibration["Vignette"])

    # Pick one of the ARW files as a EXIF source
    print("Collecting metadata")
    if len(persp_tasks) != 0:
        exif_folder = path / Path(next(iter(persp_tasks)).split(":")[0])
    else:
        exif_folder = path / Path(next(iter(vigne_tasks)).split(":")[0])
    exif_file = None
    for file in exif_folder.iterdir():
        if file.is_file() and any(
            file.name.endswith(suffix)
            for suffix in (
                ".ARW",
                ".arw",
                ".DNG",
                ".dng",
                ".NEF",
                ".nef",
                ".CR3",
                ".cr3",
            )
        ):
            exif_file = file
            break
    assert exif_file is not None
    prototype = exif.profile_prototype(exif_file, metadata)
    del exif_folder, exif_file

    # Start clibration
    persp_res: dict[str, Perspective] = {}
    for i, task in enumerate(persp_tasks):
        print(f"[{i}/{len(persp_tasks)}] Calibrating perspective {task} ...")
        image_folder = path / task.split(":")[0]
        method = task.split(":")[1]
        persp_res[task] = perspective.calibrate(image_folder, method)
    del persp_tasks

    vigne_res: dict[str, Vignette] = {}
    for i, task in enumerate(vigne_tasks):
        print(f"[{i}/{len(vigne_tasks)}] Calibrating vignette {task} ...")
        image_folder = path / task.split(":")[0]
        method = task.split(":")[1]
        vigne_res[task] = vignette.calibrate(image_folder, method)
    del vigne_tasks

    # Assemble the result
    profiles: list[Profile] = []
    for calibration in calibrations:
        profile = copy.deepcopy(prototype)
        profile.focal_length = float(calibration["FocalLength"])
        profile.focus_distance = float(calibration["FocusDistance"])
        # We need to convert from f-value to APEX aperture value
        profile.aperture_value = float(2.0 * math.log2(float(calibration["fValue"])))

        if "Perspective" in calibration:
            profile.perspective = persp_res[calibration["Perspective"]]
        else:
            # This is a zero geometry correction profile as a placeholder
            profile.perspective = Perspective(radial_distort_param1=0.0)

        if "Vignette" in calibration and profile.perspective is not None:
            profile.perspective.vignette = vigne_res[calibration["Vignette"]]

        profiles.append(profile)

    return profiles
