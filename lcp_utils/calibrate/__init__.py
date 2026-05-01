from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from lcp_utils.calibrate.metadata import metadata
from lcp_utils.parser.lcp import Profile
from lcp_utils.perspective import perspective
from lcp_utils.vignette import vignette

__all__ = ["calibrate", "metadata"]


def calibrate(path: Path) -> list[Profile]:
    res = []

    print("Collecting metadata...")
    prototype = metadata(path)
    for persp_path in (path / "perspective").iterdir():
        persp_profile = copy.deepcopy(prototype)
        
        with open(persp_path / "metadata.json") as f:
            persp_metadata = json.load(f)
            persp_profile.focal_length = float(persp_metadata["FocalLength"])
            persp_profile.focus_distance = float(persp_metadata["FocusDistance"])
            persp_profile.aperture_value = _aperture_value(
                float(persp_metadata["ApertureValue"])
            )
            del persp_metadata

        print(
            "Calibrate perspective for focal length "
            f"{persp_profile.focal_length}, focus distance "
            f"{persp_profile.focus_distance}"
        )
        persp_profile.perspective = perspective(persp_path)
        
        if not (path / "vignette").exists():
            res.append(persp_profile)
            continue
        
        for vigne_path in (path / "vignette").iterdir():
            vigne_profile = copy.deepcopy(persp_profile)
            with open(vigne_path / "metadata.json") as f:
                vigne_metadata = json.load(f)
            if vigne_profile.focal_length != float(vigne_metadata["FocalLength"]):
                continue
            if vigne_profile.focus_distance != float(vigne_metadata["FocusDistance"]):
                continue
            vigne_profile.aperture_value = _aperture_value(
                float(vigne_metadata["ApertureValue"])
            )
            print(f"Calibrate vignette for for aperture {vigne_profile.aperture_value}")
            assert vigne_profile.perspective is not None
            vigne_profile.perspective.vignette = vignette(vigne_path)
            res.append(vigne_profile)
            
    return res


def _aperture_value(f_number: float) -> float:
    return float(2.0 * math.log2(f_number))
