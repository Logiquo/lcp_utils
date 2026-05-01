from __future__ import annotations

import json
from pathlib import Path

from lcp_utils.parser.lcp import Vignette
from lcp_utils.vignette.uniform import calibrate

__all__ = ["calibrate", "vignette"]


def vignette(path: Path) -> Vignette:
    settings_path = path / "metadata.json"
    if settings_path.exists():
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
    else:
        settings = {}

    method = settings.get("method")
    if method is None:
        raise ValueError("metadata.json must define method, e.g. uniform")
    if str(method).lower() != "uniform":
        raise ValueError(f"unsupported vignette calibration method: {method}")

    return calibrate(path)
