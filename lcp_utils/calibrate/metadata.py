import json
import math
import struct
from pathlib import Path
from typing import Any

from lcp_utils.parser.lcp import Profile


def metadata(path: Path) -> Profile:
    """Build profile metadata from ``metadata.json`` and a sibling raw file.

    Values in ``metadata.json`` override EXIF values.  The JSON may use either
    LCP-style names such as ``LensPrettyName`` or dataclass field names such as
    ``lens_pretty_name``.
    """

    with open(path / "metadata.json", encoding="utf-8") as f:
        overrides = json.load(f)
    if not isinstance(overrides, dict):
        raise ValueError("metadata.json must contain a JSON object")

    exif = _read_exif(path)
    image_width = overrides.get("ImageWidth") or exif.get("ImageWidth")
    image_length = overrides.get("ImageLength") or exif.get("ImageLength")
    sensor_format_factor = overrides.get("SensorFormatFactor")
    prefer_metadata_distort = overrides.get("PreferMetadataDistort")
    camera_raw_profile = overrides.get("CameraRawProfile", True)
    lens = overrides.get("Lens") or exif.get("Lens")
    make = overrides.get("Make") or exif.get("Make")
    camera_pretty_name = (
        overrides.get("CameraPrettyName")
        or exif.get("CameraPrettyName")
        or exif.get("Model")
    )
    lens_pretty_name = (
        overrides.get("LensPrettyName")
        or exif.get("LensPrettyName")
        or exif.get("Lens")
    )
    profile_name = overrides.get("ProfileName")

    if not make:
        raise ValueError("Make is required; add it to metadata.json")
    if not camera_pretty_name:
        raise ValueError("CameraPrettyName is required; add it to metadata.json")
    if not lens_pretty_name:
        raise ValueError("LensPrettyName is required; add it to metadata.json")
    if not profile_name:
        raise ValueError("ProfileName is required; add it to metadata.json")

    return Profile(
        author=overrides.get("Author"),
        make=make,
        model=overrides.get("Model") or exif.get("Model"),
        unique_camera_model=(
            overrides.get("UniqueCameraModel")
            or exif.get("UniqueCameraModel")
            or _camera_name(exif.get("Make"), exif.get("Model"))
        ),
        camera_pretty_name=camera_pretty_name,
        lens=lens,
        lens_info=overrides.get("LensInfo") or exif.get("LensInfo"),
        lens_id=lens,
        lens_pretty_name=lens_pretty_name,
        profile_name=profile_name,
        image_width=None if image_width is None else int(image_width),
        image_length=None if image_length is None else int(image_length),
        x_resolution=None,
        y_resolution=None,
        focal_length=0.0,
        aperture_value=0.0,
        camera_raw_profile=(
            camera_raw_profile
            if isinstance(camera_raw_profile, bool)
            else str(camera_raw_profile).lower() in {"1", "true", "yes"}
        ),
        focus_distance=0.0,
        sensor_format_factor=(
            None if sensor_format_factor is None else float(sensor_format_factor)
        ),
        prefer_metadata_distort=(
            None
            if prefer_metadata_distort is None
            else (
                prefer_metadata_distort
                if isinstance(prefer_metadata_distort, bool)
                else str(prefer_metadata_distort).lower() in {"1", "true", "yes"}
            )
        ),
    )


_TYPE_SIZES = {
    1: 1,  # BYTE
    2: 1,  # ASCII
    3: 2,  # SHORT
    4: 4,  # LONG
    5: 8,  # RATIONAL
    7: 1,  # UNDEFINED
    9: 4,  # SLONG
    10: 8,  # SRATIONAL
}

_TAGS = {
    256: "ImageWidth",
    257: "ImageLength",
    271: "Make",
    272: "Model",
    282: "XResolution",
    283: "YResolution",
    33437: "FNumber",
    34665: "ExifIFD",
    34853: "GPSIFD",
    37378: "ApertureValue",
    37382: "FocusDistance",
    37386: "FocalLength",
    40962: "ImageWidth",
    40963: "ImageLength",
    42035: "LensMake",
    42036: "Lens",
    42037: "LensID",
    50708: "UniqueCameraModel",
    50736: "LensInfo",
}


def _read_exif(path: Path) -> dict[str, Any]:
    image_path = None
    for suffix in (".ARW", ".arw", ".DNG", ".dng", ".NEF", ".nef", ".CR3", ".cr3"):
        candidate = path / f"metadata{suffix}"
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        candidates = [
            item
            for item in path.iterdir()
            if item.is_file() and item.name != "metadata.json"
        ]
        if len(candidates) == 1:
            image_path = candidates[0]
        else:
            raise FileNotFoundError(f"no metadata raw file found in {path}")

    data = image_path.read_bytes()
    if len(data) < 8 or data[:2] not in {b"II", b"MM"}:
        return {}

    byte_order = "<" if data[:2] == b"II" else ">"
    if struct.unpack_from(byte_order + "H", data, 2)[0] != 42:
        return {}

    values: dict[str, Any] = {}
    seen: set[int] = set()
    first_ifd = struct.unpack_from(byte_order + "I", data, 4)[0]
    _read_ifd(data, byte_order, first_ifd, values, seen)

    if "ApertureValue" not in values and "FNumber" in values:
        values["ApertureValue"] = 2 * math.log2(float(values["FNumber"]))
    if "CameraPrettyName" not in values and "Model" in values:
        values["CameraPrettyName"] = values["Model"]
    if "LensPrettyName" not in values and "Lens" in values:
        values["LensPrettyName"] = values["Lens"]

    return values


def _read_ifd(
    data: bytes,
    byte_order: str,
    offset: int,
    values: dict[str, Any],
    seen: set[int],
) -> None:
    if offset in seen or offset <= 0 or offset + 2 > len(data):
        return
    seen.add(offset)

    count = struct.unpack_from(byte_order + "H", data, offset)[0]
    entries_end = offset + 2 + count * 12
    if entries_end + 4 > len(data):
        return

    sub_ifds: list[int] = []
    for index in range(count):
        entry = offset + 2 + index * 12
        tag, kind, item_count, value = struct.unpack_from(
            byte_order + "HHII",
            data,
            entry,
        )
        size = _TYPE_SIZES.get(kind)
        if size is None:
            continue

        byte_count = size * item_count
        value_offset = entry + 8 if byte_count <= 4 else value
        if value_offset < 0 or value_offset + byte_count > len(data):
            continue

        name = _TAGS.get(tag)
        if name in {"ExifIFD", "GPSIFD"}:
            sub_ifds.append(int(value))
            continue
        if name is not None:
            values[name] = _decode_value(
                data[value_offset : value_offset + byte_count],
                byte_order,
                kind,
                item_count,
            )

    next_offset = struct.unpack_from(byte_order + "I", data, entries_end)[0]
    if next_offset:
        sub_ifds.append(next_offset)
    for sub_ifd in sub_ifds:
        _read_ifd(data, byte_order, sub_ifd, values, seen)


def _decode_value(
    data: bytes,
    byte_order: str,
    kind: int,
    count: int,
) -> Any:
    if kind == 2:
        return data.split(b"\0", 1)[0].decode("utf-8", "replace").strip()
    if kind == 3:
        values = struct.unpack_from(byte_order + "H" * count, data, 0)
    elif kind == 4:
        values = struct.unpack_from(byte_order + "I" * count, data, 0)
    elif kind == 5:
        values = []
        for index in range(count):
            numerator, denominator = struct.unpack_from(
                byte_order + "II",
                data,
                index * 8,
            )
            values.append(None if denominator == 0 else numerator / denominator)
        values = tuple(values)
    elif kind == 9:
        values = struct.unpack_from(byte_order + "i" * count, data, 0)
    elif kind == 10:
        values = []
        for index in range(count):
            numerator, denominator = struct.unpack_from(
                byte_order + "ii",
                data,
                index * 8,
            )
            values.append(None if denominator == 0 else numerator / denominator)
        values = tuple(values)
    else:
        return data

    return values[0] if len(values) == 1 else values


def _camera_name(make: Any, model: Any) -> str | None:
    if make is None or model is None:
        return None
    make_text = str(make).title()
    model_text = str(model)
    if model_text.lower().startswith(make_text.lower()):
        return model_text
    return f"{make_text} {model_text}"
