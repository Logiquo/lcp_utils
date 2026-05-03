from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from lcp_utils.parser import index, lcp
from lcp_utils.parser.index import IndexEntry

DEFAULT_LENS_PROFILE_ROOT = Path(
    r"C:\Program Files\Adobe\Adobe Lightroom CC\Resources\LensProfiles"
)
INDEX_LOGICAL_ROOT = "/Library/Application Support/Adobe/CameraRaw/LensProfiles/1.0"


def prompt_target_directory() -> Path:
    value = input(
        f"LensProfiles directory [{DEFAULT_LENS_PROFILE_ROOT}]: "
    ).strip()
    return Path(value) if value else DEFAULT_LENS_PROFILE_ROOT


def patch_profiles(input_dir: Path, target_dir: Path) -> list[IndexEntry]:
    input_dir = input_dir.resolve()
    target_dir = target_dir.resolve()
    index_path = target_dir / "Index.dat"
    mapping_path = _mapping_path(input_dir)

    index_file = index.load(index_path.read_bytes())
    profiles = _load_mapping(mapping_path)
    entries = _entries(input_dir, profiles, index_file)

    backup_path = target_dir / f"Index.dat.bak.{datetime.now():%Y%m%d%H%M%S}"
    shutil.copy2(index_path, backup_path)

    profile_paths = {entry.path for entry in entries}
    index_file.entries = [
        entry for entry in index_file.entries if entry.path not in profile_paths
    ]
    index_file.entries.extend(entries)

    for profile_file, install_path in profiles.items():
        source = input_dir / profile_file
        destination = target_dir / "1.0" / _relative_install_path(install_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    index_path.write_bytes(index.dump(index_file))
    return entries


def _mapping_path(input_dir: Path) -> Path:
    patch_json = input_dir / "patch.json"
    if patch_json.exists():
        return patch_json
    return input_dir / "metadata.json"


def _load_mapping(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected {path} to contain a JSON object")
    return {str(profile): str(install_path) for profile, install_path in value.items()}


def _entries(
    input_dir: Path,
    profiles: dict[str, str],
    index_file: index.Index,
) -> list[IndexEntry]:
    used_identifiers = {entry.identifier for entry in index_file.entries}
    entries = []

    for profile_file, install_path in profiles.items():
        data = (input_dir / profile_file).read_bytes()
        profile = lcp.load(data.decode("utf-8-sig"))[0]
        digest = index.file_digest(data)

        entry = IndexEntry(
            path=f"{INDEX_LOGICAL_ROOT}/{_logical_install_path(install_path)}",
            identifier=_new_identifier(digest, used_identifiers),
            author=str(profile.author or ""),
            auto_scale="True",
            camera_pretty_name=profile.camera_pretty_name,
            crop_factor=str(profile.sensor_format_factor or ""),
            exif_make=profile.make,
            exif_model=str(profile.model or ""),
            file_digest=digest,
            file_name=profile_file,
            image_length=str(profile.image_length or ""),
            image_width=str(profile.image_width or ""),
            is_raw="True" if profile.camera_raw_profile else "False",
            lens_id=str(profile.lens_id or ""),
            lens_info=str(profile.lens_info or "0/0 0/0 0/0 0/0"),
            lens_name=str(profile.lens or ""),
            lens_pretty_name=profile.lens_pretty_name,
            metadata_distort=(
                "True" if profile.prefer_metadata_distort else "False"
            ),
            nn_distort="False",
            profile_name=profile.profile_name,
            unique_model=str(profile.unique_camera_model or ""),
        )
        used_identifiers.add(entry.identifier)
        entries.append(entry)

    return entries


def _new_identifier(digest: str, used: set[int]) -> int:
    identifier = int(digest[:16], 16)
    while identifier in used:
        identifier = (identifier + 1) & 0xFFFFFFFFFFFFFFFF
    return identifier


def _logical_install_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("/")


def _relative_install_path(value: str) -> Path:
    return Path(*_logical_install_path(value).split("/"))
