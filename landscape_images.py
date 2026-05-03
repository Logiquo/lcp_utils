from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

import rawpy
from PIL import Image

RAW_SUFFIXES = {".arw", ".dng", ".nef", ".cr3"}
RASTER_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
IMAGE_SUFFIXES = RAW_SUFFIXES | RASTER_SUFFIXES
TIFF_HEADER = {b"II*\x00", b"MM\x00*"}
ORIENTATION_TAG = 274
LANDSCAPE_ORIENTATION = 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rotate calibration images to landscape orientation."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak copies before modifying files.",
    )
    args = parser.parse_args()

    image_paths = _image_paths(args.path)
    if not image_paths:
        raise FileNotFoundError(f"no images found in {args.path}")

    for image_path in image_paths:
        size = _image_size(image_path)
        if size[0] >= size[1]:
            print(f"skip {image_path} ({size[0]}x{size[1]})")
            continue

        if not args.no_backup:
            _backup(image_path)

        if image_path.suffix.lower() in RAW_SUFFIXES:
            _set_raw_orientation(image_path, LANDSCAPE_ORIENTATION)
        else:
            _rotate_raster(image_path)

        new_size = _image_size(image_path)
        print(
            f"rotated {image_path} "
            f"({size[0]}x{size[1]} -> {new_size[0]}x{new_size[1]})"
        )

    return 0


def _image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in IMAGE_SUFFIXES
        and not item.name.endswith(".bak")
    )


def _image_size(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in RAW_SUFFIXES:
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        return int(rgb.shape[1]), int(rgb.shape[0])

    with Image.open(path) as image:
        return image.size


def _backup(path: Path) -> None:
    backup_path = path.with_name(f"{path.name}.bak")
    if backup_path.exists():
        return
    shutil.copy2(path, backup_path)


def _set_raw_orientation(path: Path, value: int) -> None:
    data = bytearray(path.read_bytes())
    order, entries = _orientation_value_offsets(data)
    if not entries:
        raise ValueError(f"{path} has no TIFF orientation tags")

    for offset in entries:
        struct.pack_into(order + "H", data, offset, value)
    path.write_bytes(data)


def _orientation_value_offsets(data: bytearray) -> tuple[str, list[int]]:
    if len(data) < 8 or bytes(data[:4]) not in TIFF_HEADER:
        raise ValueError("RAW file does not look like a TIFF-based file")

    order = "<" if data[:2] == b"II" else ">"
    first_ifd = struct.unpack_from(order + "I", data, 4)[0]
    seen: set[int] = set()
    offsets: list[int] = []

    def read_ifd(offset: int) -> None:
        if offset in seen or offset <= 0 or offset + 2 > len(data):
            return
        seen.add(offset)

        count = struct.unpack_from(order + "H", data, offset)[0]
        entries_end = offset + 2 + count * 12
        if entries_end + 4 > len(data):
            return

        for index in range(count):
            entry = offset + 2 + index * 12
            tag, kind, item_count, value = struct.unpack_from(
                order + "HHII",
                data,
                entry,
            )
            if tag == ORIENTATION_TAG and kind == 3 and item_count == 1:
                offsets.append(entry + 8)
            if tag in {330, 34665, 34853}:
                read_ifd(value)

        next_offset = struct.unpack_from(order + "I", data, entries_end)[0]
        if next_offset:
            read_ifd(next_offset)

    read_ifd(first_ifd)
    return order, offsets


def _rotate_raster(path: Path) -> None:
    with Image.open(path) as image:
        image.rotate(90, expand=True).save(path)


if __name__ == "__main__":
    raise SystemExit(main())
