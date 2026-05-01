from __future__ import annotations

import argparse
from pathlib import Path
from lcp_utils import __version__
import json


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcp_utils",
        description="Utilities for Lens Correction Profile (LCP) files.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    charuco_parser = subparsers.add_parser(
        "charuco",
        help="Create a ChArUco board for lens correction",
    )
    charuco_parser.add_argument("width", type=int, help="The number of chess squares horizontally.")
    charuco_parser.add_argument("height", type=int, help="The number of chess sqaure vertically.")
    charuco_parser.add_argument("-o", "--output", type=Path, help="File to write. Defaults to current working directory.")
    charuco_parser.set_defaults(func=charuco)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="create an LCP file.",
    )
    calibrate_parser.add_argument("input", type=Path, help="Folder that contains calibration images.")
    calibrate_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="File to write. Defaults to output.lcp",
    )
    calibrate_parser.set_defaults(func=calibrate)
    
    patch_parser = subparsers.add_parser(
        "patch",
        help="patch a index.dat file on desktop version",
    )
    patch_parser.add_argument("input", type=Path, help="Folder that includes Index.dat")
    patch_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="File to write. Defaults to output.dat",
    )
    patch_parser.set_defaults(func=patch)

    return parser


def charuco(args: argparse.Namespace) -> int:
    from lcp_utils.perspective.charuco import new_board
    width: int = args.width
    height: int = args.height
    output: Path = args.output or Path.cwd() / f"charuco_{width}x{height}.png"
    image = new_board(width, height)
    image.save(output)
    return 0


def calibrate(args: argparse.Namespace) -> int:
    from lcp_utils.parser.lcp import dump
    from lcp_utils.calibrate import calibrate
    path: Path = args.input
    output: Path = args.output or Path.cwd() / f"output.lcp"
    output.write_text(dump(calibrate(path)), encoding='utf-8')
    return 0

def patch(args: argparse.Namespace) -> int:
    from lcp_utils.parser import index, lcp
    from lcp_utils.parser.index import IndexEntry
    def new_identifier(digest: str, used: set[int]) -> int:
        identifier = int(digest[:16], 16)
        while identifier in used:
            identifier = (identifier + 1) & 0xFFFFFFFFFFFFFFFF
        return identifier
    
    path: Path = args.input
    output: Path = args.output or Path.cwd() / f"output.dat"
    
    index_file = index.load((path / "Index.dat").read_bytes())
    used_identifiers = {entry.identifier for entry in index_file.entries}
    
    with open(path / "metadata.json") as f:
        profiles: dict[str, str] = json.load(f)
    
    FAKE_ROOT = "/Library/Application Support/Adobe/CameraRaw/LensProfiles/1.0"
    for profile_file, fake_path in profiles.items():
        data = (path / profile_file).read_bytes()
        profile = lcp.load(data.decode("utf-8"))[0]
        digest = index.file_digest(data)
        
        entry = IndexEntry(
            path=f"{FAKE_ROOT}{fake_path}",
            identifier=new_identifier(digest, used_identifiers),
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
            metadata_distort="True" if profile.prefer_metadata_distort else "False",
            nn_distort="False",
            profile_name=profile.profile_name,
            unique_model=str(profile.unique_camera_model or ""),
        )
        used_identifiers.add(entry.identifier)
        
        index_file.entries.append(entry)
        
    output.write_bytes(index.dump(index_file))
    return 0

