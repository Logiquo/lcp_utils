from __future__ import annotations

import argparse
from pathlib import Path

from lcp_utils import __version__


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
    charuco_parser.add_argument(
        "width",
        type=int,
        help="The number of chess squares horizontally.",
    )
    charuco_parser.add_argument(
        "height",
        type=int,
        help="The number of chess sqaure vertically.",
    )
    charuco_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="File to write. Defaults to current working directory.",
    )
    charuco_parser.set_defaults(func=charuco)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="create an LCP file.",
    )
    calibrate_parser.add_argument(
        "input",
        type=Path,
        help="Folder that contains calibration images.",
    )
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
    patch_parser.add_argument(
        "input",
        type=Path,
        help="Folder that includes lens profiles and patch.json",
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
    from lcp_utils.calibrate import calibrate
    from lcp_utils.parser.lcp import dump

    path: Path = args.input
    output: Path = args.output or Path.cwd() / "output.lcp"
    output.write_text(dump(calibrate(path)), encoding="utf-8")
    return 0


def patch(args: argparse.Namespace) -> int:
    from lcp_utils.patch import patch_profiles, prompt_target_directory

    target = prompt_target_directory()
    entries = patch_profiles(args.input, target)
    print(f"Patched {target / 'Index.dat'} with {len(entries)} profile(s).")
    return 0

