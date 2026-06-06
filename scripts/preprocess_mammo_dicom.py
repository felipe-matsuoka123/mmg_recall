#!/usr/bin/env python3
"""Run single-variant mammography DICOM preprocessing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mammo_preprocessing import ImageVariant, process_dicom_dir_to_png_zip


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop mammography DICOMs and package the PNGs into a zip file."
    )
    parser.add_argument(
        "dicom_dir",
        type=Path,
        help="Directory searched recursively for .dcm and .dicom files.",
    )
    parser.add_argument("output_zip", type=Path, help="Zip file to write.")
    parser.add_argument(
        "--png-root",
        default=None,
        help="Optional top-level folder name for PNG members inside the zip.",
    )
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in ImageVariant],
        default=ImageVariant.GRAYSCALE.value,
        help="PNG channel/windowing variant to write.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip unreadable DICOM files and report them after the zip is written.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=1024,
        help="Resize each output PNG to this square size in pixels. Pass 0 to keep crop size.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_zip = process_dicom_dir_to_png_zip(
        args.dicom_dir,
        args.output_zip,
        png_root=args.png_root,
        variant=args.variant,
        output_size=args.output_size or None,
        continue_on_error=args.continue_on_error,
        failure_report=args.output_zip.with_suffix(".failures.csv")
        if args.continue_on_error
        else None,
    )
    print(output_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
