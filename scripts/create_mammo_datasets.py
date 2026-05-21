from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from preprocess_mammo_dicom import ImageVariant, process_dicom_dir_to_png_zips


DATASET_VARIANTS = {
    "grayscale": ImageVariant.GRAYSCALE,
    "rgb_replicated": ImageVariant.RGB_REPLICATED,
    "rgb_multiwindow": ImageVariant.RGB_MULTIWINDOW,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grayscale, RGB replicated, and RGB multi-window PNG dataset zips."
    )
    parser.add_argument("dicom_dir", type=Path, help="Directory searched recursively for .dcm files.")
    parser.add_argument("output_dir", type=Path, help="Directory where the three zip files are written.")
    parser.add_argument(
        "--png-root",
        default=None,
        help="Optional top-level folder name for PNG members inside each zip.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip unreadable DICOM files and report them per dataset variant.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_zips = {
        variant: args.output_dir / f"{dataset_name}.zip"
        for dataset_name, variant in DATASET_VARIANTS.items()
    }
    zip_paths = process_dicom_dir_to_png_zips(
        args.dicom_dir,
        output_zips,
        png_root=args.png_root,
        continue_on_error=args.continue_on_error,
    )
    for variant in DATASET_VARIANTS.values():
        zip_path = zip_paths[variant]
        print(zip_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
