from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from mammo_preprocessing import (
    ImageVariant,
    process_dicom_dir_to_png_zips,
    process_dicom_preview_samples,
)


DATASET_VARIANTS = {
    "grayscale": ImageVariant.GRAYSCALE,
    "rgb_multiwindow": ImageVariant.RGB_MULTIWINDOW,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grayscale and RGB multi-window PNG dataset zips."
    )
    parser.add_argument(
        "dicom_dir",
        type=Path,
        help="Directory searched recursively for .dcm and .dicom files.",
    )
    parser.add_argument("output_dir", type=Path, help="Directory where the dataset zip files are written.")
    parser.add_argument(
        "--png-root",
        default=None,
        help="Optional top-level folder name for PNG members inside each zip.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip unreadable DICOM files. This is the default for dataset creation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on the first unreadable DICOM instead of skipping it.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=1024,
        help="Resize each output PNG to this square size in pixels. Pass 0 to keep crop size.",
    )
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=3,
        help="Number of DICOMs to render per dataset when writing previews.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Directory for preview PNGs. Defaults to output_dir/previews.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Write preview PNGs and exit before creating dataset zips.",
    )
    parser.add_argument(
        "--preview-diagnostics",
        action="store_true",
        help="Also write full-window, mask, crop-overlay, and crop-window PNGs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel DICOM preprocessing workers for full dataset creation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_size = args.output_size or None
    failure_report = args.output_dir / "preprocessing_failures.csv"

    if args.preview_only:
        preview_dir = args.preview_dir or args.output_dir / "previews"
        sample_paths = process_dicom_preview_samples(
            args.dicom_dir,
            preview_dir,
            DATASET_VARIANTS,
            sample_count=args.preview_samples,
            output_size=output_size,
            write_diagnostics=args.preview_diagnostics,
            continue_on_error=args.continue_on_error or not args.strict,
            failure_report=failure_report,
        )
        for dataset_name, paths in sample_paths.items():
            for path in paths:
                print(f"{dataset_name}: {path}")
        return 0

    output_zips = {
        variant: args.output_dir / f"{dataset_name}.zip"
        for dataset_name, variant in DATASET_VARIANTS.items()
    }
    zip_paths = process_dicom_dir_to_png_zips(
        args.dicom_dir,
        output_zips,
        png_root=args.png_root,
        output_size=output_size,
        workers=args.workers,
        continue_on_error=args.continue_on_error or not args.strict,
        failure_report=failure_report,
    )
    for variant in DATASET_VARIANTS.values():
        zip_path = zip_paths[variant]
        print(zip_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
