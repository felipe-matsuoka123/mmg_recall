#!/usr/bin/env python3
"""Calculate ROI intensity stats from preprocessed mammography DICOM samples."""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mammo_preprocessing import (
    DICOM_EXTENSIONS,
    ImageVariant,
    create_breast_mask_u8,
    create_u8_crop_and_metadata,
    np,
    u16_to_mask_u8,
)

warnings.filterwarnings("ignore", module="pydicom.valuerep")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path


DATASETS = (
    DatasetConfig(
        "SPR",
        Path("/media/felipe/KINGSTON/datasets/SPR_Mammo_Recall"),
    ),
    DatasetConfig(
        "VinDrMammo",
        Path(
            "/media/felipe/KINGSTON/datasets/VinDr_Mammo/"
            "vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-"
            "detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images"
        ),
    ),
    DatasetConfig(
        "RSNA",
        Path("/media/felipe/KINGSTON/datasets/rsna_breast/train_images"),
    ),
)


CSV_FIELDS = ("manufacturer", "dataset", "roi_median", "roi_std")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate ROI median/std from preprocessed mammography samples."
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=1000,
        help="Number of DICOM files to attempt per dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/roi_stats.csv"),
        help="CSV path for manufacturer,dataset,roi_median,roi_std rows.",
    )
    return parser.parse_args(argv)


def first_dicom_files(dataset_path: Path, limit: int) -> list[Path]:
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"DICOM directory does not exist: {dataset_path}")

    dicom_paths: list[Path] = []
    for root, dirs, files in os.walk(dataset_path):
        dirs.sort()
        for filename in sorted(files):
            path = Path(root) / filename
            if path.suffix.lower() not in DICOM_EXTENSIONS:
                continue
            dicom_paths.append(path)
            if len(dicom_paths) >= limit:
                return dicom_paths
    return dicom_paths


def roi_stats_u16(image_u16: np.ndarray) -> tuple[float, float]:
    mask_u8 = create_breast_mask_u8(u16_to_mask_u8(image_u16))
    roi = image_u16[mask_u8 > 0].astype(np.float32)
    if roi.size < 100:
        roi = image_u16.astype(np.float32).ravel()
    return float(np.median(roi)), float(np.std(roi))


def calculate_dataset_rows(
    dataset: DatasetConfig,
    sample_count: int,
) -> tuple[list[dict[str, str]], list[tuple[Path, str]]]:
    rows: list[dict[str, str]] = []
    failures: list[tuple[Path, str]] = []
    dicom_files = first_dicom_files(dataset.path, sample_count)

    for index, dicom_path in enumerate(dicom_files, start=1):
        if index == 1 or index % 100 == 0:
            print(f"{dataset.name}: processing {index}/{len(dicom_files)}", flush=True)
        try:
            image_u16, metadata = create_u8_crop_and_metadata(
                dicom_path,
                ImageVariant.GRAYSCALE,
                output_size=None,
            )
            roi_median, roi_std = roi_stats_u16(image_u16)
        except Exception as exc:
            failures.append((dicom_path, str(exc)))
            continue

        rows.append(
            {
                "manufacturer": str(metadata.get("manufacturer") or "UNKNOWN"),
                "dataset": dataset.name,
                "roi_median": f"{roi_median:.6f}",
                "roi_std": f"{roi_std:.6f}",
            }
        )

    return rows, failures


def write_rows(output_path: Path, rows: Sequence[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples_per_dataset <= 0:
        raise ValueError("--samples-per-dataset must be positive")

    all_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        print(f"{dataset.name}: taking first {args.samples_per_dataset} DICOMs", flush=True)
        rows, failures = calculate_dataset_rows(dataset, args.samples_per_dataset)
        all_rows.extend(rows)
        print(f"{dataset.name}: wrote {len(rows)} stats rows", flush=True)
        if failures:
            print(f"{dataset.name}: skipped {len(failures)} unreadable files", flush=True)

    write_rows(args.output, all_rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
