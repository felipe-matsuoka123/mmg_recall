#!/usr/bin/env python3
"""Compare ROI stats before and after ROI normalization."""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mammo_preprocessing import (
    DICOM_EXTENSIONS,
    UINT16_MAX,
    create_breast_mask_u8,
    normalize_breast_roi_u16,
    np,
    read_crop_and_metadata,
    robust_window_u16,
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


DETAIL_FIELDS = (
    "dataset",
    "manufacturer",
    "stage",
    "roi_median",
    "roi_iqr",
    "roi_std",
    "dense_fraction",
)
SUMMARY_FIELDS = (
    "dataset",
    "manufacturer",
    "stage",
    "n",
    "roi_median_mean",
    "roi_median_std",
    "roi_iqr_mean",
    "roi_iqr_std",
    "roi_std_mean",
    "roi_std_std",
    "dense_fraction_mean",
    "dense_fraction_std",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ROI stats before and after ROI normalization."
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=100,
        help="Number of DICOM files to attempt per dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/roi_normalization_comparison.csv"),
        help="Per-image stats CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("processed/roi_normalization_comparison_summary.csv"),
        help="Grouped summary CSV.",
    )
    parser.add_argument(
        "--dense-threshold",
        type=float,
        default=0.75,
        help="Dense fraction threshold as a fraction of uint16 max intensity.",
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


def roi_pixels_u16(image_u16: np.ndarray) -> np.ndarray:
    mask_u8 = create_breast_mask_u8(u16_to_mask_u8(image_u16))
    roi = image_u16[mask_u8 > 0].astype(np.float32)
    if roi.size < 100:
        roi = image_u16.astype(np.float32).ravel()
    return roi


def roi_stats_row(
    *,
    dataset: str,
    manufacturer: str,
    stage: str,
    image_u16: np.ndarray,
    dense_threshold: float,
) -> dict[str, str]:
    roi = roi_pixels_u16(image_u16)
    p25, median, p75 = np.percentile(roi, [25, 50, 75])
    dense_fraction = float(np.mean(roi >= dense_threshold))
    return {
        "dataset": dataset,
        "manufacturer": manufacturer,
        "stage": stage,
        "roi_median": f"{float(median):.6f}",
        "roi_iqr": f"{float(p75 - p25):.6f}",
        "roi_std": f"{float(np.std(roi)):.6f}",
        "dense_fraction": f"{dense_fraction:.6f}",
    }


def calculate_dataset_rows(
    dataset: DatasetConfig,
    *,
    sample_count: int,
    dense_threshold: float,
) -> tuple[list[dict[str, str]], list[tuple[Path, str]]]:
    rows: list[dict[str, str]] = []
    failures: list[tuple[Path, str]] = []
    dicom_files = first_dicom_files(dataset.path, sample_count)

    for index, dicom_path in enumerate(dicom_files, start=1):
        if index == 1 or index % 25 == 0:
            print(f"{dataset.name}: processing {index}/{len(dicom_files)}", flush=True)
        try:
            cropped, metadata = read_crop_and_metadata(dicom_path)
            manufacturer = str(metadata.get("manufacturer") or "UNKNOWN")
            windowed_u16 = robust_window_u16(cropped, (1.0, 99.0))
            normalized_u16 = normalize_breast_roi_u16(windowed_u16)
        except Exception as exc:
            failures.append((dicom_path, str(exc)))
            continue

        rows.append(
            roi_stats_row(
                dataset=dataset.name,
                manufacturer=manufacturer,
                stage="windowed",
                image_u16=windowed_u16,
                dense_threshold=dense_threshold,
            )
        )
        rows.append(
            roi_stats_row(
                dataset=dataset.name,
                manufacturer=manufacturer,
                stage="roi_normalized",
                image_u16=normalized_u16,
                dense_threshold=dense_threshold,
            )
        )

    return rows, failures


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["manufacturer"], row["stage"])].append(row)

    summary_rows = []
    metric_names = ("roi_median", "roi_iqr", "roi_std", "dense_fraction")
    for (dataset, manufacturer, stage), group_rows in sorted(grouped.items()):
        summary = {
            "dataset": dataset,
            "manufacturer": manufacturer,
            "stage": stage,
            "n": str(len(group_rows)),
        }
        for metric in metric_names:
            values = np.array([float(row[metric]) for row in group_rows], dtype=np.float32)
            summary[f"{metric}_mean"] = f"{float(np.mean(values)):.6f}"
            summary[f"{metric}_std"] = f"{float(np.std(values)):.6f}"
        summary_rows.append(summary)
    return summary_rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples_per_dataset <= 0:
        raise ValueError("--samples-per-dataset must be positive")
    if not 0.0 <= args.dense_threshold <= 1.0:
        raise ValueError("--dense-threshold must be between 0 and 1")

    dense_threshold = args.dense_threshold * UINT16_MAX
    all_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        print(f"{dataset.name}: taking first {args.samples_per_dataset} DICOMs", flush=True)
        rows, failures = calculate_dataset_rows(
            dataset,
            sample_count=args.samples_per_dataset,
            dense_threshold=dense_threshold,
        )
        all_rows.extend(rows)
        print(f"{dataset.name}: wrote {len(rows)} comparison rows", flush=True)
        if failures:
            print(f"{dataset.name}: skipped {len(failures)} unreadable files", flush=True)

    write_csv(args.output, DETAIL_FIELDS, all_rows)
    write_csv(args.summary_output, SUMMARY_FIELDS, summarize_rows(all_rows))
    print(args.output)
    print(args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
