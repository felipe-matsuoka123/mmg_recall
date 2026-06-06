#!/usr/bin/env python3
"""Preview the first preprocessed images from the configured mammography datasets."""

from __future__ import annotations

import argparse
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", module="pydicom.valuerep")

import matplotlib.pyplot as plt

from mammo_preprocessing import (
    DICOM_EXTENSIONS,
    ImageVariant,
    create_u8_crop_and_metadata,
    UINT16_MAX,
)


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


@dataclass(frozen=True)
class PreviewSample:
    dataset_name: str
    dicom_path: Path
    image_u8: object


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a grid of the first preprocessed mammography DICOM samples."
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=5,
        help="Number of successfully preprocessed images per dataset.",
    )
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in ImageVariant],
        default=ImageVariant.GRAYSCALE.value,
        help="Preprocessed image variant to display.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=768,
        help="Square preview size in pixels. Pass 0 to keep crop size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("preview/preprocessing_grid.png"),
        help="Path where the rendered grid PNG is written.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Only save the grid image; do not open a matplotlib window.",
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


def sample_preprocessed_images(
    dataset: DatasetConfig,
    *,
    sample_count: int,
    variant: ImageVariant,
    output_size: int | None,
) -> tuple[list[PreviewSample], list[tuple[Path, str]]]:
    dicom_files = first_dicom_files(dataset.path, sample_count)

    samples: list[PreviewSample] = []
    failures: list[tuple[Path, str]] = []
    for dicom_path in dicom_files:
        try:
            image_u8, metadata = create_u8_crop_and_metadata(
                dicom_path,
                variant,
                output_size=output_size,
            )
        except Exception as exc:
            failures.append((dicom_path, str(exc)))
            continue

        samples.append(
            PreviewSample(
                dataset_name=dataset.name,
                dicom_path=dicom_path,
                image_u8=image_u8,
            )
        )

    return samples, failures


def render_grid(
    samples_by_dataset: dict[str, list[PreviewSample]],
    *,
    samples_per_dataset: int,
    output_path: Path,
    show: bool,
) -> None:
    rows = len(DATASETS)
    cols = samples_per_dataset
    fig_width = max(4.0 * cols, 8.0)
    fig_height = 4.6 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    for row_index, dataset in enumerate(DATASETS):
        samples = samples_by_dataset.get(dataset.name, [])
        for col_index in range(cols):
            ax = axes[row_index][col_index]
            ax.axis("off")
            if col_index == 0:
                ax.set_ylabel(dataset.name, fontsize=14, fontweight="bold")
            if col_index >= len(samples):
                ax.set_title("not available", fontsize=9)
                continue

            sample = samples[col_index]
            if sample.image_u8.ndim == 2:
                ax.imshow(sample.image_u8, cmap="jet", vmin=0, vmax=UINT16_MAX)
            else:
                ax.imshow(sample.image_u8.astype("float32") / UINT16_MAX)

    fig.suptitle("First Preprocessed Mammography Samples", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(output_path)
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.samples_per_dataset <= 0:
        raise ValueError("--samples-per-dataset must be positive")

    variant = ImageVariant(args.variant)
    output_size = args.output_size or None

    samples_by_dataset: dict[str, list[PreviewSample]] = {}
    for dataset in DATASETS:
        print(f"{dataset.name}: taking first {args.samples_per_dataset} DICOMs", flush=True)
        samples, failures = sample_preprocessed_images(
            dataset,
            sample_count=args.samples_per_dataset,
            variant=variant,
            output_size=output_size,
        )
        samples_by_dataset[dataset.name] = samples
        print(
            f"{dataset.name}: {len(samples)}/{args.samples_per_dataset} previews "
            f"from {dataset.path}"
        )
        if failures:
            print(f"{dataset.name}: skipped {len(failures)} unreadable files")

    render_grid(
        samples_by_dataset,
        samples_per_dataset=args.samples_per_dataset,
        output_path=args.output,
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
