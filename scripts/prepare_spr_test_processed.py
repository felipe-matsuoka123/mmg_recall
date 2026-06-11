#!/usr/bin/env python3
"""Preprocess only SPR competition test accessions into a small PNG subset."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mammo_preprocessing import (  # noqa: E402
    DICOM_EXTENSIONS,
    ImageVariant,
    METADATA_FIELDS,
    preprocess_dicom_payloads,
    write_failures_csv,
)


VARIANT_FOLDERS = {
    "grayscale": ImageVariant.GRAYSCALE,
    "rgb_multiwindow": ImageVariant.RGB_MULTIWINDOW,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-index",
        type=Path,
        default=Path("processed_spr/test_set_index.csv"),
        help="CSV with an AccessionNumber column.",
    )
    parser.add_argument(
        "--spr-dicom-dir",
        type=Path,
        required=True,
        help="Root folder containing one subdirectory per SPR accession.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed_spr/test_processed"),
        help="Output root for grayscale/ and rgb_multiwindow/ folders.",
    )
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-all-accessions", action="store_true")
    parser.add_argument(
        "--failure-report",
        type=Path,
        default=None,
        help="Defaults to output_root/preprocessing_failures.csv.",
    )
    return parser.parse_args(argv)


def read_test_accessions(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "AccessionNumber" not in reader.fieldnames:
            raise ValueError(f"Expected an AccessionNumber column in {path}, got {reader.fieldnames}")
        accessions = []
        seen = set()
        for row in reader:
            accession = row.get("AccessionNumber", "").strip()
            if accession and accession not in seen:
                accessions.append(accession)
                seen.add(accession)
    if not accessions:
        raise ValueError(f"No accessions found in {path}")
    return accessions


def spr_dicom_paths(root: Path, accession_number: str) -> list[Path]:
    accession_dir = root / accession_number
    if not accession_dir.is_dir():
        return []
    return sorted(
        path
        for path in accession_dir.iterdir()
        if path.is_file() and path.suffix.lower() in DICOM_EXTENSIONS
    )


def output_png_path(output_root: Path, folder: str, accession: str, dicom_path: Path) -> Path:
    return output_root / folder / accession / f"{dicom_path.stem}.png"


def preprocess_one(
    *,
    accession: str,
    dicom_path: Path,
    output_root: Path,
    output_size: int | None,
    overwrite: bool,
) -> tuple[list[dict[str, str]], tuple[Path, str] | None]:
    try:
        output_paths = {
            folder: output_png_path(output_root, folder, accession, dicom_path)
            for folder in VARIANT_FOLDERS
        }
        if not overwrite and all(path.is_file() for path in output_paths.values()):
            return (
                [
                    {
                        "accession_number": accession,
                        "dicom_path": str(dicom_path),
                        "variant": folder,
                        "processed_path": str(path.relative_to(output_root)),
                        "status": "existing",
                    }
                    for folder, path in output_paths.items()
                ],
                None,
            )

        payloads, metadata = preprocess_dicom_payloads(
            dicom_path,
            variants=tuple(VARIANT_FOLDERS.values()),
            output_size=output_size,
        )
        rows = []
        for folder, variant in VARIANT_FOLDERS.items():
            path = output_paths[folder]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payloads[variant.value])
            row = {
                "accession_number": accession,
                "dicom_path": str(dicom_path),
                "variant": folder,
                "status": "processed",
            }
            for field in METADATA_FIELDS:
                row[field] = metadata.get(field, "")
            row["processed_path"] = str(path.relative_to(output_root))
            rows.append(row)
        return rows, None
    except Exception as exc:  # noqa: BLE001
        return [], (dicom_path, str(exc))


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "accession_number",
        "variant",
        "dicom_path",
        "processed_path",
        "status",
        *[field for field in METADATA_FIELDS if field != "processed_path"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_tasks(
    *,
    accessions: Sequence[str],
    spr_dicom_dir: Path,
    output_root: Path,
    output_size: int | None,
    overwrite: bool,
) -> tuple[list[dict[str, object]], list[str]]:
    tasks = []
    missing_accessions = []
    for accession in accessions:
        dicom_paths = spr_dicom_paths(spr_dicom_dir, accession)
        if not dicom_paths:
            missing_accessions.append(accession)
            continue
        for dicom_path in dicom_paths:
            tasks.append(
                {
                    "accession": accession,
                    "dicom_path": dicom_path,
                    "output_root": output_root,
                    "output_size": output_size,
                    "overwrite": overwrite,
                }
            )
    return tasks, missing_accessions


def run_tasks(tasks: list[dict[str, object]], workers: int) -> tuple[list[dict[str, str]], list[tuple[Path, str]]]:
    if workers <= 0:
        raise ValueError("--workers must be positive")

    metadata_rows: list[dict[str, str]] = []
    failures: list[tuple[Path, str]] = []
    if workers == 1:
        iterator = (preprocess_one(**task) for task in tasks)
        for rows, failure in tqdm(iterator, total=len(tasks), desc="Preprocessing SPR test", unit="image"):
            metadata_rows.extend(rows)
            if failure is not None:
                failures.append(failure)
        return metadata_rows, failures

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(preprocess_one, **task) for task in tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Preprocessing SPR test ({workers} workers)",
            unit="image",
        ):
            rows, failure = future.result()
            metadata_rows.extend(rows)
            if failure is not None:
                failures.append(failure)
    return metadata_rows, failures


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_size = args.output_size or None
    failure_report = args.failure_report or args.output_root / "preprocessing_failures.csv"

    accessions = read_test_accessions(args.test_index)
    tasks, missing_accessions = build_tasks(
        accessions=accessions,
        spr_dicom_dir=args.spr_dicom_dir,
        output_root=args.output_root,
        output_size=output_size,
        overwrite=args.overwrite,
    )
    if missing_accessions and args.require_all_accessions:
        examples = ", ".join(missing_accessions[:5])
        raise SystemExit(
            f"Missing DICOM folders for {len(missing_accessions)} accessions under "
            f"{args.spr_dicom_dir}. Examples: {examples}"
        )
    if not tasks:
        raise SystemExit(f"No DICOMs found for test accessions under {args.spr_dicom_dir}")

    metadata_rows, failures = run_tasks(tasks, args.workers)
    metadata_rows.sort(key=lambda row: (row["accession_number"], row["variant"], row["processed_path"]))
    write_metadata(args.output_root / "metadata.csv", metadata_rows)
    if failures:
        write_failures_csv(failure_report, failures)

    successful_accessions = {row["accession_number"] for row in metadata_rows}
    print(f"read {len(accessions)} test accessions from {args.test_index}")
    print(f"found DICOMs for {len(accessions) - len(missing_accessions)}/{len(accessions)} accessions")
    print(f"wrote {len(metadata_rows)} PNG metadata rows under {args.output_root}")
    if missing_accessions:
        print(f"missing accession folders: {len(missing_accessions)}")
    if failures:
        print(f"failed DICOMs: {len(failures)}; wrote {failure_report}")
    print(f"ready for inference with --processed-root {args.output_root}")
    return 0 if successful_accessions else 1


if __name__ == "__main__":
    raise SystemExit(main())
