#!/usr/bin/env python3
"""Create a combined DICOM-to-recall-label CSV for mammography datasets."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import chain
from pathlib import Path
from typing import Any

import pydicom
import yaml


DEFAULT_PATHS_CONFIG = Path("config/env/paths_data_preprocessing.yaml")
DEFAULT_OUTPUT = Path("processed_datasets/combined_mammo_recall_labels.csv")
PATH_CONFIG_KEYS = {
    "vindr_csv",
    "vindr_dicom_dir",
    "rsna_csv",
    "rsna_dicom_dir",
    "spr_csv",
    "spr_dicom_dir",
    "output",
}

OUTPUT_FIELDS = [
    "dataset",
    "dicom_path",
    "target",
    "label_source",
    "source_label",
    "patient_id",
    "study_id",
    "series_id",
    "image_id",
    "accession_number",
    "laterality",
    "view",
    "split",
]

NEGATIVE_BIRADS = {1, 2}
POSITIVE_BIRADS = {0, 3, 4, 5}
EXCLUDED_BIRADS = {6}
SPR_DICOM_TAGS = [
    "AccessionNumber",
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "ImageLaterality",
    "Laterality",
    "ViewPosition",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one CSV relating each DICOM file to a binary mammography recall label. "
            "BI-RADS 1/2 map to 0, BI-RADS 0/3/4/5 map to 1, and BI-RADS 6 is excluded."
        )
    )
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=DEFAULT_PATHS_CONFIG,
        help=(
            "Local YAML file with dataset paths. Defaults to config/env/paths_data_preprocessing.yaml "
            "when that file exists."
        ),
    )
    parser.add_argument("--vindr-csv", type=Path, default=None)
    parser.add_argument("--vindr-dicom-dir", type=Path, default=None)
    parser.add_argument("--rsna-csv", type=Path, default=None)
    parser.add_argument("--rsna-dicom-dir", type=Path, default=None)
    parser.add_argument("--spr-csv", type=Path, default=None)
    parser.add_argument("--spr-dicom-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--rsna-label-mode",
        choices=["recall_or_cancer", "birads_only", "cancer"],
        default="recall_or_cancer",
        help=(
            "RSNA label mapping. recall_or_cancer uses BI-RADS when present and falls back "
            "to the cancer column when BI-RADS is blank. birads_only excludes blank BI-RADS rows. "
            "cancer uses the RSNA cancer column directly."
        ),
    )
    parser.add_argument(
        "--skip-missing-dicom",
        action="store_true",
        help="Skip rows whose expected DICOM file is missing instead of failing.",
    )
    parser.add_argument(
        "--spr-match-laterality",
        action="store_true",
        help=(
            "Read SPR DICOM headers and keep only images matching the SPR Laterality column. "
            "This is more precise for unilateral positive recalls but can be slow on external disks."
        ),
    )
    return apply_path_config(parser.parse_args(argv))


def read_paths_config(path: Path) -> dict[str, Path]:
    with path.open() as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise SystemExit(f"Paths config must be a mapping: {path}")
    unknown_keys = sorted(set(values) - PATH_CONFIG_KEYS)
    if unknown_keys:
        formatted = ", ".join(unknown_keys)
        raise SystemExit(f"Unknown paths config key(s) in {path}: {formatted}")
    return {
        key: Path(value)
        for key, value in values.items()
        if value is not None and str(value).strip()
    }


def apply_path_config(args: argparse.Namespace) -> argparse.Namespace:
    config_values = {}
    if args.paths_config.exists():
        config_values = read_paths_config(args.paths_config)
    elif args.paths_config != DEFAULT_PATHS_CONFIG:
        raise SystemExit(f"Paths config does not exist: {args.paths_config}")

    for key, value in config_values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.output is None:
        args.output = DEFAULT_OUTPUT

    missing = [
        key
        for key in sorted(PATH_CONFIG_KEYS - {"output"})
        if getattr(args, key) is None
    ]
    if missing:
        formatted = ", ".join(missing)
        raise SystemExit(
            f"Missing dataset path(s): {formatted}. Create config/env/paths_data_preprocessing.yaml, "
            "pass --paths-config, or pass the "
            "paths explicitly on the CLI."
        )
    return args


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def birads_to_target(value: str) -> tuple[int | None, str]:
    match = re.search(r"\d+", value or "")
    if not match:
        return None, "missing_birads"

    birads = int(match.group())
    if birads in NEGATIVE_BIRADS:
        return 0, "birads"
    if birads in POSITIVE_BIRADS:
        return 1, "birads"
    if birads in EXCLUDED_BIRADS:
        return None, "excluded_birads_6"
    return None, f"unsupported_birads_{birads}"


def cancer_to_target(value: str) -> int | None:
    value = (value or "").strip()
    if value in {"0", "1"}:
        return int(value)
    return None


def require_file(path: Path, *, skip_missing: bool, stats: Counter[str]) -> bool:
    if path.is_file():
        return True
    stats["missing_dicom"] += 1
    if skip_missing:
        return False
    raise FileNotFoundError(path)


def build_vindr_rows(args: argparse.Namespace, stats: Counter[str]) -> Iterable[dict[str, str]]:
    for row in read_csv_rows(args.vindr_csv):
        target, label_source = birads_to_target(row.get("breast_birads", ""))
        if target is None:
            stats[f"vindr_skipped_{label_source}"] += 1
            continue

        dicom_path = args.vindr_dicom_dir / row["study_id"] / f"{row['image_id']}.dicom"
        if not require_file(dicom_path, skip_missing=args.skip_missing_dicom, stats=stats):
            continue

        yield {
            "dataset": "vindr",
            "dicom_path": str(dicom_path),
            "target": str(target),
            "label_source": label_source,
            "source_label": row.get("breast_birads", ""),
            "patient_id": "",
            "study_id": row.get("study_id", ""),
            "series_id": row.get("series_id", ""),
            "image_id": row.get("image_id", ""),
            "accession_number": "",
            "laterality": row.get("laterality", ""),
            "view": row.get("view_position", ""),
            "split": row.get("split", ""),
        }


def rsna_target(row: dict[str, str], mode: str) -> tuple[int | None, str]:
    if mode == "cancer":
        target = cancer_to_target(row.get("cancer", ""))
        return target, "cancer" if target is not None else "missing_cancer"

    birads_target, birads_source = birads_to_target(row.get("BIRADS", ""))
    if birads_target is not None or mode == "birads_only" or birads_source != "missing_birads":
        return birads_target, birads_source

    target = cancer_to_target(row.get("cancer", ""))
    return target, "cancer_fallback" if target is not None else "missing_cancer"


def build_rsna_rows(args: argparse.Namespace, stats: Counter[str]) -> Iterable[dict[str, str]]:
    for row in read_csv_rows(args.rsna_csv):
        target, label_source = rsna_target(row, args.rsna_label_mode)
        if target is None:
            stats[f"rsna_skipped_{label_source}"] += 1
            continue

        dicom_path = args.rsna_dicom_dir / row["patient_id"] / f"{row['image_id']}.dcm"
        if not require_file(dicom_path, skip_missing=args.skip_missing_dicom, stats=stats):
            continue

        yield {
            "dataset": "rsna",
            "dicom_path": str(dicom_path),
            "target": str(target),
            "label_source": label_source,
            "source_label": row.get("BIRADS") or row.get("cancer", ""),
            "patient_id": row.get("patient_id", ""),
            "study_id": "",
            "series_id": "",
            "image_id": row.get("image_id", ""),
            "accession_number": "",
            "laterality": row.get("laterality", ""),
            "view": row.get("view", ""),
            "split": "train",
        }


def dicom_header(path: Path) -> dict[str, str]:
    ds = pydicom.dcmread(
        path,
        stop_before_pixels=True,
        force=True,
        specific_tags=SPR_DICOM_TAGS,
    )
    return {
        "accession_number": str(getattr(ds, "AccessionNumber", "")),
        "patient_id": str(getattr(ds, "PatientID", "")),
        "study_id": str(getattr(ds, "StudyInstanceUID", "")),
        "series_id": str(getattr(ds, "SeriesInstanceUID", "")),
        "image_id": str(getattr(ds, "SOPInstanceUID", "")),
        "laterality": str(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "")),
        "view": str(getattr(ds, "ViewPosition", "")),
    }


def spr_dicom_paths(root: Path, accession_number: str) -> list[Path]:
    accession_dir = root / accession_number
    if not accession_dir.is_dir():
        return []
    return sorted(
        path
        for path in accession_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".dcm", ".dicom"}
    )


def spr_laterality_matches(label_laterality: str, dicom_laterality: str) -> bool:
    label_laterality = (label_laterality or "").strip().upper()
    dicom_laterality = (dicom_laterality or "").strip().upper()
    if label_laterality == "N":
        return True
    if label_laterality == "B":
        return dicom_laterality in {"L", "R"}
    return label_laterality == dicom_laterality


def build_spr_rows(args: argparse.Namespace, stats: Counter[str]) -> Iterable[dict[str, str]]:
    for row in read_csv_rows(args.spr_csv):
        target = cancer_to_target(row.get("target", ""))
        if target is None:
            stats["spr_skipped_missing_target"] += 1
            continue

        accession_number = row.get("AccessionNumber", "").strip()
        dicom_paths = spr_dicom_paths(args.spr_dicom_dir, accession_number)
        if not dicom_paths:
            stats["spr_missing_accession_dir_or_dicoms"] += 1
            if args.skip_missing_dicom:
                continue
            raise FileNotFoundError(args.spr_dicom_dir / accession_number)

        matched = 0
        for dicom_path in dicom_paths:
            if args.spr_match_laterality:
                header = dicom_header(dicom_path)
                if not spr_laterality_matches(row.get("Laterality", ""), header["laterality"]):
                    stats["spr_skipped_laterality_mismatch"] += 1
                    continue
            else:
                header = {
                    "accession_number": accession_number,
                    "patient_id": row.get("PatientID", ""),
                    "study_id": "",
                    "series_id": "",
                    "image_id": dicom_path.stem,
                    "laterality": row.get("Laterality", ""),
                    "view": "",
                }

            matched += 1
            yield {
                "dataset": "spr",
                "dicom_path": str(dicom_path),
                "target": str(target),
                "label_source": "target",
                "source_label": row.get("target", ""),
                "patient_id": row.get("PatientID", "") or header["patient_id"],
                "study_id": header["study_id"],
                "series_id": header["series_id"],
                "image_id": header["image_id"],
                "accession_number": accession_number or header["accession_number"],
                "laterality": header["laterality"],
                "view": header["view"],
                "split": "train",
            }

        if matched == 0:
            stats["spr_rows_without_matching_laterality"] += 1


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> Counter[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            stats[f"{row['dataset']}_rows"] += 1
            stats[f"{row['dataset']}_target_{row['target']}"] += 1
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build_stats: Counter[str] = Counter()

    rows = chain(
        build_vindr_rows(args, build_stats),
        build_rsna_rows(args, build_stats),
        build_spr_rows(args, build_stats),
    )
    write_stats = write_rows(args.output, rows)

    total_rows = sum(value for key, value in write_stats.items() if key.endswith("_rows"))
    print(f"Wrote {total_rows} rows to {args.output}")
    for key in sorted(write_stats):
        print(f"{key}: {write_stats[key]}")
    for key in sorted(build_stats):
        print(f"{key}: {build_stats[key]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
