#!/usr/bin/env python3
"""Create a processed SPR subset containing only competition test accessions."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

PNG_EXTENSIONS = {".png"}
VARIANT_DIRS = ("grayscale", "rgb_multiwindow")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-index", type=Path, default=Path("processed_spr/test_set_index.csv"))
    parser.add_argument("--processed-root", type=Path, default=Path("processed_spr"))
    parser.add_argument("--output-root", type=Path, default=Path("processed_spr/test_subset"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_DIRS))
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-all-accessions", action="store_true")
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


def png_paths(accession_dir: Path) -> list[Path]:
    if not accession_dir.is_dir():
        return []
    return sorted(
        path
        for path in accession_dir.iterdir()
        if path.is_file() and path.suffix.lower() in PNG_EXTENSIONS
    )


def materialize_file(source: Path, destination: Path, mode: str, overwrite: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return "existing"
        destination.unlink()

    if mode == "symlink":
        os.symlink(source.resolve(), destination)
        return "symlinked"
    if mode == "hardlink":
        os.link(source, destination)
        return "hardlinked"
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copied"
    raise ValueError(f"Unsupported mode={mode!r}")


def write_subset_index(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["AccessionNumber", "variant", "source_path", "subset_path", "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_subset(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    accessions = read_test_accessions(args.test_index)
    rows: list[dict[str, str]] = []
    missing_by_variant: dict[str, list[str]] = {}

    for variant in args.variants:
        source_variant_dir = args.processed_root / variant
        destination_variant_dir = args.output_root / variant
        missing = []
        for accession in accessions:
            source_accession_dir = source_variant_dir / accession
            sources = png_paths(source_accession_dir)
            if not sources:
                missing.append(accession)
                continue
            for source in sources:
                destination = destination_variant_dir / accession / source.name
                status = materialize_file(source, destination, args.mode, args.overwrite)
                rows.append(
                    {
                        "AccessionNumber": accession,
                        "variant": variant,
                        "source_path": str(source),
                        "subset_path": str(destination),
                        "status": status,
                    }
                )
        if missing:
            missing_by_variant[variant] = missing

    return rows, missing_by_variant


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, missing_by_variant = build_subset(args)
    write_subset_index(args.output_root / "subset_index.csv", rows)

    for variant in args.variants:
        count = sum(1 for row in rows if row["variant"] == variant)
        missing = len(missing_by_variant.get(variant, []))
        print(f"{variant}: {count} PNGs, {missing} missing accessions")

    if missing_by_variant and args.require_all_accessions:
        details = "; ".join(
            f"{variant}: {len(accessions)} missing, examples={accessions[:5]}"
            for variant, accessions in missing_by_variant.items()
        )
        raise SystemExit(details)

    print(f"wrote subset index: {args.output_root / 'subset_index.csv'}")
    print(f"ready for inference with --processed-root {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
