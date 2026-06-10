#!/usr/bin/env python3
"""Download required training artifacts from an rclone remote."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DATASETS = ("rsna", "spr", "vindr")
VARIANTS = ("grayscale", "rgb_multiwindow")
LABELS = "combined_mammo_recall_labels_birads_only.csv"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote", default="s2:mammo-recall-data")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/processed_datasets"))
    parser.add_argument("--variant", choices=[*VARIANTS, "both"], required=True)
    parser.add_argument("--execute", action="store_true", help="Run rclone instead of printing commands.")


def selected_variants(variant: str) -> tuple[str, ...]:
    return VARIANTS if variant == "both" else (variant,)


def rclone_copy(remote_path: str, destination: Path, *, execute: bool) -> None:
    command = ["rclone", "copy", remote_path, str(destination), "--progress"]
    if not execute:
        print(" ".join(command))
        return
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)

    remote = args.remote.rstrip("/")
    rclone_copy(
        f"{remote}/processed_datasets/{LABELS}",
        args.data_root,
        execute=args.execute,
    )
    for dataset in DATASETS:
        for variant in selected_variants(args.variant):
            rclone_copy(
                f"{remote}/processed_datasets/{dataset}/{variant}.zip",
                args.data_root / dataset,
                execute=args.execute,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
