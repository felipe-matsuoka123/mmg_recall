#!/usr/bin/env python3
"""Check whether a training config is ready to run on the current machine."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.train_classifier import load_training_rows, parse_args as parse_train_args  # noqa: E402


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-wandb", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=20.0)


def ok(message: str) -> None:
    print(f"[ok] {message}")


def warn(message: str) -> None:
    print(f"[warn] {message}")


def fail(message: str) -> None:
    print(f"[fail] {message}")


def source_paths(args: argparse.Namespace) -> list[Path]:
    if getattr(args, "data_sources", None):
        return [Path(source["data_zip"]) for source in args.data_sources if source.get("data_zip")]
    if args.data_zip:
        return [Path(args.data_zip)]
    return []


def check_zip(path: Path, metadata_member: str) -> bool:
    if not path.exists():
        fail(f"missing data zip: {path}")
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            archive.getinfo(metadata_member)
    except (KeyError, zipfile.BadZipFile) as exc:
        fail(f"{path} is not readable or lacks {metadata_member}: {exc}")
        return False
    ok(f"data zip readable: {path}")
    return True


def check_wandb() -> bool:
    if shutil.which("wandb") is None:
        fail("wandb CLI is not on PATH")
        return False
    if os.environ.get("WANDB_MODE", "").lower() in {"disabled", "offline"}:
        fail(f"WANDB_MODE={os.environ['WANDB_MODE']}; metrics will not sync live")
        return False
    try:
        import wandb
    except ImportError:
        fail("wandb Python package is not installed")
        return False
    if not getattr(wandb.api, "api_key", None):
        fail("wandb is installed but no API key is available; run `wandb login`")
        return False

    result = subprocess.run(
        ["wandb", "status"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        fail("wandb status failed")
        print(output.strip())
        return False
    ok("wandb API key is available")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args, train_overrides = parser.parse_known_args(argv)

    train_args = parse_train_args(["--config", str(args.config), *train_overrides])
    train_args.labels_csv = Path(train_args.labels_csv) if train_args.labels_csv else None

    failures = 0
    if torch.cuda.is_available():
        ok(f"cuda available: {torch.cuda.get_device_name(0)}")
    else:
        warn("cuda is not available; smoke checks can run on CPU, full training should use GPU")

    if train_args.run_dir:
        run_dir = Path(train_args.run_dir)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(run_dir.parent).free / 1024**3
        if free_gb < args.min_free_gb:
            failures += 1
            fail(f"{run_dir.parent} has only {free_gb:.1f} GB free")
        else:
            ok(f"{run_dir.parent} has {free_gb:.1f} GB free")

    if train_args.labels_csv and train_args.labels_csv.exists():
        ok(f"labels CSV exists: {train_args.labels_csv}")
    else:
        failures += 1
        fail(f"missing labels CSV: {train_args.labels_csv}")

    split_csv = getattr(train_args, "split_csv", None)
    if split_csv:
        split_csv = Path(split_csv)
        if split_csv.exists():
            ok(f"split CSV exists: {split_csv}")
        else:
            failures += 1
            fail(f"missing split CSV: {split_csv}")

    for path in source_paths(train_args):
        if not check_zip(path, train_args.metadata_member):
            failures += 1

    try:
        rows, _ = load_training_rows(train_args)
    except Exception as exc:  # noqa: BLE001
        failures += 1
        fail(f"could not load training rows: {exc}")
    else:
        labels = Counter(row[train_args.label_col] for row in rows)
        sources = Counter(row["_dataset_name"] for row in rows)
        if len(labels) < 2:
            failures += 1
            fail(f"expected at least two classes, got {dict(labels)}")
        else:
            ok(f"loaded {len(rows)} labeled rows; labels={dict(labels)}; sources={dict(sources)}")

    if args.check_wandb and not check_wandb():
        failures += 1

    if failures:
        fail(f"preflight finished with {failures} failure(s)")
        return 1
    ok("preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
