from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "create_processed_spr_test_subset.py"
SPEC = importlib.util.spec_from_file_location("create_processed_spr_test_subset", SCRIPT_PATH)
assert SPEC and SPEC.loader
create_processed_spr_test_subset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(create_processed_spr_test_subset)


def write_test_index(path: Path) -> None:
    path.write_text(
        "AccessionNumber,target\n"
        "001,0.5\n"
        "002,0.5\n"
        "001,0.5\n"
    )


def test_read_test_accessions_deduplicates_in_order(tmp_path: Path) -> None:
    path = tmp_path / "test_set_index.csv"
    write_test_index(path)

    assert create_processed_spr_test_subset.read_test_accessions(path) == ["001", "002"]


def test_build_subset_symlinks_requested_accessions(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed_spr"
    source_dir = processed_root / "grayscale" / "001"
    source_dir.mkdir(parents=True)
    source_png = source_dir / "image.png"
    source_png.write_bytes(b"png")
    (processed_root / "grayscale" / "003").mkdir(parents=True)
    (processed_root / "grayscale" / "003" / "ignore.png").write_bytes(b"png")

    test_index = tmp_path / "test_set_index.csv"
    write_test_index(test_index)
    output_root = tmp_path / "subset"
    args = create_processed_spr_test_subset.parse_args(
        [
            "--test-index",
            str(test_index),
            "--processed-root",
            str(processed_root),
            "--output-root",
            str(output_root),
            "--variants",
            "grayscale",
        ]
    )

    rows, missing = create_processed_spr_test_subset.build_subset(args)

    subset_png = output_root / "grayscale" / "001" / "image.png"
    assert subset_png.is_symlink()
    assert subset_png.resolve() == source_png
    assert [row["AccessionNumber"] for row in rows] == ["001"]
    assert missing == {"grayscale": ["002"]}


def test_main_writes_subset_index(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed_spr"
    source_dir = processed_root / "grayscale" / "001"
    source_dir.mkdir(parents=True)
    (source_dir / "image.png").write_bytes(b"png")
    test_index = tmp_path / "test_set_index.csv"
    write_test_index(test_index)
    output_root = tmp_path / "subset"

    status = create_processed_spr_test_subset.main(
        [
            "--test-index",
            str(test_index),
            "--processed-root",
            str(processed_root),
            "--output-root",
            str(output_root),
            "--variants",
            "grayscale",
        ]
    )

    assert status == 0
    with (output_root / "subset_index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["AccessionNumber"] == "001"
    assert rows[0]["variant"] == "grayscale"
