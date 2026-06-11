from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_spr_test_processed.py"
SPEC = importlib.util.spec_from_file_location("prepare_spr_test_processed", SCRIPT_PATH)
assert SPEC and SPEC.loader
prepare_spr_test_processed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_spr_test_processed)


def test_read_test_accessions_deduplicates_in_order(tmp_path: Path) -> None:
    index = tmp_path / "test_set_index.csv"
    index.write_text(
        "AccessionNumber,target\n"
        "028236,0.5\n"
        "005409,0.5\n"
        "028236,0.5\n"
    )

    assert prepare_spr_test_processed.read_test_accessions(index) == ["028236", "005409"]


def test_read_test_accessions_requires_accession_column(tmp_path: Path) -> None:
    index = tmp_path / "bad.csv"
    index.write_text("id,target\n1,0.5\n")

    with pytest.raises(ValueError):
        prepare_spr_test_processed.read_test_accessions(index)


def test_build_tasks_finds_only_requested_accession_dicoms(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicoms"
    (dicom_root / "028236").mkdir(parents=True)
    (dicom_root / "028236" / "a.dcm").touch()
    (dicom_root / "028236" / "b.dicom").touch()
    (dicom_root / "028236" / "notes.txt").touch()
    (dicom_root / "999999").mkdir()
    (dicom_root / "999999" / "ignore.dcm").touch()

    tasks, missing = prepare_spr_test_processed.build_tasks(
        accessions=["028236", "005409"],
        spr_dicom_dir=dicom_root,
        output_root=tmp_path / "processed",
        output_size=1024,
        overwrite=False,
    )

    assert missing == ["005409"]
    assert [task["accession"] for task in tasks] == ["028236", "028236"]
    assert [Path(task["dicom_path"]).name for task in tasks] == ["a.dcm", "b.dicom"]


def test_write_metadata_outputs_expected_columns(tmp_path: Path) -> None:
    output = tmp_path / "metadata.csv"

    prepare_spr_test_processed.write_metadata(
        output,
        [
            {
                "accession_number": "028236",
                "variant": "grayscale",
                "dicom_path": "/dicoms/028236/a.dcm",
                "processed_path": "grayscale/028236/a.png",
                "status": "processed",
                "image_id": "image",
            }
        ],
    )

    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["accession_number"] == "028236"
    assert rows[0]["variant"] == "grayscale"
    assert rows[0]["processed_path"] == "grayscale/028236/a.png"
    assert "crop_xmin" in rows[0]
