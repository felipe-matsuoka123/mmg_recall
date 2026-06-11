from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "create_split_manifest.py"
SPEC = importlib.util.spec_from_file_location("create_split_manifest", SCRIPT_PATH)
assert SPEC and SPEC.loader
create_split_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(create_split_manifest)


def write_labels(path: Path) -> None:
    rows = []
    for dataset in ("spr", "rsna", "vindr"):
        for patient_index in range(8):
            target = str(patient_index % 2)
            for image_index in range(2):
                rows.append(
                    {
                        "dataset": dataset,
                        "image_id": f"{dataset}_{patient_index}_{image_index}",
                        "target": target,
                        "patient_id": f"{dataset}_patient_{patient_index}",
                        "study_id": "",
                        "accession_number": "",
                    }
                )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_create_split_manifest_is_grouped_deterministic_and_spr_test_only(tmp_path: Path) -> None:
    labels_csv = tmp_path / "labels.csv"
    output_csv = tmp_path / "splits.csv"
    second_output_csv = tmp_path / "splits_again.csv"
    write_labels(labels_csv)

    args = [
        str(labels_csv),
        str(output_csv),
        "--val-fraction",
        "0.25",
        "--test-fraction",
        "0.25",
        "--seed",
        "7",
    ]
    second_args = [
        str(labels_csv),
        str(second_output_csv),
        "--val-fraction",
        "0.25",
        "--test-fraction",
        "0.25",
        "--seed",
        "7",
    ]
    assert create_split_manifest.main(args) == 0
    assert create_split_manifest.main(second_args) == 0

    rows = read_rows(output_csv)
    assert rows == read_rows(second_output_csv)
    assert {row["experiment_split"] for row in rows} == {"train", "val", "test"}
    assert {row["dataset"] for row in rows if row["experiment_split"] == "test"} == {"spr"}

    split_by_group = {}
    for row in rows:
        split = row["experiment_split"]
        group = row["split_group"]
        split_by_group.setdefault(group, split)
        assert split_by_group[group] == split
