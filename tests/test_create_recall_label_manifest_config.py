from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "create_recall_label_manifest.py"
SPEC = importlib.util.spec_from_file_location("create_recall_label_manifest", SCRIPT_PATH)
assert SPEC and SPEC.loader
create_recall_label_manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(create_recall_label_manifest)


def write_paths_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "vindr_csv: /data/vindr/breast-level_annotations.csv",
                "vindr_dicom_dir: /data/vindr/images",
                "rsna_csv: /data/rsna/train.csv",
                "rsna_dicom_dir: /data/rsna/train_images",
                "spr_csv: /data/spr/train.csv",
                "spr_dicom_dir: /data/spr/dicoms",
                "output: processed_datasets/labels.csv",
            ]
        )
        + "\n"
    )


def test_parse_args_loads_dataset_paths_from_yaml(tmp_path: Path) -> None:
    paths_config = tmp_path / "paths.yaml"
    write_paths_config(paths_config)

    args = create_recall_label_manifest.parse_args(["--paths-config", str(paths_config)])

    assert args.vindr_csv == Path("/data/vindr/breast-level_annotations.csv")
    assert args.vindr_dicom_dir == Path("/data/vindr/images")
    assert args.rsna_csv == Path("/data/rsna/train.csv")
    assert args.rsna_dicom_dir == Path("/data/rsna/train_images")
    assert args.spr_csv == Path("/data/spr/train.csv")
    assert args.spr_dicom_dir == Path("/data/spr/dicoms")
    assert args.output == Path("processed_datasets/labels.csv")


def test_cli_paths_override_yaml_paths(tmp_path: Path) -> None:
    paths_config = tmp_path / "paths.yaml"
    write_paths_config(paths_config)

    args = create_recall_label_manifest.parse_args(
        [
            "--paths-config",
            str(paths_config),
            "--spr-dicom-dir",
            "/mnt/spr",
            "--output",
            "labels.csv",
        ]
    )

    assert args.spr_dicom_dir == Path("/mnt/spr")
    assert args.output == Path("labels.csv")


def test_parse_args_fails_without_required_paths(tmp_path: Path) -> None:
    missing_default_config = tmp_path / "missing.yaml"

    with pytest.raises(SystemExit):
        create_recall_label_manifest.parse_args(["--paths-config", str(missing_default_config)])


def test_parse_args_rejects_unknown_yaml_keys(tmp_path: Path) -> None:
    paths_config = tmp_path / "paths.yaml"
    paths_config.write_text("vindr_csv: /data/vindr.csv\nextra: value\n")

    with pytest.raises(SystemExit):
        create_recall_label_manifest.parse_args(["--paths-config", str(paths_config)])
