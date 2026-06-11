from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "experiment.py"
SPEC = importlib.util.spec_from_file_location("experiment", SCRIPT_PATH)
assert SPEC and SPEC.loader
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def test_resolve_config_accepts_unique_stem() -> None:
    path = experiment.resolve_config("all_grayscale_convnext_tiny_1024")

    assert path == experiment.REPO_ROOT / "config" / "cloud_exp" / "all_grayscale_convnext_tiny_1024.yaml"


def test_resolve_config_accepts_grouped_alias() -> None:
    path = experiment.resolve_config("cloud_exp/all_grayscale_convnext_tiny_1024")

    assert path == experiment.REPO_ROOT / "config" / "cloud_exp" / "all_grayscale_convnext_tiny_1024.yaml"


def test_resolve_config_rejects_unknown_name() -> None:
    with pytest.raises(SystemExit):
        experiment.resolve_config("does_not_exist")


def test_train_dry_run_prints_preflight_then_train(capsys: pytest.CaptureFixture[str]) -> None:
    status = experiment.main(
        [
            "train",
            "all_grayscale_convnext_tiny_1024",
            "--preflight",
            "--run-dir",
            "runs/debug",
            "--max-samples",
            "16",
            "--device",
            "cpu",
            "--no-wandb",
            "--dry-run",
        ]
    )

    assert status == 0
    output = capsys.readouterr().out
    assert "preflight.py" in output
    assert "train_classifier.py" in output
    assert "--config" in output
    assert "config/cloud_exp/all_grayscale_convnext_tiny_1024.yaml" in output
    assert "--run-dir runs/debug" in output
    assert "--max-samples 16" in output
    assert "--device cpu" in output
    assert "--no-wandb" in output
