from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_classifier.py"
SPEC = importlib.util.spec_from_file_location("train_classifier", SCRIPT_PATH)
assert SPEC and SPEC.loader
train_classifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_classifier)


def test_parse_args_loads_experiment_config_and_limited_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_zip: data.zip",
                "labels_csv: labels.csv",
                "run_dir: runs/from_config",
                "model: convnext_tiny",
                "epochs: 7",
                "batch_size: 8",
                "device: cpu",
                "wandb: false",
            ]
        )
        + "\n"
    )

    args = train_classifier.parse_args(
        [
            "--config",
            str(config_path),
            "--run-dir",
            "runs/override",
            "--max-samples",
            "32",
            "--wandb",
        ]
    )

    assert args.config == config_path
    assert args.data_zip == "data.zip"
    assert args.labels_csv == "labels.csv"
    assert args.run_dir == Path("runs/override")
    assert args.max_samples == 32
    assert args.model == "convnext_tiny"
    assert args.epochs == 7
    assert args.batch_size == 8
    assert args.device == "cpu"
    assert args.wandb is True


def test_parse_args_rejects_experiment_knobs_on_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("data_zip: data.zip\nlabels_csv: labels.csv\n")

    with pytest.raises(SystemExit):
        train_classifier.parse_args(
            ["--config", str(config_path), "--epochs", "1"]
        )


def test_parse_args_rejects_unknown_config_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("data_zip: data.zip\nepohcs: 1\n")

    with pytest.raises(SystemExit):
        train_classifier.parse_args(["--config", str(config_path)])


def test_predict_rows_adds_labels_probabilities_and_row_context() -> None:
    class FixedModel(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            positive_logit = inputs[:, 0]
            negative_logit = 1.0 - positive_logit
            return torch.stack([negative_logit, positive_logit], dim=1)

    loader = DataLoader(
        TensorDataset(
            torch.tensor([[0.0], [1.0]], dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
        ),
        batch_size=1,
        shuffle=False,
    )
    rows = [
        {"image_id": "image_a", "target": "0", "dataset": "spr"},
        {"image_id": "image_b", "target": "1", "dataset": "spr"},
    ]

    predictions = train_classifier.predict_rows(
        model=FixedModel(),
        loader=loader,
        rows=rows,
        device=torch.device("cpu"),
        label_map={"0": 0, "1": 1},
        label_col="target",
        epoch=3,
        split="val",
    )

    assert [row["image_id"] for row in predictions] == ["image_a", "image_b"]
    assert [row["pred_label"] for row in predictions] == ["0", "1"]
    assert [row["correct"] for row in predictions] == [1, 1]
    assert predictions[0]["epoch"] == 3
    assert predictions[0]["split"] == "val"
    assert "prob_0" in predictions[0]
    assert "prob_1" in predictions[0]


def test_binary_prediction_metrics_reports_threshold_metrics() -> None:
    predictions = [
        {"target_index": 0, "pred_index": 0, "prob_1": 0.10},
        {"target_index": 1, "pred_index": 1, "prob_1": 0.90},
        {"target_index": 1, "pred_index": 0, "prob_1": 0.40},
        {"target_index": 0, "pred_index": 1, "prob_1": 0.60},
    ]

    metrics = train_classifier.binary_prediction_metrics(
        predictions,
        label_map={"0": 0, "1": 1},
        positive_label="1",
    )

    assert metrics["val/precision"] == 0.5
    assert metrics["val/recall"] == 0.5
    assert metrics["val/specificity"] == 0.5
    assert metrics["val/f1"] == 0.5
    assert metrics["val/balanced_accuracy"] == 0.5
    assert metrics["val/tp"] == 1
    assert metrics["val/fp"] == 1
    assert metrics["val/tn"] == 1
    assert metrics["val/fn"] == 1
    assert "val/average_precision" in metrics


def test_log_wandb_validation_plots_logs_expected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.payload = None
            self.step = None

        def log(self, payload: dict, step: int) -> None:
            self.payload = payload
            self.step = step

    calls = []

    def fake_chart(*args, **kwargs):
        calls.append((args, kwargs))
        return {"chart": kwargs.get("title")}

    monkeypatch.setattr(wandb.plot, "roc_curve", fake_chart)
    monkeypatch.setattr(wandb.plot, "pr_curve", fake_chart)
    monkeypatch.setattr(wandb.plot, "confusion_matrix", fake_chart)

    fake_run = FakeRun()
    train_classifier.log_wandb_validation_plots(
        fake_run,
        predictions=[
            {"target_index": 0, "pred_index": 0, "prob_0": 0.8, "prob_1": 0.2},
            {"target_index": 1, "pred_index": 1, "prob_0": 0.1, "prob_1": 0.9},
        ],
        label_map={"0": 0, "1": 1},
        positive_label="1",
        step=12,
    )

    assert fake_run.step == 12
    assert set(fake_run.payload) == {
        "val/roc_curve",
        "val/pr_curve",
        "val/confusion_matrix",
        "val/positive_score_histogram",
    }
    assert len(calls) == 3
