#!/usr/bin/env python3
"""Train a mammography classifier from an experiment config."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mammorecall.data import (  # noqa: E402
    DirectoryPngDataset,
    MultiSourcePngDataset,
    ZipPngDataset,
    attach_labels,
    build_label_map,
    filter_rows_by_dataset,
    read_metadata_from_source,
    split_rows_by_group,
)
from mammorecall.engine import run_epoch  # noqa: E402
from mammorecall.models import build_model  # noqa: E402


CONFIG_DEFAULTS = {
    "data_zip": None,
    "data_root": None,
    "metadata_csv": None,
    "labels_csv": None,
    "split_csv": None,
    "split_col": "experiment_split",
    "train_split": "train",
    "val_split": "val",
    "test_split": "test",
    "metadata_member": "metadata.csv",
    "label_key": "image_id",
    "row_key": "image_id",
    "label_col": "target",
    "patient_col": "patient_id",
    "dataset_name": None,
    "dataset_col": "dataset",
    "labels_dataset_col": "dataset",
    "max_samples": None,
    "require_all_labels": False,
    "run_dir": "runs/local_baseline",
    "model": "simple_cnn",
    "input_channels": 1,
    "image_size": 512,
    "epochs": 3,
    "batch_size": 4,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "loss_weighting": "none",
    "val_fraction": 0.2,
    "num_workers": 2,
    "seed": 42,
    "device": "auto",
    "pretrained": False,
    "wandb": False,
    "wandb_project": "mammorecall",
    "wandb_entity": None,
    "wandb_run_name": None,
    "wandb_log_interval": 50,
    "positive_label": "1",
}
ALLOWED_CONFIG_KEYS = set(CONFIG_DEFAULTS) | {"data_sources"}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-run-name", default=None)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    cli_args = parser.parse_args(argv)

    with cli_args.config.open() as handle:
        config_values = yaml.safe_load(handle) or {}
    if not isinstance(config_values, dict):
        raise SystemExit(f"Training config must be a mapping: {cli_args.config}")
    unknown_keys = sorted(set(config_values) - ALLOWED_CONFIG_KEYS)
    if unknown_keys:
        formatted = ", ".join(unknown_keys)
        raise SystemExit(f"Unknown training config key(s) in {cli_args.config}: {formatted}")

    values = {**CONFIG_DEFAULTS, **config_values, "config": cli_args.config}
    for key in ("run_dir", "max_samples", "device", "wandb", "wandb_run_name"):
        value = getattr(cli_args, key)
        if value is not None:
            values[key] = value
    return argparse.Namespace(**values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def json_ready_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def read_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def source_optional_path(source: dict[str, object], key: str) -> Path | None:
    value = source.get(key)
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    value = str(value)
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return Path(value)


def attach_splits(
    rows: list[dict[str, str]],
    split_csv: str | Path | None,
    *,
    split_col: str,
    label_key: str,
    row_key: str,
    labels_dataset_col: str | None,
    dataset_name: str | None,
) -> list[dict[str, str]]:
    if split_csv is None:
        return rows

    split_rows = read_csv_rows(split_csv)
    if labels_dataset_col and dataset_name:
        split_rows = [
            row for row in split_rows if row.get(labels_dataset_col, "").strip() == dataset_name
        ]
    split_by_key = {
        row[label_key]: row[split_col]
        for row in split_rows
        if row.get(label_key, "").strip() and row.get(split_col, "").strip()
    }
    missing = []
    merged_rows = []
    for row in rows:
        key_value = row.get(row_key, "")
        split = split_by_key.get(key_value)
        if split is None:
            missing.append(key_value)
            continue
        merged_row = dict(row)
        merged_row[split_col] = split
        merged_rows.append(merged_row)
    if missing:
        examples = ", ".join(repr(key) for key in missing[:3])
        raise ValueError(
            f"Missing split assignments for {len(missing)} rows using key '{row_key}'. "
            f"First keys: {examples}"
        )
    return merged_rows


def load_source_rows(
    *,
    source: dict[str, object],
    source_index: int,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    dataset_name = source.get("dataset_name") or args.dataset_name
    row_key = source.get("row_key") or getattr(args, "row_key", None) or args.label_key
    data_zip = source_optional_path(source, "data_zip")
    data_root = source_optional_path(source, "data_root")
    metadata_csv = source_optional_path(source, "metadata_csv")
    rows = read_metadata_from_source(
        data_zip=data_zip,
        data_root=data_root,
        metadata_csv=metadata_csv,
        member=str(source.get("metadata_member") or args.metadata_member),
    )
    for row in rows:
        if row.get("processed_path"):
            row["processed_path_stem"] = Path(row["processed_path"]).stem
    rows = filter_rows_by_dataset(
        rows,
        dataset_col=args.dataset_col,
        dataset_name=str(dataset_name) if dataset_name else None,
    )
    rows = attach_labels(
        rows,
        args.labels_csv,
        label_key=args.label_key,
        label_col=args.label_col,
        row_key=str(row_key),
        labels_dataset_col=args.labels_dataset_col,
        dataset_name=str(dataset_name) if dataset_name else None,
        require_all=args.require_all_labels,
    )
    rows = attach_splits(
        rows,
        args.split_csv,
        split_col=args.split_col,
        label_key=args.label_key,
        row_key=str(row_key),
        labels_dataset_col=args.labels_dataset_col,
        dataset_name=str(dataset_name) if dataset_name else None,
    )
    for row in rows:
        row["_source_index"] = str(source_index)
        row["_dataset_name"] = str(dataset_name or source_index)
        patient_id = row.get(args.patient_col, "").strip()
        row["_split_group"] = f"{row['_dataset_name']}:{patient_id or row.get(args.label_key, '')}"
    return rows


def split_rows_for_training(
    rows: list[dict[str, str]],
    *,
    split_col: str,
    train_split: str,
    val_split: str,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if split_col in rows[0]:
        train_rows = [row for row in rows if row.get(split_col, "").strip() == train_split]
        val_rows = [row for row in rows if row.get(split_col, "").strip() == val_split]
        if not train_rows or not val_rows:
            counts = Counter(row.get(split_col, "").strip() for row in rows)
            raise ValueError(
                f"Split column {split_col!r} did not produce non-empty "
                f"{train_split!r}/{val_split!r} partitions. Counts: {dict(counts)}"
            )
        return train_rows, val_rows
    return split_rows_by_group(
        rows,
        group_col="_split_group",
        val_fraction=val_fraction,
        seed=seed,
    )


def load_training_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    if getattr(args, "data_sources", None):
        sources = list(args.data_sources)
        all_rows = []
        normalized_sources = []
        for source_index, source in enumerate(sources):
            data_zip = source_optional_path(source, "data_zip")
            data_root = source_optional_path(source, "data_root")
            if bool(data_zip) == bool(data_root):
                raise SystemExit(
                    f"data_sources[{source_index}] must set exactly one of data_zip or data_root."
                )
            normalized_sources.append({"data_zip": data_zip, "data_root": data_root})
            all_rows.extend(load_source_rows(source=source, source_index=source_index, args=args))
        return all_rows, normalized_sources

    if bool(args.data_zip) == bool(args.data_root):
        raise SystemExit("Config must set exactly one of data_zip or data_root.")
    source = {
        "dataset_name": args.dataset_name,
        "data_zip": args.data_zip,
        "data_root": args.data_root,
        "metadata_csv": args.metadata_csv,
        "metadata_member": args.metadata_member,
    }
    rows = load_source_rows(source=source, source_index=0, args=args)
    return rows, [{"data_zip": args.data_zip, "data_root": args.data_root}]


def build_dataset(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    label_map: dict[str, int],
):
    dataset_kwargs = {
        "rows": rows,
        "label_map": label_map,
        "label_col": args.label_col,
        "input_channels": args.input_channels,
        "image_size": args.image_size,
    }
    if getattr(args, "data_sources", None):
        return MultiSourcePngDataset(args.normalized_sources, **dataset_kwargs)
    if args.data_zip:
        return ZipPngDataset(args.data_zip, **dataset_kwargs)
    return DirectoryPngDataset(args.data_root, **dataset_kwargs)


def limit_rows(
    rows: list[dict[str, str]],
    *,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, str]]:
    if max_samples is None:
        return rows
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    if len(rows) <= max_samples:
        return rows
    sampled_rows = list(rows)
    random.Random(seed).shuffle(sampled_rows)
    return sampled_rows[:max_samples]


def build_cross_entropy_loss(
    *,
    loss_weighting: str,
    train_rows: list[dict[str, str]],
    label_col: str,
    label_map: dict[str, int],
    device: torch.device,
) -> tuple[nn.CrossEntropyLoss, list[float] | None]:
    if loss_weighting == "none":
        return nn.CrossEntropyLoss(), None
    if loss_weighting != "balanced":
        raise ValueError(f"Unsupported loss_weighting={loss_weighting!r}")

    counts_by_index = Counter(label_map[row[label_col]] for row in train_rows)
    missing_labels = [
        label
        for label, index in label_map.items()
        if counts_by_index[index] == 0
    ]
    if missing_labels:
        raise ValueError(
            "Cannot use balanced loss weighting because the training split has "
            f"no examples for labels: {missing_labels}"
        )

    total = len(train_rows)
    num_classes = len(label_map)
    weights = [
        total / (num_classes * counts_by_index[index])
        for index in range(num_classes)
    ]
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor), weights


def write_rows_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def probability_column(label: str) -> str:
    safe_label = "".join(char if char.isalnum() else "_" for char in str(label)).strip("_")
    return f"prob_{safe_label or 'class'}"


def positive_class_index(label_map: dict[str, int], positive_label: str) -> int:
    if positive_label in label_map:
        return label_map[positive_label]
    if len(label_map) == 2:
        return max(label_map.values())
    raise ValueError(f"Positive label {positive_label!r} not found in label map {label_map}")


def predict_rows(
    *,
    model: nn.Module,
    loader: DataLoader,
    rows: list[dict[str, str]],
    device: torch.device,
    label_map: dict[str, int],
    label_col: str,
    epoch: int,
    split: str,
) -> list[dict[str, object]]:
    index_to_label = {index: label for label, index in label_map.items()}
    probability_columns = {
        index: probability_column(index_to_label[index])
        for index in sorted(index_to_label)
    }
    predictions = []
    model.eval()
    row_offset = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            probabilities = torch.softmax(model(inputs), dim=1).cpu()
            predicted_indices = probabilities.argmax(dim=1).tolist()
            target_indices = targets.tolist()
            batch_rows = rows[row_offset: row_offset + len(target_indices)]
            row_offset += len(target_indices)

            for row, target_index, predicted_index, sample_probabilities in zip(
                batch_rows,
                target_indices,
                predicted_indices,
                probabilities.tolist(),
                strict=True,
            ):
                prediction = dict(row)
                prediction["epoch"] = epoch
                prediction["split"] = split
                prediction["target_label"] = row.get(label_col, index_to_label[target_index])
                prediction["target_index"] = target_index
                prediction["pred_label"] = index_to_label[predicted_index]
                prediction["pred_index"] = predicted_index
                prediction["correct"] = int(predicted_index == target_index)
                for index, probability in enumerate(sample_probabilities):
                    prediction[probability_columns[index]] = probability
                predictions.append(prediction)
    if row_offset != len(rows):
        raise RuntimeError(f"Predicted {row_offset} rows but expected {len(rows)} rows.")
    return predictions


def binary_average_precision(scores: list[float], targets: list[int]) -> float | None:
    positives = sum(targets)
    if positives == 0:
        return None

    ranked = sorted(zip(scores, targets, strict=False), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (_score, target) in enumerate(ranked, start=1):
        if target:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def binary_prediction_metrics(
    predictions: list[dict[str, object]],
    *,
    label_map: dict[str, int],
    positive_label: str,
) -> dict[str, float]:
    if not predictions:
        return {}

    positive_index = positive_class_index(label_map, positive_label)
    positive_probability_column = probability_column(str(positive_label))
    if positive_probability_column not in predictions[0]:
        positive_probability_column = probability_column(
            next(label for label, index in label_map.items() if index == positive_index)
        )

    targets = [int(row["target_index"]) for row in predictions]
    preds = [int(row["pred_index"]) for row in predictions]
    binary_targets = [1 if target == positive_index else 0 for target in targets]
    scores = [float(row[positive_probability_column]) for row in predictions]

    true_positive = sum(1 for target, pred in zip(targets, preds, strict=True) if target == pred == positive_index)
    false_positive = sum(1 for target, pred in zip(targets, preds, strict=True) if target != positive_index and pred == positive_index)
    true_negative = sum(1 for target, pred in zip(targets, preds, strict=True) if target != positive_index and pred != positive_index)
    false_negative = sum(1 for target, pred in zip(targets, preds, strict=True) if target == positive_index and pred != positive_index)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    average_precision = binary_average_precision(scores, binary_targets)

    metrics = {
        "val/precision": precision,
        "val/recall": recall,
        "val/specificity": specificity,
        "val/f1": f1,
        "val/balanced_accuracy": (recall + specificity) / 2.0,
        "val/tp": true_positive,
        "val/fp": false_positive,
        "val/tn": true_negative,
        "val/fn": false_negative,
    }
    if average_precision is not None:
        metrics["val/average_precision"] = average_precision
    return metrics


def log_wandb_validation_plots(
    run,
    *,
    predictions: list[dict[str, object]],
    label_map: dict[str, int],
    positive_label: str,
    step: int,
) -> None:
    if run is None or not predictions:
        return
    try:
        import wandb
    except ImportError:
        return

    labels = [label for label, _index in sorted(label_map.items(), key=lambda item: item[1])]
    y_true = [int(row["target_index"]) for row in predictions]
    y_pred = [int(row["pred_index"]) for row in predictions]
    y_probas = [
        [float(row[probability_column(label)]) for label in labels]
        for row in predictions
    ]
    payload = {
        "val/roc_curve": wandb.plot.roc_curve(
            y_true,
            y_probas,
            labels=labels,
            title="Validation ROC Curve",
            split_table=True,
        ),
        "val/pr_curve": wandb.plot.pr_curve(
            y_true,
            y_probas,
            labels=labels,
            title="Validation Precision-Recall Curve",
            split_table=True,
        ),
        "val/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true,
            preds=y_pred,
            class_names=labels,
            title="Validation Confusion Matrix",
            split_table=True,
        ),
    }
    positive_index = positive_class_index(label_map, positive_label)
    positive_label_name = next(label for label, index in label_map.items() if index == positive_index)
    positive_scores = [float(row[probability_column(positive_label_name)]) for row in predictions]
    payload["val/positive_score_histogram"] = wandb.Histogram(positive_scores)
    run.log(payload, step=step)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    label_map: dict[str, int],
    config: dict[str, object],
    val_metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "label_map": label_map,
            "config": config,
            "val_metrics": val_metrics,
        },
        path,
    )


def start_wandb(args: argparse.Namespace, config: dict[str, object]):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("Install wandb or run without --wandb.") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=config,
    )


def log_wandb_artifact(run, run_dir: Path, artifact_name: str) -> None:
    if run is None:
        return
    try:
        import wandb
    except ImportError:
        return

    artifact = wandb.Artifact(artifact_name, type="model")
    for filename in (
        "best.pt",
        "last.pt",
        "config.json",
        "label_map.json",
        "metrics.jsonl",
        "val_predictions.csv",
    ):
        path = run_dir / filename
        if path.exists():
            artifact.add_file(str(path), name=filename)
    run.log_artifact(artifact)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_zip = Path(args.data_zip) if args.data_zip else None
    args.data_root = Path(args.data_root) if args.data_root else None
    args.metadata_csv = Path(args.metadata_csv) if args.metadata_csv else None
    args.run_dir = Path(args.run_dir)
    args.labels_csv = Path(args.labels_csv) if args.labels_csv else None
    args.split_csv = Path(args.split_csv) if args.split_csv else None
    set_seed(args.seed)
    device = select_device(args.device)
    config = json_ready_config(args)

    rows, normalized_sources = load_training_rows(args)
    args.normalized_sources = normalized_sources
    if not rows:
        raise SystemExit("No rows available after metadata/label filtering.")
    source_counts = {
        dataset_name: sum(1 for row in rows if row["_dataset_name"] == dataset_name)
        for dataset_name in sorted({row["_dataset_name"] for row in rows})
    }
    rows = limit_rows(rows, max_samples=args.max_samples, seed=args.seed)
    train_rows, val_rows = split_rows_for_training(
        rows,
        split_col=args.split_col,
        train_split=args.train_split,
        val_split=args.val_split,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    test_rows = (
        [row for row in rows if row.get(args.split_col, "").strip() == args.test_split]
        if args.split_col in rows[0]
        else []
    )
    label_map = build_label_map(rows, args.label_col)

    train_dataset = build_dataset(args, train_rows, label_map)
    val_dataset = build_dataset(args, val_rows, label_map)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model,
        input_channels=args.input_channels,
        num_classes=len(label_map),
        pretrained=args.pretrained,
    ).to(device)
    criterion, loss_class_weights = build_cross_entropy_loss(
        loss_weighting=args.loss_weighting,
        train_rows=train_rows,
        label_col=args.label_col,
        label_map=label_map,
        device=device,
    )
    config["loss_class_weights"] = loss_class_weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.run_dir / "label_map.json").write_text(json.dumps(label_map, indent=2) + "\n")
    write_rows_csv(args.run_dir / "train_rows.csv", train_rows)
    write_rows_csv(args.run_dir / "val_rows.csv", val_rows)
    if test_rows:
        write_rows_csv(args.run_dir / "test_rows.csv", test_rows)
    metrics_path = args.run_dir / "metrics.jsonl"
    metrics_path.write_text("")
    print(
        f"device={device} train_images={len(train_dataset)} "
        f"val_images={len(val_dataset)} test_images={len(test_rows)} "
        f"classes={label_map} sources={source_counts} "
        f"loss_weighting={args.loss_weighting} loss_class_weights={loss_class_weights}"
    )

    run = start_wandb(args, config)
    best_accuracy = -1.0
    global_step = 0
    try:
        for epoch in range(1, args.epochs + 1):
            def log_train_batch(batch_metrics: dict[str, float]) -> None:
                nonlocal global_step
                global_step += 1
                if not run or args.wandb_log_interval <= 0:
                    return
                if global_step % args.wandb_log_interval != 0:
                    return
                run.log(
                    {
                        "epoch": epoch,
                        "train/batch_loss": batch_metrics["batch/loss"],
                        "train/running_loss": batch_metrics["running/loss"],
                        "train/running_accuracy": batch_metrics["running/accuracy"],
                        "train/images_seen_in_epoch": batch_metrics["seen"],
                    },
                    step=global_step,
                )

            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device=device,
                optimizer=optimizer,
                on_batch_end=log_train_batch,
            )
            val_metrics = run_epoch(model, val_loader, criterion, device=device)
            metrics = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
            }
            if "auroc" in train_metrics:
                metrics["train/auroc"] = train_metrics["auroc"]
            if "auroc" in val_metrics:
                metrics["val/auroc"] = val_metrics["auroc"]
            is_best = val_metrics["accuracy"] > best_accuracy
            metrics["best_checkpoint"] = is_best
            metrics["global_step"] = global_step
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_auroc={val_metrics.get('auroc', float('nan')):.4f}"
            )
            save_checkpoint(
                args.run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                label_map=label_map,
                config=config,
                val_metrics=val_metrics,
            )
            if is_best:
                best_accuracy = val_metrics["accuracy"]
                save_checkpoint(
                    args.run_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    label_map=label_map,
                    config=config,
                    val_metrics=val_metrics,
                )
                val_predictions = predict_rows(
                    model=model,
                    loader=val_loader,
                    rows=val_rows,
                    device=device,
                    label_map=label_map,
                    label_col=args.label_col,
                    epoch=epoch,
                    split=args.val_split,
                )
                write_rows_csv(args.run_dir / "val_predictions.csv", val_predictions)
                best_metrics = binary_prediction_metrics(
                    val_predictions,
                    label_map=label_map,
                    positive_label=args.positive_label,
                )
                metrics.update(best_metrics)
                log_wandb_validation_plots(
                    run,
                    predictions=val_predictions,
                    label_map=label_map,
                    positive_label=args.positive_label,
                    step=global_step,
                )
            if run:
                run.log(metrics, step=global_step)
            append_jsonl(metrics_path, metrics)
    finally:
        if run:
            artifact_name = args.wandb_run_name or args.run_dir.name
            log_wandb_artifact(run, args.run_dir, artifact_name)
            run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
