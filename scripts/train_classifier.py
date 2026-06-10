#!/usr/bin/env python3
"""Train a baseline mammography classifier from a preprocessed PNG dataset zip."""

from __future__ import annotations

import argparse
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


def optional_path(value: str | None) -> Path | None:
    if value is None or value.strip().lower() in {"", "none", "null"}:
        return None
    return Path(value)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-zip", type=optional_path, default=None)
    parser.add_argument("--data-root", type=optional_path, default=None)
    parser.add_argument("--metadata-csv", type=optional_path, default=None)
    parser.add_argument("--labels-csv", type=optional_path, default=None)
    parser.add_argument("--metadata-member", default="metadata.csv")
    parser.add_argument("--label-key", default="image_id")
    parser.add_argument("--row-key", default="image_id")
    parser.add_argument("--label-col", default="target")
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--labels-dataset-col", default="dataset")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--require-all-labels", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/local_baseline"))
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "resnet18", "convnext_tiny", "convnext_small"],
        default="simple_cnn",
    )
    parser.add_argument("--input-channels", type=int, choices=[1, 3], default=1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss-weighting",
        choices=["none", "balanced"],
        default="none",
        help="Class weighting strategy for CrossEntropyLoss.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="mammorecall")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-log-interval", type=int, default=50)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    known, _ = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    if known.config:
        with known.config.open() as handle:
            parser.set_defaults(**(yaml.safe_load(handle) or {}))
    return parser.parse_args(argv)


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
    for row in rows:
        row["_source_index"] = str(source_index)
        row["_dataset_name"] = str(dataset_name or source_index)
        patient_id = row.get(args.patient_col, "").strip()
        row["_split_group"] = f"{row['_dataset_name']}:{patient_id or row.get(args.label_key, '')}"
    return rows


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
        raise SystemExit("Pass exactly one of --data-zip or --data-root.")
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
    for filename in ("best.pt", "last.pt", "config.json", "label_map.json"):
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
    train_rows, val_rows = split_rows_by_group(
        rows,
        group_col="_split_group",
        val_fraction=args.val_fraction,
        seed=args.seed,
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
    print(
        f"device={device} train_images={len(train_dataset)} "
        f"val_images={len(val_dataset)} classes={label_map} sources={source_counts} "
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
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_auroc={val_metrics.get('auroc', float('nan')):.4f}"
            )
            if run:
                run.log(metrics, step=global_step)

            save_checkpoint(
                args.run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                label_map=label_map,
                config=config,
                val_metrics=val_metrics,
            )
            if val_metrics["accuracy"] > best_accuracy:
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
    finally:
        if run:
            artifact_name = args.wandb_run_name or args.run_dir.name
            log_wandb_artifact(run, args.run_dir, artifact_name)
            run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
