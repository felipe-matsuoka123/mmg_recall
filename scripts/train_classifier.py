#!/usr/bin/env python3
"""Train a baseline mammography classifier from a preprocessed PNG dataset zip."""

from __future__ import annotations

import argparse
import json
import random
import sys
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
    ZipPngDataset,
    attach_labels,
    build_label_map,
    read_metadata_rows,
    split_rows_by_group,
)
from mammorecall.engine import run_epoch  # noqa: E402
from mammorecall.models import build_model  # noqa: E402


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-zip", type=Path, default=None)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--metadata-member", default="metadata.csv")
    parser.add_argument("--label-key", default="image_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/local_baseline"))
    parser.add_argument("--model", choices=["simple_cnn", "resnet18"], default="simple_cnn")
    parser.add_argument("--input-channels", type=int, choices=[1, 3], default=1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mammorecall")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)


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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_zip:
        raise SystemExit("Pass --data-zip or set data_zip in --config.")
    args.data_zip = Path(args.data_zip)
    args.run_dir = Path(args.run_dir)
    args.labels_csv = Path(args.labels_csv) if args.labels_csv else None
    set_seed(args.seed)
    device = select_device(args.device)
    config = json_ready_config(args)

    rows = read_metadata_rows(args.data_zip, args.metadata_member)
    rows = attach_labels(rows, args.labels_csv, label_key=args.label_key, label_col=args.label_col)
    train_rows, val_rows = split_rows_by_group(
        rows,
        group_col=args.patient_col,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    label_map = build_label_map(rows, args.label_col)

    train_dataset = ZipPngDataset(
        args.data_zip,
        train_rows,
        label_map=label_map,
        label_col=args.label_col,
        input_channels=args.input_channels,
        image_size=args.image_size,
    )
    val_dataset = ZipPngDataset(
        args.data_zip,
        val_rows,
        label_map=label_map,
        label_col=args.label_col,
        input_channels=args.input_channels,
        image_size=args.image_size,
    )
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.run_dir / "label_map.json").write_text(json.dumps(label_map, indent=2) + "\n")
    print(
        f"device={device} train_images={len(train_dataset)} "
        f"val_images={len(val_dataset)} classes={label_map}"
    )

    run = start_wandb(args, config)
    best_accuracy = -1.0
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device=device,
                optimizer=optimizer,
            )
            val_metrics = run_epoch(model, val_loader, criterion, device=device)
            metrics = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
            }
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
            if run:
                run.log(metrics)

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
            run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
