#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from src.data.transforms import build_transforms
from src.data.mammogram_dataset import MammogramDataset, DatasetConfig
from src.models.classifier import MammogramClassifier
from src.training.engine import train_one_epoch, eval_one_epoch
from src.utils.checkpoint import save_best_if_improved, save_checkpoint
from src.utils.config import load_yaml, apply_overrides
from src.utils.seed import seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", type=str, nargs="*", default=[])
    return p.parse_args()


def _maybe_init_wandb(cfg: dict):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enable", False):
        return None
    try:
        import wandb
    except Exception:
        print("wandb not installed; continuing without W&B logging.")
        return None

    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        tags=wandb_cfg.get("tags"),
        config=cfg,
    )
    return run


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    project = cfg.get("project", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    seed_everything(int(project.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    out_dir = Path(project.get("out_dir", "runs/exp"))
    out_dir.mkdir(parents=True, exist_ok=True)

    run = _maybe_init_wandb(cfg)

    train_tf = build_transforms(int(data_cfg.get("img_size", 1024)), train=True)
    val_tf = build_transforms(int(data_cfg.get("img_size", 1024)), train=False)

    base_cfg = DatasetConfig(
        data_root=Path(data_cfg["data_root"]),
        labels_csv=Path(data_cfg["labels_csv"]),
        zip_path=Path(data_cfg["zip_path"]) if data_cfg.get("zip_path") else None,
        img_col=data_cfg.get("img_col", "png_path"),
        id_col=data_cfg.get("id_col", "sample_id"),
        label_col=data_cfg.get("label_col", "label"),
    )

    train_csv = data_cfg.get("train_csv")
    val_csv = data_cfg.get("val_csv")

    if train_csv and val_csv:
        train_ds = MammogramDataset(
            DatasetConfig(
                data_root=base_cfg.data_root,
                labels_csv=Path(train_csv),
                zip_path=base_cfg.zip_path,
                img_col=base_cfg.img_col,
                id_col=base_cfg.id_col,
                label_col=base_cfg.label_col,
            ),
            transform=train_tf,
        )
        val_ds = MammogramDataset(
            DatasetConfig(
                data_root=base_cfg.data_root,
                labels_csv=Path(val_csv),
                zip_path=base_cfg.zip_path,
                img_col=base_cfg.img_col,
                id_col=base_cfg.id_col,
                label_col=base_cfg.label_col,
            ),
            transform=val_tf,
        )
    else:
        # fallback: random split
        full_ds = MammogramDataset(base_cfg, transform=train_tf)
        n = len(full_ds)
        train_ratio = float(data_cfg.get("train_ratio", 0.9))
        n_train = int(round(train_ratio * n))
        n_val = n - n_train
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
        val_ds.dataset.transform = val_tf

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    model = MammogramClassifier(
        backbone=model_cfg.get("backbone", "resnet50"),
        num_classes=int(model_cfg.get("num_classes", 2)),
        in_chans=int(model_cfg.get("in_chans", 1)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    best_metric = None
    num_classes = int(model_cfg.get("num_classes", 2))
    metric_key = "auc" if num_classes == 1 else "acc"

    epochs = int(train_cfg.get("epochs", 10))
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            num_classes=num_classes,
            amp=bool(train_cfg.get("amp", True)),
            grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
        )
        val_metrics = eval_one_epoch(
            model,
            val_loader,
            device,
            num_classes=num_classes,
        )
        elapsed = time.time() - t0

        log = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/acc": train_metrics["acc"],
            "val/loss": val_metrics["loss"],
            "val/acc": val_metrics["acc"],
            "time/epoch_sec": elapsed,
        }
        if num_classes == 1:
            log["train/auc"] = train_metrics.get("auc")
            log["val/auc"] = val_metrics.get("auc")

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | {elapsed:.1f}s"
        )

        current_metric = val_metrics.get(metric_key)
        if current_metric is None:
            current_metric = val_metrics["acc"]

        best_metric, improved = save_best_if_improved(
            current_value=current_metric,
            best_value=best_metric,
            out_path=out_dir / "best.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            extra={"val_metrics": val_metrics},
            mode="max",
        )

        save_checkpoint(
            out_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            extra={"val_metrics": val_metrics},
        )

        if run is not None:
            if improved:
                log["best/" + metric_key] = best_metric
            run.log(log)

    if run is not None:
        run.finish()

    print(f"Done. Best {metric_key}={best_metric}. Saved to {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
