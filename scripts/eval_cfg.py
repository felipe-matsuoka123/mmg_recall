#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.transforms import build_transforms
from src.data.mammogram_dataset import MammogramDataset, DatasetConfig
from src.models.classifier import MammogramClassifier
from src.training.engine import eval_one_epoch
from src.utils.config import load_yaml, apply_overrides


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", type=str, nargs="*", default=[])
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.override)
    project = cfg.get("project", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    out_dir = Path(project.get("out_dir", "runs/exp"))
    ckpt_path = Path(args.ckpt) if args.ckpt else (out_dir / "best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = MammogramClassifier(
        backbone=model_cfg.get("backbone", "resnet50"),
        num_classes=int(model_cfg.get("num_classes", 2)),
        in_chans=int(model_cfg.get("in_chans", 1)),
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    split_csv = data_cfg.get(f"{args.split}_csv") or data_cfg.get("labels_csv")
    ds_cfg = DatasetConfig(
        data_root=Path(data_cfg["data_root"]),
        labels_csv=Path(split_csv),
        zip_path=Path(data_cfg["zip_path"]) if data_cfg.get("zip_path") else None,
        img_col=data_cfg.get("img_col", "png_path"),
        id_col=data_cfg.get("id_col", "sample_id"),
        label_col=data_cfg.get("label_col", "label"),
    )
    tf = build_transforms(int(data_cfg.get("img_size", 1024)), train=False)
    ds = MammogramDataset(ds_cfg, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=int(train_cfg.get("batch_size", data_cfg.get("eval_batch_size", 8))),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    metrics = eval_one_epoch(model, loader, device, num_classes=int(model_cfg.get("num_classes", 2)))
    print(f"{args.split} metrics: {metrics}")

    all_ids = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y, sample_ids in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            if int(model_cfg.get("num_classes", 2)) == 1:
                probs = torch.sigmoid(logits).squeeze(1).cpu()
                preds = (probs >= 0.5).long()
            else:
                preds = logits.argmax(dim=1).cpu()
            all_ids.extend(sample_ids)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    out_csv = out_dir / f"preds_{args.split}.csv"
    pd.DataFrame({"sample_id": all_ids, "pred": all_preds, "label": all_labels}).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
