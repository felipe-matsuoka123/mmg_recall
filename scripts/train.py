from __future__ import annotations
import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.data.transforms import build_transforms
from src.data.mammogram_dataset import MammogramDataset, DatasetConfig
from src.models.classifier import MammogramClassifier

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--labels_csv", type=str, required=True)
    p.add_argument("--zip_path", type=str, default=None)

    p.add_argument("--img_size", type=int, default=1024)
    p.add_argument("--backbone", type=str, default="resnet50")
    p.add_argument("--num_classes", type=int, default=2)  # set 1 for BCE
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_dir", type=str, default="runs/exp01")
    return p.parse_args()

@torch.no_grad()
def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_tf = build_transforms(args.img_size, train=True)
    val_tf = build_transforms(args.img_size, train=False)

    # If you already have split CSVs: point train/val to different CSVs.
    # For now: use labels_csv for both and split later properly.
    cfg = DatasetConfig(
        data_root=Path(args.data_root),
        labels_csv=Path(args.labels_csv),
        zip_path=Path(args.zip_path) if args.zip_path else None,
    )
    full_ds = MammogramDataset(cfg, transform=train_tf)

    # Minimal split (replace with scripts/make_splits.py)
    n = len(full_ds)
    n_train = int(0.9 * n)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    val_ds.dataset.transform = val_tf  # swap transforms for val

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model
    model = MammogramClassifier(
        backbone=args.backbone,
        num_classes=args.num_classes,
        in_chans=1,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    # Loss
    # If binary with BCEWithLogitsLoss: set num_classes=1 and make y float in loop.
    if args.num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for x, y, _ids in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)

            if args.num_classes == 1:
                # y expected in {0,1}
                yb = y.float().unsqueeze(1)  # [B,1]
                loss = criterion(logits, yb)
            else:
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # Val
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_seen = 0

        with torch.no_grad():
            for x, y, _ids in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)

                if args.num_classes == 1:
                    yb = y.float().unsqueeze(1)
                    loss = criterion(logits, yb)
                    # accuracy for binary:
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).long().squeeze(1)
                    acc = (preds == y).float().mean().item()
                else:
                    loss = criterion(logits, y)
                    acc = accuracy(logits, y)

                bs = x.size(0)
                val_loss += loss.item() * bs
                val_acc += acc * bs
                n_seen += bs

        val_loss /= n_seen
        val_acc /= n_seen

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {elapsed:.1f}s")

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "best_val_acc": best_val,
            }
            torch.save(ckpt, out_dir / "best.pt")

    print(f"Done. Best val_acc={best_val:.4f}. Saved to {out_dir/'best.pt'}")

if __name__ == "__main__":
    main()
