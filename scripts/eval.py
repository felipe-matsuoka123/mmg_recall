from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.transforms import build_transforms
from src.data.mammogram_dataset import MammogramDataset, DatasetConfig
from src.models.classifier import MammogramClassifier

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--labels_csv", type=str, required=True)
    p.add_argument("--zip_path", type=str, default=None)
    p.add_argument("--img_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    margs = ckpt["args"]

    model = MammogramClassifier(
        backbone=margs["backbone"],
        num_classes=margs["num_classes"],
        in_chans=1,
        pretrained=False,
        dropout=margs["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tf = build_transforms(args.img_size, train=False)
    cfg = DatasetConfig(
        data_root=Path(args.data_root),
        labels_csv=Path(args.labels_csv),
        zip_path=Path(args.zip_path) if args.zip_path else None,
    )
    ds = MammogramDataset(cfg, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for x, y, sample_ids in loader:
            x = x.to(device)
            logits = model(x)
            if margs["num_classes"] == 1:
                probs = torch.sigmoid(logits).squeeze(1).cpu()
                preds = (probs >= 0.5).long()
            else:
                preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_ids.extend(sample_ids)

    preds = torch.cat(all_preds).numpy()
    # Save a simple CSV
    out_path = Path(args.ckpt).parent / "preds.csv"
    import pandas as pd
    pd.DataFrame({"sample_id": all_ids, "pred": preds}).to_csv(out_path, index=False)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
