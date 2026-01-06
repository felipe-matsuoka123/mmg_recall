#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip_dir", type=str, default=None, help="Directory with .zip files.")
    p.add_argument("--zip_paths", type=str, nargs="*", default=None, help="One or more .zip files.")
    p.add_argument("--out_dir", type=str, required=True, help="Output dataset root.")

    p.add_argument("--labels_csv", type=str, required=True, help="Input labels CSV.")
    p.add_argument("--out_labels_csv", type=str, default=None, help="Output labels CSV.")
    p.add_argument("--img_col", type=str, default="filename",
                   help="Column in labels_csv with image filename (used if png_path missing).")
    p.add_argument("--png_col", type=str, default="png_path",
                   help="Column name to write relative png path.")
    p.add_argument("--path_prefix", type=str, default="images",
                   help="Folder (relative to out_dir) where images are extracted.")
    p.add_argument("--rewrite_paths", action="store_true",
                   help="Rewrite png_col even if it already exists.")
    p.add_argument("--verify", action="store_true",
                   help="Verify that png paths exist after extraction.")

    p.add_argument("--make_splits", action="store_true")
    p.add_argument("--id_col", type=str, default="patient_id")
    p.add_argument("--stratify_col", type=str, default="label")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.80)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.10)
    return p.parse_args()


def _safe_extract(zf: zipfile.ZipFile, out_dir: Path):
    for member in zf.infolist():
        target = out_dir / member.filename
        if not str(target.resolve()).startswith(str(out_dir.resolve())):
            raise ValueError(f"Unsafe path in zip: {member.filename}")
    zf.extractall(out_dir)


def _validate_ratios(train, val, test):
    s = train + val + test
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0. Got {s:.6f}")


def _stratified_split_ids(ids, labels, rng, train_ratio, val_ratio, test_ratio):
    train_ids, val_ids, test_ids = [], [], []
    for cls in sorted(set(labels)):
        cls_ids = ids[labels == cls]
        rng.shuffle(cls_ids)
        n = len(cls_ids)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_ids.append(cls_ids[:n_train])
        val_ids.append(cls_ids[n_train:n_train + n_val])
        test_ids.append(cls_ids[n_train + n_val:])
    import numpy as np
    train_ids = np.concatenate(train_ids) if train_ids else np.array([])
    val_ids = np.concatenate(val_ids) if val_ids else np.array([])
    test_ids = np.concatenate(test_ids) if test_ids else np.array([])
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)
    return train_ids, val_ids, test_ids


def _mode(series):
    vc = series.value_counts()
    top = vc[vc == vc.max()].index.astype(str)
    return sorted(top)[0]


def _write_splits(df, out_dir: Path, id_col: str, stratify_col: str,
                  seed: int, train_ratio: float, val_ratio: float, test_ratio: float):
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    import numpy as np

    rng = np.random.default_rng(seed)
    has_group = id_col in df.columns
    has_strat = stratify_col in df.columns

    if has_group:
        group_df = (
            df.groupby(id_col, as_index=False)[stratify_col]
            .agg(_mode if has_strat else (lambda s: "ALL"))
            .rename(columns={stratify_col: "__group_label"})
        )
        ids = group_df[id_col].astype(str).to_numpy()
        if has_strat:
            labs = group_df["__group_label"].astype(str).to_numpy()
            train_ids, val_ids, test_ids = _stratified_split_ids(
                ids, labs, rng, train_ratio, val_ratio, test_ratio
            )
        else:
            rng.shuffle(ids)
            n = len(ids)
            n_train = int(round(train_ratio * n))
            n_val = int(round(val_ratio * n))
            n_train = min(n_train, n)
            n_val = min(n_val, n - n_train)
            train_ids = ids[:n_train]
            val_ids = ids[n_train:n_train + n_val]
            test_ids = ids[n_train + n_val:]

        df["__split"] = "UNASSIGNED"
        df.loc[df[id_col].astype(str).isin(set(train_ids)), "__split"] = "train"
        df.loc[df[id_col].astype(str).isin(set(val_ids)), "__split"] = "val"
        df.loc[df[id_col].astype(str).isin(set(test_ids)), "__split"] = "test"
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if has_strat:
            df["__split"] = "UNASSIGNED"
            for cls, sub in df.groupby(stratify_col):
                idx = sub.index.to_numpy()
                rng.shuffle(idx)
                n = len(idx)
                n_train = int(round(train_ratio * n))
                n_val = int(round(val_ratio * n))
                n_train = min(n_train, n)
                n_val = min(n_val, n - n_train)
                train_idx = idx[:n_train]
                val_idx = idx[n_train:n_train + n_val]
                test_idx = idx[n_train + n_val:]
                df.loc[train_idx, "__split"] = "train"
                df.loc[val_idx, "__split"] = "val"
                df.loc[test_idx, "__split"] = "test"
        else:
            n = len(df)
            n_train = int(round(train_ratio * n))
            n_val = int(round(val_ratio * n))
            df["__split"] = "test"
            df.loc[:n_train - 1, "__split"] = "train"
            df.loc[n_train:n_train + n_val - 1, "__split"] = "val"

    assert (df["__split"] != "UNASSIGNED").all(), "Some rows were not assigned to a split."
    for split in ["train", "val", "test"]:
        out_path = out_dir / f"{split}.csv"
        df[df["__split"] == split].drop(columns=["__split"]).to_csv(out_path, index=False)
        print(f"Wrote {out_path}")


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    images_dir = out_dir / args.path_prefix
    images_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = []
    if args.zip_dir:
        zip_paths.extend(sorted(Path(args.zip_dir).glob("*.zip")))
    if args.zip_paths:
        zip_paths.extend([Path(p) for p in args.zip_paths])

    for zp in zip_paths:
        with zipfile.ZipFile(zp, "r") as zf:
            _safe_extract(zf, images_dir)
        print(f"Extracted {zp} -> {images_dir}")

    df = pd.read_csv(args.labels_csv)
    if args.png_col in df.columns and not args.rewrite_paths:
        pass
    else:
        if args.img_col not in df.columns:
            raise ValueError(f"Missing column '{args.img_col}' in {args.labels_csv}")
        rel_paths = df[args.img_col].astype(str).map(lambda s: f"{args.path_prefix}/{s}")
        df[args.png_col] = rel_paths

    out_labels = Path(args.out_labels_csv) if args.out_labels_csv else (out_dir / "labels.csv")
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_labels, index=False)
    print(f"Wrote {out_labels}")

    if args.verify:
        missing = []
        for p in df[args.png_col].astype(str).tolist():
            if not (out_dir / p).exists():
                missing.append(p)
                if len(missing) >= 10:
                    break
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} example files. First: {missing[:3]}")

    if args.make_splits:
        _write_splits(
            df,
            out_dir,
            id_col=args.id_col,
            stratify_col=args.stratify_col,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )


if __name__ == "__main__":
    main()
