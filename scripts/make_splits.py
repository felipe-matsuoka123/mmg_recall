#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--id_col", type=str, default="patient_id",
                   help="Group ID column for leakage-safe split. If missing, falls back to row-level.")
    p.add_argument("--stratify_col", type=str, default="label",
                   help="Column used for stratification (usually label). If missing, no stratification.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train_ratio", type=float, default=0.80)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--test_ratio", type=float, default=0.10)

    return p.parse_args()

def _validate_ratios(train, val, test):
    s = train + val + test
    if not np.isclose(s, 1.0):
        raise ValueError(f"Ratios must sum to 1.0. Got {s:.6f}")

def _stratified_split_ids(ids: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
                          train_ratio: float, val_ratio: float, test_ratio: float):
    """
    Stratified split of unique IDs given a single label per ID.
    """
    train_ids, val_ids, test_ids = [], [], []
    for cls in np.unique(labels):
        cls_ids = ids[labels == cls]
        rng.shuffle(cls_ids)
        n = len(cls_ids)
        n_train = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        # ensure sum <= n, remaining goes to test
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train_ids.append(cls_ids[:n_train])
        val_ids.append(cls_ids[n_train:n_train + n_val])
        test_ids.append(cls_ids[n_train + n_val:])

    train_ids = np.concatenate(train_ids) if train_ids else np.array([], dtype=ids.dtype)
    val_ids = np.concatenate(val_ids) if val_ids else np.array([], dtype=ids.dtype)
    test_ids = np.concatenate(test_ids) if test_ids else np.array([], dtype=ids.dtype)

    # shuffle final sets (mix classes)
    rng.shuffle(train_ids); rng.shuffle(val_ids); rng.shuffle(test_ids)
    return train_ids, val_ids, test_ids

def _mode(series: pd.Series):
    # deterministic "mode": if tie, choose smallest string representation
    vc = series.value_counts()
    top = vc[vc == vc.max()].index.astype(str)
    return sorted(top)[0]

def main():
    args = parse_args()
    _validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    labels_csv = Path(args.labels_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_csv)
    rng = np.random.default_rng(args.seed)

    has_group = args.id_col in df.columns
    has_strat = args.stratify_col in df.columns

    if has_group:
        # patient-level: one label per patient_id via mode (works for binary/multiclass)
        group_df = (
            df.groupby(args.id_col, as_index=False)[args.stratify_col]
              .agg(_mode if has_strat else (lambda s: "ALL"))
              .rename(columns={args.stratify_col: "__group_label"})
        )

        ids = group_df[args.id_col].astype(str).to_numpy()
        if has_strat:
            labs = group_df["__group_label"].astype(str).to_numpy()
            train_ids, val_ids, test_ids = _stratified_split_ids(
                ids, labs, rng, args.train_ratio, args.val_ratio, args.test_ratio
            )
        else:
            rng.shuffle(ids)
            n = len(ids)
            n_train = int(round(args.train_ratio * n))
            n_val = int(round(args.val_ratio * n))
            n_train = min(n_train, n)
            n_val = min(n_val, n - n_train)
            train_ids = ids[:n_train]
            val_ids = ids[n_train:n_train + n_val]
            test_ids = ids[n_train + n_val:]

        df["__split"] = "UNASSIGNED"
        df.loc[df[args.id_col].astype(str).isin(set(train_ids)), "__split"] = "train"
        df.loc[df[args.id_col].astype(str).isin(set(val_ids)), "__split"] = "val"
        df.loc[df[args.id_col].astype(str).isin(set(test_ids)), "__split"] = "test"

    else:
        # row-level fallback
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        if has_strat:
            # stratify rows by label
            df["__split"] = "UNASSIGNED"
            for cls, sub in df.groupby(args.stratify_col):
                idx = sub.index.to_numpy()
                rng.shuffle(idx)
                n = len(idx)
                n_train = int(round(args.train_ratio * n))
                n_val = int(round(args.val_ratio * n))
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
            n_train = int(round(args.train_ratio * n))
            n_val = int(round(args.val_ratio * n))
            df["__split"] = "test"
            df.loc[:n_train - 1, "__split"] = "train"
            df.loc[n_train:n_train + n_val - 1, "__split"] = "val"

    # sanity
    assert (df["__split"] != "UNASSIGNED").all(), "Some rows were not assigned to a split."

    # write split csvs (keep all columns)
    for split in ["train", "val", "test"]:
        out_path = out_dir / f"{split}.csv"
        df[df["__split"] == split].drop(columns=["__split"]).to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({(df['__split'] == split).sum()} rows)")

if __name__ == "__main__":
    main()
