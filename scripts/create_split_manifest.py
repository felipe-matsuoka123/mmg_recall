#!/usr/bin/env python3
"""Create a fixed patient-grouped train/validation/test split manifest."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--label-col", default="target")
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--study-col", default="study_id")
    parser.add_argument("--fallback-group-col", default="accession_number")
    parser.add_argument("--image-col", default="image_id")
    parser.add_argument("--split-col", default="experiment_split")
    parser.add_argument("--group-col", default="split_group")
    parser.add_argument("--test-dataset", default="spr")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def group_id(row: dict[str, str], args: argparse.Namespace) -> str:
    dataset = row.get(args.dataset_col, "").strip() or "unknown"
    for column in (args.patient_col, args.study_col, args.fallback_group_col, args.image_col):
        value = row.get(column, "").strip()
        if value:
            return f"{dataset}:{value}"
    raise ValueError(f"Row has no usable group key: {row}")


def group_target(rows: list[dict[str, str]], label_col: str) -> str:
    labels = {row.get(label_col, "").strip() for row in rows}
    labels.discard("")
    if not labels:
        raise ValueError(f"Group has no labels in column {label_col!r}")
    if labels <= {"0", "1"}:
        return "1" if "1" in labels else "0"
    return "|".join(sorted(labels))


def take_fraction(group_ids: list[str], fraction: float, rng: random.Random) -> set[str]:
    if fraction <= 0 or len(group_ids) < 2:
        return set()
    count = round(len(group_ids) * fraction)
    count = max(1, min(count, len(group_ids) - 1))
    shuffled = list(group_ids)
    rng.shuffle(shuffled)
    return set(shuffled[:count])


def assign_splits(
    grouped_rows: dict[str, list[dict[str, str]]],
    *,
    args: argparse.Namespace,
) -> dict[str, str]:
    rng = random.Random(args.seed)
    group_info = {
        group: (
            rows[0].get(args.dataset_col, "").strip() or "unknown",
            group_target(rows, args.label_col),
        )
        for group, rows in grouped_rows.items()
    }
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group, key in group_info.items():
        buckets[key].append(group)

    assignments = {group: "train" for group in grouped_rows}
    for (dataset, _target), group_ids in sorted(buckets.items()):
        if dataset == args.test_dataset:
            for group in take_fraction(group_ids, args.test_fraction, rng):
                assignments[group] = "test"

    val_buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group, (dataset, target) in group_info.items():
        if assignments[group] != "test":
            val_buckets[(dataset, target)].append(group)
    for group_ids in val_buckets.values():
        for group in take_fraction(group_ids, args.val_fraction, rng):
            assignments[group] = "val"

    return assignments


def print_summary(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    counts = Counter(
        (
            row.get(args.split_col, ""),
            row.get(args.dataset_col, ""),
            row.get(args.label_col, ""),
        )
        for row in rows
    )
    for (split, dataset, label), count in sorted(counts.items()):
        print(f"{split:5s} dataset={dataset or 'unknown':7s} label={label or 'missing':7s} rows={count}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in [0, 1).")
    if not 0.0 <= args.test_fraction < 1.0:
        raise SystemExit("--test-fraction must be in [0, 1).")

    rows, fieldnames = read_csv_rows(args.labels_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.labels_csv}")
    for column in (args.dataset_col, args.label_col, args.image_col):
        if column not in fieldnames:
            raise SystemExit(f"Missing required column {column!r} in {args.labels_csv}")

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[group_id(row, args)].append(row)

    assignments = assign_splits(grouped_rows, args=args)
    output_rows = []
    for row in rows:
        row = dict(row)
        split_group = group_id(row, args)
        row[args.group_col] = split_group
        row[args.split_col] = assignments[split_group]
        output_rows.append(row)

    output_fields = list(fieldnames)
    for column in (args.group_col, args.split_col):
        if column not in output_fields:
            output_fields.append(column)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"wrote {len(output_rows)} rows across {len(grouped_rows)} groups to {args.output_csv}")
    print_summary(output_rows, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
