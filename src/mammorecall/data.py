from __future__ import annotations

import csv
import io
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_metadata_rows(data_zip: str | Path, member: str = "metadata.csv") -> list[dict[str, str]]:
    with zipfile.ZipFile(data_zip) as archive:
        with archive.open(member) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            return list(csv.DictReader(text))


def attach_labels(
    rows: list[dict[str, str]],
    labels_csv: str | Path | None,
    *,
    label_key: str,
    label_col: str,
) -> list[dict[str, str]]:
    if labels_csv is None:
        labeled = [row for row in rows if row.get(label_col, "").strip()]
        if len(labeled) != len(rows):
            raise ValueError(
                f"{len(rows) - len(labeled)} metadata rows have empty '{label_col}'. "
                "Populate metadata.csv or pass --labels-csv."
            )
        return rows

    label_rows = read_csv_rows(labels_csv)
    labels_by_key = {
        row[label_key]: row[label_col]
        for row in label_rows
        if row.get(label_key, "").strip() and row.get(label_col, "").strip()
    }
    missing = []
    merged_rows = []
    for row in rows:
        row_key = row.get(label_key, "")
        label = labels_by_key.get(row_key)
        if label is None:
            missing.append(row_key)
            continue
        merged_row = dict(row)
        merged_row[label_col] = label
        merged_rows.append(merged_row)

    if missing:
        examples = ", ".join(repr(key) for key in missing[:3])
        raise ValueError(
            f"Missing labels for {len(missing)} metadata rows using key '{label_key}'. "
            f"First keys: {examples}"
        )
    return merged_rows


def build_label_map(rows: list[dict[str, str]], label_col: str) -> dict[str, int]:
    labels = sorted({row[label_col] for row in rows})
    if len(labels) < 2:
        raise ValueError(f"Expected at least two labels in '{label_col}', got {labels}")
    return {label: index for index, label in enumerate(labels)}


def split_rows_by_group(
    rows: list[dict[str, str]],
    *,
    group_col: str,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        group_id = row.get(group_col, "").strip() or f"__row_{index}"
        grouped_rows[group_id].append(row)

    group_ids = list(grouped_rows)
    random.Random(seed).shuffle(group_ids)
    val_count = max(1, round(len(group_ids) * val_fraction))
    val_group_ids = set(group_ids[:val_count])

    train_rows = []
    val_rows = []
    for group_id, group_rows in grouped_rows.items():
        (val_rows if group_id in val_group_ids else train_rows).extend(group_rows)

    if not train_rows or not val_rows:
        raise ValueError(
            "Patient split produced an empty partition. "
            "Use more patients or change --val-fraction."
        )
    return train_rows, val_rows


def build_transform(image_size: int, input_channels: int) -> transforms.Compose:
    if input_channels not in {1, 3}:
        raise ValueError(f"input_channels must be 1 or 3, got {input_channels}")
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * input_channels, [0.5] * input_channels),
        ]
    )


class ZipPngDataset(Dataset):
    def __init__(
        self,
        data_zip: str | Path,
        rows: list[dict[str, str]],
        *,
        label_map: dict[str, int],
        label_col: str,
        input_channels: int,
        image_size: int,
    ) -> None:
        self.data_zip = Path(data_zip)
        self.rows = rows
        self.label_map = label_map
        self.label_col = label_col
        self.input_channels = input_channels
        self.transform = build_transform(image_size, input_channels)
        self._archive: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def _zip(self) -> zipfile.ZipFile:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.data_zip)
        return self._archive

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with self._zip().open(row["processed_path"]) as handle:
            with Image.open(handle) as image:
                mode = "L" if self.input_channels == 1 else "RGB"
                tensor = self.transform(image.convert(mode))

        target = torch.tensor(self.label_map[row[self.label_col]], dtype=torch.long)
        return tensor, target
