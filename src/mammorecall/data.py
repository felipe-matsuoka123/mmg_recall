from __future__ import annotations

import csv
import io
import random
import zipfile
from abc import ABC, abstractmethod
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


def read_metadata_from_source(
    *,
    data_zip: str | Path | None = None,
    data_root: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    member: str = "metadata.csv",
) -> list[dict[str, str]]:
    if metadata_csv:
        return read_csv_rows(metadata_csv)
    if data_zip and data_root:
        raise ValueError("Pass only one of data_zip or data_root.")
    if data_zip:
        return read_metadata_rows(data_zip, member)
    if data_root:
        return read_csv_rows(Path(data_root) / member)
    raise ValueError("Pass data_zip or data_root.")


def attach_labels(
    rows: list[dict[str, str]],
    labels_csv: str | Path | None,
    *,
    label_key: str,
    label_col: str,
    row_key: str | None = None,
    labels_dataset_col: str | None = None,
    dataset_name: str | None = None,
    require_all: bool = True,
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
    if labels_dataset_col and dataset_name:
        label_rows = [
            row for row in label_rows if row.get(labels_dataset_col, "").strip() == dataset_name
        ]
    labels_by_key = {
        row[label_key]: row[label_col]
        for row in label_rows
        if row.get(label_key, "").strip() and row.get(label_col, "").strip()
    }
    missing = []
    merged_rows = []
    row_key = row_key or label_key
    for row in rows:
        key_value = row.get(row_key, "")
        label = labels_by_key.get(key_value)
        if label is None:
            missing.append(key_value)
            continue
        merged_row = dict(row)
        merged_row[label_col] = label
        merged_rows.append(merged_row)

    if missing and require_all:
        examples = ", ".join(repr(key) for key in missing[:3])
        raise ValueError(
            f"Missing labels for {len(missing)} metadata rows using key '{label_key}'. "
            f"First keys: {examples}"
        )
    return merged_rows


def filter_rows_by_dataset(
    rows: list[dict[str, str]],
    *,
    dataset_col: str | None,
    dataset_name: str | None,
) -> list[dict[str, str]]:
    if not rows or not dataset_col or not dataset_name or dataset_col not in rows[0]:
        return rows
    return [row for row in rows if row.get(dataset_col, "").strip() == dataset_name]


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


class PngDataset(Dataset, ABC):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        label_map: dict[str, int],
        label_col: str,
        input_channels: int,
        image_size: int,
    ) -> None:
        self.rows = rows
        self.label_map = label_map
        self.label_col = label_col
        self.input_channels = input_channels
        self.transform = build_transform(image_size, input_channels)

    def __len__(self) -> int:
        return len(self.rows)

    @abstractmethod
    def _open_image(self, row: dict[str, str]) -> Image.Image:
        """Open one processed PNG as a PIL image."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with self._open_image(row) as image:
            mode = "L" if self.input_channels == 1 else "RGB"
            tensor = self.transform(image.convert(mode))

        target = torch.tensor(self.label_map[row[self.label_col]], dtype=torch.long)
        return tensor, target


class ZipPngDataset(PngDataset):
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
        super().__init__(
            rows,
            label_map=label_map,
            label_col=label_col,
            input_channels=input_channels,
            image_size=image_size,
        )
        self.data_zip = Path(data_zip)
        self._archive: zipfile.ZipFile | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def _zip(self) -> zipfile.ZipFile:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.data_zip)
        return self._archive

    def _open_image(self, row: dict[str, str]) -> Image.Image:
        with self._zip().open(row["processed_path"]) as handle:
            return Image.open(handle).copy()


class DirectoryPngDataset(PngDataset):
    def __init__(
        self,
        data_root: str | Path,
        rows: list[dict[str, str]],
        *,
        label_map: dict[str, int],
        label_col: str,
        input_channels: int,
        image_size: int,
    ) -> None:
        super().__init__(
            rows,
            label_map=label_map,
            label_col=label_col,
            input_channels=input_channels,
            image_size=image_size,
        )
        self.data_root = Path(data_root)

    def _open_image(self, row: dict[str, str]) -> Image.Image:
        return Image.open(self.data_root / row["processed_path"])


class MultiSourcePngDataset(PngDataset):
    def __init__(
        self,
        sources: list[dict[str, str | Path | None]],
        rows: list[dict[str, str]],
        *,
        label_map: dict[str, int],
        label_col: str,
        input_channels: int,
        image_size: int,
    ) -> None:
        super().__init__(
            rows,
            label_map=label_map,
            label_col=label_col,
            input_channels=input_channels,
            image_size=image_size,
        )
        self.sources = [
            {
                "data_zip": Path(source["data_zip"]) if source.get("data_zip") else None,
                "data_root": Path(source["data_root"]) if source.get("data_root") else None,
            }
            for source in sources
        ]
        self._archives: dict[int, zipfile.ZipFile] = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_archives"] = {}
        return state

    def _zip(self, source_index: int) -> zipfile.ZipFile:
        if source_index not in self._archives:
            data_zip = self.sources[source_index]["data_zip"]
            if data_zip is None:
                raise ValueError(f"Source {source_index} does not have data_zip.")
            self._archives[source_index] = zipfile.ZipFile(data_zip)
        return self._archives[source_index]

    def _open_image(self, row: dict[str, str]) -> Image.Image:
        source_index = int(row["_source_index"])
        source = self.sources[source_index]
        if source["data_zip"]:
            with self._zip(source_index).open(row["processed_path"]) as handle:
                return Image.open(handle).copy()
        data_root = source["data_root"]
        if data_root is None:
            raise ValueError(f"Source {source_index} does not have data_root.")
        return Image.open(data_root / row["processed_path"])
