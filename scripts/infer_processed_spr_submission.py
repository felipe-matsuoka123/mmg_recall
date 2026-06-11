#!/usr/bin/env python3
"""Run inference from already-processed SPR PNGs and write submission CSVs.

Expected processed layout by default:

    <processed-root>/grayscale/<AccessionNumber>/*.png
    <processed-root>/rgb_multiwindow/<AccessionNumber>/*.png

Use --grayscale-dir and --rgb-dir if your directories have different names.
The output CSVs keep the same row order and columns as sample_submissionA.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mammorecall.data import build_transform  # noqa: E402
from mammorecall.models import build_model  # noqa: E402


PNG_EXTENSIONS = {".png"}


class ProcessedPngDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        input_channels: int,
        image_size: int,
    ) -> None:
        self.rows = rows
        self.input_channels = input_channels
        self.transform = build_transform(image_size, input_channels)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.rows[index]
        with Image.open(row["png_path"]) as image:
            mode = "L" if self.input_channels == 1 else "RGB"
            tensor = self.transform(image.convert(mode))
        return tensor, row["AccessionNumber"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-submission", type=Path, default=Path("sample_submissionA.csv"))
    parser.add_argument("--processed-root", type=Path, default=Path("processed_datasets/spr"))
    parser.add_argument("--grayscale-dir", type=Path, default=None)
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument(
        "--grayscale-checkpoint",
        type=Path,
        default=Path("runs/all_grayscale_convnext_tiny_1024/best.pt"),
    )
    parser.add_argument(
        "--rgb-checkpoint",
        type=Path,
        default=Path("runs/all_rgb_multiwindow_convnext_tiny_1024_weighed_loss/best.pt"),
    )
    parser.add_argument(
        "--grayscale-output",
        type=Path,
        default=Path("outputs/submissions/spr_test_grayscale_submission.csv"),
    )
    parser.add_argument(
        "--rgb-output",
        type=Path,
        default=Path("outputs/submissions/spr_test_rgb_multiwindow_submission.csv"),
    )
    parser.add_argument("--skip-grayscale", action="store_true")
    parser.add_argument("--skip-rgb", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--positive-label", default="1")
    parser.add_argument("--aggregate", choices=["mean", "max"], default="mean")
    parser.add_argument("--missing-score", type=float, default=0.5)
    parser.add_argument("--require-all-accessions", action="store_true")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return parser.parse_args(argv)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def read_sample_accessions(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["AccessionNumber", "target"]:
            raise ValueError(
                f"Expected sample columns ['AccessionNumber', 'target'], got {reader.fieldnames}"
            )
        return [row["AccessionNumber"] for row in reader]


def find_png_rows(
    accessions: Sequence[str],
    processed_dir: Path,
    *,
    require_all_accessions: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for accession_number in accessions:
        accession_dir = processed_dir / accession_number
        png_paths = (
            sorted(path for path in accession_dir.iterdir() if path.suffix.lower() in PNG_EXTENSIONS)
            if accession_dir.is_dir()
            else []
        )
        if not png_paths:
            missing.append(accession_number)
            continue
        rows.extend(
            {"AccessionNumber": accession_number, "png_path": str(path)}
            for path in png_paths
        )

    if missing and require_all_accessions:
        examples = ", ".join(missing[:5])
        raise ValueError(
            f"Missing processed PNGs for {len(missing)} accessions under {processed_dir}. "
            f"Examples: {examples}"
        )
    return rows, missing


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[dict, dict, dict[str, int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config") or {}
    label_map = checkpoint.get("label_map") or {}
    if not label_map:
        label_map_path = checkpoint_path.with_name("label_map.json")
        if label_map_path.is_file():
            label_map = json.loads(label_map_path.read_text())
    if not label_map:
        raise ValueError(f"Could not resolve label_map from {checkpoint_path}")
    return checkpoint, config, {str(label): int(index) for label, index in label_map.items()}


def positive_class_index(label_map: dict[str, int], positive_label: str) -> int:
    if positive_label in label_map:
        return label_map[positive_label]
    if len(label_map) == 2:
        return max(label_map.values())
    raise ValueError(f"Positive label {positive_label!r} not found in label map {label_map}")


def checkpoint_setting(config: dict, key: str, checkpoint_path: Path) -> object:
    value = config.get(key)
    if value is None:
        raise ValueError(f"Checkpoint {checkpoint_path} config is missing {key!r}")
    return value


def predict_variant(
    *,
    rows: list[dict[str, str]],
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    positive_label: str,
) -> dict[str, list[float]]:
    checkpoint, config, label_map = load_checkpoint(checkpoint_path, device)
    model_name = str(checkpoint_setting(config, "model", checkpoint_path))
    image_size = int(checkpoint_setting(config, "image_size", checkpoint_path))
    input_channels = int(checkpoint_setting(config, "input_channels", checkpoint_path))
    pos_index = positive_class_index(label_map, positive_label)

    model = build_model(
        model_name,
        input_channels=input_channels,
        num_classes=len(label_map),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = ProcessedPngDataset(
        rows,
        input_channels=input_channels,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    scores_by_accession: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for inputs, accessions in tqdm(loader, desc=f"Predicting {checkpoint_path.parent.name}"):
            probs = torch.softmax(model(inputs.to(device, non_blocking=True)), dim=1)[:, pos_index]
            for accession_number, score in zip(accessions, probs.detach().cpu().tolist(), strict=True):
                scores_by_accession[accession_number].append(float(score))
    return scores_by_accession


def aggregate(scores: list[float], method: str) -> float:
    if method == "mean":
        return sum(scores) / len(scores)
    if method == "max":
        return max(scores)
    raise ValueError(f"Unsupported aggregate={method!r}")


def write_submission(
    *,
    accessions: Sequence[str],
    scores_by_accession: dict[str, list[float]],
    output_path: Path,
    aggregate_method: str,
    missing_score: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["AccessionNumber", "target"])
        writer.writeheader()
        for accession_number in accessions:
            scores = scores_by_accession.get(accession_number)
            target = aggregate(scores, aggregate_method) if scores else missing_score
            writer.writerow({"AccessionNumber": accession_number, "target": f"{target:.8f}"})


def run_one(
    *,
    name: str,
    accessions: Sequence[str],
    processed_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    rows, missing = find_png_rows(
        accessions,
        processed_dir,
        require_all_accessions=args.require_all_accessions,
    )
    if not rows:
        raise SystemExit(f"No processed PNGs found for {name} under {processed_dir}")

    scores_by_accession = predict_variant(
        rows=rows,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        positive_label=args.positive_label,
    )
    write_submission(
        accessions=accessions,
        scores_by_accession=scores_by_accession,
        output_path=output_path,
        aggregate_method=args.aggregate,
        missing_score=args.missing_score,
    )
    scored = sum(1 for accession_number in accessions if accession_number in scores_by_accession)
    print(
        f"{name}: wrote {output_path} with {scored}/{len(accessions)} scored accessions; "
        f"{len(missing)} used missing_score={args.missing_score}."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = select_device(args.device)
    accessions = read_sample_accessions(args.sample_submission)

    grayscale_dir = args.grayscale_dir or args.processed_root / "grayscale"
    rgb_dir = args.rgb_dir or args.processed_root / "rgb_multiwindow"

    if args.skip_grayscale and args.skip_rgb:
        raise ValueError("At least one variant must be enabled.")

    if not args.skip_grayscale:
        run_one(
            name="grayscale",
            accessions=accessions,
            processed_dir=grayscale_dir,
            checkpoint_path=args.grayscale_checkpoint,
            output_path=args.grayscale_output,
            args=args,
            device=device,
        )
    if not args.skip_rgb:
        run_one(
            name="rgb-multiwindow",
            accessions=accessions,
            processed_dir=rgb_dir,
            checkpoint_path=args.rgb_checkpoint,
            output_path=args.rgb_output,
            args=args,
            device=device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
