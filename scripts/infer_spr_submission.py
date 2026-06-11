#!/usr/bin/env python3
"""Run SPR test-set inference and write a sample-submission shaped CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mammorecall.data import build_transform  # noqa: E402
from mammorecall.models import build_model  # noqa: E402
from mammo_preprocessing import ImageVariant, preprocess_dicom_payloads, write_failures_csv  # noqa: E402


class SubmissionPngDataset(Dataset):
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, str]:
        row = self.rows[index]
        with Image.open(row["png_path"]) as image:
            mode = "L" if self.input_channels == 1 else "RGB"
            tensor = self.transform(image.convert(mode))
        return tensor, row["AccessionNumber"], row["dicom_path"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-submission", type=Path, default=Path("sample_submissionA.csv"))
    parser.add_argument(
        "--spr-dicom-dir",
        type=Path,
        default=Path("/media/felipe/KINGSTON/datasets/SPR_Mammo_Recall"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=[ImageVariant.GRAYSCALE.value, ImageVariant.RGB_MULTIWINDOW.value],
        required=True,
    )
    parser.add_argument(
        "--cache-variant",
        action="append",
        choices=[ImageVariant.GRAYSCALE.value, ImageVariant.RGB_MULTIWINDOW.value],
        default=[],
        help="Additional PNG variant to cache while reading each DICOM.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/spr_test_preprocessed"))
    parser.add_argument("--failure-report", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "resnet18", "convnext_tiny", "convnext_small"],
        default=None,
    )
    parser.add_argument("--input-channels", type=int, choices=[1, 3], default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preprocess-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--positive-label", default="1")
    parser.add_argument("--aggregate", choices=["mean", "max"], default="mean")
    parser.add_argument("--missing-score", type=float, default=0.5)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return parser.parse_args(argv)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def sample_accessions(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["AccessionNumber", "target"]:
            raise ValueError(
                f"Expected sample columns ['AccessionNumber', 'target'], got {reader.fieldnames}"
            )
        return [row["AccessionNumber"] for row in reader]


def spr_dicom_paths(root: Path, accession_number: str) -> list[Path]:
    accession_dir = root / accession_number
    if not accession_dir.is_dir():
        return []
    return sorted(
        path
        for path in accession_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".dcm", ".dicom"}
    )


def cache_png_path(cache_dir: Path, variant: str, accession_number: str, dicom_path: Path) -> Path:
    return cache_dir / variant / accession_number / f"{dicom_path.stem}.png"


def preprocess_one(
    accession_number: str,
    dicom_path: Path,
    variant: str,
    cache_variants: tuple[str, ...],
    cache_dir: Path,
    image_size: int,
) -> tuple[dict[str, str] | None, tuple[Path, str] | None]:
    try:
        png_paths = {
            cache_variant: cache_png_path(cache_dir, cache_variant, accession_number, dicom_path)
            for cache_variant in cache_variants
        }
        missing_variants = [
            cache_variant
            for cache_variant, png_path in png_paths.items()
            if not png_path.is_file()
        ]
        if missing_variants:
            payloads, _ = preprocess_dicom_payloads(
                dicom_path,
                variants=tuple(missing_variants),
                output_size=image_size,
            )
            for cache_variant in missing_variants:
                png_path = png_paths[cache_variant]
                png_path.parent.mkdir(parents=True, exist_ok=True)
                png_path.write_bytes(payloads[cache_variant])

        target_png_path = png_paths[variant]
        return (
            {
                "AccessionNumber": accession_number,
                "dicom_path": str(dicom_path),
                "png_path": str(target_png_path),
            },
            None,
        )
    except Exception as exc:
        return None, (dicom_path, str(exc))


def preprocess_rows(
    *,
    accessions: Sequence[str],
    spr_dicom_dir: Path,
    variant: str,
    cache_dir: Path,
    image_size: int,
    failure_report: Path | None,
    workers: int,
    cache_variants: Sequence[str],
) -> tuple[list[dict[str, str]], list[str]]:
    if workers <= 0:
        raise ValueError("--preprocess-workers must be positive")

    rows: list[dict[str, str]] = []
    failures: list[tuple[Path, str]] = []
    missing_accessions: list[str] = []
    normalized_cache_variants = tuple(dict.fromkeys((variant, *cache_variants)))
    tasks: list[tuple[str, Path, str, tuple[str, ...], Path, int]] = []

    for accession_number in accessions:
        dicom_paths = spr_dicom_paths(spr_dicom_dir, accession_number)
        if not dicom_paths:
            missing_accessions.append(accession_number)
            continue

        for dicom_path in dicom_paths:
            tasks.append(
                (
                    accession_number,
                    dicom_path,
                    variant,
                    normalized_cache_variants,
                    cache_dir,
                    image_size,
                )
            )

    successful_accessions: set[str] = set()
    if workers == 1:
        iterator = (
            preprocess_one(
                accession_number,
                dicom_path,
                task_variant,
                task_cache_variants,
                task_cache_dir,
                task_image_size,
            )
            for (
                accession_number,
                dicom_path,
                task_variant,
                task_cache_variants,
                task_cache_dir,
                task_image_size,
            ) in tasks
        )
        for row, failure in tqdm(
            iterator,
            total=len(tasks),
            desc=f"Preprocessing {variant}",
            unit="image",
        ):
            if row is not None:
                rows.append(row)
                successful_accessions.add(row["AccessionNumber"])
            if failure is not None:
                failures.append(failure)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(preprocess_one, *task) for task in tasks]
            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Preprocessing {variant} ({workers} workers)",
                unit="image",
            )
            for future in progress:
                row, failure = future.result()
                if row is not None:
                    rows.append(row)
                    successful_accessions.add(row["AccessionNumber"])
                if failure is not None:
                    failures.append(failure)

    for accession_number in accessions:
        if accession_number not in successful_accessions and accession_number not in missing_accessions:
            missing_accessions.append(accession_number)

    if failures and failure_report is not None:
        write_failures_csv(failure_report, failures)
    if failures:
        print(f"{len(failures)} DICOM files failed preprocessing", file=sys.stderr)
    return rows, missing_accessions


def load_checkpoint_config(checkpoint_path: Path, device: torch.device) -> tuple[dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config") or {}
    label_map = checkpoint.get("label_map") or {}
    if not label_map:
        label_map_path = checkpoint_path.with_name("label_map.json")
        if label_map_path.is_file():
            label_map = json.loads(label_map_path.read_text())
    return checkpoint, {"config": config, "label_map": label_map}


def resolve_model_settings(args: argparse.Namespace, checkpoint_meta: dict) -> tuple[str, int, int]:
    config = checkpoint_meta["config"]
    model_name = args.model or config.get("model")
    image_size = args.image_size or config.get("image_size")
    input_channels = args.input_channels or config.get("input_channels")
    if model_name is None or image_size is None or input_channels is None:
        raise ValueError(
            "Checkpoint config is missing model/image_size/input_channels; pass them explicitly."
        )
    return str(model_name), int(image_size), int(input_channels)


def positive_class_index(label_map: dict[str, int], positive_label: str) -> int:
    if positive_label in label_map:
        return int(label_map[positive_label])
    if len(label_map) == 2:
        return max(int(index) for index in label_map.values())
    raise ValueError(f"Positive label {positive_label!r} not found in label map {label_map}")


def predict_rows(
    *,
    rows: list[dict[str, str]],
    checkpoint: dict,
    checkpoint_meta: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[float]]:
    model_name, image_size, input_channels = resolve_model_settings(args, checkpoint_meta)
    label_map = checkpoint_meta["label_map"]
    if not label_map:
        raise ValueError("Could not resolve label_map from checkpoint or label_map.json")
    pos_index = positive_class_index(label_map, args.positive_label)

    model = build_model(
        model_name,
        input_channels=input_channels,
        num_classes=len(label_map),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = SubmissionPngDataset(rows, input_channels=input_channels, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    scores_by_accession: dict[str, list[float]] = defaultdict(list)
    with torch.inference_mode():
        for inputs, accessions, _ in tqdm(loader, desc="Predicting", unit="batch"):
            probs = torch.softmax(model(inputs.to(device)), dim=1)[:, pos_index]
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = select_device(args.device)
    accessions = sample_accessions(args.sample_submission)
    checkpoint, checkpoint_meta = load_checkpoint_config(args.checkpoint, device)
    _, image_size, _ = resolve_model_settings(args, checkpoint_meta)
    rows, missing_accessions = preprocess_rows(
        accessions=accessions,
        spr_dicom_dir=args.spr_dicom_dir,
        variant=args.variant,
        cache_dir=args.cache_dir,
        image_size=image_size,
        failure_report=args.failure_report,
        workers=args.preprocess_workers,
        cache_variants=args.cache_variant,
    )
    if not rows:
        raise SystemExit("No usable DICOM images were found for inference.")

    scores_by_accession = predict_rows(
        rows=rows,
        checkpoint=checkpoint,
        checkpoint_meta=checkpoint_meta,
        args=args,
        device=device,
    )
    write_submission(
        accessions=accessions,
        scores_by_accession=scores_by_accession,
        output_path=args.output,
        aggregate_method=args.aggregate,
        missing_score=args.missing_score,
    )
    scored = sum(1 for accession_number in accessions if accession_number in scores_by_accession)
    print(
        f"Wrote {args.output} with {scored}/{len(accessions)} scored accessions; "
        f"{len(missing_accessions)} used missing_score={args.missing_score}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
