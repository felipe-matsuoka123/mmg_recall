#!/usr/bin/env python3
"""Convert mammography DICOM files into cropped 8-bit PNGs in zip archives."""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from enum import Enum
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pydicom
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - depends on the runtime environment.
    raise SystemExit(
        "DICOM preprocessing requires numpy, pydicom, opencv-python, and tqdm."
    ) from exc


def invert_if_needed(pixels: np.ndarray, dicom: pydicom.Dataset) -> np.ndarray:
    """Make bright breast tissue consistently bright for MONOCHROME1 inputs."""
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        return pixels.max() - pixels
    return pixels


def create_breast_mask_u8(image_u8: np.ndarray) -> np.ndarray:
    """Return a 0/255 breast silhouette mask using the notebook workflow."""
    blurred = cv2.GaussianBlur(image_u8, (5, 5), 0)
    _, foreground = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)

    height, width = foreground.shape
    inverted_foreground = cv2.bitwise_not(foreground)
    background = np.zeros_like(inverted_foreground)
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)

    seeds = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    for seed_x, seed_y in seeds:
        if inverted_foreground[seed_y, seed_x] == 255:
            flood_source = inverted_foreground.copy()
            flood_mask[:] = 0
            cv2.floodFill(flood_source, flood_mask, (seed_x, seed_y), 128)
            background |= ((flood_source == 128).astype(np.uint8) * 255)

    breast = cv2.bitwise_not(background)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    breast = cv2.morphologyEx(breast, cv2.MORPH_CLOSE, close_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (breast > 0).astype(np.uint8), connectivity=8
    )
    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        breast = (labels == largest).astype(np.uint8) * 255

    inverted_breast = cv2.bitwise_not(breast)
    hole_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(inverted_breast, hole_mask, (0, 0), 128)
    holes = ((inverted_breast != 128).astype(np.uint8) * 255)
    breast = cv2.bitwise_or(breast, holes)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.dilate(breast, dilation_kernel, iterations=1)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, mask.shape[1], mask.shape[0])
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


def window_u8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    clipped = np.clip(image, low, high)
    normalized = (clipped - low) / (high - low + 1e-6)
    return (normalized * 255.0).astype(np.uint8)


METADATA_FIELDS = [
    "image_id",
    "patient_id",
    "study_uid",
    "series_uid",
    "laterality",
    "view_position",
    "label",
    "manufacturer",
    "pixel_spacing",
    "bits_stored",
    "original_height",
    "original_width",
    "crop_xmin",
    "crop_ymin",
    "crop_xmax",
    "crop_ymax",
    "processed_path",
]


class ImageVariant(str, Enum):
    GRAYSCALE = "grayscale"
    RGB_REPLICATED = "rgb-replicated"
    RGB_MULTIWINDOW = "rgb-multiwindow"


def dicom_value(dicom: pydicom.Dataset, keyword: str, default: str = "") -> str:
    value = getattr(dicom, keyword, default)
    return str(value).strip() if value is not None else default


def pixel_spacing_value(dicom: pydicom.Dataset) -> str:
    spacing = getattr(dicom, "PixelSpacing", None)
    if spacing is None:
        spacing = getattr(dicom, "ImagerPixelSpacing", None)
    if spacing is None:
        return ""
    if isinstance(spacing, str):
        return spacing.strip()
    return "\\".join(str(value).strip() for value in spacing)


def image_metadata(
    dicom: pydicom.Dataset,
    *,
    original_shape: tuple[int, int],
    crop_bbox_xywh: tuple[int, int, int, int],
) -> dict[str, str | int]:
    x, y, width, height = crop_bbox_xywh
    original_height, original_width = original_shape
    laterality = dicom_value(dicom, "ImageLaterality") or dicom_value(dicom, "Laterality")

    return {
        "image_id": dicom_value(dicom, "SOPInstanceUID"),
        "patient_id": dicom_value(dicom, "PatientID"),
        "study_uid": dicom_value(dicom, "StudyInstanceUID"),
        "series_uid": dicom_value(dicom, "SeriesInstanceUID"),
        "laterality": laterality,
        "view_position": dicom_value(dicom, "ViewPosition"),
        "label": "",
        "manufacturer": dicom_value(dicom, "Manufacturer"),
        "pixel_spacing": pixel_spacing_value(dicom),
        "bits_stored": dicom_value(dicom, "BitsStored"),
        "original_height": original_height,
        "original_width": original_width,
        "crop_xmin": x,
        "crop_ymin": y,
        "crop_xmax": x + width,
        "crop_ymax": y + height,
        "processed_path": "",
    }


def create_u8_images(
    cropped: np.ndarray,
    variants: Sequence[ImageVariant | str],
) -> dict[ImageVariant, np.ndarray]:
    """Create one or more PNG payloads while sharing crop windowing work."""
    requested_variants = tuple(ImageVariant(variant) for variant in variants)
    if not requested_variants:
        raise ValueError("At least one image variant is required")

    base_low, base_high = np.percentile(cropped, (0.5, 99.5))
    base_image = window_u8(cropped, float(base_low), float(base_high))
    images = {}
    if ImageVariant.GRAYSCALE in requested_variants:
        images[ImageVariant.GRAYSCALE] = base_image
    if ImageVariant.RGB_REPLICATED in requested_variants:
        images[ImageVariant.RGB_REPLICATED] = np.repeat(base_image[..., None], 3, axis=2)

    if ImageVariant.RGB_MULTIWINDOW in requested_variants:
        bright_low, bright_high = np.percentile(cropped, (95, 99.9))
        tissue_low, tissue_high = np.percentile(cropped, (10, 90))
        bright_image = window_u8(cropped, float(bright_low), float(bright_high))
        tissue_image = window_u8(cropped, float(tissue_low), float(tissue_high))
        images[ImageVariant.RGB_MULTIWINDOW] = np.dstack(
            (base_image, bright_image, tissue_image)
        )

    return {variant: images[variant] for variant in requested_variants}


def create_u8_image(cropped: np.ndarray, variant: ImageVariant | str) -> np.ndarray:
    image_variant = ImageVariant(variant)
    return create_u8_images(cropped, (image_variant,))[image_variant]


def read_crop_and_metadata(
    dicom_path: str | Path,
) -> tuple[np.ndarray, dict[str, str | int]]:
    """Read one DICOM and return its crop before PNG windowing plus metadata."""
    dicom = pydicom.dcmread(str(dicom_path), force=True)
    raw = np.squeeze(dicom.pixel_array).astype(np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {raw.shape}")

    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
    image = invert_if_needed(raw * slope + intercept, dicom)

    view_u8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask_u8 = create_breast_mask_u8(view_u8)
    x, y, width, height = bbox_from_mask(mask_u8)
    cropped = image[y : y + height, x : x + width]
    if cropped.size == 0:
        raise ValueError("Computed an empty breast crop")

    metadata = image_metadata(
        dicom,
        original_shape=raw.shape,
        crop_bbox_xywh=(x, y, width, height),
    )
    return cropped, metadata


def create_u8_crop_and_metadata(
    dicom_path: str | Path,
    variant: ImageVariant | str = ImageVariant.GRAYSCALE,
) -> tuple[np.ndarray, dict[str, str | int]]:
    """Read one DICOM and return the requested cropped PNG payload plus metadata."""
    cropped, metadata = read_crop_and_metadata(dicom_path)
    return create_u8_image(cropped, ImageVariant(variant)), metadata


def create_u8_crop(dicom_path: str | Path) -> np.ndarray:
    image_u8, _ = create_u8_crop_and_metadata(dicom_path)
    return image_u8


def discover_dicom_files(dicom_dir: str | Path) -> list[Path]:
    input_dir = Path(dicom_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"DICOM directory does not exist: {input_dir}")

    return sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".dcm"
    )


def png_archive_path(dicom_path: Path, dicom_dir: Path, png_root: str | None) -> Path:
    relative_png = dicom_path.relative_to(dicom_dir).with_suffix(".png")
    return Path(png_root) / relative_png if png_root else relative_png


def encode_png(image_u8: np.ndarray) -> bytes:
    png_image = image_u8
    if image_u8.ndim == 3:
        # cv2 encodes 3-channel arrays as BGR; flip channel order so PNG readers see RGB.
        png_image = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)

    ok, encoded = cv2.imencode(".png", png_image)
    if not ok:
        raise RuntimeError("cv2.imencode returned false")
    return encoded.tobytes()


def write_metadata_csv(
    archive: zipfile.ZipFile,
    metadata_rows: Sequence[dict[str, str | int]],
) -> None:
    metadata_buffer = io.StringIO()
    metadata_writer = csv.DictWriter(metadata_buffer, fieldnames=METADATA_FIELDS)
    metadata_writer.writeheader()
    metadata_writer.writerows(metadata_rows)
    archive.writestr("metadata.csv", metadata_buffer.getvalue())


def report_failures(processed: int, total: int, failures: Sequence[str]) -> None:
    if not failures:
        return

    print(
        f"Converted {processed}/{total} DICOM files; "
        f"{len(failures)} failed:",
        file=sys.stderr,
    )
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)


def process_dicom_dir_to_png_zip(
    dicom_dir: str | Path,
    output_zip: str | Path,
    *,
    png_root: str | None = None,
    variant: ImageVariant | str = ImageVariant.GRAYSCALE,
    continue_on_error: bool = False,
) -> Path:
    """Write every recursive ``*.dcm`` under ``dicom_dir`` as a PNG in ``output_zip``."""
    input_dir = Path(dicom_dir).resolve()
    output_path = Path(output_zip)
    image_variant = ImageVariant(variant)
    dicom_files = discover_dicom_files(input_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm files found under {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    failures: list[str] = []
    metadata_rows: list[dict[str, str | int]] = []

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for dicom_path in tqdm(
            dicom_files,
            desc=f"Preprocessing {image_variant.value}",
            unit="dicom",
        ):
            try:
                image_u8, metadata = create_u8_crop_and_metadata(dicom_path, image_variant)

                processed_path = png_archive_path(dicom_path, input_dir, png_root).as_posix()
                archive.writestr(
                    processed_path,
                    encode_png(image_u8),
                )
                metadata["processed_path"] = processed_path
                metadata_rows.append(metadata)
                processed += 1
            except Exception as exc:
                if not continue_on_error:
                    raise RuntimeError(f"Failed to preprocess {dicom_path}: {exc}") from exc
                failures.append(f"{dicom_path}: {exc}")

        write_metadata_csv(archive, metadata_rows)

    if not processed:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("No DICOM files were successfully converted")

    report_failures(processed, len(dicom_files), failures)

    return output_path


def process_dicom_dir_to_png_zips(
    dicom_dir: str | Path,
    output_zips: Mapping[ImageVariant | str, str | Path],
    *,
    png_root: str | None = None,
    continue_on_error: bool = False,
) -> dict[ImageVariant, Path]:
    """Write multiple PNG variants while decoding and cropping each DICOM once."""
    input_dir = Path(dicom_dir).resolve()
    output_paths = {
        ImageVariant(variant): Path(output_zip)
        for variant, output_zip in output_zips.items()
    }
    if not output_paths:
        raise ValueError("At least one output zip is required")

    dicom_files = discover_dicom_files(input_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm files found under {input_dir}")

    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    failures: list[str] = []
    metadata_rows = {variant: [] for variant in output_paths}

    with ExitStack() as stack:
        archives = {
            variant: stack.enter_context(
                zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED)
            )
            for variant, output_path in output_paths.items()
        }
        for dicom_path in tqdm(
            dicom_files,
            desc="Preprocessing dataset variants",
            unit="dicom",
        ):
            try:
                cropped, metadata = read_crop_and_metadata(dicom_path)
                images = create_u8_images(cropped, tuple(output_paths))
                processed_path = png_archive_path(dicom_path, input_dir, png_root).as_posix()

                for variant, archive in archives.items():
                    archive.writestr(processed_path, encode_png(images[variant]))
                    variant_metadata = metadata.copy()
                    variant_metadata["processed_path"] = processed_path
                    metadata_rows[variant].append(variant_metadata)
                processed += 1
            except Exception as exc:
                if not continue_on_error:
                    raise RuntimeError(f"Failed to preprocess {dicom_path}: {exc}") from exc
                failures.append(f"{dicom_path}: {exc}")

        for variant, archive in archives.items():
            write_metadata_csv(archive, metadata_rows[variant])

    if not processed:
        for output_path in output_paths.values():
            output_path.unlink(missing_ok=True)
        raise RuntimeError("No DICOM files were successfully converted")

    report_failures(processed, len(dicom_files), failures)
    return output_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop mammography DICOMs and package the PNGs into a zip file."
    )
    parser.add_argument("dicom_dir", type=Path, help="Directory searched recursively for .dcm files.")
    parser.add_argument("output_zip", type=Path, help="Zip file to write.")
    parser.add_argument(
        "--png-root",
        default=None,
        help="Optional top-level folder name for PNG members inside the zip.",
    )
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in ImageVariant],
        default=ImageVariant.GRAYSCALE.value,
        help="PNG channel/windowing variant to write.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip unreadable DICOM files and report them after the zip is written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_zip = process_dicom_dir_to_png_zip(
        args.dicom_dir,
        args.output_zip,
        png_root=args.png_root,
        variant=args.variant,
        continue_on_error=args.continue_on_error,
    )
    print(output_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
