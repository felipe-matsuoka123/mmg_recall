#!/usr/bin/env python3
"""Reusable mammography DICOM preprocessing functions."""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from enum import Enum
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pydicom
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "DICOM preprocessing requires numpy, pydicom, opencv-python, and tqdm."
    ) from exc


DICOM_EXTENSIONS = {".dcm", ".dicom"}
UINT16_MAX = 65535.0
UINT8_TO_UINT16 = UINT16_MAX / 255.0


def invert_if_needed(pixels: np.ndarray, dicom: pydicom.Dataset) -> np.ndarray:
    """Make bright breast tissue consistently bright for inverted DICOM inputs."""
    photometric = str(getattr(dicom, "PhotometricInterpretation", "")).upper()
    presentation = str(getattr(dicom, "PresentationLUTShape", "")).upper()
    if photometric == "MONOCHROME1" or presentation == "INVERSE":
        return np.nanmax(pixels) + np.nanmin(pixels) - pixels
    return pixels


def image_border_pixels(image: np.ndarray) -> np.ndarray:
    border = max(1, min(image.shape) // 20)
    return np.concatenate(
        (
            image[:border, :].ravel(),
            image[-border:, :].ravel(),
            image[:, :border].ravel(),
            image[:, -border:].ravel(),
        )
    )


def ensure_dark_background(image: np.ndarray) -> np.ndarray:
    """Invert images whose border/background is brighter than the breast field."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return image

    border = image_border_pixels(image)
    border = border[np.isfinite(border)]
    if border.size == 0:
        return image

    border_median = float(np.median(border))
    height, width = image.shape
    center = image[height // 4 : height * 3 // 4, width // 4 : width * 3 // 4]
    center = center[np.isfinite(center)]
    if center.size == 0:
        return image

    center_median = float(np.median(center))
    if border_median > center_median:
        return np.nanmax(finite) + np.nanmin(finite) - image
    return image


def largest_component_u8(mask_u8: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_u8 > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask_u8)

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def fill_mask_holes_u8(mask_u8: np.ndarray) -> np.ndarray:
    height, width = mask_u8.shape
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood_source = cv2.bitwise_not(padded)
    flood_mask = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(flood_source, flood_mask, (0, 0), 128)
    holes = ((flood_source != 128).astype(np.uint8) * 255)[1:-1, 1:-1]
    return cv2.bitwise_or(mask_u8, holes)


def candidate_breast_mask_u8(blurred_u8: np.ndarray, threshold: int) -> np.ndarray:
    _, foreground = cv2.threshold(blurred_u8, threshold, 255, cv2.THRESH_BINARY)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, close_kernel)
    breast = largest_component_u8(foreground)
    if not np.any(breast):
        return breast

    breast = fill_mask_holes_u8(breast)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.dilate(breast, dilation_kernel, iterations=1)


def create_breast_mask_u8(image_u8: np.ndarray) -> np.ndarray:
    """Return a 0/255 breast silhouette mask."""
    blurred = cv2.GaussianBlur(image_u8, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thresholds = []
    for threshold in (5, int(otsu_threshold), 10, 15, 20, 30, 40):
        if 0 <= threshold <= 254 and threshold not in thresholds:
            thresholds.append(threshold)

    best_mask = np.zeros_like(image_u8)
    best_area_ratio = 0.0
    for threshold in thresholds:
        candidate = candidate_breast_mask_u8(blurred, threshold)
        area_ratio = float(np.count_nonzero(candidate)) / candidate.size
        if 0.01 <= area_ratio < 0.95:
            return candidate
        if area_ratio > best_area_ratio:
            best_mask = candidate
            best_area_ratio = area_ratio

    return best_mask


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, mask.shape[1], mask.shape[0])
    return cv2.boundingRect(max(contours, key=cv2.contourArea))


def window_u8(image: np.ndarray, low: float, high: float) -> np.ndarray:
    clipped = np.clip(image, low, high)
    normalized = (clipped - low) / (high - low + 1e-6)
    return (normalized * 255.0).astype(np.uint8)


def window_u16(image: np.ndarray, low: float, high: float) -> np.ndarray:
    clipped = np.clip(image, low, high)
    normalized = (clipped - low) / (high - low + 1e-6)
    return (normalized * UINT16_MAX).astype(np.uint16)


def robust_window_values(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return finite

    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        return finite

    preliminary_u8 = window_u8(image, float(low), float(high))
    mask_u8 = create_breast_mask_u8(preliminary_u8)
    area_ratio = float(np.count_nonzero(mask_u8)) / mask_u8.size
    if not 0.01 <= area_ratio < 0.98:
        return finite

    masked = image[(mask_u8 > 0) & np.isfinite(image)]
    return masked if masked.size else finite


def robust_window_limits(
    image: np.ndarray,
    percentiles: tuple[float, float],
) -> tuple[float, float]:
    values = robust_window_values(image)
    if values.size == 0:
        return 0.0, 1.0

    low, high = np.percentile(values, percentiles)
    if high <= low:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return 0.0, 1.0
        low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def robust_window_u8(
    image: np.ndarray,
    percentiles: tuple[float, float],
) -> np.ndarray:
    low, high = robust_window_limits(image, percentiles)
    return window_u8(image, low, high)


def robust_window_u16(
    image: np.ndarray,
    percentiles: tuple[float, float],
) -> np.ndarray:
    low, high = robust_window_limits(image, percentiles)
    return window_u16(image, low, high)


def normalize_for_mask_u8(image: np.ndarray) -> np.ndarray:
    return robust_window_u8(image, (0.5, 99.5))


def u16_to_mask_u8(image_u16: np.ndarray) -> np.ndarray:
    return np.clip(image_u16.astype(np.float32) / UINT8_TO_UINT16, 0, 255).astype(
        np.uint8
    )


def normalize_breast_roi_u16(
    image_u16: np.ndarray,
    target_median: float = 100.0 * UINT8_TO_UINT16,
    target_iqr: float = 70.0 * UINT8_TO_UINT16,
) -> np.ndarray:
    """Mild ROI normalization using only breast pixels."""
    mask_u8 = create_breast_mask_u8(u16_to_mask_u8(image_u16))
    roi = image_u16[mask_u8 > 0].astype(np.float32)

    if roi.size < 100:
        return image_u16

    p25, median, p75 = np.percentile(roi, [25, 50, 75])
    iqr = max(p75 - p25, 1.0)

    normalized = (image_u16.astype(np.float32) - median) * (target_iqr / iqr)
    normalized = normalized + target_median

    return np.clip(normalized, 0, UINT16_MAX).astype(np.uint16)


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


def normalize_variants(variants: Sequence[ImageVariant | str]) -> tuple[ImageVariant, ...]:
    requested_variants = tuple(ImageVariant(variant) for variant in variants)
    if not requested_variants:
        raise ValueError("At least one image variant is required")
    return requested_variants


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
    requested_variants = normalize_variants(variants)

    base_image = normalize_breast_roi_u16(robust_window_u16(cropped, (1.0, 99.0)))

    images = {}
    if ImageVariant.GRAYSCALE in requested_variants:
        images[ImageVariant.GRAYSCALE] = base_image
    if ImageVariant.RGB_REPLICATED in requested_variants:
        images[ImageVariant.RGB_REPLICATED] = np.repeat(
            base_image[..., None],
            3,
            axis=2,
        )

    if ImageVariant.RGB_MULTIWINDOW in requested_variants:
        dense_image = normalize_breast_roi_u16(
            robust_window_u16(cropped, (60.0, 99.5))
        )
        tissue_image = normalize_breast_roi_u16(
            robust_window_u16(cropped, (5.0, 95.0))
        )
        images[ImageVariant.RGB_MULTIWINDOW] = np.dstack(
            (base_image, dense_image, tissue_image)
        )

    return {variant: images[variant] for variant in requested_variants}


def create_u8_image(cropped: np.ndarray, variant: ImageVariant | str) -> np.ndarray:
    image_variant = ImageVariant(variant)
    return create_u8_images(cropped, (image_variant,))[image_variant]


def resize_image_u8(image_u8: np.ndarray, output_size: int | None) -> np.ndarray:
    if output_size is None:
        return image_u8
    if output_size <= 0:
        raise ValueError("output_size must be a positive integer or None")

    height, width = image_u8.shape[:2]
    if (height, width) == (output_size, output_size):
        return image_u8

    interpolation = (
        cv2.INTER_AREA
        if height > output_size or width > output_size
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        image_u8,
        (output_size, output_size),
        interpolation=interpolation,
    )


def read_preprocessing_state(
    dicom_path: str | Path,
) -> tuple[np.ndarray, dict[str, str | int], np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Read one DICOM and return crop state plus intermediate mask images."""
    dicom = pydicom.dcmread(str(dicom_path), force=True)
    raw = np.squeeze(dicom.pixel_array).astype(np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {raw.shape}")

    slope = float(getattr(dicom, "RescaleSlope", 1.0))
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
    image = ensure_dark_background(invert_if_needed(raw * slope + intercept, dicom))

    view_u8 = normalize_for_mask_u8(image)
    mask_u8 = create_breast_mask_u8(view_u8)
    crop_bbox = bbox_from_mask(mask_u8)
    x, y, width, height = crop_bbox
    cropped = image[y : y + height, x : x + width]
    if cropped.size == 0:
        raise ValueError("Computed an empty breast crop")

    metadata = image_metadata(
        dicom,
        original_shape=raw.shape,
        crop_bbox_xywh=crop_bbox,
    )
    return cropped, metadata, view_u8, mask_u8, crop_bbox


def read_crop_and_metadata(
    dicom_path: str | Path,
) -> tuple[np.ndarray, dict[str, str | int]]:
    """Read one DICOM and return its crop before PNG windowing plus metadata."""
    cropped, metadata, _, _, _ = read_preprocessing_state(dicom_path)
    return cropped, metadata


def create_u8_crop_and_metadata(
    dicom_path: str | Path,
    variant: ImageVariant | str = ImageVariant.GRAYSCALE,
    output_size: int | None = None,
) -> tuple[np.ndarray, dict[str, str | int]]:
    """Read one DICOM and return the requested cropped PNG payload plus metadata."""
    cropped, metadata = read_crop_and_metadata(dicom_path)
    image_variant = ImageVariant(variant)
    image_u8 = create_u8_images(cropped, (image_variant,))[image_variant]
    return resize_image_u8(image_u8, output_size), metadata


def create_u8_crop(dicom_path: str | Path) -> np.ndarray:
    image_u8, _ = create_u8_crop_and_metadata(dicom_path)
    return image_u8


def discover_dicom_files(dicom_dir: str | Path) -> list[Path]:
    input_dir = Path(dicom_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"DICOM directory does not exist: {input_dir}")

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DICOM_EXTENSIONS
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


def write_failures_csv(
    failure_report: str | Path,
    failures: Sequence[tuple[Path, str]],
) -> None:
    report_path = Path(failure_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dicom_path", "error"))
        writer.writeheader()
        for dicom_path, error in failures:
            writer.writerow({"dicom_path": dicom_path, "error": error})


def write_png_file(path: str | Path, image_u8: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encode_png(image_u8))
    return output_path


def preprocess_dicom_payloads(
    dicom_path: str | Path,
    variants: Sequence[ImageVariant | str],
    output_size: int | None,
) -> tuple[dict[str, bytes], dict[str, str | int]]:
    cropped, metadata = read_crop_and_metadata(dicom_path)
    requested_variants = normalize_variants(variants)
    images = create_u8_images(cropped, requested_variants)
    payloads = {
        variant.value: encode_png(resize_image_u8(image_u8, output_size))
        for variant, image_u8 in images.items()
    }
    return payloads, metadata


def crop_overlay_image(
    view_u8: np.ndarray,
    mask_u8: np.ndarray,
    crop_bbox_xywh: tuple[int, int, int, int],
) -> np.ndarray:
    overlay = np.repeat(view_u8[..., None], 3, axis=2)
    mask_pixels = mask_u8 > 0
    overlay[mask_pixels] = (
        overlay[mask_pixels].astype(np.float32) * 0.65
        + np.array([0, 255, 0], dtype=np.float32) * 0.35
    ).astype(np.uint8)

    x, y, width, height = crop_bbox_xywh
    thickness = max(2, min(view_u8.shape[:2]) // 200)
    cv2.rectangle(
        overlay,
        (x, y),
        (x + width - 1, y + height - 1),
        (255, 0, 0),
        thickness=thickness,
    )
    return overlay


def png_preview_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint16:
        return np.clip(image.astype(np.float32) / (UINT16_MAX / 255.0), 0, 255).astype(
            np.uint8
        )
    return image


def write_crop_diagnostics(
    output_dir: str | Path,
    sample_name: str,
    cropped: np.ndarray,
    view_u8: np.ndarray,
    mask_u8: np.ndarray,
    crop_bbox_xywh: tuple[int, int, int, int],
) -> list[Path]:
    diagnostics_dir = Path(output_dir) / "diagnostics"
    cropped_preview_u8 = png_preview_u8(create_u8_image(cropped, ImageVariant.GRAYSCALE))
    outputs = [
        write_png_file(diagnostics_dir / f"{sample_name}_01_full_window.png", view_u8),
        write_png_file(diagnostics_dir / f"{sample_name}_02_mask.png", mask_u8),
        write_png_file(
            diagnostics_dir / f"{sample_name}_03_crop_overlay.png",
            crop_overlay_image(view_u8, mask_u8, crop_bbox_xywh),
        ),
        write_png_file(
            diagnostics_dir / f"{sample_name}_04_crop_window.png",
            cropped_preview_u8,
        ),
    ]
    return outputs


def write_variant_channel_diagnostics(
    output_dir: str | Path,
    dataset_name: str,
    sample_name: str,
    image_u8: np.ndarray,
) -> list[Path]:
    if image_u8.ndim != 3:
        return []

    channel_names = ("red_base", "green_dense", "blue_tissue")
    diagnostics_dir = Path(output_dir) / "diagnostics" / dataset_name
    outputs = []
    for channel_index, channel_name in enumerate(channel_names[: image_u8.shape[2]]):
        outputs.append(
            write_png_file(
                diagnostics_dir / f"{sample_name}_{channel_index + 1}_{channel_name}.png",
                image_u8[..., channel_index],
            )
        )
    return outputs


def process_dicom_preview_samples(
    dicom_dir: str | Path,
    output_dir: str | Path,
    variants: Mapping[str, ImageVariant | str],
    *,
    sample_count: int = 3,
    output_size: int | None = 1024,
    write_diagnostics: bool = False,
    continue_on_error: bool = True,
    failure_report: str | Path | None = None,
) -> dict[str, list[Path]]:
    """Write preview PNG samples for each requested variant without creating zips."""
    if sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")

    input_dir = Path(dicom_dir).resolve()
    preview_dir = Path(output_dir)
    image_variants = {name: ImageVariant(variant) for name, variant in variants.items()}
    if not image_variants:
        raise ValueError("At least one image variant is required")

    dicom_files = discover_dicom_files(input_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm or .dicom files found under {input_dir}")

    sample_paths = {name: [] for name in image_variants}
    failures: list[tuple[Path, str]] = []
    processed = 0

    for dicom_path in tqdm(
        dicom_files,
        desc="Writing preview samples",
        unit="dicom",
    ):
        if processed >= sample_count:
            break
        try:
            cropped, _, view_u8, mask_u8, crop_bbox = read_preprocessing_state(dicom_path)
            requested_variants = tuple(image_variants.values())
            images = create_u8_images(cropped, requested_variants)
            processed += 1
            sample_name = f"sample_{processed:03d}_{dicom_path.stem}"

            if write_diagnostics:
                write_crop_diagnostics(
                    preview_dir,
                    sample_name,
                    cropped,
                    view_u8,
                    mask_u8,
                    crop_bbox,
                )

            for dataset_name, variant in image_variants.items():
                image_u8 = resize_image_u8(images[variant], output_size)
                sample_path = preview_dir / dataset_name / f"{sample_name}.png"
                sample_paths[dataset_name].append(write_png_file(sample_path, image_u8))
                if write_diagnostics:
                    write_variant_channel_diagnostics(
                        preview_dir,
                        dataset_name,
                        sample_name,
                        image_u8,
                    )
        except Exception as exc:
            if not continue_on_error:
                raise RuntimeError(f"Failed to preprocess {dicom_path}: {exc}") from exc
            failures.append((dicom_path, str(exc)))

    if not processed:
        if failures and failure_report is not None:
            write_failures_csv(failure_report, failures)
        raise RuntimeError("No DICOM files were successfully converted")

    report_failures(processed, len(dicom_files), failures)
    if failures and failure_report is not None:
        write_failures_csv(failure_report, failures)
    return sample_paths


def report_failures(processed: int, total: int, failures: Sequence[tuple[Path, str]]) -> None:
    if not failures:
        return

    print(
        f"Converted {processed}/{total} DICOM files; "
        f"{len(failures)} failed:",
        file=sys.stderr,
    )
    for dicom_path, error in failures:
        print(f"  {dicom_path}: {error}", file=sys.stderr)


def process_dicom_dir_to_png_zip(
    dicom_dir: str | Path,
    output_zip: str | Path,
    *,
    png_root: str | None = None,
    variant: ImageVariant | str = ImageVariant.GRAYSCALE,
    output_size: int | None = 1024,
    continue_on_error: bool = False,
    failure_report: str | Path | None = None,
) -> Path:
    """Write every recursive ``*.dcm``/``*.dicom`` under ``dicom_dir`` as PNGs."""
    input_dir = Path(dicom_dir).resolve()
    output_path = Path(output_zip)
    image_variant = ImageVariant(variant)
    dicom_files = discover_dicom_files(input_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm or .dicom files found under {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    failures: list[tuple[Path, str]] = []
    metadata_rows: list[dict[str, str | int]] = []

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for dicom_path in tqdm(
            dicom_files,
            desc=f"Preprocessing {image_variant.value}",
            unit="dicom",
        ):
            try:
                image_u8, metadata = create_u8_crop_and_metadata(
                    dicom_path,
                    image_variant,
                    output_size=output_size,
                )

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
                failures.append((dicom_path, str(exc)))

        write_metadata_csv(archive, metadata_rows)

    if not processed:
        output_path.unlink(missing_ok=True)
        if failures and failure_report is not None:
            write_failures_csv(failure_report, failures)
        raise RuntimeError("No DICOM files were successfully converted")

    report_failures(processed, len(dicom_files), failures)
    if failures and failure_report is not None:
        write_failures_csv(failure_report, failures)

    return output_path


def process_dicom_dir_to_png_zips(
    dicom_dir: str | Path,
    output_zips: Mapping[ImageVariant | str, str | Path],
    *,
    png_root: str | None = None,
    output_size: int | None = 1024,
    workers: int = 1,
    continue_on_error: bool = False,
    failure_report: str | Path | None = None,
) -> dict[ImageVariant, Path]:
    """Write multiple PNG variants while decoding and cropping each DICOM once."""
    if workers <= 0:
        raise ValueError("workers must be a positive integer")

    input_dir = Path(dicom_dir).resolve()
    output_paths = {
        ImageVariant(variant): Path(output_zip)
        for variant, output_zip in output_zips.items()
    }
    if not output_paths:
        raise ValueError("At least one output zip is required")

    dicom_files = discover_dicom_files(input_dir)
    if not dicom_files:
        raise FileNotFoundError(f"No .dcm or .dicom files found under {input_dir}")

    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    failures: list[tuple[Path, str]] = []
    metadata_rows = {variant: [] for variant in output_paths}
    variants = tuple(output_paths)

    with ExitStack() as stack:
        archives = {
            variant: stack.enter_context(
                zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED)
            )
            for variant, output_path in output_paths.items()
        }

        def write_result(
            dicom_path: Path,
            payloads: Mapping[str, bytes],
            metadata: dict[str, str | int],
        ) -> None:
            processed_path = png_archive_path(dicom_path, input_dir, png_root).as_posix()
            for variant, archive in archives.items():
                archive.writestr(processed_path, payloads[variant.value])
                variant_metadata = metadata.copy()
                variant_metadata["processed_path"] = processed_path
                metadata_rows[variant].append(variant_metadata)

        if workers == 1:
            progress = tqdm(
                dicom_files,
                desc="Preprocessing dataset variants",
                unit="dicom",
            )
            for dicom_path in progress:
                try:
                    payloads, metadata = preprocess_dicom_payloads(
                        dicom_path,
                        variants,
                        output_size,
                    )
                    write_result(dicom_path, payloads, metadata)
                    processed += 1
                except Exception as exc:
                    if not continue_on_error:
                        raise RuntimeError(f"Failed to preprocess {dicom_path}: {exc}") from exc
                    failures.append((dicom_path, str(exc)))
                progress.set_postfix(processed=processed, failed=len(failures))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                dicom_iter = iter(dicom_files)
                futures = {}
                max_in_flight = workers * 2

                def submit_next() -> bool:
                    try:
                        dicom_path = next(dicom_iter)
                    except StopIteration:
                        return False

                    future = executor.submit(
                        preprocess_dicom_payloads,
                        dicom_path,
                        tuple(variant.value for variant in variants),
                        output_size,
                    )
                    futures[future] = dicom_path
                    return True

                for _ in range(min(max_in_flight, len(dicom_files))):
                    submit_next()

                progress = tqdm(
                    total=len(dicom_files),
                    desc=f"Preprocessing dataset variants ({workers} workers)",
                    unit="dicom",
                )
                with progress:
                    while futures:
                        for future in as_completed(futures):
                            dicom_path = futures.pop(future)
                            try:
                                payloads, metadata = future.result()
                                write_result(dicom_path, payloads, metadata)
                                processed += 1
                            except Exception as exc:
                                if not continue_on_error:
                                    raise RuntimeError(
                                        f"Failed to preprocess {dicom_path}: {exc}"
                                    ) from exc
                                failures.append((dicom_path, str(exc)))
                            progress.update(1)
                            progress.set_postfix(
                                processed=processed,
                                failed=len(failures),
                            )
                            submit_next()
                            break

        for variant, archive in archives.items():
            write_metadata_csv(archive, metadata_rows[variant])

    if not processed:
        for output_path in output_paths.values():
            output_path.unlink(missing_ok=True)
        if failures and failure_report is not None:
            write_failures_csv(failure_report, failures)
        raise RuntimeError("No DICOM files were successfully converted")

    report_failures(processed, len(dicom_files), failures)
    if failures and failure_report is not None:
        write_failures_csv(failure_report, failures)
    return output_paths
