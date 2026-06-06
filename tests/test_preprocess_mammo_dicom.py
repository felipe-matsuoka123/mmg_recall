from __future__ import annotations

import csv
import importlib.util
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")
pytest.importorskip("pydicom")

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "mammo_preprocessing.py"
SPEC = importlib.util.spec_from_file_location("mammo_preprocessing", SCRIPT_PATH)
assert SPEC and SPEC.loader
mammo_preprocessing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mammo_preprocessing)


def fake_preprocess_dicom_payloads_for_parallel_test(
    dicom_path: Path,
    variants: tuple[str, ...],
    output_size: int | None,
) -> tuple[dict[str, bytes], dict[str, str]]:
    image = mammo_preprocessing.np.zeros(
        (8, 8), dtype=mammo_preprocessing.np.uint16
    )
    payloads = {
        variant: mammo_preprocessing.encode_png(image)
        for variant in variants
    }
    metadata = {field: "" for field in mammo_preprocessing.METADATA_FIELDS}
    return payloads, metadata


def test_discover_dicom_files_is_recursive_and_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "study").mkdir()
    expected = [
        tmp_path / "a.dcm",
        tmp_path / "study" / "b.DCM",
        tmp_path / "study" / "c.dicom",
        tmp_path / "study" / "d.DICOM",
    ]
    for path in expected:
        path.touch()
    (tmp_path / "study" / "ignore.txt").touch()

    assert mammo_preprocessing.discover_dicom_files(tmp_path) == expected


def test_png_archive_path_preserves_dicom_relative_layout(tmp_path: Path) -> None:
    input_dir = tmp_path / "dicoms"
    dicom_path = input_dir / "accession" / "image.dcm"

    assert mammo_preprocessing.png_archive_path(dicom_path, input_dir, None) == Path(
        "accession/image.png"
    )
    assert mammo_preprocessing.png_archive_path(
        dicom_path, input_dir, "pngs"
    ) == Path("pngs/accession/image.png")


def test_bbox_from_mask_falls_back_to_full_image_for_empty_mask() -> None:
    mask = mammo_preprocessing.np.zeros((12, 20), dtype=mammo_preprocessing.np.uint8)

    assert mammo_preprocessing.bbox_from_mask(mask) == (0, 0, 20, 12)


def test_breast_mask_rejects_full_image_background_noise() -> None:
    image = mammo_preprocessing.np.full(
        (128, 128), 7, dtype=mammo_preprocessing.np.uint8
    )
    mammo_preprocessing.cv2.ellipse(
        image,
        center=(82, 64),
        axes=(36, 50),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=90,
        thickness=-1,
    )

    mask = mammo_preprocessing.create_breast_mask_u8(image)
    area_ratio = mammo_preprocessing.np.count_nonzero(mask) / mask.size

    assert 0.1 < area_ratio < 0.7


def test_normalize_for_mask_is_robust_to_bright_outlier() -> None:
    image = mammo_preprocessing.np.full(
        (128, 128), 10, dtype=mammo_preprocessing.np.float32
    )
    image[32:96, 48:112] = 80
    image[0, 0] = 10000

    normalized = mammo_preprocessing.normalize_for_mask_u8(image)

    assert normalized[64, 64] > 200
    assert normalized[10, 10] == 0


def test_ensure_dark_background_inverts_bright_border_image() -> None:
    image = mammo_preprocessing.np.full(
        (128, 128), 4000, dtype=mammo_preprocessing.np.float32
    )
    image[24:104, 32:112] = 600

    normalized = mammo_preprocessing.ensure_dark_background(image)

    assert normalized[0, 0] < normalized[64, 64]


def test_robust_window_uses_breast_foreground_for_contrast() -> None:
    image = mammo_preprocessing.np.zeros(
        (128, 128), dtype=mammo_preprocessing.np.float32
    )
    image[24:104, 32:112] = 100
    image[44:84, 52:92] = 140
    image[0, 0] = 10000

    windowed = mammo_preprocessing.robust_window_u8(image, (1.0, 99.0))

    assert windowed[64, 64] > windowed[30, 40]
    assert windowed[30, 40] > windowed[0, 1]


def test_robust_window_u16_outputs_uint16() -> None:
    image = mammo_preprocessing.np.arange(
        100, dtype=mammo_preprocessing.np.float32
    ).reshape((10, 10))

    windowed = mammo_preprocessing.robust_window_u16(image, (1.0, 99.0))

    assert windowed.dtype == mammo_preprocessing.np.uint16
    assert windowed.max() > 65000


def test_create_u8_images_uses_roi_normalized_uint16_windows() -> None:
    cropped = mammo_preprocessing.np.arange(
        100,
        dtype=mammo_preprocessing.np.float32,
    ).reshape((10, 10))

    images = mammo_preprocessing.create_u8_images(
        cropped,
        (
            mammo_preprocessing.ImageVariant.GRAYSCALE,
            mammo_preprocessing.ImageVariant.RGB_MULTIWINDOW,
        ),
    )

    grayscale = images[mammo_preprocessing.ImageVariant.GRAYSCALE]
    rgb_multiwindow = images[mammo_preprocessing.ImageVariant.RGB_MULTIWINDOW]

    assert grayscale.dtype == mammo_preprocessing.np.uint16
    assert rgb_multiwindow.dtype == mammo_preprocessing.np.uint16
    assert rgb_multiwindow[..., 0].tolist() == grayscale.tolist()


def test_normalize_breast_roi_u16_targets_roi_distribution() -> None:
    image = mammo_preprocessing.np.zeros(
        (128, 128), dtype=mammo_preprocessing.np.uint16
    )
    image[32:96, 32:96] = int(70 * mammo_preprocessing.UINT8_TO_UINT16)
    image[44:84, 44:84] = int(130 * mammo_preprocessing.UINT8_TO_UINT16)

    original_mask = mammo_preprocessing.create_breast_mask_u8(
        mammo_preprocessing.u16_to_mask_u8(image)
    )
    normalized = mammo_preprocessing.normalize_breast_roi_u16(
        image,
        target_median=100.0 * mammo_preprocessing.UINT8_TO_UINT16,
        target_iqr=70.0 * mammo_preprocessing.UINT8_TO_UINT16,
    )
    roi = normalized[original_mask > 0]
    p25, median, p75 = mammo_preprocessing.np.percentile(roi, [25, 50, 75])

    assert normalized.dtype == mammo_preprocessing.np.uint16
    assert 95 <= median / mammo_preprocessing.UINT8_TO_UINT16 <= 105
    assert 65 <= (p75 - p25) / mammo_preprocessing.UINT8_TO_UINT16 <= 75
    assert normalized[0, 0] < median


def test_image_metadata_uses_requested_csv_fields() -> None:
    dicom = mammo_preprocessing.pydicom.Dataset()
    dicom.SOPInstanceUID = "image"
    dicom.PatientID = "patient"
    dicom.StudyInstanceUID = "study"
    dicom.SeriesInstanceUID = "series"
    dicom.ImageLaterality = "L"
    dicom.ViewPosition = "MLO"
    dicom.Manufacturer = "vendor"
    dicom.PixelSpacing = [0.07, 0.07]
    dicom.BitsStored = 12

    metadata = mammo_preprocessing.image_metadata(
        dicom,
        original_shape=(100, 80),
        crop_bbox_xywh=(5, 10, 25, 40),
    )

    assert list(metadata) == mammo_preprocessing.METADATA_FIELDS
    assert metadata["pixel_spacing"] == "0.07\\0.07"
    assert metadata["crop_xmax"] == 30
    assert metadata["crop_ymax"] == 50
    assert metadata["label"] == ""


def test_create_u8_image_variants_keep_crop_size() -> None:
    cropped = mammo_preprocessing.np.arange(100, dtype=mammo_preprocessing.np.float32)
    cropped = cropped.reshape((10, 10))

    grayscale = mammo_preprocessing.create_u8_image(
        cropped, mammo_preprocessing.ImageVariant.GRAYSCALE
    )
    rgb_replicated = mammo_preprocessing.create_u8_image(
        cropped, mammo_preprocessing.ImageVariant.RGB_REPLICATED
    )
    rgb_multiwindow = mammo_preprocessing.create_u8_image(
        cropped, mammo_preprocessing.ImageVariant.RGB_MULTIWINDOW
    )

    assert grayscale.shape == (10, 10)
    assert rgb_replicated.shape == (10, 10, 3)
    assert rgb_multiwindow.shape == (10, 10, 3)
    assert grayscale.dtype == mammo_preprocessing.np.uint16
    assert rgb_replicated.dtype == mammo_preprocessing.np.uint16
    assert rgb_multiwindow.dtype == mammo_preprocessing.np.uint16
    assert (rgb_replicated[..., 0] == grayscale).all()
    assert (rgb_replicated[..., 1] == grayscale).all()
    assert (rgb_replicated[..., 2] == grayscale).all()


def test_create_u8_images_shares_requested_variant_output() -> None:
    cropped = mammo_preprocessing.np.arange(
        100, dtype=mammo_preprocessing.np.float32
    )
    cropped = cropped.reshape((10, 10))

    images = mammo_preprocessing.create_u8_images(
        cropped,
        (
            mammo_preprocessing.ImageVariant.GRAYSCALE,
            mammo_preprocessing.ImageVariant.RGB_REPLICATED,
        ),
    )

    assert list(images) == [
        mammo_preprocessing.ImageVariant.GRAYSCALE,
        mammo_preprocessing.ImageVariant.RGB_REPLICATED,
    ]
    assert (
        images[mammo_preprocessing.ImageVariant.RGB_REPLICATED][..., 0]
        == images[mammo_preprocessing.ImageVariant.GRAYSCALE]
    ).all()


def test_resize_image_u8_outputs_requested_square_size() -> None:
    image = mammo_preprocessing.np.zeros(
        (12, 20, 3), dtype=mammo_preprocessing.np.uint16
    )

    resized = mammo_preprocessing.resize_image_u8(image, 16)

    assert resized.shape == (16, 16, 3)
    assert mammo_preprocessing.resize_image_u8(image, None) is image


def test_multi_zip_writer_reads_crop_once_per_dicom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dicom_paths = [tmp_path / "dicoms" / "a.dcm", tmp_path / "dicoms" / "nested" / "b.dcm"]
    for dicom_path in dicom_paths:
        dicom_path.parent.mkdir(parents=True, exist_ok=True)
        dicom_path.touch()

    read_paths = []

    def fake_read_crop_and_metadata(dicom_path: Path) -> tuple[object, dict[str, str]]:
        read_paths.append(dicom_path)
        metadata = {field: "" for field in mammo_preprocessing.METADATA_FIELDS}
        cropped = mammo_preprocessing.np.arange(
            100, dtype=mammo_preprocessing.np.float32
        ).reshape((10, 10))
        return cropped, metadata

    monkeypatch.setattr(
        mammo_preprocessing,
        "read_crop_and_metadata",
        fake_read_crop_and_metadata,
    )

    output_paths = mammo_preprocessing.process_dicom_dir_to_png_zips(
        tmp_path / "dicoms",
        {
            mammo_preprocessing.ImageVariant.GRAYSCALE: tmp_path / "gray.zip",
            mammo_preprocessing.ImageVariant.RGB_REPLICATED: tmp_path / "rgb.zip",
        },
    )

    assert read_paths == dicom_paths
    for output_path in output_paths.values():
        with zipfile.ZipFile(output_path) as archive:
            assert archive.namelist() == ["a.png", "nested/b.png", "metadata.csv"]


def test_multi_zip_writer_accepts_parallel_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dicom_paths = [tmp_path / "dicoms" / "a.dcm", tmp_path / "dicoms" / "b.dcm"]
    for dicom_path in dicom_paths:
        dicom_path.parent.mkdir(parents=True, exist_ok=True)
        dicom_path.touch()

    monkeypatch.setattr(
        mammo_preprocessing,
        "preprocess_dicom_payloads",
        fake_preprocess_dicom_payloads_for_parallel_test,
    )

    output_paths = mammo_preprocessing.process_dicom_dir_to_png_zips(
        tmp_path / "dicoms",
        {mammo_preprocessing.ImageVariant.GRAYSCALE: tmp_path / "gray.zip"},
        workers=2,
    )

    with zipfile.ZipFile(output_paths[mammo_preprocessing.ImageVariant.GRAYSCALE]) as archive:
        assert sorted(archive.namelist()) == ["a.png", "b.png", "metadata.csv"]


def test_zip_writer_resizes_encoded_png_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dicom_path = tmp_path / "dicoms" / "a.dcm"
    dicom_path.parent.mkdir()
    dicom_path.touch()

    def fake_read_preprocessing_state(
        dicom_path: Path,
    ) -> tuple[object, dict[str, str], object, object, tuple[int, int, int, int]]:
        metadata = {field: "" for field in mammo_preprocessing.METADATA_FIELDS}
        cropped = mammo_preprocessing.np.arange(
            12 * 20, dtype=mammo_preprocessing.np.float32
        ).reshape((12, 20))
        view = mammo_preprocessing.np.zeros(
            (24, 32), dtype=mammo_preprocessing.np.uint8
        )
        mask = mammo_preprocessing.np.zeros(
            (24, 32), dtype=mammo_preprocessing.np.uint8
        )
        mask[4:20, 6:26] = 255
        return cropped, metadata, view, mask, (6, 4, 20, 16)

    monkeypatch.setattr(
        mammo_preprocessing,
        "read_preprocessing_state",
        fake_read_preprocessing_state,
    )

    output_path = mammo_preprocessing.process_dicom_dir_to_png_zip(
        tmp_path / "dicoms",
        tmp_path / "dataset.zip",
    )

    with zipfile.ZipFile(output_path) as archive:
        png_bytes = archive.read("a.png")
    encoded = mammo_preprocessing.np.frombuffer(
        png_bytes, dtype=mammo_preprocessing.np.uint8
    )
    image = mammo_preprocessing.cv2.imdecode(
        encoded, mammo_preprocessing.cv2.IMREAD_UNCHANGED
    )

    assert image.shape == (1024, 1024)
    assert image.dtype == mammo_preprocessing.np.uint16


def test_zip_writer_can_keep_encoded_png_crop_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dicom_path = tmp_path / "dicoms" / "a.dcm"
    dicom_path.parent.mkdir()
    dicom_path.touch()

    def fake_read_preprocessing_state(
        dicom_path: Path,
    ) -> tuple[object, dict[str, str], object, object, tuple[int, int, int, int]]:
        metadata = {field: "" for field in mammo_preprocessing.METADATA_FIELDS}
        cropped = mammo_preprocessing.np.arange(
            12 * 20, dtype=mammo_preprocessing.np.float32
        ).reshape((12, 20))
        view = mammo_preprocessing.np.zeros(
            (24, 32), dtype=mammo_preprocessing.np.uint8
        )
        mask = mammo_preprocessing.np.zeros(
            (24, 32), dtype=mammo_preprocessing.np.uint8
        )
        mask[4:20, 6:26] = 255
        return cropped, metadata, view, mask, (6, 4, 20, 16)

    monkeypatch.setattr(
        mammo_preprocessing,
        "read_preprocessing_state",
        fake_read_preprocessing_state,
    )

    output_path = mammo_preprocessing.process_dicom_dir_to_png_zip(
        tmp_path / "dicoms",
        tmp_path / "dataset.zip",
        output_size=None,
    )

    with zipfile.ZipFile(output_path) as archive:
        png_bytes = archive.read("a.png")
    encoded = mammo_preprocessing.np.frombuffer(
        png_bytes, dtype=mammo_preprocessing.np.uint8
    )
    image = mammo_preprocessing.cv2.imdecode(
        encoded, mammo_preprocessing.cv2.IMREAD_UNCHANGED
    )

    assert image.shape == (12, 20)
    assert image.dtype == mammo_preprocessing.np.uint16


def test_multi_zip_writer_reports_skipped_dicoms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dicom_paths = [tmp_path / "dicoms" / "good.dcm", tmp_path / "dicoms" / "bad.dcm"]
    for dicom_path in dicom_paths:
        dicom_path.parent.mkdir(parents=True, exist_ok=True)
        dicom_path.touch()

    def fake_read_crop_and_metadata(dicom_path: Path) -> tuple[object, dict[str, str]]:
        if dicom_path.name == "bad.dcm":
            raise ValueError("pixel data is shorter than expected")
        metadata = {field: "" for field in mammo_preprocessing.METADATA_FIELDS}
        cropped = mammo_preprocessing.np.arange(
            12 * 20, dtype=mammo_preprocessing.np.float32
        ).reshape((12, 20))
        return cropped, metadata

    monkeypatch.setattr(
        mammo_preprocessing,
        "read_crop_and_metadata",
        fake_read_crop_and_metadata,
    )

    failure_report = tmp_path / "failures.csv"
    output_paths = mammo_preprocessing.process_dicom_dir_to_png_zips(
        tmp_path / "dicoms",
        {mammo_preprocessing.ImageVariant.GRAYSCALE: tmp_path / "gray.zip"},
        continue_on_error=True,
        failure_report=failure_report,
    )

    with zipfile.ZipFile(output_paths[mammo_preprocessing.ImageVariant.GRAYSCALE]) as archive:
        assert archive.namelist() == ["good.png", "metadata.csv"]

    with failure_report.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "dicom_path": str(tmp_path / "dicoms" / "bad.dcm"),
            "error": "pixel data is shorter than expected",
        }
    ]


def test_preview_samples_writes_requested_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dicom_paths = [
        tmp_path / "dicoms" / "a.dcm",
        tmp_path / "dicoms" / "b.dcm",
        tmp_path / "dicoms" / "c.dcm",
    ]
    for dicom_path in dicom_paths:
        dicom_path.parent.mkdir(parents=True, exist_ok=True)
        dicom_path.touch()

    def fake_read_preprocessing_state(
        dicom_path: Path,
    ) -> tuple[object, dict[str, str], object, object, tuple[int, int, int, int]]:
        metadata = {field: "" for field in mammo_preprocessing.METADATA_FIELDS}
        cropped = mammo_preprocessing.np.arange(
            12 * 20, dtype=mammo_preprocessing.np.float32
        ).reshape((12, 20))
        view = mammo_preprocessing.np.zeros(
            (24, 32), dtype=mammo_preprocessing.np.uint8
        )
        mask = mammo_preprocessing.np.zeros(
            (24, 32), dtype=mammo_preprocessing.np.uint8
        )
        mask[4:20, 6:26] = 255
        return cropped, metadata, view, mask, (6, 4, 20, 16)

    monkeypatch.setattr(
        mammo_preprocessing,
        "read_preprocessing_state",
        fake_read_preprocessing_state,
    )

    sample_paths = mammo_preprocessing.process_dicom_preview_samples(
        tmp_path / "dicoms",
        tmp_path / "previews",
        {
            "grayscale": mammo_preprocessing.ImageVariant.GRAYSCALE,
            "rgb_multiwindow": mammo_preprocessing.ImageVariant.RGB_MULTIWINDOW,
        },
        sample_count=3,
        output_size=16,
        write_diagnostics=True,
    )

    assert len(sample_paths["grayscale"]) == 3
    assert len(sample_paths["rgb_multiwindow"]) == 3
    assert sample_paths["grayscale"][0].is_file()
    assert sample_paths["rgb_multiwindow"][0].is_file()

    grayscale = mammo_preprocessing.cv2.imdecode(
        mammo_preprocessing.np.frombuffer(
            sample_paths["grayscale"][0].read_bytes(),
            dtype=mammo_preprocessing.np.uint8,
        ),
        mammo_preprocessing.cv2.IMREAD_UNCHANGED,
    )
    rgb_multiwindow = mammo_preprocessing.cv2.imdecode(
        mammo_preprocessing.np.frombuffer(
            sample_paths["rgb_multiwindow"][0].read_bytes(),
            dtype=mammo_preprocessing.np.uint8,
        ),
        mammo_preprocessing.cv2.IMREAD_UNCHANGED,
    )

    assert grayscale.shape == (16, 16)
    assert rgb_multiwindow.shape == (16, 16, 3)
    assert (
        tmp_path
        / "previews"
        / "diagnostics"
        / "sample_001_a_01_full_window.png"
    ).is_file()
    assert (
        tmp_path
        / "previews"
        / "diagnostics"
        / "sample_001_a_02_mask.png"
    ).is_file()
    assert (
        tmp_path
        / "previews"
        / "diagnostics"
        / "sample_001_a_03_crop_overlay.png"
    ).is_file()
    assert (
        tmp_path
        / "previews"
        / "diagnostics"
        / "sample_001_a_04_crop_window.png"
    ).is_file()
    assert (
        tmp_path
        / "previews"
        / "diagnostics"
        / "rgb_multiwindow"
        / "sample_001_a_1_red_base.png"
    ).is_file()
    assert (
        tmp_path
        / "previews"
        / "diagnostics"
        / "rgb_multiwindow"
        / "sample_001_a_2_green_dense.png"
    ).is_file()
    assert (
        tmp_path
        / "previews"
        / "diagnostics"
        / "rgb_multiwindow"
        / "sample_001_a_3_blue_tissue.png"
    ).is_file()
