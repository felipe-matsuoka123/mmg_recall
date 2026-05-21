from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")
pytest.importorskip("pydicom")

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "preprocess_mammo_dicom.py"
SPEC = importlib.util.spec_from_file_location("preprocess_mammo_dicom", SCRIPT_PATH)
assert SPEC and SPEC.loader
preprocess_mammo_dicom = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preprocess_mammo_dicom)


def test_discover_dicom_files_is_recursive_and_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "study").mkdir()
    expected = [tmp_path / "a.dcm", tmp_path / "study" / "b.DCM"]
    for path in expected:
        path.touch()
    (tmp_path / "study" / "ignore.txt").touch()

    assert preprocess_mammo_dicom.discover_dicom_files(tmp_path) == expected


def test_png_archive_path_preserves_dicom_relative_layout(tmp_path: Path) -> None:
    input_dir = tmp_path / "dicoms"
    dicom_path = input_dir / "accession" / "image.dcm"

    assert preprocess_mammo_dicom.png_archive_path(dicom_path, input_dir, None) == Path(
        "accession/image.png"
    )
    assert preprocess_mammo_dicom.png_archive_path(
        dicom_path, input_dir, "pngs"
    ) == Path("pngs/accession/image.png")


def test_bbox_from_mask_falls_back_to_full_image_for_empty_mask() -> None:
    mask = preprocess_mammo_dicom.np.zeros((12, 20), dtype=preprocess_mammo_dicom.np.uint8)

    assert preprocess_mammo_dicom.bbox_from_mask(mask) == (0, 0, 20, 12)


def test_image_metadata_uses_requested_csv_fields() -> None:
    dicom = preprocess_mammo_dicom.pydicom.Dataset()
    dicom.SOPInstanceUID = "image"
    dicom.PatientID = "patient"
    dicom.StudyInstanceUID = "study"
    dicom.SeriesInstanceUID = "series"
    dicom.ImageLaterality = "L"
    dicom.ViewPosition = "MLO"
    dicom.Manufacturer = "vendor"
    dicom.PixelSpacing = [0.07, 0.07]
    dicom.BitsStored = 12

    metadata = preprocess_mammo_dicom.image_metadata(
        dicom,
        original_shape=(100, 80),
        crop_bbox_xywh=(5, 10, 25, 40),
    )

    assert list(metadata) == preprocess_mammo_dicom.METADATA_FIELDS
    assert metadata["pixel_spacing"] == "0.07\\0.07"
    assert metadata["crop_xmax"] == 30
    assert metadata["crop_ymax"] == 50
    assert metadata["label"] == ""


def test_create_u8_image_variants_keep_crop_size() -> None:
    cropped = preprocess_mammo_dicom.np.arange(100, dtype=preprocess_mammo_dicom.np.float32)
    cropped = cropped.reshape((10, 10))

    grayscale = preprocess_mammo_dicom.create_u8_image(
        cropped, preprocess_mammo_dicom.ImageVariant.GRAYSCALE
    )
    rgb_replicated = preprocess_mammo_dicom.create_u8_image(
        cropped, preprocess_mammo_dicom.ImageVariant.RGB_REPLICATED
    )
    rgb_multiwindow = preprocess_mammo_dicom.create_u8_image(
        cropped, preprocess_mammo_dicom.ImageVariant.RGB_MULTIWINDOW
    )

    assert grayscale.shape == (10, 10)
    assert rgb_replicated.shape == (10, 10, 3)
    assert rgb_multiwindow.shape == (10, 10, 3)
    assert (rgb_replicated[..., 0] == grayscale).all()
    assert (rgb_replicated[..., 1] == grayscale).all()
    assert (rgb_replicated[..., 2] == grayscale).all()
