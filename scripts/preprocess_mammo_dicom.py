import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pydicom
import cv2
from tqdm import tqdm


def invert_if_needed(pixels, dicom):
    return (pixels.max() - pixels) if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1" else pixels


def resize_u8(img_u8, size=2048):
    return cv2.resize(img_u8, (size, size), interpolation=cv2.INTER_AREA)


def create_breast_mask_u8(image_u8):
    img = image_u8.copy()
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, fg = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)

    h, w = fg.shape
    inv = cv2.bitwise_not(fg)
    bg = np.zeros_like(inv)
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for sx, sy in seeds:
        if inv[sy, sx] == 255:
            tmp = inv.copy()
            flood_mask[:] = 0
            cv2.floodFill(tmp, flood_mask, (sx, sy), 128)
            bg |= ((tmp == 128).astype(np.uint8) * 255)

    breast = cv2.bitwise_not(bg)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    breast = cv2.morphologyEx(breast, cv2.MORPH_CLOSE, k_close)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((breast > 0).astype(np.uint8), connectivity=8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        breast = (labels == largest).astype(np.uint8) * 255

    inv_b = cv2.bitwise_not(breast)
    hole_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv_b, hole_mask, (0, 0), 128)
    holes = ((inv_b != 128).astype(np.uint8) * 255)
    breast = cv2.bitwise_or(breast, holes)

    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    breast = cv2.dilate(breast, k_dil, iterations=1)

    return breast


def bbox_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, mask.shape[1], mask.shape[0])
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (x, y, w, h)


def window_u8(img, lo, hi):
    img = np.clip(img, lo, hi)
    out = (img - lo) / (hi - lo + 1e-6)
    return (out * 255.0).astype(np.uint8)


def _parse_yes_no(v):
    """Normalize DICOM YES/NO-ish values to True/False/None."""
    if v is None:
        return None
    s = str(v).strip().upper()
    if s in {"Y", "YES", "TRUE", "T", "1"}:
        return True
    if s in {"N", "NO", "FALSE", "F", "0"}:
        return False
    return None


def get_breast_implant_present(dcm):
    """
    DICOM (0018,1300) Breast Implant Present
    Returns: True/False/None (if absent/unparseable)
    """
    v = getattr(dcm, "BreastImplantPresent", None)
    if v is None:
        tag = pydicom.tag.Tag(0x0018, 0x1300)
        if tag in dcm:
            v = dcm.get(tag).value

    return _parse_yes_no(v)


def create_u8_crop_and_meta(dicom_path, out_size=2048):
    dcm = pydicom.dcmread(dicom_path, force=True)
    raw = dcm.pixel_array.astype(np.float32)

    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    img = raw * slope + intercept
    img = invert_if_needed(img, dcm)

    H, W = img.shape[:2]

    view_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask = create_breast_mask_u8(view_u8)
    x, y, w, h = bbox_from_mask(mask)

    img_crop = img[y:y+h, x:x+w]

    lo_base, hi_base = np.percentile(img_crop, 0.5), np.percentile(img_crop, 99.5)
    img_u8 = window_u8(img_crop, lo_base, hi_base)
    img_u8 = resize_u8(img_u8, size=out_size)

    lo0, hi0 = lo_base, hi_base
    lo1, hi1 = np.percentile(img_crop, 95), np.percentile(img_crop, 99.9)
    lo2, hi2 = np.percentile(img_crop, 10), np.percentile(img_crop, 90)

    manufacturer = str(getattr(dcm, "Manufacturer", "UNKNOWN")).strip() or "UNKNOWN"
    implant_present = get_breast_implant_present(dcm)

    meta = {
        "orig_shape": [int(H), int(W)],
        "crop_bbox_xywh": [int(x), int(y), int(w), int(h)],
        "photometric": str(getattr(dcm, "PhotometricInterpretation", "")),
        "slope": slope,
        "intercept": intercept,
        "manufacturer": manufacturer,
        "breast_implant_present": implant_present,  # <-- NEW
        "base_window": [float(lo_base), float(hi_base)],
        "window_params": {
            "ch0": [float(lo0), float(hi0)],
            "ch1": [float(lo1), float(hi1)],
            "ch2": [float(lo2), float(hi2)],
        },
        "output": {"size": int(out_size), "dtype": "uint8"}
    }
    return img_u8, meta


def process_to_dir_png(input_dir, output_dir, out_size=2048):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    dicom_files = glob(str(input_dir / "**" / "*.dcm"), recursive=True)
    dicom_files.sort()

    for p in tqdm(dicom_files, desc="Processing DICOMs", unit="file"):
        p = Path(p)
        rel = p.relative_to(input_dir)               # preserve folder structure
        base = rel.with_suffix("")                   # remove .dcm

        out_png = output_dir / "images" / (str(base) + ".png")
        out_json = output_dir / "meta" / (str(base) + ".json")

        out_png.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)

        try:
            img_u8, meta = create_u8_crop_and_meta(p, out_size=out_size)

            # Save grayscale PNG
            ok = cv2.imwrite(str(out_png), img_u8)
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed for {out_png}")

            # Save metadata JSON sidecar
            out_json.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        except Exception as e:
            tqdm.write(f"Erro em {p}: {e}")



process_to_dir_png(
     input_dir="/home/felipe/spr-mmg-1",
     output_dir="/home/felipe/projects/MammoRecall/data/processed/spr-mmg-01_u8_png",
     out_size=2048
 )
