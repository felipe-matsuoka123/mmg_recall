from __future__ import annotations

from pathlib import Path
import zipfile
from typing import Dict

from PIL import Image


class ImageSource:
    """
    Load PNGs either from an extracted folder or from one or more zip files.

    - If zip_path is None: read from root / rel_path
    - If zip_path is a file: read rel_path inside that zip
    - If zip_path is a dir: index all .zip files and map names to zip files
    """

    def __init__(self, root: Path, zip_path: Path | None = None):
        self.root = Path(root)
        self.zip_path = Path(zip_path) if zip_path else None
        self._zip_index: Dict[str, Path] | None = None
        self._zip_cache: Dict[Path, zipfile.ZipFile] = {}

    def _get_zip(self, path: Path) -> zipfile.ZipFile:
        zf = self._zip_cache.get(path)
        if zf is None:
            zf = zipfile.ZipFile(path, "r")
            self._zip_cache[path] = zf
        return zf

    def _build_index(self):
        if self.zip_path is None or self.zip_path.is_file():
            return
        index: Dict[str, Path] = {}
        for zp in sorted(self.zip_path.glob("*.zip")):
            with zipfile.ZipFile(zp, "r") as zf:
                for name in zf.namelist():
                    # Keep first occurrence to avoid silent overwrites.
                    index.setdefault(name, zp)
        self._zip_index = index

    def open_png(self, rel_path: str) -> Image.Image:
        if self.zip_path is None:
            path = self.root / rel_path
            return Image.open(path).convert("L")

        if self.zip_path.is_file():
            zf = self._get_zip(self.zip_path)
            with zf.open(rel_path) as fp:
                return Image.open(fp).convert("L")

        if self._zip_index is None:
            self._build_index()

        if self._zip_index is None or rel_path not in self._zip_index:
            raise FileNotFoundError(f"'{rel_path}' not found in zip index at {self.zip_path}")

        zf = self._get_zip(self._zip_index[rel_path])
        with zf.open(rel_path) as fp:
            return Image.open(fp).convert("L")
