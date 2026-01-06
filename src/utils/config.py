from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)

def _coerce(v: str):
    # simple coercion: bool/int/float/null/str
    s = v.strip()
    if s.lower() in ("null", "none"):
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s

def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    overrides: ["train.batch_size=16", "model.backbone=resnet18"]
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use key=value.")
        key, val = item.split("=", 1)
        val = _coerce(val)

        keys = key.split(".")
        cur = cfg
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = val
    return cfg
