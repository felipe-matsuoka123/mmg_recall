from __future__ import annotations
from pathlib import Path
import torch

def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    epoch: int | None = None,
    cfg: dict | None = None,
    extra: dict | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if cfg is not None:
        payload["cfg"] = cfg
    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, path)

def load_checkpoint(path: str | Path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(str(path), map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

def save_best_if_improved(
    current_value: float,
    best_value: float | None,
    out_path: str | Path,
    model,
    optimizer=None,
    epoch: int | None = None,
    cfg: dict | None = None,
    extra: dict | None = None,
    mode: str = "max",
):
    """
    mode: "max" (e.g., accuracy/AUC) or "min" (e.g., loss)
    """
    improved = False
    if best_value is None:
        improved = True
    elif mode == "max" and current_value > best_value:
        improved = True
    elif mode == "min" and current_value < best_value:
        improved = True

    if improved:
        save_checkpoint(out_path, model, optimizer, epoch=epoch, cfg=cfg, extra=extra)
        return current_value, True

    return best_value, False
