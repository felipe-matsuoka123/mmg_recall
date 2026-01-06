from __future__ import annotations
import numpy as np
import torch

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Multiclass accuracy from logits [B,C] and y [B] long.
    """
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def binary_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor, thresh: float = 0.5) -> float:
    """
    Binary accuracy from logits [B,1] or [B] and y [B] in {0,1}.
    """
    if logits.ndim == 2 and logits.size(1) == 1:
        logits = logits.squeeze(1)
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).long()
    return (preds == y.long()).float().mean().item()

def try_binary_auc_from_logits(logits: torch.Tensor, y: torch.Tensor):
    """
    Returns float AUC if sklearn is available and both classes present, else None.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None

    if logits.ndim == 2 and logits.size(1) == 1:
        logits = logits.squeeze(1)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    yy = y.detach().cpu().numpy().astype(int)

    # AUC undefined if only one class present in y
    if len(np.unique(yy)) < 2:
        return None
    return float(roc_auc_score(yy, probs))
