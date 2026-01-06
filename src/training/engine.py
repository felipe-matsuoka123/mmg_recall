from __future__ import annotations
import torch
import torch.nn as nn

from src.training.metrics import (
    accuracy_from_logits,
    binary_accuracy_from_logits,
    try_binary_auc_from_logits,
)

def _compute_loss_and_metrics(logits, y, criterion, num_classes: int):
    if num_classes == 1:
        # BCE: logits [B,1], y [B] -> [B,1]
        yb = y.float().unsqueeze(1)
        loss = criterion(logits, yb)
        acc = binary_accuracy_from_logits(logits, y)
        auc = try_binary_auc_from_logits(logits, y)
        return loss, {"acc": acc, "auc": auc}
    else:
        loss = criterion(logits, y)
        acc = accuracy_from_logits(logits, y)
        return loss, {"acc": acc}

def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    device,
    num_classes: int,
    amp: bool = True,
    grad_accum_steps: int = 1,
):
    model.train()

    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_n = 0

    # accumulate metrics (weighted by batch size)
    sum_acc = 0.0
    sum_auc = 0.0
    auc_n = 0

    for step, (x, y, _ids) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss, metrics = _compute_loss_and_metrics(logits, y, criterion, num_classes)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = x.size(0)
        total_loss += loss.item() * bs * grad_accum_steps
        total_n += bs

        sum_acc += metrics["acc"] * bs
        if num_classes == 1 and metrics.get("auc") is not None:
            sum_auc += metrics["auc"] * bs
            auc_n += bs

    out = {
        "loss": total_loss / max(total_n, 1),
        "acc": sum_acc / max(total_n, 1),
    }
    if num_classes == 1:
        out["auc"] = (sum_auc / auc_n) if auc_n > 0 else None
    return out

@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    loader,
    device,
    num_classes: int,
):
    model.eval()

    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_n = 0

    sum_acc = 0.0
    sum_auc = 0.0
    auc_n = 0

    for x, y, _ids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss, metrics = _compute_loss_and_metrics(logits, y, criterion, num_classes)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        sum_acc += metrics["acc"] * bs
        if num_classes == 1 and metrics.get("auc") is not None:
            sum_auc += metrics["auc"] * bs
            auc_n += bs

    out = {
        "loss": total_loss / max(total_n, 1),
        "acc": sum_acc / max(total_n, 1),
    }
    if num_classes == 1:
        out["auc"] = (sum_auc / auc_n) if auc_n > 0 else None
    return out
