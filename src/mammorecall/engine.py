from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn


def binary_auroc(scores: list[float], targets: list[int]) -> float | None:
    positives = sum(targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(scores, targets, strict=False), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        rank_sum += average_rank * sum(target for _, target in ranked[index:end])
        rank += end - index
        index = end

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def run_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    on_batch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    correct = 0
    seen = 0
    positive_scores: list[float] = []
    all_targets: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += batch_size
            if logits.size(1) == 2:
                positive_scores.extend(torch.softmax(logits.detach(), dim=1)[:, 1].cpu().tolist())
                all_targets.extend(targets.detach().cpu().tolist())
            if on_batch_end is not None:
                on_batch_end(
                    {
                        "batch/loss": loss.item(),
                        "running/loss": loss_sum / seen,
                        "running/accuracy": correct / seen,
                        "seen": seen,
                    }
                )

    metrics = {"loss": loss_sum / seen, "accuracy": correct / seen}
    auroc = binary_auroc(positive_scores, all_targets) if positive_scores else None
    if auroc is not None:
        metrics["auroc"] = auroc
    return metrics
