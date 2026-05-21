from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def run_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    correct = 0
    seen = 0
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

    return {"loss": loss_sum / seen, "accuracy": correct / seen}
