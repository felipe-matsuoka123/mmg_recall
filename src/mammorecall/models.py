from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SimpleMammoCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(input_channels, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def _adapt_resnet_input(model: nn.Module, input_channels: int, pretrained: bool) -> None:
    if input_channels == 3:
        return
    if input_channels != 1:
        raise ValueError(f"input_channels must be 1 or 3 for ResNet, got {input_channels}")

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        input_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    model.conv1 = new_conv


def build_model(
    model_name: str,
    *,
    input_channels: int,
    num_classes: int,
    pretrained: bool = False,
) -> nn.Module:
    if model_name == "simple_cnn":
        return SimpleMammoCNN(input_channels, num_classes)
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        _adapt_resnet_input(model, input_channels, pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model '{model_name}'. Use 'simple_cnn' or 'resnet18'.")
