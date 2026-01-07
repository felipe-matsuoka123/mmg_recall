from __future__ import annotations
import torch
import torch.nn as nn
import timm

class MammogramClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        in_chans: int = 1,
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,        
            global_pool="avg",
        )
        feat_dim = self.net.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net(x)
        logits = self.head(feats)
        return logits
