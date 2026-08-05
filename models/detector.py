from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone


class DiscHeatmapDetector(nn.Module):
    def __init__(self, config: dict, num_levels: int = 5, out_size: int = 56, hidden: int = 128):
        super().__init__()
        self.num_levels = num_levels
        self.out_size = out_size
        self.scale = 224 / out_size

        self.backbone = build_backbone(config)
        c = getattr(self.backbone, "embed_dim", config.get("backbone_dim", 384))

        self.reduce = nn.Sequential(nn.Conv2d(c, hidden, 1), nn.GELU())
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_levels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        h = self.reduce(feat)
        h = F.interpolate(h, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return self.head(h)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_detector(config: dict, out_size: int = 56) -> DiscHeatmapDetector:
    return DiscHeatmapDetector(config, num_levels=config.get("num_levels", 5), out_size=out_size)
