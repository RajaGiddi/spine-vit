"""Learned disc-level detector — heatmap regression on RSNA coordinate annotations.

Replaces the ground-truth coordinates the grader currently consumes at inference. Shares
the same frozen DINOv2 ViT-S/14 backbone as the grader (so features are shared and the
oracle-vs-detected comparison stays clean), with a lightweight upsampling decoder that
predicts one Gaussian heatmap per disc level (L1/L2 … L5/S1).

Output contract: forward(x: (B,3,224,224)) -> heatmaps (B, 5, out_size, out_size) in
[0,1] (sigmoid). Sub-pixel centers come from soft-argmax (see utils/detector_metrics.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone


class DiscHeatmapDetector(nn.Module):
    def __init__(self, config: dict, num_levels: int = 5, out_size: int = 56, hidden: int = 128):
        super().__init__()
        self.num_levels = num_levels
        self.out_size = out_size  # 224 / 4 = 56  -> clean scale factor of 4 to input space
        self.scale = 224 / out_size

        self.backbone = build_backbone(config)  # frozen DINOv2 (same as grader)
        c = getattr(self.backbone, "embed_dim", config.get("backbone_dim", 384))

        # Lightweight decoder: 1x1 reduce -> bilinear upsample to out_size -> two convs.
        # interpolate+conv (not transposed conv) avoids checkerboard artifacts. ~0.13M params.
        self.reduce = nn.Sequential(nn.Conv2d(c, hidden, 1), nn.GELU())
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_levels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns raw heatmap LOGITS; decoding uses spatial-softmax (see detector_metrics)
        feat = self.backbone(x)                       # (B, C, 16, 16)
        h = self.reduce(feat)                          # (B, hidden, 16, 16)
        h = F.interpolate(h, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return self.head(h)                            # (B, num_levels, out_size, out_size)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_detector(config: dict, out_size: int = 56) -> DiscHeatmapDetector:
    return DiscHeatmapDetector(config, num_levels=config.get("num_levels", 5), out_size=out_size)
