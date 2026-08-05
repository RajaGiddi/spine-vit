import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone


class DiscHeatmapDetector(nn.Module):
    def __init__(self, config, num_levels=5, out_size=56, hidden=128):
        super().__init__()
        self.num_levels = num_levels
        self.out_size = out_size
        self.scale = 224 / out_size

        self.backbone = build_backbone(config)
        backbone_dim = getattr(self.backbone, "embed_dim", config.get("backbone_dim", 384))

        self.reduce = nn.Sequential(nn.Conv2d(backbone_dim, hidden, 1), nn.GELU())
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_levels, 3, padding=1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        hidden = self.reduce(feat)
        hidden = F.interpolate(hidden, size=(self.out_size, self.out_size), mode="bilinear",
                          align_corners=False)
        return self.head(hidden)

    def count_trainable_params(self):
        total = 0
        for parameter in self.parameters():
            if parameter.requires_grad:
                total = total + parameter.numel()
        return total


def build_detector(config, out_size=56):
    return DiscHeatmapDetector(config, num_levels=config.get("num_levels", 5),
                               out_size=out_size)
