from __future__ import annotations

import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    def __init__(self, model_name: str = "dinov2_vits14", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.spatial_scale = 1.0 / self.patch_size

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode regardless of the parent's mode."""
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        hp, wp = h // self.patch_size, w // self.patch_size
        tokens = self.model.get_intermediate_layers(x, n=1)[0]
        feat = tokens.transpose(1, 2).reshape(b, self.embed_dim, hp, wp)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                return self._extract(x)
        return self._extract(x)


class MockBackbone(nn.Module):
    """Offline stand-in with the DINOv2 output contract (no downloads)."""

    def __init__(self, embed_dim: int = 384, patch_size: int = 14, freeze: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.spatial_scale = 1.0 / patch_size
        self.freeze = freeze
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.stem.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.stem.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        hp, wp = h // self.patch_size, w // self.patch_size
        feat = self.stem(x)
        feat = nn.functional.adaptive_avg_pool2d(feat, (hp, wp))
        if self.freeze:
            feat = feat.detach()
        return feat


def build_backbone(config: dict) -> nn.Module:
    name = config.get("backbone", "dinov2_vits14")
    freeze = config.get("freeze_backbone", True)
    if name == "mock":
        return MockBackbone(
            embed_dim=config.get("backbone_dim", 384),
            patch_size=config.get("patch_size", 14),
            freeze=freeze,
        )
    return DINOv2Backbone(model_name=name, freeze=freeze)
