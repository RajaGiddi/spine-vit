"""Tokenizers: three ways to turn a backbone feature map into per-level tokens.

All three share one output contract so they are drop-in interchangeable:

    forward(feature_map, boxes, level_indices, num_levels) -> (N_total, embed_dim)

where feature_map is (B, C, H', W'), boxes is (N_total, 5) = [batch_idx, x1, y1, x2, y2]
in INPUT-image coordinates, level_indices is (N_total,), and num_levels is a list of
per-sample token counts summing to N_total.

- AnatomyTokenizer  (ours): ROI-Align on the precise anatomical boxes.
- UniformStripTokenizer   : ROI-Align on K equal horizontal strips (ignores box x/y);
                            tests whether *precise* localization matters.
- PatchTokenizer          : learned per-level query attends over all patch tokens; no
                            spatial localization prior at all.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torchvision.ops import roi_align


class _Projection(nn.Module):
    """Shared ROI feature -> token projection: pool -> Linear -> LayerNorm -> GELU."""

    def __init__(self, backbone_dim: int, embed_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(backbone_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, roi_feats: torch.Tensor) -> torch.Tensor:
        # roi_feats: (N, C, s, s) -> (N, C) -> (N, embed_dim)
        x = self.pool(roi_feats).flatten(1)
        return self.act(self.norm(self.proj(x)))


class AnatomyTokenizer(nn.Module):
    """ROI-Align pooling from the precise per-level anatomical boxes (our method)."""

    def __init__(self, backbone_dim=384, embed_dim=256, roi_output_size=7, spatial_scale=1 / 14, image_size=224):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.spatial_scale = spatial_scale
        self.image_size = image_size
        self.projection = _Projection(backbone_dim, embed_dim)

    def forward(self, feature_map, boxes, level_indices=None, num_levels=None, images=None):
        roi_feats = roi_align(
            feature_map,
            boxes,
            output_size=self.roi_output_size,
            spatial_scale=self.spatial_scale,
            aligned=True,
        )  # (N_total, C, s, s)
        return self.projection(roi_feats)


class UniformStripTokenizer(nn.Module):
    """ROI-Align on K equal, full-width horizontal strips (top -> bottom)."""

    def __init__(self, backbone_dim=384, embed_dim=256, roi_output_size=7, spatial_scale=1 / 14, image_size=224):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.spatial_scale = spatial_scale
        self.image_size = image_size
        self.projection = _Projection(backbone_dim, embed_dim)

    def _strip_boxes(self, num_levels: List[int], device) -> torch.Tensor:
        H = W = float(self.image_size)
        rows = []
        for bi, k in enumerate(num_levels):
            if k <= 0:
                continue
            step = H / k
            for j in range(k):  # j=0 is the topmost strip -> level index 0
                rows.append([float(bi), 0.0, j * step, W, (j + 1) * step])
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def forward(self, feature_map, boxes, level_indices=None, num_levels=None, images=None):
        assert num_levels is not None, "UniformStripTokenizer needs num_levels"
        strip_boxes = self._strip_boxes(num_levels, feature_map.device)
        roi_feats = roi_align(
            feature_map,
            strip_boxes,
            output_size=self.roi_output_size,
            spatial_scale=self.spatial_scale,
            aligned=True,
        )
        return self.projection(roi_feats)


class PatchTokenizer(nn.Module):
    """Learned per-level query cross-attends over all patch tokens (no localization).

    This is the "does structured localization help at all?" baseline. Each token gets a
    learned query selected by its ordinal level index; the query attends over every
    patch token of its own sample. Only precise-vs-learned localization differs from the
    other tokenizers; the downstream encoder/heads are identical.
    """

    def __init__(self, backbone_dim=384, embed_dim=256, max_levels=12, num_heads=4, image_size=224, spatial_scale=1 / 14):
        super().__init__()
        self.image_size = image_size
        self.spatial_scale = spatial_scale
        self.kv_proj = nn.Linear(backbone_dim, embed_dim)
        self.query_embed = nn.Embedding(max_levels, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, feature_map, boxes, level_indices=None, num_levels=None, images=None):
        assert level_indices is not None and num_levels is not None
        b, c, hp, wp = feature_map.shape
        patches = feature_map.flatten(2).transpose(1, 2)  # (B, N_patch, C)
        kv = self.kv_proj(patches)  # (B, N_patch, embed)

        out_tokens = []
        offset = 0
        for bi, k in enumerate(num_levels):
            if k == 0:
                continue
            idx = level_indices[offset : offset + k]  # (k,)
            q = self.query_embed(idx).unsqueeze(0)  # (1, k, embed)
            kv_b = kv[bi : bi + 1]  # (1, N_patch, embed)
            attended, _ = self.attn(q, kv_b, kv_b)  # (1, k, embed)
            out_tokens.append(attended.squeeze(0))
            offset += k
        tokens = torch.cat(out_tokens, dim=0)  # (N_total, embed)
        return self.act(self.norm(tokens))


class CASTCropTokenizer(nn.Module):
    """CAST-style baseline: crop each level's ROI from the ORIGINAL image and encode each
    crop INDEPENDENTLY with a frozen ImageNet ResNet-18 — versus our ROI-Align pooling from
    a shared DINOv2 feature map. Same boxes, same downstream encoder/head; the ONLY
    difference is per-level token extraction (independent CNN crop vs shared-map ROI-Align).
    Tests whether the shared-feature-map mechanism is the architectural novelty.
    """

    def __init__(self, embed_dim=256, crop_size=112, freeze=True, image_size=224):
        super().__init__()
        import torchvision

        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # -> (N, 512, 1, 1)
        self.freeze = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        self.crop_size = crop_size
        self.image_size = image_size
        self.proj = nn.Linear(512, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, feature_map, boxes, level_indices=None, num_levels=None, images=None):
        assert images is not None, "cast_crop tokenizer requires the original images"
        # crop each box from the 224-space image and resize to crop_size (spatial_scale=1.0)
        crops = roi_align(images, boxes, output_size=self.crop_size, spatial_scale=1.0, aligned=True)
        if self.freeze:
            with torch.no_grad():
                feat = self.encoder(crops).flatten(1)   # (N_total, 512)
        else:
            feat = self.encoder(crops).flatten(1)
        return self.act(self.norm(self.proj(feat)))


def build_tokenizer(config: dict) -> nn.Module:
    kind = config.get("tokenizer", "anatomy")
    backbone_dim = config.get("backbone_dim", 384)
    embed_dim = config.get("embed_dim", 256)
    image_size = config.get("image_size", 224)
    spatial_scale = 1.0 / config.get("patch_size", 14)
    if kind == "anatomy":
        return AnatomyTokenizer(backbone_dim, embed_dim, config.get("roi_output_size", 7), spatial_scale, image_size)
    if kind == "strips":
        return UniformStripTokenizer(backbone_dim, embed_dim, config.get("roi_output_size", 7), spatial_scale, image_size)
    if kind == "patches":
        return PatchTokenizer(
            backbone_dim, embed_dim, config.get("max_levels", 12), config.get("encoder_heads", 4), image_size, spatial_scale
        )
    if kind == "cast_crop":
        return CASTCropTokenizer(
            embed_dim=embed_dim, crop_size=config.get("crop_size", 112),
            freeze=config.get("freeze_backbone", True), image_size=image_size,
        )
    raise ValueError(f"Unknown tokenizer '{kind}' (expected anatomy|strips|patches|cast_crop)")
