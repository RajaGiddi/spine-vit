"""Full Spine-ViT model assembly and the build_model() factory.

Pipeline: backbone -> tokenizer -> encoder -> grading heads. The tokenizer and the
encoder's positional-encoding are swappable via config, which is what the ablation
study varies.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .backbone import build_backbone
from .tokenizer import build_tokenizer
from .encoder import AnatomyEncoder
from .heads import GradingHeads


class SpineGrader(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.task = config.get("task", "stenosis")

        self.backbone = build_backbone(config)
        # Keep tokenizer/encoder dims consistent with the actual backbone.
        config = dict(config)
        config["backbone_dim"] = getattr(self.backbone, "embed_dim", config.get("backbone_dim", 384))
        config["patch_size"] = getattr(self.backbone, "patch_size", config.get("patch_size", 14))

        self.tokenizer = build_tokenizer(config)
        self.encoder = AnatomyEncoder(
            embed_dim=config.get("embed_dim", 256),
            num_heads=config.get("encoder_heads", 4),
            num_layers=config.get("encoder_layers", 2),
            dropout=config.get("dropout", 0.1),
            max_levels=config.get("max_levels", 24),
            pos_encoding=config.get("pos_encoding", "ordinal"),
        )
        self.heads = GradingHeads(
            embed_dim=config.get("embed_dim", 256),
            num_stenosis_classes=config.get("num_stenosis_classes", 3),
            num_pfirrmann_classes=config.get("num_pfirrmann_classes", 5),
            dropout=config.get("dropout", 0.1),
            head_type=config.get("head", "ce"),
        )

    def forward(self, batch: Dict) -> Dict:
        # cast_crop uses its own ResNet on image crops; skip the (unused) DINOv2 forward.
        feature_map = None if self.config.get("tokenizer") == "cast_crop" else self.backbone(batch["images"])
        tokens = self.tokenizer(
            feature_map, batch["boxes"], batch["level_indices"], batch["num_levels"], images=batch["images"]
        )
        encoded = self.encoder(
            tokens, batch["level_indices"], batch["level_types"], batch["num_levels"]
        )
        logits, disc_mask = self.heads(encoded, batch["level_types"], task=self.task)
        return {
            "logits": logits,                       # (N_disc, num_classes)
            "disc_mask": disc_mask,                  # (N_total,) bool
            "encoded_tokens": encoded,               # (N_total, embed_dim)
            "disc_level_indices": batch["level_indices"][disc_mask],
        }

    @torch.no_grad()
    def forward_with_attention(self, batch: Dict):
        """Return (output_dict, attention_maps) for visualization."""
        feature_map = None if self.config.get("tokenizer") == "cast_crop" else self.backbone(batch["images"])
        tokens = self.tokenizer(
            feature_map, batch["boxes"], batch["level_indices"], batch["num_levels"], images=batch["images"]
        )
        encoded, attn = self.encoder.forward_with_attention(
            tokens, batch["level_indices"], batch["level_types"], batch["num_levels"]
        )
        logits, disc_mask = self.heads(encoded, batch["level_types"], task=self.task)
        out = {
            "logits": logits,
            "disc_mask": disc_mask,
            "encoded_tokens": encoded,
            "disc_level_indices": batch["level_indices"][disc_mask],
        }
        return out, attn

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: Dict) -> SpineGrader:
    """Factory: construct a SpineGrader from a config dict."""
    return SpineGrader(config)
