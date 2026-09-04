import torch
import torch.nn as nn

from .backbone import build_backbone
from .tokenizer import build_tokenizer
from .encoder import AnatomyEncoder
from .heads import GradingHeads


class SpineGrader(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.task = config.get("task", "stenosis")

        self.backbone = build_backbone(config)

        # The backbone decides the feature width, so copy those onto the config the tokenizer sees. copy first, we do not want to edit the caller's dict
        config = dict(config)
        config["backbone_dim"] = getattr(self.backbone, "embed_dim",
                                         config.get("backbone_dim", 384))
        config["patch_size"] = getattr(self.backbone, "patch_size",
                                       config.get("patch_size", 14))

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

    def make_tokens(self, batch):
        # The cast_crop baseline cuts from the raw image, so it needs no feature map
        if self.config.get("tokenizer") == "cast_crop":
            feature_map = None
        else:
            feature_map = self.backbone(batch["images"])

        return self.tokenizer(feature_map, batch["boxes"], batch["level_indices"],
                              batch["num_levels"], images=batch["images"])

    def forward(self, batch):
        tokens = self.make_tokens(batch)
        encoded = self.encoder(tokens, batch["level_indices"], batch["level_types"],
                               batch["num_levels"])
        logits, disc_mask = self.heads(encoded, batch["level_types"], task=self.task)

        disc_levels = batch["level_indices"][disc_mask]
        return {
            "logits": logits,
            "disc_mask": disc_mask,
            "encoded_tokens": encoded,
            "disc_level_indices": disc_levels,
        }

    def forward_with_attention(self, batch):
        # Same as forward but keeps the attention maps for the overlay figures
        with torch.no_grad():
            tokens = self.make_tokens(batch)
            encoded, attention = self.encoder.forward_with_attention(
                tokens, batch["level_indices"], batch["level_types"], batch["num_levels"])
            logits, disc_mask = self.heads(encoded, batch["level_types"], task=self.task)

            disc_levels = batch["level_indices"][disc_mask]
            out = {
                "logits": logits,
                "disc_mask": disc_mask,
                "encoded_tokens": encoded,
                "disc_level_indices": disc_levels,
            }
            return out, attention

    def count_trainable_params(self):
        total = 0
        for parameter in self.parameters():
            if parameter.requires_grad:
                total = total + parameter.numel()
        return total


def build_model(config):
    return SpineGrader(config)
