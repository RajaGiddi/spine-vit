import torch
import torch.nn as nn

from .backbone import build_backbone
from .tokenizer import build_tokenizer
from .encoder import AnatomyEncoder
from .heads import GradingHeads


class SpineFusionGrader(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.task = config.get("task", "stenosis")
        self.views = config.get("views", "both")
        self.fusion = config.get("fusion", "attn")
        self.sag_slices = int(config.get("sag_slices", 1))
        embed_dim = config.get("embed_dim", 256)

        self.backbone = build_backbone(config)

        config = dict(config)
        config["backbone_dim"] = getattr(self.backbone, "embed_dim",
                                         config.get("backbone_dim", 384))
        config["patch_size"] = getattr(self.backbone, "patch_size",
                                       config.get("patch_size", 14))
        config["tokenizer"] = "anatomy"
        self.tokenizer = build_tokenizer(config)

        self.encoder = AnatomyEncoder(
            embed_dim=embed_dim,
            num_heads=config.get("encoder_heads", 4),
            num_layers=config.get("encoder_layers", 2),
            dropout=config.get("dropout", 0.1),
            max_levels=config.get("max_levels", 24),
            pos_encoding=config.get("pos_encoding", "ordinal"),
        )
        self.heads = GradingHeads(
            embed_dim=embed_dim,
            num_stenosis_classes=config.get("num_stenosis_classes", 3),
            num_pfirrmann_classes=config.get("num_pfirrmann_classes", 5),
            dropout=config.get("dropout", 0.1),
            head_type=config.get("head", "ce"),
        )

        self.view_embedding = nn.Embedding(2, embed_dim)
        nn.init.zeros_(self.view_embedding.weight)
        self.missing_axial = nn.Parameter(torch.zeros(embed_dim))

        if self.fusion == "concat" and self.views == "both":
            self.concat_proj = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )

    def sag_tokens(self, batch):
        if self.sag_slices > 1 and "sag_multi_images" in batch:
            return self.sag_tokens_multi(batch)

        feature_map = self.backbone(batch["images"])
        return self.tokenizer(feature_map, batch["boxes"], batch["level_indices"],
                              batch["num_levels"], images=batch["images"])

    def sag_tokens_multi(self, batch):
        # We read several parasagittal slices and average the tokens so that the sagittal side gets the same slice count as the axial side
        multi = batch["sag_multi_images"]
        num_slices = multi.shape[1]

        total = None
        for j in range(num_slices):
            slice_images = multi[:, j]
            feature_map = self.backbone(slice_images)
            tokens = self.tokenizer(feature_map, batch["boxes"], batch["level_indices"],
                                    batch["num_levels"], images=slice_images)
            if total is None:
                total = tokens
            else:
                total = total + tokens

        return total / num_slices

    def axial_tokens(self, batch):
        axial_images = batch["axial_images"]

        # Some batches have no axial slice at all, hand back an empty block
        if axial_images.shape[0] == 0:
            return axial_images.new_zeros(0, self.view_embedding.embedding_dim)

        feature_map = self.backbone(axial_images)
        return self.tokenizer(feature_map, batch["axial_boxes"],
                              batch["axial_level_indices"], None, images=axial_images)

    def align_axial(self, axial_tokens, slot, n_total):
        # Slot says which axial token belongs to which sagittal one, -1 means none.
        # Levels with no axial slice get the learned placeholder instead.
        covered = slot >= 0
        aligned = self.missing_axial.unsqueeze(0).expand(n_total, -1).clone()
        if covered.any() and axial_tokens.shape[0] > 0:
            aligned[covered] = axial_tokens[slot[covered]]
        return aligned, covered

    def view_embed(self, view_id, reference):
        # 0 for sagittal, 1 for axial
        indices = torch.full((reference.shape[0],), view_id, dtype=torch.long,
                             device=reference.device)
        return self.view_embedding(indices)

    def attention_forward(self, batch, sag, axial):
        num_levels = batch["num_levels"]
        axial_num = batch["axial_num"]

        sag = sag + self.view_embed(0, sag)
        if axial.shape[0] > 0:
            axial = axial + self.view_embed(1, axial)

        combined_tokens = []
        combined_levels = []
        combined_counts = []
        readout_positions = []

        sag_offset = 0
        axial_offset = 0
        position = 0

        # Build one sequence per study: its sagittal tokens then its axial ones, and note where the sagittal ones landed so we can read them back out.
        for i in range(len(num_levels)):
            count = num_levels[i]
            axial_count = axial_num[i]

            combined_tokens.append(sag[sag_offset:sag_offset + count])
            combined_levels.append(batch["level_indices"][sag_offset:sag_offset + count])
            for j in range(count):
                readout_positions.append(position + j)
            position = position + count

            if axial_count > 0:
                combined_tokens.append(axial[axial_offset:axial_offset + axial_count])
                axial_levels = batch["axial_level_indices"][axial_offset:axial_offset + axial_count]
                combined_levels.append(axial_levels)
                position = position + axial_count

            combined_counts.append(count + axial_count)
            sag_offset = sag_offset + count
            axial_offset = axial_offset + axial_count

        tokens = torch.cat(combined_tokens, 0)
        level_indices = torch.cat(combined_levels, 0)
        level_types = torch.ones_like(level_indices)

        # Attention mixes across view and level, then we take the sagittal slots back
        encoded = self.encoder(tokens, level_indices, level_types, combined_counts)
        readout = torch.tensor(readout_positions, dtype=torch.long, device=encoded.device)
        return encoded[readout]

    def forward(self, batch):
        n_total = batch["level_indices"].shape[0]
        level_indices = batch["level_indices"]
        level_types = batch["level_types"]
        num_levels = batch["num_levels"]

        if self.views == "sag":
            sag = self.sag_tokens(batch)
            fused = sag + self.view_embed(0, sag)
            encoded = self.encoder(fused, level_indices, level_types, num_levels)

        elif self.views == "axial":
            axial = self.axial_tokens(batch)
            aligned, covered = self.align_axial(axial, batch["axial_slot"], n_total)
            fused = aligned + self.view_embed(1, aligned)
            encoded = self.encoder(fused, level_indices, level_types, num_levels)

        elif self.fusion == "concat":
            sag = self.sag_tokens(batch)
            axial = self.axial_tokens(batch)
            aligned, covered = self.align_axial(axial, batch["axial_slot"], n_total)
            sag = sag + self.view_embed(0, sag)
            aligned = aligned + self.view_embed(1, aligned)
            fused = self.concat_proj(torch.cat([sag, aligned], dim=-1))
            encoded = self.encoder(fused, level_indices, level_types, num_levels)

        else:
            sag = self.sag_tokens(batch)
            axial = self.axial_tokens(batch)
            encoded = self.attention_forward(batch, sag, axial)

        logits, disc_mask = self.heads(encoded, level_types, task=self.task)
        return {
            "logits": logits,
            "disc_mask": disc_mask,
            "encoded_tokens": encoded,
            "disc_level_indices": level_indices[disc_mask],
        }

    def count_trainable_params(self):
        total = 0
        for parameter in self.parameters():
            if parameter.requires_grad:
                total = total + parameter.numel()
        return total


def build_fusion_model(config):
    return SpineFusionGrader(config)
