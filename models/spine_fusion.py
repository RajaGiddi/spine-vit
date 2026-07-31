"""Two-view (sagittal + axial) canal-stenosis grader — v2 Task 2.

Reuses the v1 parts unchanged: a SHARED frozen DINOv2 backbone and a SHARED anatomy
ROI-Align tokenizer encode BOTH views, so the only thing distinguishing a sagittal token
from the axial token AT THE SAME LEVEL is the learned view-type embedding (both are disc
tokens, same ordinal level index, same level-type). The view embedding is therefore
load-bearing, not decorative — zero-initialized so it must earn its contribution.

Modes (config `views` x `fusion`):
  - views="sag"                : sagittal token per level (v1 control, matched no-aug).
  - views="axial"              : axial token per level (missing -> placeholder).
  - views="both", fusion="concat" (A): per level, proj([sag ; axial]) -> one fused token.
  - views="both", fusion="attn"   (B): sag + axial tokens share ONE transformer sequence;
                                      self-attention mixes them across view AND level;
                                      per-level readout comes from the sagittal position.

Masked fusion: coverage is partial, so a level with no axial slice uses `missing_axial`
(concat) or simply contributes no axial token (attn) — every study still trains on its
sagittal tokens. Output contract matches SpineGrader (logits over the per-level sagittal
tokens, disc_mask all-True) so train.py's run_epoch/evaluate_split are reused unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .backbone import build_backbone
from .tokenizer import build_tokenizer
from .encoder import AnatomyEncoder
from .heads import GradingHeads


class SpineFusionGrader(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.task = config.get("task", "stenosis")
        self.views = config.get("views", "both")        # "sag" | "axial" | "both"
        self.fusion = config.get("fusion", "attn")       # "concat" | "attn" (views=="both")
        self.sag_slices = int(config.get("sag_slices", 1))  # >1 -> parasagittal budget control
        embed_dim = config.get("embed_dim", 256)

        self.backbone = build_backbone(config)           # shared, frozen
        config = dict(config)
        config["backbone_dim"] = getattr(self.backbone, "embed_dim", config.get("backbone_dim", 384))
        config["patch_size"] = getattr(self.backbone, "patch_size", config.get("patch_size", 14))
        config["tokenizer"] = "anatomy"                  # fusion tests the anatomy tokenizer
        self.tokenizer = build_tokenizer(config)         # shared across views

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

        # view-type embedding: 0=sagittal, 1=axial. THE feature that separates the two
        # views' tokens. Zero-init -> starts as a no-op, must be learned.
        self.view_embedding = nn.Embedding(2, embed_dim)
        nn.init.zeros_(self.view_embedding.weight)
        # learned placeholder for a level with no axial slice (masked fusion).
        self.missing_axial = nn.Parameter(torch.zeros(embed_dim))

        if self.fusion == "concat" and self.views == "both":
            self.concat_proj = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim)
            )

    # ---- view tokenization (shared backbone + tokenizer) ------------------------------
    def _sag_tokens(self, batch: Dict) -> torch.Tensor:
        if self.sag_slices > 1 and "sag_multi_images" in batch:
            return self._sag_tokens_multi(batch)
        fmap = self.backbone(batch["images"])                          # (B,384,16,16)
        return self.tokenizer(fmap, batch["boxes"], batch["level_indices"],
                              batch["num_levels"], images=batch["images"])  # (N_total,D)

    def _sag_tokens_multi(self, batch: Dict) -> torch.Tensor:
        """Mean-pool the per-level token over K parasagittal slices (budget control). Same
        (N_total, D) output as the single-slice path."""
        multi = batch["sag_multi_images"]                              # (B,K,3,H,W)
        K = multi.shape[1]
        acc = None
        for j in range(K):
            fmap = self.backbone(multi[:, j])
            tok = self.tokenizer(fmap, batch["boxes"], batch["level_indices"],
                                 batch["num_levels"], images=multi[:, j])
            acc = tok if acc is None else acc + tok
        return acc / K

    def _axial_tokens(self, batch: Dict) -> torch.Tensor:
        """(M, D) axial tokens, one per covered level, ordered study-by-study. Empty if none."""
        ax_imgs = batch["axial_images"]
        if ax_imgs.shape[0] == 0:
            return ax_imgs.new_zeros(0, self.view_embedding.embedding_dim)
        fmap = self.backbone(ax_imgs)                                   # (M,384,16,16)
        return self.tokenizer(fmap, batch["axial_boxes"], batch["axial_level_indices"],
                              None, images=ax_imgs)                     # (M,D)

    def _axial_aligned(self, ax_tokens: torch.Tensor, slot: torch.Tensor,
                       n_total: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatter axial tokens into sagittal-aligned (N_total, D); missing -> placeholder.
        Returns (aligned, coverage_mask)."""
        cov = slot >= 0
        aligned = self.missing_axial.unsqueeze(0).expand(n_total, -1).clone()
        if cov.any() and ax_tokens.shape[0] > 0:
            aligned[cov] = ax_tokens[slot[cov]]
        return aligned, cov

    def _emb(self, view_id: int, ref: torch.Tensor) -> torch.Tensor:
        idx = torch.full((ref.shape[0],), view_id, dtype=torch.long, device=ref.device)
        return self.view_embedding(idx)

    # ---- fusion-B: one transformer sequence over sag+axial tokens ---------------------
    def _attn_forward(self, batch: Dict, sag: torch.Tensor, ax: torch.Tensor):
        """Build per-study [sag block | axial block], run the shared encoder, read out the
        sagittal positions. view_embedding is what lets attention tell the views apart."""
        num_levels: List[int] = batch["num_levels"]
        axial_num: List[int] = batch["axial_num"]
        lvl_sag = batch["level_indices"]
        lvl_ax = batch["axial_level_indices"]

        sag = sag + self._emb(0, sag)
        if ax.shape[0] > 0:
            ax = ax + self._emb(1, ax)

        comb_tokens, comb_lvl, comb_num, sag_readout = [], [], [], []
        sag_off = ax_off = run = 0
        for k, na in zip(num_levels, axial_num):
            comb_tokens.append(sag[sag_off:sag_off + k])
            comb_lvl.append(lvl_sag[sag_off:sag_off + k])
            sag_readout.extend(range(run, run + k))       # sag positions in combined stream
            run += k
            if na > 0:
                comb_tokens.append(ax[ax_off:ax_off + na])
                comb_lvl.append(lvl_ax[ax_off:ax_off + na])
                run += na
            comb_num.append(k + na)
            sag_off += k
            ax_off += na

        comb = torch.cat(comb_tokens, 0)
        comb_lvl_idx = torch.cat(comb_lvl, 0)
        comb_types = torch.ones_like(comb_lvl_idx)        # all disc
        encoded = self.encoder(comb, comb_lvl_idx, comb_types, comb_num)  # (N_comb, D)
        readout = torch.tensor(sag_readout, dtype=torch.long, device=encoded.device)
        return encoded[readout]                           # (N_total, D) sag-aligned

    def forward(self, batch: Dict) -> Dict:
        n_total = batch["level_indices"].shape[0]
        level_indices = batch["level_indices"]
        level_types = batch["level_types"]
        num_levels = batch["num_levels"]

        # -- compute per-level fused token stream (one token per sagittal level) --
        if self.views == "sag":
            sag = self._sag_tokens(batch)
            fused = sag + self._emb(0, sag)
            encoded = self.encoder(fused, level_indices, level_types, num_levels)

        elif self.views == "axial":
            ax = self._axial_tokens(batch)
            aligned, _ = self._axial_aligned(ax, batch["axial_slot"], n_total)
            fused = aligned + self._emb(1, aligned)
            encoded = self.encoder(fused, level_indices, level_types, num_levels)

        elif self.fusion == "concat":                     # fusion-A
            sag = self._sag_tokens(batch)
            ax = self._axial_tokens(batch)
            aligned, _ = self._axial_aligned(ax, batch["axial_slot"], n_total)
            sag = sag + self._emb(0, sag)
            aligned = aligned + self._emb(1, aligned)
            fused = self.concat_proj(torch.cat([sag, aligned], dim=-1))
            encoded = self.encoder(fused, level_indices, level_types, num_levels)

        else:                                             # fusion-B (attn)
            sag = self._sag_tokens(batch)
            ax = self._axial_tokens(batch)
            encoded = self._attn_forward(batch, sag, ax)

        logits, disc_mask = self.heads(encoded, level_types, task=self.task)
        return {
            "logits": logits,
            "disc_mask": disc_mask,
            "encoded_tokens": encoded,
            "disc_level_indices": level_indices[disc_mask],
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_fusion_model(config: Dict) -> SpineFusionGrader:
    return SpineFusionGrader(config)
