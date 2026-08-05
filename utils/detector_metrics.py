from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

RSNA_LEVEL_NAMES = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


def make_heatmaps(centers_out: torch.Tensor, valid: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    b, k, _ = centers_out.shape
    dev = centers_out.device
    xs = torch.arange(size, device=dev).view(1, 1, 1, size)
    ys = torch.arange(size, device=dev).view(1, 1, size, 1)
    cx = centers_out[..., 0].view(b, k, 1, 1)
    cy = centers_out[..., 1].view(b, k, 1, 1)
    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    return g * valid.view(b, k, 1, 1)


def spatial_softmax(logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """(B,K,H,W) logits -> per-channel probability map summing to 1."""
    b, k, h, w = logits.shape
    return torch.softmax(logits.view(b, k, -1) / temp, dim=-1).view(b, k, h, w)


def soft_argmax(logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """DSNT-style expected coordinate from heatmap LOGITS. (B,K,H,W) -> (B,K,2)."""
    p = spatial_softmax(logits, temp)
    _, _, h, w = logits.shape
    xs = torch.arange(w, device=logits.device).float()
    ys = torch.arange(h, device=logits.device).float()
    ex = (p.sum(dim=2) * xs).sum(dim=-1)
    ey = (p.sum(dim=3) * ys).sum(dim=-1)
    return torch.stack([ex, ey], dim=-1)


def coord_loss(logits: torch.Tensor, gt_coords_out: torch.Tensor, valid: torch.Tensor,
               reg: float = 0.0) -> torch.Tensor:
    p = spatial_softmax(logits)
    _, _, h, w = logits.shape
    xs = torch.arange(w, device=logits.device).float()
    ys = torch.arange(h, device=logits.device).float()
    ex = (p.sum(dim=2) * xs).sum(dim=-1)
    ey = (p.sum(dim=3) * ys).sum(dim=-1)
    d = (ex - gt_coords_out[..., 0]) ** 2 + (ey - gt_coords_out[..., 1]) ** 2
    loss = (d * valid).sum() / valid.sum().clamp_min(1.0)
    if reg > 0:
        vx = (p.sum(dim=2) * (xs[None, None] - ex[..., None]) ** 2).sum(dim=-1)
        vy = (p.sum(dim=3) * (ys[None, None] - ey[..., None]) ** 2).sum(dim=-1)
        var = (vx + vy)
        loss = loss + reg * (var * valid).sum() / valid.sum().clamp_min(1.0)
    return loss


def localization_error_mm(pred_c224: torch.Tensor, gt_c224: torch.Tensor, mm_scale: torch.Tensor) -> torch.Tensor:
    """Per-level Euclidean error in mm. All (B,K,2)/(B,2). Returns (B,K)."""
    d = pred_c224 - gt_c224
    dx = d[..., 0] * mm_scale[:, None, 0]
    dy = d[..., 1] * mm_scale[:, None, 1]
    return torch.sqrt(dx ** 2 + dy ** 2)


class LocalizationReport:
    """Accumulate per-level mm errors and summarize (overall + per level + within-5/10mm)."""

    def __init__(self, level_names: List[str] = None):
        self.level_names = level_names or RSNA_LEVEL_NAMES
        self.err: List[np.ndarray] = []
        self.valid: List[np.ndarray] = []

    def update(self, err_mm: torch.Tensor, valid: torch.Tensor):
        self.err.append(err_mm.detach().cpu().numpy())
        self.valid.append(valid.detach().cpu().numpy())

    def compute(self) -> Dict:
        err = np.concatenate(self.err, axis=0)
        val = np.concatenate(self.valid, axis=0).astype(bool)
        flat = err[val]
        out = {
            "n_levels": int(flat.size),
            "mean_mm": float(flat.mean()),
            "median_mm": float(np.median(flat)),
            "pct_within_5mm": float((flat <= 5).mean() * 100),
            "pct_within_10mm": float((flat <= 10).mean() * 100),
            "per_level": {},
        }
        for li, name in enumerate(self.level_names):
            m = val[:, li]
            if m.sum() == 0:
                continue
            e = err[m, li]
            out["per_level"][name] = {
                "n": int(m.sum()),
                "mean_mm": float(e.mean()),
                "median_mm": float(np.median(e)),
                "pct_within_5mm": float((e <= 5).mean() * 100),
                "pct_within_10mm": float((e <= 10).mean() * 100),
            }
        return out
