"""Local pre-Modal gate for the fusion architecture (v2 Task 2).

Prints forward-pass shapes for all four modes and runs the coverage-masked training check
the user asked for BEFORE any Modal spend:
  (1) a batch containing a full-coverage, a partial-coverage, and (if present) a zero-axial
      study flows through every mode with NO NaN/Inf;
  (2) the alignment invariant holds (aligned axial token == its source token at each slot);
  (3) a few optimizer steps reduce the loss and the view-embedding + missing-axial
      placeholder actually receive gradient (they are load-bearing, not decorative).

Run:  ./.venv/bin/python scripts/fusion_smoke.py --data_dir data/rsna
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.rsna_fusion import make_rsna_fusion_splits, rsna_fusion_collate_fn
from models.spine_fusion import build_fusion_model


def _pick_batch(train_ds):
    """Choose studies exercising full / partial / zero axial coverage."""
    full = partial = zero = None
    for i, s in enumerate(train_ds.samples):
        k = len(s["levels"])
        na = len(train_ds.axial_index.get(s["study_id"], {})
                 .keys() & {lv["level_idx"] for lv in s["levels"]})
        if na == k and full is None:
            full = i
        elif 0 < na < k and partial is None:
            partial = i
        elif na == 0 and zero is None:
            zero = i
    idxs = [j for j in (full, partial, zero) if j is not None]
    # pad to >=3 with leading samples so packing/masking is non-trivial
    for i in range(len(train_ds)):
        if len(idxs) >= 3:
            break
        if i not in idxs:
            idxs.append(i)
    tags = {full: "full", partial: "partial", zero: "zero"}
    return idxs, tags


def _finite(name, t):
    ok = torch.isfinite(t).all().item()
    print(f"    {name:<22} shape {tuple(t.shape)}  finite={ok}")
    assert ok, f"{name} has NaN/Inf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)

    cfg = dict(dataset="rsna", data_dir=args.data_dir, task="stenosis", seed=42,
               image_size=224, box_size=32, axial_box_size=32,
               embed_dim=256, encoder_layers=2, encoder_heads=4, pos_encoding="ordinal",
               num_stenosis_classes=3, num_pfirrmann_classes=5, num_classes=3,
               backbone="dinov2_vits14", freeze_backbone=True, head="ce")

    print("[1] building fusion splits (seed 42) ...")
    train_ds, val_ds, test_ds = make_rsna_fusion_splits(args.data_dir, cfg)
    print(f"    train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)} studies")

    idxs, tags = _pick_batch(train_ds)
    items = [train_ds[i] for i in idxs]
    print("    batch studies (coverage):")
    for i, it in zip(idxs, items):
        print(f"      study {it['study_id']:>10}  sag_levels={it['num_levels']} "
              f"axial={it['axial_num']}  [{tags.get(i,'fill')}]")
    batch = rsna_fusion_collate_fn(items)
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)

    N = batch["level_indices"].shape[0]
    M = batch["axial_images"].shape[0]
    cov = (batch["axial_slot"] >= 0)
    print(f"\n[2] batch tensors: N_sag_tokens={N}  M_axial_slices={M}  "
          f"axial-covered levels={int(cov.sum())}/{N}  masked levels={int((~cov).sum())}")
    print(f"    images {tuple(batch['images'].shape)}  boxes {tuple(batch['boxes'].shape)}  "
          f"axial_images {tuple(batch['axial_images'].shape)}  axial_boxes {tuple(batch['axial_boxes'].shape)}")
    assert (~cov).any(), "no masked level in batch — masking path not exercised"

    print("\n[3] forward-pass shapes per mode (all must be finite):")
    for views, fusion in [("sag", None), ("axial", None), ("both", "concat"), ("both", "attn")]:
        c = dict(cfg, views=views, fusion=fusion or "attn")
        model = build_fusion_model(c).to(device).eval()
        name = f"{views}" + (f"/{fusion}" if views == "both" else "")
        with torch.no_grad():
            out = model(batch)
        print(f"  -- mode {name}  (trainable params {model.count_trainable_params()/1e6:.3f}M)")
        _finite("logits", out["logits"])
        _finite("encoded_tokens", out["encoded_tokens"])
        assert out["logits"].shape == (N, 3), f"expected per-level logits (N,3), got {tuple(out['logits'].shape)}"
        assert out["disc_mask"].sum().item() == N

    print("\n[4] alignment invariant: aligned axial token == source token at each slot")
    model = build_fusion_model(dict(cfg, views="both", fusion="attn")).to(device).eval()
    with torch.no_grad():
        ax = model._axial_tokens(batch)
        aligned, cov2 = model._axial_aligned(ax, batch["axial_slot"], N)
        slot = batch["axial_slot"]
        max_err = 0.0
        for i in range(N):
            if slot[i] >= 0:
                max_err = max(max_err, (aligned[i] - ax[slot[i]]).abs().max().item())
        # masked positions must equal the (untrained) missing placeholder
        miss_err = (aligned[~cov2] - model.missing_axial).abs().max().item() if (~cov2).any() else 0.0
    print(f"    covered scatter max|err| = {max_err:.2e}   masked==placeholder max|err| = {miss_err:.2e}")
    assert max_err < 1e-5 and miss_err < 1e-5

    print("\n[5] coverage-masked TRAINING check: 4 steps, loss must fall; the view-embedding")
    print("    must receive gradient in BOTH modes. Masking mechanism differs by design:")
    print("      - attn (fusion-B): masks by ABSENCE (no axial token in the seq) -> no placeholder")
    print("      - concat (fusion-A): masks by SUBSTITUTION -> missing-axial placeholder gets grad")
    crit = torch.nn.CrossEntropyLoss(ignore_index=-1)
    targets = batch["targets"]
    for fusion in ("attn", "concat"):
        model = build_fusion_model(dict(cfg, views="both", fusion=fusion)).to(device).train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        losses, ve_norm, ma_norm = [], None, None
        for step in range(4):
            out = model(batch)
            loss = crit(out["logits"], targets[out["disc_mask"]])
            opt.zero_grad()
            loss.backward()
            if step == 0:
                ve_g = model.view_embedding.weight.grad
                ma_g = model.missing_axial.grad
                ve_norm = ve_g.norm().item() if ve_g is not None else 0.0
                ma_norm = ma_g.norm().item() if ma_g is not None else 0.0
            opt.step()
            losses.append(float(loss.item()))
        print(f"    [{fusion:>6}] loss {[round(l,4) for l in losses]}  "
              f"view_emb.grad={ve_norm:.3e}  missing_axial.grad={ma_norm:.3e}")
        assert np.isfinite(losses).all() and losses[-1] < losses[0], f"{fusion}: loss did not decrease"
        assert ve_norm > 0, f"{fusion}: view-embedding got NO gradient (not load-bearing!)"
        if fusion == "concat":
            assert ma_norm > 0, "concat: missing-axial placeholder got NO gradient despite masked levels"

    print("\n[6] sag-only study (axial_num==0) flows through both fusion modes without NaN")
    if any(it["axial_num"] == 0 for it in items):
        for fusion in ("concat", "attn"):
            m = build_fusion_model(dict(cfg, views="both", fusion=fusion)).to(device).eval()
            with torch.no_grad():
                o = m(batch)
            assert torch.isfinite(o["logits"]).all()
            print(f"    {fusion}: OK")
    else:
        print("    (no zero-axial study in this batch; partial-coverage masking already exercised in [5])")

    print("\nALL CHECKS PASSED — shapes + coverage-masked training verified locally.")


if __name__ == "__main__":
    main()
