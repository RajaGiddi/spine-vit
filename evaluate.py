"""Standalone evaluation + figure generation for Spine-ViT.

Two modes:
  * default: reload each experiment's best_model.pt, run it on the test split, compute
    metrics + level-attribution, and generate every paper figure.
  * --from_saved: skip inference and aggregate each experiment's already-written
    test_results.json / history.json (no data or GPU needed). Handy for regenerating
    the comparison table and curves.

Usage:
    python evaluate.py --experiments_dir outputs --data_dir /path/to/rsna-2024 --generate_figures
    python evaluate.py --experiments_dir outputs --from_saved --generate_figures
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List

import numpy as np
import torch

from models import build_model
from utils.metrics import compute_metrics, LevelAttributionAnalyzer, RSNA_LEVEL_NAMES
from utils import visualization as viz

CLASS_NAMES = {
    "stenosis": ["Normal/Mild", "Moderate", "Severe"],
    "pfirrmann": ["I", "II", "III", "IV", "V"],
}


def find_experiments(experiments_dir: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(experiments_dir)):
        d = os.path.join(experiments_dir, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
            out.append(d)
    return out


def _load_json(path: str):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# --------------------------------------------------------------------------------------
# Inference over the test split
# --------------------------------------------------------------------------------------
def run_test_inference(exp_dir: str, config: Dict, device: torch.device):
    """Return (metrics, attribution, preds, targets, levels, sample_batch, attn)."""
    from train import get_dataloaders, move_batch

    _, _, _, test_loader = get_dataloaders(config)
    model = build_model(config).to(device)
    ckpt = torch.load(os.path.join(exp_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    level_names = RSNA_LEVEL_NAMES if config["dataset"] == "rsna" else None
    analyzer = LevelAttributionAnalyzer(level_names=level_names, num_classes=config["num_classes"])
    all_preds, all_targets, all_levels = [], [], []
    sample_batch, attn = None, None

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = move_batch(batch, device)
            if i == 0:  # capture attention on the first batch for figures
                out, attn = model.forward_with_attention(batch)
                sample_batch = batch
            else:
                out = model(batch)
            targets = batch["targets"][out["disc_mask"]]
            preds = out["logits"].argmax(-1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_levels.append(out["disc_level_indices"].cpu().numpy())

    preds = np.concatenate(all_preds) if all_preds else np.array([])
    targets = np.concatenate(all_targets) if all_targets else np.array([])
    levels = np.concatenate(all_levels) if all_levels else np.array([])
    analyzer.update(preds, targets, levels)

    metrics = compute_metrics(preds, targets, config["num_classes"])
    attribution = analyzer.compute()
    return metrics, attribution, preds, targets, levels, sample_batch, attn


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def make_attention_figures(exp_dir, config, sample_batch, attn, eval_dir):
    """Attention heatmap + overlay for the first sample of the test batch."""
    if sample_batch is None or not attn:
        return
    # First sample: how many tokens does it have?
    k0 = sample_batch["num_levels"][0]
    level_types0 = sample_batch["level_types"][:k0].cpu().numpy()
    level_idx0 = sample_batch["level_indices"][:k0].cpu().numpy()

    if config["dataset"] == "rsna":
        labels = [RSNA_LEVEL_NAMES[i] if i < len(RSNA_LEVEL_NAMES) else f"L{i}" for i in level_idx0]
    else:
        labels = [f"{'V' if t == 0 else 'D'}{i}" for i, t in zip(level_idx0, level_types0)]

    last_attn = attn[-1][0, :k0, :k0].cpu().numpy()  # last layer, first sample
    viz.plot_attention_weights(last_attn, labels, os.path.join(eval_dir, "attention_weights.png"))

    # Overlay: received attention per box (column-sum).
    received = last_attn.sum(axis=0)
    img0 = sample_batch["images"][0].cpu().numpy()
    boxes0 = sample_batch["boxes"][sample_batch["boxes"][:, 0] == 0][:, 1:].cpu().numpy()
    viz.plot_attention_overlay(img0, boxes0, received, labels, os.path.join(eval_dir, "attention_overlay.png"))


def generate_per_experiment_figures(exp_dir, config, preds, targets, eval_dir):
    name = os.path.basename(exp_dir)
    class_names = CLASS_NAMES[config["task"]]
    viz.plot_grade_confusion_matrix(
        targets, preds, class_names, f"Confusion — {name}", os.path.join(eval_dir, f"confusion_{name}.png")
    )
    history = _load_json(os.path.join(exp_dir, "history.json"))
    if history:
        viz.plot_training_curves(history, os.path.join(eval_dir, f"curves_{name}.png"))


# --------------------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------------------
def print_comparison_table(rows: List[Dict]):
    header = (f"{'Experiment':40s} {'Tokenizer':9s} {'PosEnc':8s} {'MacroF1':>8s} {'Kappa':>7s} "
              f"{'BalAcc':>7s} {'WorstLvl':>8s} {'PathP':>6s} {'PathR':>6s}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:40s} {r['tokenizer']:9s} {r['pos_encoding']:8s} "
            f"{r['macro_f1']:8.3f} {r['kappa']:7.3f} {r['balanced_acc']:7.3f} "
            f"{r['worst_lvl']:8.3f} {r['path_prec']:6.3f} {r['path_rec']:6.3f}"
        )
    print("-" * len(header))
    print(f"{'LumbarDISC framework (ref)':40s} {'Cuboid':9s} {'Context':8s} "
          f"{0.783:8.3f} {0.765:7.3f} {'—':>7s} {'—':>8s} {'—':>6s} {'—':>6s}\n")


def save_markdown_table(rows: List[Dict], path: str):
    lines = [
        "| Experiment | Tokenizer | Pos Enc | Macro F1 | κ | Bal Acc | Worst-Level Acc | Pathology P | Pathology R |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['tokenizer']} | {r['pos_encoding']} | "
            f"{r['macro_f1']:.3f} | {r['kappa']:.3f} | {r['balanced_acc']:.3f} | "
            f"{r['worst_lvl']:.3f} | {r['path_prec']:.3f} | {r['path_rec']:.3f} |"
        )
    lines.append("| LumbarDISC framework (ref) | Cuboid | Context | 0.783 | 0.765 | — | — | — | — |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------
# Human-readable labels + ordering for figures
# --------------------------------------------------------------------------------------
_TOK = {"anatomy": "Anatomy", "strips": "Strips", "patches": "Patches", "cast_crop": "CAST-crop"}
_PE = {"ordinal": "Ordinal", "learned": "Learned", "none": "None"}


def pretty_label(a: Dict) -> str:
    """Clean figure label from an aggregated row, e.g. 'Anatomy+Ordinal (ours)'."""
    ft = a["group"].endswith("_ft")
    lab = f"{_TOK.get(a['tokenizer'], a['tokenizer'])}+{_PE.get(a['pos_encoding'], a['pos_encoding'])}"
    if ft:
        return lab + " (FT)"
    if a["tokenizer"] == "anatomy" and a["pos_encoding"] == "ordinal":
        return lab + " (ours)"
    if a["tokenizer"] in ("strips", "patches", "cast_crop"):
        return lab + " (baseline)"
    return lab


def _order_key(a: Dict) -> int:
    """Fixed logical order: ours, learned, none, fine-tuned, patches, strips."""
    if a["group"].endswith("_ft"):
        return 3
    return {("anatomy", "ordinal"): 0, ("anatomy", "learned"): 1, ("anatomy", "none"): 2,
            ("cast_crop", "ordinal"): 4, ("patches", "ordinal"): 5, ("strips", "ordinal"): 6}.get(
        (a["tokenizer"], a["pos_encoding"]), 9)


def aggregate_perlevel(analyzer_results: Dict[str, Dict], rows: List[Dict]) -> Dict[str, Dict]:
    """Average per-level exact_acc across seeds -> {pretty_label: {'per_level': {...}}}."""
    meta = {r["name"]: r for r in rows}  # name -> row (has group/tokenizer/pos_encoding)
    groups: Dict[str, List] = {}
    for name, attr in analyzer_results.items():
        if name in meta:
            groups.setdefault(meta[name]["group"], []).append((meta[name], attr))

    entries = []
    for group, items in groups.items():
        row0 = items[0][0]
        levels: List[str] = []
        for _, attr in items:
            for lv in attr.get("per_level", {}):
                if lv not in levels:
                    levels.append(lv)
        merged = {}
        for lv in levels:
            vals = [attr["per_level"][lv]["exact_acc"] for _, attr in items if lv in attr.get("per_level", {})]
            if vals:
                merged[lv] = {"exact_acc": float(np.mean(vals))}
        entries.append((row0, {"per_level": merged}))

    entries.sort(key=lambda e: _order_key(e[0]))
    return {pretty_label(row): res for row, res in entries}


def aggregate_over_seeds(rows: List[Dict]) -> List[Dict]:
    """Group runs by config (seed-stripped) and return mean/std per metric across seeds."""
    metrics = ["macro_f1", "kappa", "balanced_acc", "worst_lvl", "path_prec", "path_rec"]
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    out = []
    for g, rs in groups.items():
        agg = {"group": g, "tokenizer": rs[0]["tokenizer"], "pos_encoding": rs[0]["pos_encoding"], "n_seeds": len(rs)}
        for m in metrics:
            vals = np.array([r[m] for r in rs], dtype=float)
            agg[f"{m}_mean"], agg[f"{m}_std"] = float(vals.mean()), float(vals.std())
        out.append(agg)
    return out


def print_aggregated_table(agg: List[Dict]):
    if not agg:
        return
    print("\n=== Aggregated over seeds (mean ± std) ===")
    header = f"{'Config (seed-stripped)':34s} {'seeds':>5s} {'MacroF1':>15s} {'Kappa':>15s} {'WorstLvl':>15s}"
    print(header)
    print("-" * len(header))
    for a in sorted(agg, key=lambda x: -x["kappa_mean"]):
        print(
            f"{a['group']:34s} {a['n_seeds']:5d} "
            f"{a['macro_f1_mean']:.3f}±{a['macro_f1_std']:.3f}   "
            f"{a['kappa_mean']:.3f}±{a['kappa_std']:.3f}   "
            f"{a['worst_lvl_mean']:.3f}±{a['worst_lvl_std']:.3f}"
        )
    print("-" * len(header))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    eval_dir = os.path.join(args.experiments_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    rows: List[Dict] = []
    analyzer_results: Dict[str, Dict] = {}

    for exp_dir in find_experiments(args.experiments_dir):
        name = os.path.basename(exp_dir)
        config = _load_json(os.path.join(exp_dir, "config.json"))
        if config is None:
            continue
        if args.data_dir:
            config["data_dir"] = args.data_dir

        if args.from_saved:
            test_out = _load_json(os.path.join(exp_dir, "test_results.json"))
            if test_out is None:
                print(f"[skip] {name}: no test_results.json")
                continue
            metrics, attribution = test_out["metrics"], test_out["attribution"]
        else:
            if not os.path.exists(os.path.join(exp_dir, "best_model.pt")):
                print(f"[skip] {name}: no best_model.pt")
                continue
            print(f"[eval] {name}")
            metrics, attribution, preds, targets, levels, sample_batch, attn = run_test_inference(exp_dir, config, device)
            if args.generate_figures:
                generate_per_experiment_figures(exp_dir, config, preds, targets, eval_dir)
                make_attention_figures(exp_dir, config, sample_batch, attn, eval_dir)

        rows.append(
            {
                "name": name,
                # config signature with the seed suffix stripped -> groups seeds together
                "group": re.sub(r"_s\d+$", "", name),
                "tokenizer": config.get("tokenizer", "?"),
                "pos_encoding": config.get("pos_encoding", "?"),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "kappa": metrics.get("kappa", 0.0),
                "balanced_acc": metrics.get("balanced_acc", 0.0),
                # honest level metrics: non-gameable worst-level id + detection precision/recall
                "worst_lvl": attribution.get("worst_level_accuracy") or 0.0,
                "path_prec": attribution.get("pathology_precision", 0.0),
                "path_rec": attribution.get("pathology_recall", 0.0),
            }
        )
        analyzer_results[name] = attribution

    if not rows:
        print("[warn] no experiments found to evaluate.")
        return

    print_comparison_table(rows)
    agg = aggregate_over_seeds(rows)
    print_aggregated_table(agg)
    save_markdown_table(rows, os.path.join(eval_dir, "comparison_table.md"))
    with open(os.path.join(eval_dir, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(eval_dir, "comparison_aggregated.json"), "w") as f:
        json.dump(agg, f, indent=2)

    if args.generate_figures:
        agg_sorted = sorted(agg, key=_order_key)
        # bar charts: one bar per config (seed mean), clean labels, std error bars
        for metric in ("macro_f1", "kappa", "worst_lvl"):
            means = {pretty_label(a): a[f"{metric}_mean"] for a in agg_sorted}
            errs = {pretty_label(a): a[f"{metric}_std"] for a in agg_sorted}
            viz.plot_ablation_comparison(means, metric, os.path.join(eval_dir, f"ablation_{metric}.png"), errors=errs)
        # per-level heatmap: 6 configs (seed-averaged), clean labels
        heat = aggregate_perlevel(analyzer_results, rows)
        viz.plot_level_attribution_heatmap(heat, os.path.join(eval_dir, "level_attribution_heatmap.png"))

    print(f"[info] evaluation artifacts -> {eval_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Spine-ViT experiments")
    p.add_argument("--experiments_dir", type=str, default="outputs")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--from_saved", action="store_true", help="aggregate saved test_results.json instead of re-running")
    p.add_argument("--generate_figures", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    main()
