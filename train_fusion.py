"""Training entry point for the two-view (sagittal + axial) fusion grader — v2 Task 2.

Reuses train.py's epoch loop / metrics wholesale; only the dataset (RSNAFusionDataset),
the model (SpineFusionGrader), and the reporting differ. Canal-stenosis metrics are
reported TWICE, as required: on the FULL test set and on the AXIAL-AVAILABLE subset (so a
fusion gain is not confounded by which levels happen to have an axial slice).

Modes:
    --views sag                    # sagittal-only control (matched, un-augmented)
    --views axial                  # axial-only
    --views both --fusion concat   # fusion-A (early concat + proj)
    --views both --fusion attn     # fusion-B (joint self-attention, view-embedding load-bearing)

Example:
    python train_fusion.py --data_dir data/rsna --views both --fusion attn --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from models.spine_fusion import build_fusion_model
from train import (load_config, resolve_device, set_seed, run_epoch, evaluate_split, _fmt)
from utils.metrics import (compute_metrics, compute_class_weights, LevelAttributionAnalyzer,
                           RSNA_LEVEL_NAMES)


def fusion_experiment_name(config: Dict) -> str:
    views = config.get("views", "both")
    fus = f"_{config.get('fusion', 'attn')}" if views == "both" else ""
    ab = config.get("axial_box_size", 32)
    abs_tag = f"_ab{ab}" if ab != 32 else ""
    aug_tag = "_aug" if config.get("augment") else ""   # never overwrite un-augmented runs
    ks = int(config.get("sag_slices", 1))
    sag_tag = f"_sag{ks}" if ks > 1 else ""             # parasagittal budget control
    return (f"{config['dataset']}_fusion_{views}{fus}_{config['pos_encoding']}"
            f"_{config['embed_dim']}_{config['encoder_layers']}{abs_tag}{sag_tag}{aug_tag}_s{config['seed']}")


def get_fusion_dataloaders(config: Dict):
    from data.rsna_fusion import make_rsna_fusion_splits, rsna_fusion_collate_fn
    from torch.utils.data import DataLoader

    train_ds, val_ds, test_ds = make_rsna_fusion_splits(config["data_dir"], config)
    limit = config.get("limit_samples")
    if limit:
        train_ds.samples = train_ds.samples[:limit]
        val_ds.samples = val_ds.samples[: max(2, limit // 2)]
        test_ds.samples = test_ds.samples[: max(2, limit // 2)]
    nw = config.get("num_workers", 4)
    collate = rsna_fusion_collate_fn
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate, num_workers=nw)
    return train_ds, train_loader, val_loader, test_loader


def _axial_coverage_set(data_dir: str) -> set:
    """{(study_id, level_idx)} that have an axial subarticular slice (for subset metrics)."""
    from data.rsna_axial import build_axial_index
    axial = build_axial_index(data_dir)
    return {(int(sid), int(li)) for sid, lv in axial.items() for li in lv}


def _subset_metrics(test_res: Dict, cov_set: set, num_classes: int):
    """Metrics restricted to axial-available (study, level) pairs."""
    sids, lvls = test_res["studyids"], test_res["levels"]
    mask = np.array([(int(s), int(l)) in cov_set for s, l in zip(sids, lvls)], dtype=bool)
    if mask.sum() == 0:
        return None, 0
    m = compute_metrics(test_res["preds"][mask], test_res["targets"][mask], num_classes)
    return m, int(mask.sum())


def main():
    args = parse_args()
    config = load_config(vars(args))
    # fusion-specific defaults not in default.yaml
    config.setdefault("views", "both")
    config.setdefault("fusion", "attn")
    config.setdefault("axial_box_size", config.get("box_size", 32))
    config.setdefault("augment", False)
    config.setdefault("sag_slices", 1)
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    out_dir = os.path.join(config["output_dir"], config.get("experiment_name") or fusion_experiment_name(config))
    os.makedirs(out_dir, exist_ok=True)
    if config.get("skip_if_done") and os.path.exists(os.path.join(out_dir, "test_results.json")):
        print(f"[skip] {out_dir} already complete")
        return
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in config.items() if k not in ("detected_centers",)}, f, indent=2, default=str)
    print(f"[info] experiment -> {out_dir}")
    print(f"[info] device={device} views={config['views']} fusion={config['fusion']} "
          f"axial_box={config['axial_box_size']} pos={config['pos_encoding']}")

    train_ds, train_loader, val_loader, test_loader = get_fusion_dataloaders(config)

    model = build_fusion_model(config).to(device)
    print(f"[info] trainable params: {model.count_trainable_params()/1e6:.3f}M")

    class_weights = compute_class_weights(train_ds, config["num_classes"],
                                          scheme=config.get("class_weight", "sqrt_inverse")).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    predict_fn = None

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    history = {k: [] for k in ("train_loss", "val_loss", "val_kappa", "val_worst_lvl")}
    select_metric = config.get("select_metric", "kappa")
    select_window = max(1, config.get("select_window", 3))
    val_select_hist = []
    best_score, best_epoch, patience_ctr = -1e9, -1, 0

    for epoch in range(1, config["epochs"] + 1):
        train_res = run_epoch(model, train_loader, criterion, device, config, optimizer, desc=f"train {epoch}", predict_fn=predict_fn)
        val_res = evaluate_split(model, val_loader, criterion, device, config, desc=f"val {epoch}", predict_fn=predict_fn)
        scheduler.step()

        vm, attr = val_res["metrics"], val_res["attribution"]
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_kappa"].append(vm.get("kappa", 0.0))
        history["val_worst_lvl"].append(attr.get("worst_level_accuracy"))
        print(f"epoch {epoch:3d} | train_loss {train_res['loss']:.4f} | val_loss {val_res['loss']:.4f} "
              f"kappa {vm.get('kappa',0):.3f} | worst_lvl {_fmt(attr.get('worst_level_accuracy'))}")
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        val_select_hist.append(vm.get(select_metric, 0.0))
        cur_score = float(np.mean(val_select_hist[-select_window:]))
        if cur_score > best_score:
            best_score, best_epoch, patience_ctr = cur_score, epoch, 0
            torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch,
                        "val_metrics": vm, "smoothed_score": cur_score}, os.path.join(out_dir, "best_model.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                print(f"[info] early stopping at epoch {epoch} (best smoothed val {select_metric} "
                      f"{best_score:.3f} @ {best_epoch})")
                break

    # Test with best checkpoint; report FULL + axial-available-subset metrics.
    ckpt = torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_res = evaluate_split(model, test_loader, criterion, device, config, desc="test", predict_fn=predict_fn)

    cov_set = _axial_coverage_set(config["data_dir"])
    sub_metrics, n_sub = _subset_metrics(test_res, cov_set, config["num_classes"])
    test_out = {
        "metrics_full": test_res["metrics"],
        "attribution_full": test_res["attribution"],
        "metrics_axial_subset": sub_metrics,
        "n_axial_subset_tokens": n_sub,
        "n_full_tokens": int(len(test_res["preds"])),
        "best_epoch": best_epoch,
    }
    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump(test_out, f, indent=2)
    with open(os.path.join(out_dir, "test_predictions.json"), "w") as f:
        json.dump({k: [int(x) for x in test_res[k]] for k in ("studyids", "levels", "preds", "targets")}, f)

    tm, ta = test_res["metrics"], test_res["attribution"]
    print(f"[info] TEST (full, n={test_out['n_full_tokens']})  kappa {tm.get('kappa',0):.3f} "
          f"macro_f1 {tm.get('macro_f1',0):.3f} worst_lvl {_fmt(ta.get('worst_level_accuracy'))} "
          f"(n={ta.get('n_studies_with_pathology',0)})")
    if sub_metrics:
        print(f"[info] TEST (axial-subset, n={n_sub})  kappa {sub_metrics.get('kappa',0):.3f} "
              f"macro_f1 {sub_metrics.get('macro_f1',0):.3f}")
    print(f"[info] results -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Spine-ViT fusion (sagittal + axial)")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--dataset", type=str, default="rsna", choices=["rsna"])
    p.add_argument("--views", type=str, default=None, choices=["sag", "axial", "both"])
    p.add_argument("--fusion", type=str, default=None, choices=["concat", "attn"])
    p.add_argument("--pos_encoding", type=str, default=None, choices=["ordinal", "learned", "none"])
    p.add_argument("--box_size", type=int, default=None, help="sagittal ROI extent (224-space px)")
    p.add_argument("--axial_box_size", type=int, default=None, help="axial ROI extent (224-space px)")
    p.add_argument("--augment", action="store_true", default=None,
                   help="per-view augmentation (sagittal hflip OFF, axial hflip ON); tags runs _aug")
    p.add_argument("--sag_slices", type=int, default=None,
                   help="parasagittal budget control: K primary sag slices (mean-pooled). 1=default; 5 matches axial")
    p.add_argument("--embed_dim", type=int, default=None)
    p.add_argument("--encoder_layers", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--class_weight", type=str, default=None, choices=["inverse", "sqrt_inverse", "none"])
    p.add_argument("--select_metric", type=str, default=None, choices=["kappa", "macro_f1", "balanced_acc"])
    p.add_argument("--select_window", type=int, default=None)
    p.add_argument("--skip_if_done", action="store_true", default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--limit_samples", type=int, default=None)
    p.add_argument("--experiment_name", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
