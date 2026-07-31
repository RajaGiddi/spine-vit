"""Training entry point for Spine-ViT.

Loads configs/default.yaml, applies CLI overrides, builds the requested
dataset/model, and trains with early stopping on validation macro-F1. Each run writes
config.json, best_model.pt, history.json, and test_results.json to a uniquely named
output directory.

Example:
    python train.py --data_dir /path/to/rsna-2024 --dataset rsna \
        --tokenizer anatomy --pos_encoding ordinal
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from utils.metrics import compute_metrics, compute_class_weights, LevelAttributionAnalyzer
from utils.metrics import RSNA_LEVEL_NAMES

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
def load_config(overrides: Dict) -> Dict:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            config[k] = v
    # num_classes derived from task
    config["num_classes"] = (
        config["num_pfirrmann_classes"] if config["task"] == "pfirrmann" else config["num_stenosis_classes"]
    )
    # load learned detector centers when grading with detected boxes
    if config.get("box_source") == "detected":
        path = config.get("detected_centers_path") or os.path.join(config["output_dir"], "detector", "detected_centers.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"--box_source detected needs {path}; run train_detector.py --export first")
        with open(path) as f:
            config["detected_centers"] = json.load(f)
        print(f"[info] loaded detected centers for {len(config['detected_centers'])} studies from {path}")
    return config


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if name in ("cuda", "mps"):
        print(f"[warn] device '{name}' unavailable; falling back to cpu")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experiment_name(config: Dict) -> str:
    # seed + fine-tune + box-source markers so runs of the same tokenizer/pos-encoding
    # don't overwrite each other (oracle vs detected boxes are separate results).
    ft = "" if config.get("freeze_backbone", True) else "_ft"
    src = "_det" if config.get("box_source", "oracle") == "detected" else ""
    bs = config.get("box_size", 32)
    box = f"_b{bs}" if bs != 32 else ""   # box-size sweep marker (default 32 unmarked)
    head = "_coral" if config.get("head", "ce") == "coral" else ""
    return (f"{config['dataset']}_{config['tokenizer']}_{config['pos_encoding']}"
            f"_{config['embed_dim']}_{config['encoder_layers']}{src}{ft}{box}{head}_s{config['seed']}")


def _fmt(v, spec: str = ".3f") -> str:
    """Format a metric that may be None (e.g. worst_level_accuracy with no pathology)."""
    return format(v, spec) if isinstance(v, (int, float)) else "n/a"


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def get_dataloaders(config: Dict):
    if config["dataset"] == "rsna":
        from data.rsna_dataset import make_rsna_splits, rsna_collate_fn

        train_ds, val_ds, test_ds = make_rsna_splits(config["data_dir"], config)
        collate = rsna_collate_fn
    elif config["dataset"] == "spider":
        from data.spider_dataset import make_spider_splits, spider_collate_fn

        train_ds, val_ds, test_ds = make_spider_splits(config["data_dir"], config)
        collate = spider_collate_fn
    else:
        raise ValueError(f"Unknown dataset '{config['dataset']}'")

    limit = config.get("limit_samples")
    if limit:  # tiny-subset sanity mode
        train_ds.samples = train_ds.samples[:limit]
        val_ds.samples = val_ds.samples[: max(2, limit // 2)]
        test_ds.samples = test_ds.samples[: max(2, limit // 2)]

    nw = config.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate, num_workers=nw)
    return train_ds, train_loader, val_loader, test_loader


def move_batch(batch: Dict, device: torch.device) -> Dict:
    out = dict(batch)
    # v1 sagittal tensors (always present) + optional fusion axial tensors (present only
    # for the two-view dataset). Moving a superset is a no-op for v1 batches.
    keys = ("images", "boxes", "level_indices", "level_types", "targets",
            "axial_images", "axial_boxes", "axial_level_indices", "axial_slot",
            "sag_multi_images")   # parasagittal budget control (fusion sag_slices>1)
    for k in keys:
        if k in batch and torch.is_tensor(batch[k]):
            out[k] = batch[k].to(device)
    return out


# --------------------------------------------------------------------------------------
# Epoch loop
# --------------------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, config, optimizer=None, desc="", predict_fn=None) -> Dict:
    is_train = optimizer is not None
    predict_fn = predict_fn or (lambda lg: lg.argmax(-1))   # ce: argmax; coral: threshold count
    model.train(is_train)
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets, all_levels, all_studyids = [], [], [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        # per-token study id (aligned to the flat token stream) for worst-level grouping
        tok_studyids = np.repeat(np.asarray(batch["study_ids"]), batch["num_levels"])
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            out = model(batch)
            logits = out["logits"]
            disc_mask = out["disc_mask"]
            targets = batch["targets"][disc_mask]

            if (targets != -1).sum() == 0:
                continue
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], config.get("grad_clip", 1.0)
                )
                optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        all_preds.append(predict_fn(logits).detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_levels.append(out["disc_level_indices"].detach().cpu().numpy())
        all_studyids.append(tok_studyids[disc_mask.detach().cpu().numpy()])

    preds = np.concatenate(all_preds) if all_preds else np.array([])
    targets = np.concatenate(all_targets) if all_targets else np.array([])
    levels = np.concatenate(all_levels) if all_levels else np.array([])
    studyids = np.concatenate(all_studyids) if all_studyids else np.array([])
    return {
        "loss": total_loss / max(1, n_batches),
        "preds": preds,
        "targets": targets,
        "levels": levels,
        "studyids": studyids,
    }


def evaluate_split(model, loader, criterion, device, config, desc="val", predict_fn=None) -> Dict:
    res = run_epoch(model, loader, criterion, device, config, optimizer=None, desc=desc, predict_fn=predict_fn)
    metrics = compute_metrics(res["preds"], res["targets"], config["num_classes"])
    level_names = RSNA_LEVEL_NAMES if config["dataset"] == "rsna" else None
    analyzer = LevelAttributionAnalyzer(level_names=level_names, num_classes=config["num_classes"])
    analyzer.update(res["preds"], res["targets"], res["levels"], patient_id=res["studyids"])
    attribution = analyzer.compute()
    return {"loss": res["loss"], "metrics": metrics, "attribution": attribution,
            "preds": res["preds"], "targets": res["targets"], "levels": res["levels"], "studyids": res["studyids"]}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    config = load_config(vars(args))
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    out_dir = os.path.join(config["output_dir"], config.get("experiment_name") or experiment_name(config))
    os.makedirs(out_dir, exist_ok=True)
    if config.get("skip_if_done") and os.path.exists(os.path.join(out_dir, "test_results.json")):
        print(f"[skip] {out_dir} already complete (test_results.json exists)")
        return
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[info] experiment -> {out_dir}")
    print(f"[info] device={device} dataset={config['dataset']} task={config['task']} "
          f"tokenizer={config['tokenizer']} pos_encoding={config['pos_encoding']}")

    train_ds, train_loader, val_loader, test_loader = get_dataloaders(config)

    model = build_model(config).to(device)
    print(f"[info] trainable params: {model.count_trainable_params()/1e6:.3f}M")

    class_weights = compute_class_weights(
        train_ds, config["num_classes"], scheme=config.get("class_weight", "inverse")
    ).to(device)
    print(f"[info] class weights ({config.get('class_weight','inverse')}): "
          f"{[round(w, 3) for w in class_weights.tolist()]}")
    # head-dependent loss + decode: CE (softmax/argmax) vs CORAL (ordinal thresholds)
    if config.get("head", "ce") == "coral":
        from utils.metrics import coral_loss, coral_predict, coral_pos_weights
        pos_weight = coral_pos_weights(train_ds, config["num_classes"]).to(device)  # per-threshold balance
        criterion = lambda lg, tg: coral_loss(lg, tg, pos_weight)  # noqa: E731
        predict_fn = coral_predict
        print(f"[info] head: CORAL ordinal, per-threshold pos_weight {[round(w,2) for w in pos_weight.tolist()]}")
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        predict_fn = None  # run_epoch defaults to argmax

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    history = {k: [] for k in ("train_loss", "val_loss", "val_macro_f1", "val_kappa",
                               "val_balanced_acc", "val_worst_lvl", "val_path_prec", "val_path_rec")}
    select_metric = config.get("select_metric", "kappa")
    select_window = max(1, config.get("select_window", 3))
    val_select_hist = []
    best_score, best_epoch, patience_ctr = -1e9, -1, 0

    for epoch in range(1, config["epochs"] + 1):
        train_res = run_epoch(model, train_loader, criterion, device, config, optimizer, desc=f"train {epoch}", predict_fn=predict_fn)
        train_metrics = compute_metrics(train_res["preds"], train_res["targets"], config["num_classes"])
        val_res = evaluate_split(model, val_loader, criterion, device, config, desc=f"val {epoch}", predict_fn=predict_fn)
        scheduler.step()

        vm = val_res["metrics"]
        attr = val_res["attribution"]
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_macro_f1"].append(vm.get("macro_f1", 0.0))
        history["val_kappa"].append(vm.get("kappa", 0.0))
        history["val_balanced_acc"].append(vm.get("balanced_acc", 0.0))
        history["val_worst_lvl"].append(attr.get("worst_level_accuracy"))
        history["val_path_prec"].append(attr.get("pathology_precision"))
        history["val_path_rec"].append(attr.get("pathology_recall"))

        print(
            f"epoch {epoch:3d} | train_loss {train_res['loss']:.4f} (f1 {train_metrics.get('macro_f1',0):.3f}) "
            f"| val_loss {val_res['loss']:.4f} f1 {vm.get('macro_f1',0):.3f} "
            f"kappa {vm.get('kappa',0):.3f} bal_acc {vm.get('balanced_acc',0):.3f} "
            f"| worst_lvl {_fmt(attr.get('worst_level_accuracy'))} "
            f"path_P/R {_fmt(attr.get('pathology_precision'))}/{_fmt(attr.get('pathology_recall'))}"
        )

        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Select on a trailing moving average of the val metric, not the raw value, so a
        # single lucky epoch can't win — the model must hold up over `select_window`
        # epochs. At small val sizes this is what makes selection defensible.
        val_select_hist.append(vm.get(select_metric, 0.0))
        cur_score = float(np.mean(val_select_hist[-select_window:]))
        if cur_score > best_score:
            best_score, best_epoch, patience_ctr = cur_score, epoch, 0
            torch.save(
                {"model_state": model.state_dict(), "config": config, "epoch": epoch,
                 "val_metrics": vm, "smoothed_score": cur_score},
                os.path.join(out_dir, "best_model.pt"),
            )
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                print(f"[info] early stopping at epoch {epoch} "
                      f"(best {select_window}-epoch smoothed val {select_metric} "
                      f"{best_score:.3f} @ epoch {best_epoch})")
                break

    # Final test with the best checkpoint.
    ckpt = torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_res = evaluate_split(model, test_loader, criterion, device, config, desc="test", predict_fn=predict_fn)
    test_out = {"metrics": test_res["metrics"], "attribution": test_res["attribution"], "best_epoch": best_epoch}
    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump(test_out, f, indent=2)
    # per-study/-level predictions -> lets analyze_localization.py relate grading to detection error
    with open(os.path.join(out_dir, "test_predictions.json"), "w") as f:
        json.dump({k: [int(x) for x in test_res[k]]
                   for k in ("studyids", "levels", "preds", "targets")}, f)
    ta = test_res["attribution"]
    print(f"[info] TEST macro_f1 {test_res['metrics'].get('macro_f1',0):.3f} "
          f"kappa {test_res['metrics'].get('kappa',0):.3f} "
          f"| worst_lvl_acc {_fmt(ta.get('worst_level_accuracy'))} "
          f"(n={ta.get('n_studies_with_pathology',0)}) "
          f"| pathology P/R {_fmt(ta.get('pathology_precision'))}/{_fmt(ta.get('pathology_recall'))} "
          f"fp_rate {_fmt(ta.get('pathology_fp_rate'))}")
    print(f"[info] results saved to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Spine-ViT")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None, choices=["rsna", "spider"])
    p.add_argument("--tokenizer", type=str, default=None, choices=["anatomy", "strips", "patches", "cast_crop"])
    p.add_argument("--pos_encoding", type=str, default=None, choices=["ordinal", "learned", "none"])
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--embed_dim", type=int, default=None)
    p.add_argument("--encoder_layers", type=int, default=None)
    p.add_argument("--encoder_heads", type=int, default=None)
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--class_weight", type=str, default=None, choices=["inverse", "sqrt_inverse", "none"])
    p.add_argument("--select_metric", type=str, default=None, choices=["kappa", "macro_f1", "balanced_acc"])
    p.add_argument("--select_window", type=int, default=None, help="epochs in the val-metric moving average for selection (1 = raw peak)")
    p.add_argument("--skip_if_done", action="store_true", default=None, help="skip if this run's test_results.json already exists (resumable sweeps)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--task", type=str, default=None, choices=["stenosis", "pfirrmann"])
    p.add_argument("--box_size", type=int, default=None, help="ROI extent in 224-space px (default 32; sweep for the tolerance curve)")
    p.add_argument("--head", type=str, default=None, choices=["ce", "coral"], help="ce=softmax cross-entropy; coral=ordinal thresholds")
    p.add_argument("--box_source", type=str, default=None, choices=["oracle", "detected"])
    p.add_argument("--detected_centers_path", type=str, default=None, help="path to detected_centers.json (box_source=detected)")
    p.add_argument("--use_oracle", dest="use_oracle_regions", action="store_true", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--limit_samples", type=int, default=None, help="train on a tiny subset for a sanity check")
    p.add_argument("--experiment_name", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()
