import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.spine_fusion import build_fusion_model
from train import load_config, resolve_device, set_seed, run_epoch, evaluate_split, format_metric
from utils.metrics import compute_metrics, compute_class_weights
from data.rsna_fusion import make_rsna_fusion_splits, rsna_fusion_collate_fn
from data.rsna_axial import build_axial_index


def fusion_experiment_name(config):
    views = config.get("views", "both")
    name = f"{config['dataset']}_fusion_{views}"

    if views == "both":
        name = f"{name}_{config.get('fusion', 'attn')}"

    name = f"{name}_{config['pos_encoding']}"
    name = f"{name}_{config['embed_dim']}_{config['encoder_layers']}"

    axial_box = config.get("axial_box_size", 32)
    if axial_box != 32:
        name = f"{name}_ab{axial_box}"

    sag_slices = int(config.get("sag_slices", 1))
    if sag_slices > 1:
        name = f"{name}_sag{sag_slices}"

    if config.get("augment"):
        name = f"{name}_aug"

    return f"{name}_s{config['seed']}"


def get_fusion_dataloaders(config):
    train_ds, val_ds, test_ds = make_rsna_fusion_splits(config["data_dir"], config)

    limit = config.get("limit_samples")
    if limit:
        small = max(2, limit // 2)
        train_ds.samples = train_ds.samples[:limit]
        val_ds.samples = val_ds.samples[:small]
        test_ds.samples = test_ds.samples[:small]

    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 4)
    collate = rsna_fusion_collate_fn
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate, num_workers=num_workers)
    return train_ds, train_loader, val_loader, test_loader


def axial_coverage_set(data_dir):
    axial = build_axial_index(data_dir)
    covered = set()
    for study_id in axial:
        for level_index in axial[study_id]:
            covered.add((int(study_id), int(level_index)))
    return covered


def subset_metrics(test_res, covered, num_classes):
    study_ids = test_res["studyids"]
    levels = test_res["levels"]

    keep = []
    for i in range(len(study_ids)):
        pair = (int(study_ids[i]), int(levels[i]))
        keep.append(pair in covered)
    keep = np.array(keep, dtype=bool)

    if keep.sum() == 0:
        return None, 0

    metrics = compute_metrics(test_res["predictions"][keep], test_res["targets"][keep], num_classes)
    return metrics, int(keep.sum())


def main():
    args = parse_args()
    config = load_config(vars(args))
    config.setdefault("views", "both")
    config.setdefault("fusion", "attn")
    config.setdefault("axial_box_size", config.get("box_size", 32))
    config.setdefault("augment", False)
    config.setdefault("sag_slices", 1)

    set_seed(config["seed"])
    device = resolve_device(config["device"])

    name = config.get("experiment_name")
    if not name:
        name = fusion_experiment_name(config)
    out_dir = os.path.join(config["output_dir"], name)
    os.makedirs(out_dir, exist_ok=True)

    if config.get("skip_if_done") and os.path.exists(os.path.join(out_dir, "test_results.json")):
        print(out_dir, "is already complete")
        return

    saveable = {}
    for key in config:
        if key != "detected_centers":
            saveable[key] = config[key]
    with open(os.path.join(out_dir, "config.json"), "w") as config_file:
        json.dump(saveable, config_file, indent=2, default=str)

    print("experiment ->", out_dir)
    print("device", device, "views", config["views"], "fusion", config["fusion"])
    print("axial_box", config["axial_box_size"], "pos", config["pos_encoding"])

    train_ds, train_loader, val_loader, test_loader = get_fusion_dataloaders(config)

    model = build_fusion_model(config).to(device)
    print("trainable params: %.3fM" % (model.count_trainable_params() / 1e6))

    weight_scheme = config.get("class_weight", "sqrt_inverse")
    class_weights = compute_class_weights(train_ds, config["num_classes"], scheme=weight_scheme)
    class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    trainable = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(trainable, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"],
                                                           eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_kappa": [], "val_worst_lvl": []}
    select_metric = config.get("select_metric", "kappa")
    select_window = max(1, config.get("select_window", 3))
    val_scores = []
    best_score = -1e9
    best_epoch = -1
    epochs_since_best = 0

    for epoch in range(1, config["epochs"] + 1):
        train_res = run_epoch(model, train_loader, criterion, device, config, optimizer,
                              desc=f"train {epoch}")
        val_res = evaluate_split(model, val_loader, criterion, device, config,
                                 desc=f"val {epoch}")
        scheduler.step()

        val_metrics = val_res["metrics"]
        attribution = val_res["attribution"]
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_kappa"].append(val_metrics.get("kappa", 0.0))
        history["val_worst_lvl"].append(attribution.get("worst_level_accuracy"))

        print("epoch %3d | train_loss %.4f | val_loss %.4f kappa %.3f | worst_lvl %s" % (
            epoch, train_res["loss"], val_res["loss"], val_metrics.get("kappa", 0),
            format_metric(attribution.get("worst_level_accuracy"))))

        with open(os.path.join(out_dir, "history.json"), "w") as history_file:
            json.dump(history, history_file, indent=2)

        val_scores.append(val_metrics.get(select_metric, 0.0))
        recent = val_scores[-select_window:]
        smoothed = float(np.mean(recent))

        if smoothed > best_score:
            best_score = smoothed
            best_epoch = epoch
            epochs_since_best = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "smoothed_score": smoothed,
            }
            torch.save(checkpoint, os.path.join(out_dir, "best_model.pt"))
        else:
            epochs_since_best = epochs_since_best + 1
            if epochs_since_best >= config["patience"]:
                print("early stopping at epoch", epoch, "- best", select_metric,
                      "%.3f" % best_score, "at epoch", best_epoch)
                break

    checkpoint = torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_res = evaluate_split(model, test_loader, criterion, device, config, desc="test")

    covered = axial_coverage_set(config["data_dir"])
    axial_metrics, n_axial = subset_metrics(test_res, covered, config["num_classes"])

    test_out = {
        "metrics_full": test_res["metrics"],
        "attribution_full": test_res["attribution"],
        "metrics_axial_subset": axial_metrics,
        "n_axial_subset_tokens": n_axial,
        "n_full_tokens": int(len(test_res["predictions"])),
        "best_epoch": best_epoch,
    }
    with open(os.path.join(out_dir, "test_results.json"), "w") as results_file:
        json.dump(test_out, results_file, indent=2)

    predictions = {}
    for key in ["studyids", "levels", "predictions", "targets"]:
        values = []
        for value in test_res[key]:
            values.append(int(value))
        predictions[key] = values
    with open(os.path.join(out_dir, "test_predictions.json"), "w") as predictions_file:
        json.dump(predictions, predictions_file)

    test_metrics = test_res["metrics"]
    test_attribution = test_res["attribution"]
    print("TEST full (n=%d) kappa %.3f macro_f1 %.3f worst_lvl %s" % (
        test_out["n_full_tokens"], test_metrics.get("kappa", 0),
        test_metrics.get("macro_f1", 0),
        format_metric(test_attribution.get("worst_level_accuracy"))))

    if axial_metrics:
        print("TEST axial subset (n=%d) kappa %.3f macro_f1 %.3f" % (
            n_axial, axial_metrics.get("kappa", 0), axial_metrics.get("macro_f1", 0)))

    print("results saved to", out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Spine-ViT with sagittal + axial views")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="rsna", choices=["rsna"])
    parser.add_argument("--views", type=str, default=None, choices=["sag", "axial", "both"])
    parser.add_argument("--fusion", type=str, default=None, choices=["concat", "attn"])
    parser.add_argument("--pos_encoding", type=str, default=None,
                        choices=["ordinal", "learned", "none"])
    parser.add_argument("--box_size", type=int, default=None,
                        help="sagittal ROI size in resized pixels")
    parser.add_argument("--axial_box_size", type=int, default=None,
                        help="axial ROI size in resized pixels")
    parser.add_argument("--augment", action="store_true", default=None,
                        help="per-view augmentation, sagittal hflip off and axial hflip on")
    parser.add_argument("--sag_slices", type=int, default=None,
                        help="number of parasagittal slices to mean-pool, 1 by default")
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--encoder_layers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--class_weight", type=str, default=None,
                        choices=["inverse", "sqrt_inverse", "none"])
    parser.add_argument("--select_metric", type=str, default=None,
                        choices=["kappa", "macro_f1", "balanced_acc"])
    parser.add_argument("--select_window", type=int, default=None)
    parser.add_argument("--skip_if_done", action="store_true", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
