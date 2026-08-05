from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.rsna_dataset import RSNADataset, build_rsna_index
from data.rsna_detector import RSNADetectorDataset, detector_collate_fn, make_rsna_detector_splits
from models.detector import build_detector
from utils.detector_metrics import (
    soft_argmax, coord_loss, localization_error_mm, LocalizationReport, RSNA_LEVEL_NAMES,
)
from train import load_config, resolve_device, set_seed


def move(batch: Dict, device):
    out = dict(batch)
    for k in ("image", "centers", "valid", "mm_scale"):
        out[k] = batch[k].to(device)
    return out


def evaluate(model, loader, device, scale, desc="val"):
    model.eval()
    report = LocalizationReport()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = move(batch, device)
            pred = model(batch["image"])
            total_loss += float(coord_loss(pred, batch["centers"] / scale, batch["valid"])); n += 1
            pred_c = soft_argmax(pred) * scale
            err = localization_error_mm(pred_c, batch["centers"], batch["mm_scale"])
            report.update(err, batch["valid"])
    return total_loss / max(1, n), report.compute()


def save_overlays(model, dataset, device, scale, out_dir, n=20):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for i in range(min(n, len(dataset))):
        item = dataset[i]
        with torch.no_grad():
            pred = model(item["image"].unsqueeze(0).to(device))
            pred_c = (soft_argmax(pred) * scale)[0].cpu().numpy()
        img = item["image"].numpy()
        disp = img[1] if img.ndim == 3 else img
        gt, valid = item["centers"].numpy(), item["valid"].numpy()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(disp, cmap="gray")
        for L in range(5):
            if valid[L]:
                ax.plot(gt[L, 0], gt[L, 1], "r+", markersize=12, markeredgewidth=2)
            ax.plot(pred_c[L, 0], pred_c[L, 1], "bx", markersize=9, markeredgewidth=2)
            ax.text(pred_c[L, 0] + 3, pred_c[L, 1], RSNA_LEVEL_NAMES[L], color="cyan", fontsize=7)
        ax.set_title(f"study {item['study_id']}  (red=GT, blue=pred)")
        ax.axis("off")
        fig.savefig(os.path.join(out_dir, f"overlay_{i}_{item['study_id']}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)


def export_detected_centers(model, data_dir, config, device, scale, out_path):
    """Run the detector over ALL studies and save predicted centers in ORIGINAL coords."""
    base = RSNADataset(data_dir, augment=False, image_size=config.get("image_size", 224),
                       box_size=config.get("box_size", 32), use_25d=config.get("use_25d", True))
    ds = RSNADetectorDataset(base)
    loader = DataLoader(ds, batch_size=config.get("batch_size", 16), shuffle=False,
                        collate_fn=detector_collate_fn, num_workers=config.get("num_workers", 4))
    model.eval()
    detected: Dict[str, Dict[str, list]] = {}
    per_study_err: Dict[str, float] = {}
    s = config.get("image_size", 224)
    with torch.no_grad():
        for batch in tqdm(loader, desc="export", leave=False):
            pred_t = soft_argmax(model(batch["image"].to(device))) * scale
            err = localization_error_mm(pred_t.cpu(), batch["centers"], batch["mm_scale"])
            pred_c = pred_t.cpu().numpy()
            oh = batch["orig_hw"].numpy()
            valid = batch["valid"].bool()
            for b, sid in enumerate(batch["study_ids"]):
                H, W = float(oh[b, 0]), float(oh[b, 1])
                detected[str(sid)] = {
                    str(L): [float(pred_c[b, L, 0] * W / s), float(pred_c[b, L, 1] * H / s)]
                    for L in range(pred_c.shape[1])
                }
                e = err[b][valid[b]]
                per_study_err[str(sid)] = float(e.mean()) if e.numel() else None
    with open(out_path, "w") as f:
        json.dump(detected, f)
    err_path = os.path.join(os.path.dirname(out_path), "localization_per_study.json")
    with open(err_path, "w") as f:
        json.dump(per_study_err, f)
    print(f"[info] exported detected centers + per-study mm error for {len(detected)} studies -> {out_path}")


def main():
    args = parse_args()
    config = load_config(vars(args))
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    out_dir = os.path.join(config["output_dir"], "detector")
    os.makedirs(out_dir, exist_ok=True)

    out_size = args.out_size
    scale = config["image_size"] / out_size

    train_ds, val_ds, test_ds = make_rsna_detector_splits(config["data_dir"], config)
    nw = config.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=detector_collate_fn, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            collate_fn=detector_collate_fn, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False,
                             collate_fn=detector_collate_fn, num_workers=nw)

    model = build_detector(config, out_size=out_size).to(device)
    print(f"[info] detector trainable params: {model.count_trainable_params()/1e6:.3f}M  "
          f"(out {out_size}x{out_size}, scale {scale:g}, reg {args.reg})")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    best_median, best_epoch = 1e9, -1
    history = {"train_loss": [], "val_loss": [], "val_median_mm": [], "val_mean_mm": []}
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        tl, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"train {epoch}", leave=False):
            batch = move(batch, device)
            pred = model(batch["image"])
            loss = coord_loss(pred, batch["centers"] / scale, batch["valid"], reg=args.reg)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += float(loss.item()); n += 1
        scheduler.step()

        vloss, vrep = evaluate(model, val_loader, device, scale, desc=f"val {epoch}")
        history["train_loss"].append(tl / max(1, n)); history["val_loss"].append(vloss)
        history["val_median_mm"].append(vrep["median_mm"]); history["val_mean_mm"].append(vrep["mean_mm"])
        print(f"epoch {epoch:3d} | train_loss {tl/max(1,n):.5f} | val_loss {vloss:.5f} "
              f"| median {vrep['median_mm']:.2f}mm mean {vrep['mean_mm']:.2f}mm "
              f"| <=5mm {vrep['pct_within_5mm']:.0f}% <=10mm {vrep['pct_within_10mm']:.0f}%")
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if vrep["median_mm"] < best_median:
            best_median, best_epoch = vrep["median_mm"], epoch
            torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch,
                        "out_size": out_size, "val": vrep}, os.path.join(out_dir, "best_detector.pt"))

    ckpt = torch.load(os.path.join(out_dir, "best_detector.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    _, test_rep = evaluate(model, test_loader, device, scale, desc="test")
    with open(os.path.join(out_dir, "localization_report.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "test": test_rep}, f, indent=2)
    print(f"\n[info] TEST  median {test_rep['median_mm']:.2f}mm  mean {test_rep['mean_mm']:.2f}mm  "
          f"<=5mm {test_rep['pct_within_5mm']:.0f}%  <=10mm {test_rep['pct_within_10mm']:.0f}%")
    print("[info] per-level (median mm):", {k: round(v["median_mm"], 2) for k, v in test_rep["per_level"].items()})

    save_overlays(model, test_ds, device, scale, os.path.join(out_dir, "overlays"), n=args.overlay_n)
    print(f"[info] overlays -> {os.path.join(out_dir, 'overlays')}  (inspect all {args.overlay_n})")

    if args.export:
        export_detected_centers(model, config["data_dir"], config, device, scale,
                                os.path.join(out_dir, "detected_centers.json"))
    print(f"[info] detector artifacts -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train the disc-level heatmap detector")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--dataset", type=str, default="rsna")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--out_size", type=int, default=56)
    p.add_argument("--reg", type=float, default=0.0, help="variance regularizer weight (keeps heatmap peaks tight)")
    p.add_argument("--overlay_n", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--export", action="store_true", help="export detected_centers.json over all studies")
    return p.parse_args()


if __name__ == "__main__":
    main()
