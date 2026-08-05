import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.rsna_dataset import RSNADataset
from data.rsna_detector import RSNADetectorDataset, detector_collate_fn, make_rsna_detector_splits
from models.detector import build_detector
from utils.detector_metrics import (
    soft_argmax, coord_loss, localization_error_mm, LocalizationReport, RSNA_LEVEL_NAMES,
)
from train import load_config, resolve_device, set_seed


def move_batch(batch, device):
    moved = dict(batch)
    for key in ["image", "centers", "valid", "mm_scale"]:
        moved[key] = batch[key].to(device)
    return moved


def evaluate(model, loader, device, scale, desc="val"):
    model.eval()
    report = LocalizationReport()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = move_batch(batch, device)
            pred = model(batch["image"])

            loss = coord_loss(pred, batch["centers"] / scale, batch["valid"])
            total_loss = total_loss + float(loss)
            num_batches = num_batches + 1

            pred_centers = soft_argmax(pred) * scale
            error = localization_error_mm(pred_centers, batch["centers"], batch["mm_scale"])
            report.update(error, batch["valid"])

    return total_loss / max(1, num_batches), report.compute()


def save_overlays(model, dataset, device, scale, out_dir, n=20):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    for i in range(min(n, len(dataset))):
        item = dataset[i]

        with torch.no_grad():
            pred = model(item["image"].unsqueeze(0).to(device))
            pred_centers = (soft_argmax(pred) * scale)[0].cpu().numpy()

        image = item["image"].numpy()
        if image.ndim == 3:
            display = image[1]
        else:
            display = image

        true_centers = item["centers"].numpy()
        valid = item["valid"].numpy()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(display, cmap="gray")
        for level in range(5):
            if valid[level]:
                ax.plot(true_centers[level, 0], true_centers[level, 1], "r+",
                        markersize=12, markeredgewidth=2)
            ax.plot(pred_centers[level, 0], pred_centers[level, 1], "bx",
                    markersize=9, markeredgewidth=2)
            ax.text(pred_centers[level, 0] + 3, pred_centers[level, 1],
                    RSNA_LEVEL_NAMES[level], color="cyan", fontsize=7)
        ax.set_title(f"study {item['study_id']}  (red=true, blue=pred)")
        ax.axis("off")

        filename = f"overlay_{i}_{item['study_id']}.png"
        fig.savefig(os.path.join(out_dir, filename), dpi=130, bbox_inches="tight")
        plt.close(fig)


def export_detected_centers(model, data_dir, config, device, scale, out_path):
    image_size = config.get("image_size", 224)
    base = RSNADataset(data_dir, augment=False, image_size=image_size,
                       box_size=config.get("box_size", 32),
                       use_25d=config.get("use_25d", True))
    dataset = RSNADetectorDataset(base)
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 16), shuffle=False,
                        collate_fn=detector_collate_fn,
                        num_workers=config.get("num_workers", 4))

    model.eval()
    detected = {}
    per_study_error = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="export", leave=False):
            pred_centers = soft_argmax(model(batch["image"].to(device))) * scale
            error = localization_error_mm(pred_centers.cpu(), batch["centers"],
                                          batch["mm_scale"])
            pred_centers = pred_centers.cpu().numpy()
            original_size = batch["orig_hw"].numpy()
            valid = batch["valid"].bool()

            for sample_index in range(len(batch["study_ids"])):
                study_id = batch["study_ids"][sample_index]
                height = float(original_size[sample_index, 0])
                width = float(original_size[sample_index, 1])

                centers = {}
                for level in range(pred_centers.shape[1]):
                    x = float(pred_centers[sample_index, level, 0] * width / image_size)
                    y = float(pred_centers[sample_index, level, 1] * height / image_size)
                    centers[str(level)] = [x, y]
                detected[str(study_id)] = centers

                study_error = error[sample_index][valid[sample_index]]
                if study_error.numel():
                    per_study_error[str(study_id)] = float(study_error.mean())
                else:
                    per_study_error[str(study_id)] = None

    with open(out_path, "w") as centers_file:
        json.dump(detected, centers_file)

    error_path = os.path.join(os.path.dirname(out_path), "localization_per_study.json")
    with open(error_path, "w") as error_file:
        json.dump(per_study_error, error_file)

    print("exported detected centers for", len(detected), "studies ->", out_path)


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
    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=detector_collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=detector_collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=detector_collate_fn, num_workers=num_workers)

    model = build_detector(config, out_size=out_size).to(device)
    print("detector trainable params: %.3fM" % (model.count_trainable_params() / 1e6))
    print("output %dx%d, scale %g, reg %g" % (out_size, out_size, scale, args.reg))

    trainable = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(trainable, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"],
                                                           eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_median_mm": [], "val_mean_mm": []}
    best_median = 1e9
    best_epoch = -1

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"train {epoch}", leave=False):
            batch = move_batch(batch, device)
            pred = model(batch["image"])
            loss = coord_loss(pred, batch["centers"] / scale, batch["valid"], reg=args.reg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + float(loss.item())
            num_batches = num_batches + 1

        scheduler.step()
        train_loss = total_loss / max(1, num_batches)

        val_loss, val_report = evaluate(model, val_loader, device, scale,
                                        desc=f"val {epoch}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_median_mm"].append(val_report["median_mm"])
        history["val_mean_mm"].append(val_report["mean_mm"])

        print("epoch %3d | train_loss %.5f | val_loss %.5f | median %.2fmm mean %.2fmm" % (
            epoch, train_loss, val_loss, val_report["median_mm"], val_report["mean_mm"]))
        print("          within 5mm %.0f%%  within 10mm %.0f%%" % (
            val_report["pct_within_5mm"], val_report["pct_within_10mm"]))

        with open(os.path.join(out_dir, "history.json"), "w") as history_file:
            json.dump(history, history_file, indent=2)

        if val_report["median_mm"] < best_median:
            best_median = val_report["median_mm"]
            best_epoch = epoch
            checkpoint = {
                "model_state": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "out_size": out_size,
                "val": val_report,
            }
            torch.save(checkpoint, os.path.join(out_dir, "best_detector.pt"))

    checkpoint = torch.load(os.path.join(out_dir, "best_detector.pt"), map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_report = evaluate(model, test_loader, device, scale, desc="test")

    with open(os.path.join(out_dir, "localization_report.json"), "w") as report_file:
        json.dump({"best_epoch": best_epoch, "test": test_report}, report_file, indent=2)

    print("\nTEST median %.2fmm mean %.2fmm" % (test_report["median_mm"],
                                                       test_report["mean_mm"]))
    print("within 5mm %.0f%%  within 10mm %.0f%%" % (test_report["pct_within_5mm"],
                                                            test_report["pct_within_10mm"]))

    per_level = {}
    for level in test_report["per_level"]:
        per_level[level] = round(test_report["per_level"][level]["median_mm"], 2)
    print("per-level median mm:", per_level)

    overlay_dir = os.path.join(out_dir, "overlays")
    save_overlays(model, test_ds, device, scale, overlay_dir, n=args.overlay_n)
    print("overlays ->", overlay_dir)

    if args.export:
        export_detected_centers(model, config["data_dir"], config, device, scale,
                                os.path.join(out_dir, "detected_centers.json"))

    print("detector artifacts ->", out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the disc-level heatmap detector")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="rsna")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--out_size", type=int, default=56)
    parser.add_argument("--reg", type=float, default=0.0,
                        help="variance regularizer that keeps heatmap peaks tight")
    parser.add_argument("--overlay_n", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--export", action="store_true",
                        help="export detected_centers.json for all studies")
    return parser.parse_args()


if __name__ == "__main__":
    main()
