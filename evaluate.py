import argparse
import json
import os
import re

import numpy as np
import torch

from models import build_model
from utils.metrics import compute_metrics, LevelAttributionAnalyzer, RSNA_LEVEL_NAMES
from utils import visualization as viz

CLASS_NAMES = {
    "stenosis": ["Normal/Mild", "Moderate", "Severe"],
    "pfirrmann": ["I", "II", "III", "IV", "V"],
}

TOKENIZER_LABELS = {
    "anatomy": "Anatomy",
    "strips": "Strips",
    "patches": "Patches",
    "cast_crop": "CAST-crop",
}

POS_ENCODING_LABELS = {
    "ordinal": "Ordinal",
    "learned": "Learned",
    "none": "None",
}

DISPLAY_ORDER = {
    ("anatomy", "ordinal"): 0,
    ("anatomy", "learned"): 1,
    ("anatomy", "none"): 2,
    ("cast_crop", "ordinal"): 4,
    ("patches", "ordinal"): 5,
    ("strips", "ordinal"): 6,
}


def find_experiments(experiments_dir):
    found = []
    for name in sorted(os.listdir(experiments_dir)):
        path = os.path.join(experiments_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            found.append(path)
    return found


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as json_file:
        return json.load(json_file)


def run_test_inference(exp_dir, config, device):
    from train import get_dataloaders, move_batch

    train_ds, train_loader, val_loader, test_loader = get_dataloaders(config)

    model = build_model(config).to(device)
    checkpoint = torch.load(os.path.join(exp_dir, "best_model.pt"), map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if config["dataset"] == "rsna":
        level_names = RSNA_LEVEL_NAMES
    else:
        level_names = None
    analyzer = LevelAttributionAnalyzer(level_names=level_names, num_classes=config["num_classes"])

    all_preds = []
    all_targets = []
    all_levels = []
    sample_batch = None
    attention = None

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = move_batch(batch, device)
            if i == 0:
                out, attention = model.forward_with_attention(batch)
                sample_batch = batch
            else:
                out = model(batch)

            targets = batch["targets"][out["disc_mask"]]
            preds = out["logits"].argmax(-1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_levels.append(out["disc_level_indices"].cpu().numpy())

    if all_preds:
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        levels = np.concatenate(all_levels)
    else:
        preds = np.array([])
        targets = np.array([])
        levels = np.array([])

    analyzer.update(preds, targets, levels)
    metrics = compute_metrics(preds, targets, config["num_classes"])
    attribution = analyzer.compute()
    return metrics, attribution, preds, targets, levels, sample_batch, attention


def make_attention_figures(config, sample_batch, attention, eval_dir):
    if sample_batch is None or not attention:
        return

    num_levels = sample_batch["num_levels"][0]
    level_types = sample_batch["level_types"][:num_levels].cpu().numpy()
    level_indices = sample_batch["level_indices"][:num_levels].cpu().numpy()

    labels = []
    if config["dataset"] == "rsna":
        for index in level_indices:
            if index < len(RSNA_LEVEL_NAMES):
                labels.append(RSNA_LEVEL_NAMES[index])
            else:
                labels.append(f"L{index}")
    else:
        for index, level_type in zip(level_indices, level_types):
            if level_type == 0:
                labels.append(f"V{index}")
            else:
                labels.append(f"D{index}")

    last_attention = attention[-1][0, :num_levels, :num_levels].cpu().numpy()
    viz.plot_attention_weights(last_attention, labels,
                               os.path.join(eval_dir, "attention_weights.png"))

    received = last_attention.sum(axis=0)
    image = sample_batch["images"][0].cpu().numpy()
    first_sample = sample_batch["boxes"][:, 0] == 0
    boxes = sample_batch["boxes"][first_sample][:, 1:].cpu().numpy()
    viz.plot_attention_overlay(image, boxes, received, labels,
                               os.path.join(eval_dir, "attention_overlay.png"))


def generate_per_experiment_figures(exp_dir, config, preds, targets, eval_dir):
    name = os.path.basename(exp_dir)
    class_names = CLASS_NAMES[config["task"]]
    viz.plot_grade_confusion_matrix(targets, preds, class_names, f"Confusion - {name}",
                                    os.path.join(eval_dir, f"confusion_{name}.png"))

    history = load_json(os.path.join(exp_dir, "history.json"))
    if history:
        viz.plot_training_curves(history, os.path.join(eval_dir, f"curves_{name}.png"))


def print_comparison_table(rows):
    header = ("%-40s %-9s %-8s %8s %7s %7s %8s %6s %6s" % (
        "Experiment", "Tokenizer", "PosEnc", "MacroF1", "Kappa", "BalAcc",
        "WorstLvl", "PathP", "PathR"))
    print("\n" + header)
    print("-" * len(header))

    for row in rows:
        print("%-40s %-9s %-8s %8.3f %7.3f %7.3f %8.3f %6.3f %6.3f" % (
            row["name"], row["tokenizer"], row["pos_encoding"], row["macro_f1"],
            row["kappa"], row["balanced_acc"], row["worst_lvl"],
            row["path_prec"], row["path_rec"]))

    print("-" * len(header))
    print("%-40s %-9s %-8s %8.3f %7.3f %7s %8s %6s %6s\n" % (
        "LumbarDISC framework (ref)", "Cuboid", "Context", 0.783, 0.765, "-", "-", "-", "-"))


def save_markdown_table(rows, path):
    lines = [
        "| Experiment | Tokenizer | Pos Enc | Macro F1 | kappa | Bal Acc | "
        "Worst-Level Acc | Pathology P | Pathology R |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append("| %s | %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            row["name"], row["tokenizer"], row["pos_encoding"], row["macro_f1"],
            row["kappa"], row["balanced_acc"], row["worst_lvl"],
            row["path_prec"], row["path_rec"]))

    lines.append("| LumbarDISC framework (ref) | Cuboid | Context | 0.783 | 0.765 | "
                 "- | - | - | - |")

    with open(path, "w") as table_file:
        table_file.write("\n".join(lines) + "\n")


def pretty_label(row):
    tokenizer = TOKENIZER_LABELS.get(row["tokenizer"], row["tokenizer"])
    pos_encoding = POS_ENCODING_LABELS.get(row["pos_encoding"], row["pos_encoding"])
    label = f"{tokenizer}+{pos_encoding}"

    if row["group"].endswith("_ft"):
        return f"{label} (FT)"
    if row["tokenizer"] == "anatomy" and row["pos_encoding"] == "ordinal":
        return f"{label} (ours)"
    if row["tokenizer"] in ("strips", "patches", "cast_crop"):
        return f"{label} (baseline)"
    return label


def order_key(row):
    if row["group"].endswith("_ft"):
        return 3
    return DISPLAY_ORDER.get((row["tokenizer"], row["pos_encoding"]), 9)


def entry_order_key(entry):
    return order_key(entry[0])


def negative_kappa(row):
    return -row["kappa_mean"]


def aggregate_perlevel(analyzer_results, rows):
    meta = {}
    for row in rows:
        meta[row["name"]] = row

    groups = {}
    for name in analyzer_results:
        if name in meta:
            group = meta[name]["group"]
            if group not in groups:
                groups[group] = []
            groups[group].append((meta[name], analyzer_results[name]))

    entries = []
    for group in groups:
        items = groups[group]
        first_row = items[0][0]

        level_names = []
        for row, attribution in items:
            for level in attribution.get("per_level", {}):
                if level not in level_names:
                    level_names.append(level)

        merged = {}
        for level in level_names:
            values = []
            for row, attribution in items:
                per_level = attribution.get("per_level", {})
                if level in per_level:
                    values.append(per_level[level]["exact_acc"])
            if values:
                merged[level] = {"exact_acc": float(np.mean(values))}

        entries.append((first_row, {"per_level": merged}))

    entries.sort(key=entry_order_key)

    result = {}
    for row, per_level in entries:
        result[pretty_label(row)] = per_level
    return result


def aggregate_over_seeds(rows):
    metric_names = ["macro_f1", "kappa", "balanced_acc", "worst_lvl", "path_prec", "path_rec"]

    groups = {}
    for row in rows:
        group = row["group"]
        if group not in groups:
            groups[group] = []
        groups[group].append(row)

    aggregated = []
    for group in groups:
        group_rows = groups[group]
        summary = {
            "group": group,
            "tokenizer": group_rows[0]["tokenizer"],
            "pos_encoding": group_rows[0]["pos_encoding"],
            "n_seeds": len(group_rows),
        }
        for metric in metric_names:
            values = []
            for row in group_rows:
                values.append(row[metric])
            values = np.array(values, dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
        aggregated.append(summary)

    return aggregated


def print_aggregated_table(aggregated):
    if not aggregated:
        return

    print("\n=== Aggregated over seeds (mean +/- std) ===")
    header = "%-34s %5s %15s %15s %15s" % ("Config (seed-stripped)", "seeds", "MacroF1",
                                           "Kappa", "WorstLvl")
    print(header)
    print("-" * len(header))

    for row in sorted(aggregated, key=negative_kappa):
        print("%-34s %5d %.3f+/-%.3f   %.3f+/-%.3f   %.3f+/-%.3f" % (
            row["group"], row["n_seeds"],
            row["macro_f1_mean"], row["macro_f1_std"],
            row["kappa_mean"], row["kappa_std"],
            row["worst_lvl_mean"], row["worst_lvl_std"]))

    print("-" * len(header))


def main():
    args = parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    eval_dir = os.path.join(args.experiments_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    rows = []
    analyzer_results = {}

    for exp_dir in find_experiments(args.experiments_dir):
        name = os.path.basename(exp_dir)
        config = load_json(os.path.join(exp_dir, "config.json"))
        if config is None:
            continue
        if args.data_dir:
            config["data_dir"] = args.data_dir

        if args.from_saved:
            test_out = load_json(os.path.join(exp_dir, "test_results.json"))
            if test_out is None:
                print(name, "has no test_results.json")
                continue
            metrics = test_out["metrics"]
            attribution = test_out["attribution"]
        else:
            if not os.path.exists(os.path.join(exp_dir, "best_model.pt")):
                print(name, "has no best_model.pt")
                continue
            print(name)
            results = run_test_inference(exp_dir, config, device)
            metrics, attribution, preds, targets, levels, sample_batch, attention = results
            if args.generate_figures:
                generate_per_experiment_figures(exp_dir, config, preds, targets, eval_dir)
                make_attention_figures(config, sample_batch, attention, eval_dir)

        worst_lvl = attribution.get("worst_level_accuracy")
        if worst_lvl is None:
            worst_lvl = 0.0

        rows.append({
            "name": name,
            "group": re.sub(r"_s\d+$", "", name),
            "tokenizer": config.get("tokenizer", "?"),
            "pos_encoding": config.get("pos_encoding", "?"),
            "macro_f1": metrics.get("macro_f1", 0.0),
            "kappa": metrics.get("kappa", 0.0),
            "balanced_acc": metrics.get("balanced_acc", 0.0),
            "worst_lvl": worst_lvl,
            "path_prec": attribution.get("pathology_precision", 0.0),
            "path_rec": attribution.get("pathology_recall", 0.0),
        })
        analyzer_results[name] = attribution

    if not rows:
        print("no experiments found to evaluate")
        return

    print_comparison_table(rows)
    aggregated = aggregate_over_seeds(rows)
    print_aggregated_table(aggregated)

    save_markdown_table(rows, os.path.join(eval_dir, "comparison_table.md"))
    with open(os.path.join(eval_dir, "comparison.json"), "w") as comparison_file:
        json.dump(rows, comparison_file, indent=2)
    with open(os.path.join(eval_dir, "comparison_aggregated.json"), "w") as aggregated_file:
        json.dump(aggregated, aggregated_file, indent=2)

    if args.generate_figures:
        ordered = sorted(aggregated, key=order_key)
        for metric in ["macro_f1", "kappa", "worst_lvl"]:
            means = {}
            errors = {}
            for row in ordered:
                label = pretty_label(row)
                means[label] = row[f"{metric}_mean"]
                errors[label] = row[f"{metric}_std"]
            viz.plot_ablation_comparison(means, metric,
                                         os.path.join(eval_dir, f"ablation_{metric}.png"),
                                         errors=errors)

        heatmap_data = aggregate_perlevel(analyzer_results, rows)
        viz.plot_level_attribution_heatmap(heatmap_data,
                                           os.path.join(eval_dir,
                                                        "level_attribution_heatmap.png"))

    print("evaluation artifacts ->", eval_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Spine-ViT experiments")
    parser.add_argument("--experiments_dir", type=str, default="outputs")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--from_saved", action="store_true",
                        help="use saved test_results.json instead of re-running the model")
    parser.add_argument("--generate_figures", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    main()
