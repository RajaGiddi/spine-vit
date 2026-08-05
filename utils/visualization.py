from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def _ensure_dir(save_path: str):
    d = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(d, exist_ok=True)


def plot_grade_confusion_matrix(true, pred, class_names, title, save_path):
    """Heatmap of predicted vs true grades."""
    from sklearn.metrics import confusion_matrix

    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    valid = true != -1
    cm = confusion_matrix(true[valid], pred[valid], labels=list(range(len(class_names))))

    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True
    )
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_level_attribution_heatmap(analyzer_results: Dict[str, Dict], save_path):
    models = list(analyzer_results.keys())
    level_names, matrix = [], []
    for res in analyzer_results.values():
        for lv in res.get("per_level", {}):
            if lv not in level_names:
                level_names.append(lv)
    for m in models:
        per_level = analyzer_results[m].get("per_level", {})
        matrix.append([per_level.get(lv, {}).get("exact_acc", np.nan) for lv in level_names])
    matrix = np.array(matrix, dtype=float)

    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(max(6, len(level_names) * 1.2 + 3), max(2.5, len(models) * 0.62 + 1)))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=level_names, yticklabels=models,
        vmin=0, vmax=1, ax=ax, annot_kws={"fontsize": 9},
        cbar_kws={"label": "exact-grade accuracy"},
    )
    ax.set_xlabel("Vertebral level")
    ax.set_title("Per-level exact-grade accuracy")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_attention_weights(attention_matrix, level_labels, save_path, title="Encoder self-attention"):
    """Transformer attention weights as a level x level heatmap."""
    attention_matrix = np.asarray(attention_matrix)
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(attention_matrix, cmap="magma", xticklabels=level_labels, yticklabels=level_labels, ax=ax, square=True)
    ax.set_xlabel("Key (attended-to level)")
    ax.set_ylabel("Query (attending level)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_attention_overlay(image, boxes, attention_weights, level_labels, save_path, title="Attention overlay"):
    img = np.asarray(image)
    if img.ndim == 3:
        img = img[img.shape[0] // 2] if img.shape[0] <= 4 else img.mean(0)
    boxes = np.asarray(boxes)
    aw = np.asarray(attention_weights, dtype=float)
    if aw.size and aw.max() > aw.min():
        aw = (aw - aw.min()) / (aw.max() - aw.min())

    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    cmap = plt.get_cmap("autumn")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = float(aw[i]) if i < len(aw) else 0.0
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=cmap(w), facecolor=cmap(w), alpha=0.35
        )
        ax.add_patch(rect)
        label = level_labels[i] if level_labels and i < len(level_labels) else str(i)
        ax.text(x1, y1 - 2, label, color="cyan", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_training_curves(history: Dict, save_path):
    _ensure_dir(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    if "train_loss" in history:
        axes[0].plot(epochs, history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    for key in ("val_macro_f1", "val_kappa", "val_balanced_acc"):
        if key in history:
            axes[1].plot(range(1, len(history[key]) + 1), history[key], label=key.replace("val_", ""))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_ablation_comparison(results_dict: Dict[str, float], metric_name: str, save_path, errors=None):
    _ensure_dir(save_path)
    items = sorted(results_dict.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    errs = [float(errors.get(n, 0.0)) for n in names] if errors else [0.0] * len(names)

    fig, ax = plt.subplots(figsize=(8.5, max(2.5, len(names) * 0.6 + 0.8)))
    colors = sns.color_palette("crest", len(names))
    ax.barh(names, values, xerr=errs, color=colors, capsize=4,
            error_kw={"ecolor": "0.35", "lw": 1.2})
    ax.set_xlabel(metric_name)
    ax.set_title(f"{metric_name} across ablation variants (mean ± std over seeds)")
    ax.set_xlim(0, max(1.0, (max(values) + max(errs)) * 1.15))
    for i, (v, e) in enumerate(zip(values, errs)):
        ax.text(v + e + 0.012, i, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
    return fig
