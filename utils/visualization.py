import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def ensure_dir(save_path):
    folder = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(folder, exist_ok=True)


def plot_grade_confusion_matrix(true, pred, class_names, title, save_path):
    from sklearn.metrics import confusion_matrix

    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    valid = true != -1
    confusion = confusion_matrix(true[valid], pred[valid], labels=list(range(len(class_names))))

    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_level_attribution_heatmap(analyzer_results, save_path):
    model_names = list(analyzer_results.keys())

    level_names = []
    for name in model_names:
        for level in analyzer_results[name].get("per_level", {}):
            if level not in level_names:
                level_names.append(level)

    matrix = []
    for name in model_names:
        per_level = analyzer_results[name].get("per_level", {})
        row = []
        for level in level_names:
            if level in per_level:
                row.append(per_level[level].get("exact_acc", np.nan))
            else:
                row.append(np.nan)
        matrix.append(row)
    matrix = np.array(matrix, dtype=float)

    width = max(6, len(level_names) * 1.2 + 3)
    height = max(2.5, len(model_names) * 0.62 + 1)

    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", xticklabels=level_names,
                yticklabels=model_names, vmin=0, vmax=1, ax=ax,
                annot_kws={"fontsize": 9},
                cbar_kws={"label": "exact-grade accuracy"})
    ax.set_xlabel("Vertebral level")
    ax.set_title("Per-level exact-grade accuracy")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_attention_weights(attention_matrix, level_labels, save_path,
                           title="Encoder self-attention"):
    attention_matrix = np.asarray(attention_matrix)

    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(attention_matrix, cmap="magma", xticklabels=level_labels,
                yticklabels=level_labels, ax=ax, square=True)
    ax.set_xlabel("Key (attended-to level)")
    ax.set_ylabel("Query (attending level)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_attention_overlay(image, boxes, attention_weights, level_labels, save_path,
                           title="Attention overlay"):
    image = np.asarray(image)
    if image.ndim == 3:
        if image.shape[0] <= 4:
            image = image[image.shape[0] // 2]
        else:
            image = image.mean(0)

    boxes = np.asarray(boxes)
    weights = np.asarray(attention_weights, dtype=float)
    if weights.size and weights.max() > weights.min():
        weights = (weights - weights.min()) / (weights.max() - weights.min())

    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray")
    colormap = plt.get_cmap("autumn")

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        if i < len(weights):
            weight = float(weights[i])
        else:
            weight = 0.0

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                 edgecolor=colormap(weight),
                                 facecolor=colormap(weight), alpha=0.35)
        ax.add_patch(rect)

        if level_labels and i < len(level_labels):
            label = level_labels[i]
        else:
            label = str(i)
        ax.text(x1, y1 - 2, label, color="cyan", fontsize=8)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_training_curves(history, save_path):
    ensure_dir(save_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if "train_loss" in history:
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label="train")
    if "val_loss" in history:
        epochs = range(1, len(history["val_loss"]) + 1)
        axes[0].plot(epochs, history["val_loss"], label="val")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    for key in ["val_macro_f1", "val_kappa", "val_balanced_acc"]:
        if key in history:
            epochs = range(1, len(history[key]) + 1)
            axes[1].plot(epochs, history[key], label=key.replace("val_", ""))

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def sort_by_value(item):
    return item[1]


def plot_ablation_comparison(results_dict, metric_name, save_path, errors=None):
    ensure_dir(save_path)

    items = sorted(results_dict.items(), key=sort_by_value)
    names = []
    values = []
    for name, value in items:
        names.append(name)
        values.append(value)

    bar_errors = []
    for name in names:
        if errors:
            bar_errors.append(float(errors.get(name, 0.0)))
        else:
            bar_errors.append(0.0)

    height = max(2.5, len(names) * 0.6 + 0.8)
    fig, ax = plt.subplots(figsize=(8.5, height))
    colors = sns.color_palette("crest", len(names))
    ax.barh(names, values, xerr=bar_errors, color=colors, capsize=4,
            error_kw={"ecolor": "0.35", "lw": 1.2})

    ax.set_xlabel(metric_name)
    ax.set_title(metric_name + " across ablation variants (mean and std over seeds)")
    ax.set_xlim(0, max(1.0, (max(values) + max(bar_errors)) * 1.15))

    for i in range(len(values)):
        ax.text(values[i] + bar_errors[i] + 0.012, i, "%.3f" % values[i],
                va="center", fontsize=9)

    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig
