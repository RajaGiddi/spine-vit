from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    balanced_accuracy_score,
    confusion_matrix,
)

IGNORE_INDEX = -1
RSNA_LEVEL_NAMES = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


def compute_metrics(preds, targets, num_classes: int, task_name: str = "") -> Dict:
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    valid = targets != IGNORE_INDEX
    preds, targets = preds[valid], targets[valid]

    if targets.size == 0:
        return {f"{task_name}_n": 0}

    labels = list(range(num_classes))
    per_class_f1 = f1_score(targets, preds, labels=labels, average=None, zero_division=0)
    metrics = {
        "n": int(targets.size),
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(targets, preds, labels=labels, weights="quadratic"))
        if len(np.unique(targets)) > 1
        else 0.0,
        "kappa_linear": float(cohen_kappa_score(targets, preds, labels=labels, weights="linear"))
        if len(np.unique(targets)) > 1
        else 0.0,
        "kappa_unweighted": float(cohen_kappa_score(targets, preds, labels=labels))
        if len(np.unique(targets)) > 1
        else 0.0,
        "mae": float(np.abs(preds - targets).mean()),
        "balanced_acc": float(balanced_accuracy_score(targets, preds)),
    }
    for c in range(num_classes):
        metrics[f"f1_class_{c}"] = float(per_class_f1[c])
    if task_name:
        metrics = {f"{task_name}_{k}": v for k, v in metrics.items()}
    return metrics


def compute_class_weights(dataset_or_targets, num_classes: int, scheme: str = "inverse") -> torch.Tensor:
    if hasattr(dataset_or_targets, "get_all_targets"):
        targets = dataset_or_targets.get_all_targets()
    else:
        targets = np.asarray(dataset_or_targets)
    targets = targets[targets != IGNORE_INDEX]

    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    inv = np.ones(num_classes, dtype=np.float64)
    nz = counts > 0
    inv[nz] = total / (num_classes * counts[nz])

    if scheme == "none":
        w = np.ones(num_classes, dtype=np.float64)
    elif scheme == "inverse":
        w = inv
    elif scheme == "sqrt_inverse":
        w = np.sqrt(inv)
        w = w * (inv.mean() / w.mean())
    else:
        raise ValueError(f"Unknown class_weight scheme '{scheme}' (inverse|sqrt_inverse|none)")
    return torch.tensor(w, dtype=torch.float32)


def coral_pos_weights(dataset_or_targets, num_classes: int) -> torch.Tensor:
    if hasattr(dataset_or_targets, "get_all_targets"):
        targets = dataset_or_targets.get_all_targets()
    else:
        targets = np.asarray(dataset_or_targets)
    targets = targets[targets != IGNORE_INDEX]
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    pw = np.ones(num_classes - 1, dtype=np.float64)
    for k in range(num_classes - 1):
        n_pos = counts[k + 1:].sum()
        n_neg = total - n_pos
        pw[k] = n_neg / max(1.0, n_pos)
    return torch.tensor(pw, dtype=torch.float32)


def coral_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor = None,
               ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    valid = targets != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0
    logits, targets = logits[valid], targets[valid]
    k = torch.arange(logits.shape[1], device=logits.device)
    bin_t = (targets[:, None] > k[None, :]).float()
    return F.binary_cross_entropy_with_logits(logits, bin_t, pos_weight=pos_weight, reduction="mean")


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


class LevelAttributionAnalyzer:
    """Accumulate per-level predictions and analyze level-specific detection quality."""

    def __init__(self, level_names: Optional[List[str]] = None, num_classes: int = 3):
        self.level_names = level_names
        self.num_classes = num_classes
        self.pred_grades: List[int] = []
        self.true_grades: List[int] = []
        self.level_indices: List[int] = []
        self.patient_ids: List = []

    def update(self, pred_grades, true_grades, level_indices, patient_id=None):
        pred_grades = _to_list(pred_grades)
        true_grades = _to_list(true_grades)
        level_indices = _to_list(level_indices)
        self.pred_grades.extend(pred_grades)
        self.true_grades.extend(true_grades)
        self.level_indices.extend(level_indices)
        self.patient_ids.extend(
            [patient_id] * len(pred_grades) if not isinstance(patient_id, (list, tuple, np.ndarray)) else list(patient_id)
        )

    def compute(self, pathology_threshold: int = 1) -> Dict:
        pred = np.asarray(self.pred_grades)
        true = np.asarray(self.true_grades)
        lvl = np.asarray(self.level_indices)
        pid = np.asarray(self.patient_ids, dtype=object)

        valid = true != IGNORE_INDEX
        pred, true, lvl, pid = pred[valid], true[valid], lvl[valid], pid[valid]
        if true.size == 0:
            return {"n": 0}

        pred_pos = pred >= pathology_threshold
        true_pos = true >= pathology_threshold
        tp = int(np.sum(pred_pos & true_pos))
        fp = int(np.sum(pred_pos & ~true_pos))
        fn = int(np.sum(~pred_pos & true_pos))
        tn = int(np.sum(~pred_pos & ~true_pos))
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) else 0.0

        has_ids = pid.size > 0 and not all(p is None for p in pid)
        worst_correct = worst_total = 0
        if has_ids:
            for p in _unique_objects(pid):
                m = pid == p
                t_g, p_g, l_g = true[m], pred[m], lvl[m]
                if t_g.max() < pathology_threshold:
                    continue
                worst_total += 1
                true_worst_levels = set(l_g[t_g == t_g.max()].tolist())
                pred_worst_level = int(l_g[int(np.argmax(p_g))])
                worst_correct += int(pred_worst_level in true_worst_levels)
        worst_level_acc = (worst_correct / worst_total) if worst_total else None

        per_level = {}
        for li in sorted(np.unique(lvl)):
            m = lvl == li
            pm, tm = pred[m] >= pathology_threshold, true[m] >= pathology_threshold
            tp_l = int(np.sum(pm & tm)); fp_l = int(np.sum(pm & ~tm)); fn_l = int(np.sum(~pm & tm))
            per_level[self._name(int(li))] = {
                "exact_acc": float((pred[m] == true[m]).mean()),
                "n": int(m.sum()),
                "n_pathological": int(tm.sum()),
                "recall": (tp_l / (tp_l + fn_l)) if (tp_l + fn_l) else None,
                "precision": (tp_l / (tp_l + fp_l)) if (tp_l + fp_l) else None,
            }

        cm = confusion_matrix(true, pred, labels=list(range(self.num_classes)))
        return {
            "n": int(true.size),
            "n_pathological": int(true_pos.sum()),
            "worst_level_accuracy": worst_level_acc,
            "n_studies_with_pathology": worst_total,
            "pathology_precision": precision,
            "pathology_recall": recall,
            "pathology_f1": f1,
            "pathology_fp_rate": fp_rate,
            "per_level": per_level,
            "grade_confusion_matrix": cm.tolist(),
        }

    def _name(self, li: int) -> str:
        if self.level_names and 0 <= li < len(self.level_names):
            return self.level_names[li]
        return f"level_{li}"

    def reset(self):
        self.pred_grades.clear()
        self.true_grades.clear()
        self.level_indices.clear()
        self.patient_ids.clear()


def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1).tolist()
    if isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _unique_objects(arr: np.ndarray) -> list:
    seen, out = set(), []
    for x in arr.tolist():
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
