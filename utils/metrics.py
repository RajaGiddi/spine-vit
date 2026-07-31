"""Metrics: standard grading metrics, class weights, and level-attribution analysis.

The LevelAttributionAnalyzer is the paper's signature diagnostic: for pathological
findings (grade >= threshold), does the model flag pathology *at the correct level*?
This is what standard macro-F1 / kappa cannot reveal about level-misattribution.
"""

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
    """Compute macro-F1, Cohen's kappa, balanced accuracy, accuracy, and per-class F1.

    Invalid targets (== IGNORE_INDEX) are filtered out first.
    """
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
        # headline "kappa" stays QUADRATIC-weighted (ordinal-appropriate; matches prior work)
        "kappa": float(cohen_kappa_score(targets, preds, labels=labels, weights="quadratic"))
        if len(np.unique(targets)) > 1
        else 0.0,
        "kappa_linear": float(cohen_kappa_score(targets, preds, labels=labels, weights="linear"))
        if len(np.unique(targets)) > 1
        else 0.0,
        "kappa_unweighted": float(cohen_kappa_score(targets, preds, labels=labels))
        if len(np.unique(targets)) > 1
        else 0.0,
        "mae": float(np.abs(preds - targets).mean()),   # mean |pred - true| in grade units
        "balanced_acc": float(balanced_accuracy_score(targets, preds)),
    }
    for c in range(num_classes):
        metrics[f"f1_class_{c}"] = float(per_class_f1[c])
    if task_name:
        metrics = {f"{task_name}_{k}": v for k, v in metrics.items()}
    return metrics


def compute_class_weights(dataset_or_targets, num_classes: int, scheme: str = "inverse") -> torch.Tensor:
    """Class weights from training targets (ignores IGNORE_INDEX).

    scheme:
      "inverse"      - full inverse frequency: total / (num_classes * count_c). Strong
                       rebalancing; on rare classes this drives the model to over-flag
                       (high recall, low precision).
      "sqrt_inverse" - sqrt of the inverse weights, rescaled to the same mean. Softer:
                       trades a little recall for meaningfully better precision.
      "none"         - uniform weights (no rebalancing).
    """
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
        w = w * (inv.mean() / w.mean())  # keep overall magnitude comparable to inverse
    else:
        raise ValueError(f"Unknown class_weight scheme '{scheme}' (inverse|sqrt_inverse|none)")
    return torch.tensor(w, dtype=torch.float32)


def coral_pos_weights(dataset_or_targets, num_classes: int) -> torch.Tensor:
    """Per-threshold positive weights for CORAL = N_neg / N_pos at each rank threshold k
    (positive = y > k). Threshold 0 balances {Mod,Sev} vs {Normal}; threshold 1 balances
    {Sev} vs {Normal,Mod}. Each binary task is weighted INDEPENDENTLY — reusing the 3-class
    per-sample weights instead squeezes the middle class to zero (that was the bug).
    """
    if hasattr(dataset_or_targets, "get_all_targets"):
        targets = dataset_or_targets.get_all_targets()
    else:
        targets = np.asarray(dataset_or_targets)
    targets = targets[targets != IGNORE_INDEX]
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    pw = np.ones(num_classes - 1, dtype=np.float64)
    for k in range(num_classes - 1):
        n_pos = counts[k + 1:].sum()   # y > k
        n_neg = total - n_pos          # y <= k
        pw[k] = n_neg / max(1.0, n_pos)
    return torch.tensor(pw, dtype=torch.float32)


def coral_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor = None,
               ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """CORAL ordinal loss. logits (N, K-1) rank-threshold logits; targets (N,) grades in
    [0, K-1]. Binary target at threshold k is 1 iff y > k. BCE per threshold with an
    INDEPENDENT per-threshold pos_weight (see coral_pos_weights).
    """
    valid = targets != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0
    logits, targets = logits[valid], targets[valid]
    k = torch.arange(logits.shape[1], device=logits.device)
    bin_t = (targets[:, None] > k[None, :]).float()                       # (N, K-1)
    return F.binary_cross_entropy_with_logits(logits, bin_t, pos_weight=pos_weight, reduction="mean")


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """CORAL decode: predicted grade = number of thresholds passed. (N,K-1) -> (N,) in
    [0, K-1]. The shared weight vector keeps thresholds monotonic, so this is coherent."""
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
        """Level-attribution analysis.

        Two things prior work does not report, framed to be *honest* rather than
        inflatable:

        1. worst_level_accuracy (headline, clinical): per study, does argmax over the
           levels of the predicted grade land on a truly worst-affected level? This is
           "which level do you operate on?" and it CANNOT be gamed by predicting
           pathology everywhere — doing so makes the prediction argmax arbitrary. Only
           computed over studies that actually have pathology, and only when per-finding
           study ids were supplied via update(..., patient_id=...).

        2. Pathology detection reported as precision AND recall (+ FP rate). Recall
           alone ("of pathological levels, how many were flagged") is inflatable by
           over-flagging; precision exposes exactly that. Report the pair.
        """
        pred = np.asarray(self.pred_grades)
        true = np.asarray(self.true_grades)
        lvl = np.asarray(self.level_indices)
        pid = np.asarray(self.patient_ids, dtype=object)

        valid = true != IGNORE_INDEX
        pred, true, lvl, pid = pred[valid], true[valid], lvl[valid], pid[valid]
        if true.size == 0:
            return {"n": 0}

        # --- Pathology detection as a level-wise binary problem (grade >= threshold) ---
        pred_pos = pred >= pathology_threshold
        true_pos = true >= pathology_threshold
        tp = int(np.sum(pred_pos & true_pos))
        fp = int(np.sum(pred_pos & ~true_pos))
        fn = int(np.sum(~pred_pos & true_pos))
        tn = int(np.sum(~pred_pos & ~true_pos))
        recall = tp / (tp + fn) if (tp + fn) else 0.0          # inflatable alone
        precision = tp / (tp + fp) if (tp + fp) else 0.0       # exposes over-flagging
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) else 0.0         # of normal levels, fraction flagged

        # --- Worst-level identification (per study; non-gameable) ---
        has_ids = pid.size > 0 and not all(p is None for p in pid)
        worst_correct = worst_total = 0
        if has_ids:
            for p in _unique_objects(pid):
                m = pid == p
                t_g, p_g, l_g = true[m], pred[m], lvl[m]
                if t_g.max() < pathology_threshold:
                    continue  # no pathology in this study -> nothing to localize
                worst_total += 1
                true_worst_levels = set(l_g[t_g == t_g.max()].tolist())
                pred_worst_level = int(l_g[int(np.argmax(p_g))])
                worst_correct += int(pred_worst_level in true_worst_levels)
        worst_level_acc = (worst_correct / worst_total) if worst_total else None

        # --- Per-level breakdown (exact accuracy + detection precision/recall) ---
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
            # Clinical headline — which level is worst-affected (per study). Non-gameable.
            "worst_level_accuracy": worst_level_acc,
            "n_studies_with_pathology": worst_total,
            # Pathology detection — report the PAIR; recall alone is dishonest.
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
    """Order-preserving unique for an object array (study ids), avoiding np.unique
    edge cases with mixed/None values."""
    seen, out = set(), []
    for x in arr.tolist():
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
