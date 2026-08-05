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


def compute_metrics(preds, targets, num_classes, task_name=""):
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    valid = targets != IGNORE_INDEX
    preds = preds[valid]
    targets = targets[valid]

    if targets.size == 0:
        return {f"{task_name}_n": 0}

    labels = list(range(num_classes))

    if len(np.unique(targets)) > 1:
        kappa = float(cohen_kappa_score(targets, preds, labels=labels, weights="quadratic"))
        kappa_linear = float(cohen_kappa_score(targets, preds, labels=labels, weights="linear"))
        kappa_unweighted = float(cohen_kappa_score(targets, preds, labels=labels))
    else:
        kappa = 0.0
        kappa_linear = 0.0
        kappa_unweighted = 0.0

    metrics = {
        "n": int(targets.size),
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, labels=labels, average="macro",
                                   zero_division=0)),
        "kappa": kappa,
        "kappa_linear": kappa_linear,
        "kappa_unweighted": kappa_unweighted,
        "mae": float(np.abs(preds - targets).mean()),
        "balanced_acc": float(balanced_accuracy_score(targets, preds)),
    }

    per_class_f1 = f1_score(targets, preds, labels=labels, average=None, zero_division=0)
    for class_index in range(num_classes):
        metrics[f"f1_class_{class_index}"] = float(per_class_f1[class_index])

    if task_name:
        renamed = {}
        for key in metrics:
            renamed[f"{task_name}_{key}"] = metrics[key]
        return renamed

    return metrics


def get_targets(dataset_or_targets):
    if hasattr(dataset_or_targets, "get_all_targets"):
        targets = dataset_or_targets.get_all_targets()
    else:
        targets = np.asarray(dataset_or_targets)
    return targets[targets != IGNORE_INDEX]


def compute_class_weights(dataset_or_targets, num_classes, scheme="inverse"):
    targets = get_targets(dataset_or_targets)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    total = counts.sum()

    inverse = np.ones(num_classes, dtype=np.float64)
    non_empty = counts > 0
    inverse[non_empty] = total / (num_classes * counts[non_empty])

    if scheme == "none":
        weights = np.ones(num_classes, dtype=np.float64)
    elif scheme == "inverse":
        weights = inverse
    elif scheme == "sqrt_inverse":
        weights = np.sqrt(inverse)
        weights = weights * (inverse.mean() / weights.mean())
    else:
        raise ValueError(f"Unknown class_weight scheme: {scheme}")

    return torch.tensor(weights, dtype=torch.float32)


def coral_pos_weights(dataset_or_targets, num_classes):
    targets = get_targets(dataset_or_targets)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    total = counts.sum()

    weights = np.ones(num_classes - 1, dtype=np.float64)
    for threshold_index in range(num_classes - 1):
        n_positive = counts[threshold_index + 1:].sum()
        n_negative = total - n_positive
        weights[threshold_index] = n_negative / max(1.0, n_positive)

    return torch.tensor(weights, dtype=torch.float32)


def coral_loss(logits, targets, pos_weight=None, ignore_index=IGNORE_INDEX):
    valid = targets != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0

    logits = logits[valid]
    targets = targets[valid]

    thresholds = torch.arange(logits.shape[1], device=logits.device)
    binary_targets = (targets[:, None] > thresholds[None, :]).float()

    return F.binary_cross_entropy_with_logits(logits, binary_targets,
                                              pos_weight=pos_weight, reduction="mean")


def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1).tolist()
    if isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def unique_in_order(values):
    seen = set()
    out = []
    for value in values.tolist():
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


class LevelAttributionAnalyzer:
    def __init__(self, level_names=None, num_classes=3):
        self.level_names = level_names
        self.num_classes = num_classes
        self.pred_grades = []
        self.true_grades = []
        self.level_indices = []
        self.patient_ids = []

    def update(self, pred_grades, true_grades, level_indices, patient_id=None):
        pred_grades = to_list(pred_grades)
        true_grades = to_list(true_grades)
        level_indices = to_list(level_indices)

        self.pred_grades.extend(pred_grades)
        self.true_grades.extend(true_grades)
        self.level_indices.extend(level_indices)

        if isinstance(patient_id, (list, tuple, np.ndarray)):
            self.patient_ids.extend(list(patient_id))
        else:
            self.patient_ids.extend([patient_id] * len(pred_grades))

    def level_name(self, index):
        if self.level_names and 0 <= index < len(self.level_names):
            return self.level_names[index]
        return f"level_{index}"

    def compute(self, pathology_threshold=1):
        pred = np.asarray(self.pred_grades)
        true = np.asarray(self.true_grades)
        levels = np.asarray(self.level_indices)
        patients = np.asarray(self.patient_ids, dtype=object)

        valid = true != IGNORE_INDEX
        pred = pred[valid]
        true = true[valid]
        levels = levels[valid]
        patients = patients[valid]

        if true.size == 0:
            return {"n": 0}

        pred_positive = pred >= pathology_threshold
        true_positive = true >= pathology_threshold

        tp = int(np.sum(pred_positive & true_positive))
        fp = int(np.sum(pred_positive & ~true_positive))
        fn = int(np.sum(~pred_positive & true_positive))
        tn = int(np.sum(~pred_positive & ~true_positive))

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        if fp + tn > 0:
            fp_rate = fp / (fp + tn)
        else:
            fp_rate = 0.0

        worst_level_acc = self.worst_level_accuracy(pred, true, levels, patients,
                                                    pathology_threshold)
        per_level = self.per_level_stats(pred, true, levels, pathology_threshold)
        confusion = confusion_matrix(true, pred, labels=list(range(self.num_classes)))

        return {
            "n": int(true.size),
            "n_pathological": int(true_positive.sum()),
            "worst_level_accuracy": worst_level_acc[0],
            "n_studies_with_pathology": worst_level_acc[1],
            "pathology_precision": precision,
            "pathology_recall": recall,
            "pathology_f1": f1,
            "pathology_fp_rate": fp_rate,
            "per_level": per_level,
            "grade_confusion_matrix": confusion.tolist(),
        }

    def worst_level_accuracy(self, pred, true, levels, patients, pathology_threshold):
        has_ids = patients.size > 0
        if has_ids:
            all_none = True
            for patient in patients:
                if patient is not None:
                    all_none = False
                    break
            if all_none:
                has_ids = False

        if not has_ids:
            return None, 0

        correct = 0
        total = 0
        for patient in unique_in_order(patients):
            mask = patients == patient
            true_grades = true[mask]
            pred_grades = pred[mask]
            study_levels = levels[mask]

            if true_grades.max() < pathology_threshold:
                continue

            total = total + 1
            worst_true_levels = set(study_levels[true_grades == true_grades.max()].tolist())
            worst_pred_level = int(study_levels[int(np.argmax(pred_grades))])
            if worst_pred_level in worst_true_levels:
                correct = correct + 1

        if total == 0:
            return None, 0
        return correct / total, total

    def per_level_stats(self, pred, true, levels, pathology_threshold):
        per_level = {}

        for level_index in sorted(np.unique(levels)):
            mask = levels == level_index
            pred_positive = pred[mask] >= pathology_threshold
            true_positive = true[mask] >= pathology_threshold

            tp = int(np.sum(pred_positive & true_positive))
            fp = int(np.sum(pred_positive & ~true_positive))
            fn = int(np.sum(~pred_positive & true_positive))

            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = None

            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = None

            per_level[self.level_name(int(level_index))] = {
                "exact_acc": float((pred[mask] == true[mask]).mean()),
                "n": int(mask.sum()),
                "n_pathological": int(true_positive.sum()),
                "recall": recall,
                "precision": precision,
            }

        return per_level

    def reset(self):
        self.pred_grades = []
        self.true_grades = []
        self.level_indices = []
        self.patient_ids = []
