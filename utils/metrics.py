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


def compute_metrics(predictions, targets, num_classes, task_name=""):
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    # Levels with no label are marked -1, we drop them before scoring
    valid = targets != IGNORE_INDEX
    predictions = predictions[valid]
    targets = targets[valid]

    if targets.size == 0:
        return {f"{task_name}_n": 0}

    labels = []
    for i in range(num_classes):
        labels.append(i)

    # Kappa needs at least two different true labels or sklearn causes errrors
    unique_targets = np.unique(targets)
    if len(unique_targets) > 1:
        kappa = float(cohen_kappa_score(targets, predictions, labels=labels, weights="quadratic"))
        kappa_linear = float(cohen_kappa_score(targets, predictions, labels=labels, weights="linear"))
        kappa_unweighted = float(cohen_kappa_score(targets, predictions, labels=labels))
    else:
        kappa = 0.0
        kappa_linear = 0.0
        kappa_unweighted = 0.0

    correct = predictions == targets
    accuracy = float(correct.mean())

    macro_f1 = f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
    macro_f1 = float(macro_f1)

    errors = np.abs(predictions - targets)
    mae = float(errors.mean())

    balanced = float(balanced_accuracy_score(targets, predictions))

    metrics = {
        "n": int(targets.size),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "kappa_linear": kappa_linear,
        "kappa_unweighted": kappa_unweighted,
        "mae": mae,
        "balanced_acc": balanced,
    }

    # One f1 per class as well, so we can see which grade is being missed
    f1s = f1_score(targets, predictions, labels=labels, average=None, zero_division=0)
    for i in range(num_classes):
        metrics[f"f1_class_{i}"] = float(f1s[i])

    if task_name:
        renamed = {}
        for key in metrics:
            renamed[f"{task_name}_{key}"] = metrics[key]
        return renamed
    else:
        return metrics


def get_targets(dataset_or_targets):
    # We accept either a dataset object or a plain array of labels
    if hasattr(dataset_or_targets, "get_all_targets"):
        targets = dataset_or_targets.get_all_targets()
    else:
        targets = np.asarray(dataset_or_targets)

    keep = targets != IGNORE_INDEX
    return targets[keep]


def compute_class_weights(dataset_or_targets, num_classes, scheme="inverse"):
    targets = get_targets(dataset_or_targets)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    total = counts.sum()

    # Rare grades get a bigger weight, empty ones stay at 1 so we never divide by zero
    inverse = np.ones(num_classes, dtype=np.float64)
    non_empty = counts > 0
    inverse[non_empty] = total / (num_classes * counts[non_empty])

    if scheme == "none":
        weights = np.ones(num_classes, dtype=np.float64)
    elif scheme == "inverse":
        weights = inverse
    elif scheme == "sqrt_inverse":
        # Sqrt softens the weighting, then we rescale so the mean matches inverse
        weights = np.sqrt(inverse)
        scale = inverse.mean() / weights.mean()
        weights = weights * scale
    else:
        raise ValueError(f"Unknown class_weight scheme: {scheme}")

    return torch.tensor(weights, dtype=torch.float32)


def coral_pos_weights(dataset_or_targets, num_classes):
    targets = get_targets(dataset_or_targets)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    total = counts.sum()

    # Coral turns the grade into num_classes-1 yes/no questions, each needs its own balance
    weights = np.ones(num_classes - 1, dtype=np.float64)
    for k in range(num_classes - 1):
        n_positive = counts[k + 1:].sum()
        n_negative = total - n_positive
        weights[k] = n_negative / max(1.0, n_positive)

    return torch.tensor(weights, dtype=torch.float32)


def coral_loss(logits, targets, pos_weight=None, ignore_index=IGNORE_INDEX):
    valid = targets != ignore_index
    if valid.sum() == 0:
        return logits.sum() * 0.0

    logits = logits[valid]
    targets = targets[valid]

    # Question k is "is the grade above k", so the target is a row of 1s then 0s
    thresholds = torch.arange(logits.shape[1], device=logits.device)
    binary_targets = (targets[:, None] > thresholds[None, :]).float()

    return F.binary_cross_entropy_with_logits(logits, binary_targets,
                                              pos_weight=pos_weight, reduction="mean")


def coral_predict(logits):
    # Count how many thresholds the model said yes to
    above = torch.sigmoid(logits) > 0.5
    return above.sum(dim=1)


def to_list(x):
    if isinstance(x, torch.Tensor):
        flat = x.detach().cpu().numpy().reshape(-1)
        return flat.tolist()
    elif isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()
    elif isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x]


def unique_in_order(values):
    # Keeps first-seen order, which set() would not
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
        self.prediction_grades = []
        self.true_grades = []
        self.level_indices = []
        self.patient_ids = []

    def update(self, prediction_grades, true_grades, level_indices, patient_id=None):
        prediction_grades = to_list(prediction_grades)
        true_grades = to_list(true_grades)
        level_indices = to_list(level_indices)

        self.prediction_grades.extend(prediction_grades)
        self.true_grades.extend(true_grades)
        self.level_indices.extend(level_indices)

        # patient_id can be one id for the whole batch or one per token
        if isinstance(patient_id, (list, tuple, np.ndarray)):
            self.patient_ids.extend(list(patient_id))
        else:
            repeated = [patient_id] * len(prediction_grades)
            self.patient_ids.extend(repeated)

    def level_name(self, index):
        if self.level_names and 0 <= index < len(self.level_names):
            return self.level_names[index]
        else:
            return f"level_{index}"

    def compute(self, pathology_threshold=1):
        prediction = np.asarray(self.prediction_grades)
        true = np.asarray(self.true_grades)
        levels = np.asarray(self.level_indices)
        patients = np.asarray(self.patient_ids, dtype=object)

        valid = true != IGNORE_INDEX
        prediction = prediction[valid]
        true = true[valid]
        levels = levels[valid]
        patients = patients[valid]

        if true.size == 0:
            return {"n": 0}

        # Anything at or above the threshold counts as pathology
        prediction_positive = prediction >= pathology_threshold
        true_positive = true >= pathology_threshold

        tp = int(np.sum(prediction_positive & true_positive))
        fp = int(np.sum(prediction_positive & ~true_positive))
        fn = int(np.sum(~prediction_positive & true_positive))
        tn = int(np.sum(~prediction_positive & ~true_positive))

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

        worst_level_acc = self.worst_level_accuracy(prediction, true, levels, patients,
                                                    pathology_threshold)
        per_level = self.per_level_stats(prediction, true, levels, pathology_threshold)

        class_labels = []
        for i in range(self.num_classes):
            class_labels.append(i)
        confusion = confusion_matrix(true, prediction, labels=class_labels)

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

    def worst_level_accuracy(self, prediction, true, levels, patients, pathology_threshold):
        # This is the metric the paper cares about: did we point at the right level
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
            prediction_grades = prediction[mask]
            study_levels = levels[mask]

            # Studies with nothing wrong have no worst level to find
            if true_grades.max() < pathology_threshold:
                continue

            total = total + 1

            # There can be a tie for worst, any of them counts as correct
            worst_true = []
            for i in range(len(study_levels)):
                if true_grades[i] == true_grades.max():
                    worst_true.append(study_levels[i])
            worst_true = set(worst_true)

            best_guess = int(np.argmax(prediction_grades))
            worst_prediction_level = int(study_levels[best_guess])
            if worst_prediction_level in worst_true:
                correct = correct + 1

        if total == 0:
            return None, 0
        else:
            return correct / total, total

    def per_level_stats(self, prediction, true, levels, pathology_threshold):
        per_level = {}

        for level_index in sorted(np.unique(levels)):
            mask = levels == level_index
            prediction_positive = prediction[mask] >= pathology_threshold
            true_positive = true[mask] >= pathology_threshold

            tp = int(np.sum(prediction_positive & true_positive))
            fp = int(np.sum(prediction_positive & ~true_positive))
            fn = int(np.sum(~prediction_positive & true_positive))

            # None rather than 0 here, so an empty level shows as missing not as a real zero
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = None

            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = None

            hits = prediction[mask] == true[mask]
            name = self.level_name(int(level_index))
            per_level[name] = {
                "exact_acc": float(hits.mean()),
                "n": int(mask.sum()),
                "n_pathological": int(true_positive.sum()),
                "recall": recall,
                "precision": precision,
            }

        return per_level

    def reset(self):
        self.prediction_grades = []
        self.true_grades = []
        self.level_indices = []
        self.patient_ids = []
