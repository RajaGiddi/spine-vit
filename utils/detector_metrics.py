import numpy as np
import torch

RSNA_LEVEL_NAMES = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


def make_heatmaps(centers, valid, size, sigma):
    batch_size = centers.shape[0]
    num_levels = centers.shape[1]
    device = centers.device

    x_positions = torch.arange(size, device=device).view(1, 1, 1, size)
    y_positions = torch.arange(size, device=device).view(1, 1, size, 1)
    center_x = centers[..., 0].view(batch_size, num_levels, 1, 1)
    center_y = centers[..., 1].view(batch_size, num_levels, 1, 1)

    squared_distance = (x_positions - center_x) ** 2 + (y_positions - center_y) ** 2
    gaussian = torch.exp(-squared_distance / (2 * sigma ** 2))
    return gaussian * valid.view(batch_size, num_levels, 1, 1)


def spatial_softmax(logits, temp=1.0):
    batch_size, num_levels, height, width = logits.shape
    flat = logits.view(batch_size, num_levels, -1) / temp
    return torch.softmax(flat, dim=-1).view(batch_size, num_levels, height, width)


def expected_coords(probs, height, width, device):
    x_positions = torch.arange(width, device=device).float()
    y_positions = torch.arange(height, device=device).float()
    expected_x = (probs.sum(dim=2) * x_positions).sum(dim=-1)
    expected_y = (probs.sum(dim=3) * y_positions).sum(dim=-1)
    return expected_x, expected_y


def soft_argmax(logits, temp=1.0):
    probs = spatial_softmax(logits, temp)
    height = logits.shape[2]
    width = logits.shape[3]
    expected_x, expected_y = expected_coords(probs, height, width, logits.device)
    return torch.stack([expected_x, expected_y], dim=-1)


def coord_loss(logits, gt_coords, valid, reg=0.0):
    probs = spatial_softmax(logits)
    height = logits.shape[2]
    width = logits.shape[3]
    expected_x, expected_y = expected_coords(probs, height, width, logits.device)

    squared_error = (expected_x - gt_coords[..., 0]) ** 2 + (expected_y - gt_coords[..., 1]) ** 2
    loss = (squared_error * valid).sum() / valid.sum().clamp_min(1.0)

    if reg > 0:
        x_positions = torch.arange(width, device=logits.device).float()
        y_positions = torch.arange(height, device=logits.device).float()
        var_x = (probs.sum(dim=2) * (x_positions[None, None] - expected_x[..., None]) ** 2).sum(dim=-1)
        var_y = (probs.sum(dim=3) * (y_positions[None, None] - expected_y[..., None]) ** 2).sum(dim=-1)
        variance = var_x + var_y
        loss = loss + reg * (variance * valid).sum() / valid.sum().clamp_min(1.0)

    return loss


def localization_error_mm(pred_centers, gt_centers, mm_scale):
    difference = pred_centers - gt_centers
    dx = difference[..., 0] * mm_scale[:, None, 0]
    dy = difference[..., 1] * mm_scale[:, None, 1]
    return torch.sqrt(dx ** 2 + dy ** 2)


class LocalizationReport:
    def __init__(self, level_names=None):
        if level_names is None:
            level_names = RSNA_LEVEL_NAMES
        self.level_names = level_names
        self.errors = []
        self.valid = []

    def update(self, error_mm, valid):
        self.errors.append(error_mm.detach().cpu().numpy())
        self.valid.append(valid.detach().cpu().numpy())

    def compute(self):
        errors = np.concatenate(self.errors, axis=0)
        valid = np.concatenate(self.valid, axis=0).astype(bool)
        flat = errors[valid]

        report = {
            "n_levels": int(flat.size),
            "mean_mm": float(flat.mean()),
            "median_mm": float(np.median(flat)),
            "pct_within_5mm": float((flat <= 5).mean() * 100),
            "pct_within_10mm": float((flat <= 10).mean() * 100),
            "per_level": {},
        }

        for level_index in range(len(self.level_names)):
            name = self.level_names[level_index]
            mask = valid[:, level_index]
            if mask.sum() == 0:
                continue

            level_errors = errors[mask, level_index]
            report["per_level"][name] = {
                "n": int(mask.sum()),
                "mean_mm": float(level_errors.mean()),
                "median_mm": float(np.median(level_errors)),
                "pct_within_5mm": float((level_errors <= 5).mean() * 100),
                "pct_within_10mm": float((level_errors <= 10).mean() * 100),
            }

        return report
