import numpy as np
import torch

RSNA_LEVEL_NAMES = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]


def make_heatmaps(centers, valid, size, sigma):
    batch_size = centers.shape[0]
    num_levels = centers.shape[1]
    device = centers.device

    # A grid of x and y positions we can measure distance from
    xs = torch.arange(size, device=device).view(1, 1, 1, size)
    ys = torch.arange(size, device=device).view(1, 1, size, 1)

    center_x = centers[..., 0].view(batch_size, num_levels, 1, 1)
    center_y = centers[..., 1].view(batch_size, num_levels, 1, 1)

    dx = xs - center_x
    dy = ys - center_y
    squared = dx ** 2 + dy ** 2

    # A blob centred on each disc, zeroed out for levels we have no label for
    gaussian = torch.exp(-squared / (2 * sigma ** 2))
    mask = valid.view(batch_size, num_levels, 1, 1)
    return gaussian * mask


def spatial_softmax(logits, temp=1.0):
    batch_size, num_levels, height, width = logits.shape

    # Flatten the map so softmax runs over the whole image, not per row
    flat = logits.view(batch_size, num_levels, -1) / temp
    probs = torch.softmax(flat, dim=-1)
    return probs.view(batch_size, num_levels, height, width)


def expected_coords(probs, height, width, device):
    xs = torch.arange(width, device=device).float()
    ys = torch.arange(height, device=device).float()

    # Sum away one axis first, then take the weighted average along the other
    col_probs = probs.sum(dim=2)
    row_probs = probs.sum(dim=3)
    expected_x = (col_probs * xs).sum(dim=-1)
    expected_y = (row_probs * ys).sum(dim=-1)
    return expected_x, expected_y


def soft_argmax(logits, temp=1.0):
    # Centre of mass of the heatmap, so it can land between pixels and stay differentiable
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

    # Squared distance from the predicted point to the labelled one
    dx = expected_x - gt_coords[..., 0]
    dy = expected_y - gt_coords[..., 1]
    squared_error = dx ** 2 + dy ** 2

    denominator = valid.sum().clamp_min(1.0)
    loss = (squared_error * valid).sum() / denominator

    if reg > 0:
        # Optional extra term that punishes a spread out heatmap, keeps the peak tight
        xs = torch.arange(width, device=logits.device).float()
        ys = torch.arange(height, device=logits.device).float()
        col_probs = probs.sum(dim=2)
        row_probs = probs.sum(dim=3)
        spread_x = (xs[None, None] - expected_x[..., None]) ** 2
        spread_y = (ys[None, None] - expected_y[..., None]) ** 2
        var_x = (col_probs * spread_x).sum(dim=-1)
        var_y = (row_probs * spread_y).sum(dim=-1)
        variance = var_x + var_y
        loss = loss + reg * (variance * valid).sum() / denominator

    return loss


def localization_error_mm(prediction_centers, gt_centers, mm_scale):
    # Pixels mean different distances in different studies, so convert before measuring
    difference = prediction_centers - gt_centers
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
        # Collect batches now, work out the numbers at the end
        self.errors.append(error_mm.detach().cpu().numpy())
        self.valid.append(valid.detach().cpu().numpy())

    def compute(self):
        errors = np.concatenate(self.errors, axis=0)
        valid = np.concatenate(self.valid, axis=0).astype(bool)
        flat = errors[valid]

        within_5 = flat <= 5
        within_10 = flat <= 10

        report = {
            "n_levels": int(flat.size),
            "mean_mm": float(flat.mean()),
            "median_mm": float(np.median(flat)),
            "pct_within_5mm": float(within_5.mean() * 100),
            "pct_within_10mm": float(within_10.mean() * 100),
            "per_level": {},
        }

        # Same numbers again but split by level, L1/L2 is usually the worst
        for i in range(len(self.level_names)):
            name = self.level_names[i]
            mask = valid[:, i]
            if mask.sum() == 0:
                continue

            level_errors = errors[mask, i]
            level_within_5 = level_errors <= 5
            level_within_10 = level_errors <= 10

            report["per_level"][name] = {
                "n": int(mask.sum()),
                "mean_mm": float(level_errors.mean()),
                "median_mm": float(np.median(level_errors)),
                "pct_within_5mm": float(level_within_5.mean() * 100),
                "pct_within_10mm": float(level_within_10.mean() * 100),
            }

        return report
