"""Integration test for the training loop on synthetic data (instructions.md Phase 3).

Drives the real train.run_epoch / evaluate helpers and utils.metrics over a tiny fixed
synthetic dataset (mock backbone, no DICOM/NIfTI needed). Verifies that:
  * the loss decreases as the model overfits the tiny set,
  * metrics and the LevelAttributionAnalyzer produce non-degenerate values,
  * class-weight computation and checkpoint save/load work.

Run:  python tests/test_train_loop.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.rsna_dataset import rsna_collate_fn
from models import build_model
from utils.metrics import compute_class_weights, compute_metrics, LevelAttributionAnalyzer
from train import run_epoch, move_batch

NUM_CLASSES = 3
IMG = 224


class SyntheticSpineDataset(Dataset):
    """Fixed synthetic samples in the exact per-sample format the RSNA dataset emits."""

    def __init__(self, n=8, k=5, seed=0):
        rng = np.random.RandomState(seed)
        self.items = []
        for i in range(n):
            img = torch.from_numpy(rng.randn(3, IMG, IMG).astype(np.float32))
            centers = rng.uniform(40, IMG - 40, size=(k, 2))
            boxes = np.stack(
                [centers[:, 0] - 20, centers[:, 1] - 20, centers[:, 0] + 20, centers[:, 1] + 20], axis=1
            ).astype(np.float32)
            targets = rng.randint(0, NUM_CLASSES, size=k).astype(np.int64)
            self.items.append(
                {
                    "image": img,
                    "boxes": torch.from_numpy(boxes),
                    "level_indices": torch.arange(k, dtype=torch.long),
                    "level_types": torch.ones(k, dtype=torch.long),
                    "targets": torch.from_numpy(targets),
                    "num_levels": k,
                    "study_id": i,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def get_all_targets(self):
        return np.concatenate([it["targets"].numpy() for it in self.items])


def config():
    return {
        "backbone": "mock", "backbone_dim": 384, "patch_size": 14, "freeze_backbone": True,
        "embed_dim": 256, "encoder_layers": 2, "encoder_heads": 4, "dropout": 0.0,
        "max_levels": 24, "image_size": IMG, "roi_output_size": 7,
        "num_stenosis_classes": NUM_CLASSES, "num_pfirrmann_classes": 5,
        "task": "stenosis", "grad_clip": 1.0, "num_classes": NUM_CLASSES,
    }


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = config()

    ds = SyntheticSpineDataset(n=8, k=5)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=rsna_collate_fn)

    model = build_model(cfg).to(device)
    print(f"trainable params: {model.count_trainable_params()/1e6:.3f}M")

    weights = compute_class_weights(ds, NUM_CLASSES).to(device)
    print(f"class weights: {weights.tolist()}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    losses = []
    for epoch in range(30):
        res = run_epoch(model, loader, criterion, device, cfg, optimizer, desc="")
        losses.append(res["loss"])

    first, last = np.mean(losses[:3]), np.mean(losses[-3:])
    print(f"loss: first3={first:.4f} last3={last:.4f}")
    assert last < first, f"loss did not decrease ({first:.4f} -> {last:.4f})"

    # Metrics + attribution on the (overfit) training set.
    eval_res = run_epoch(model, loader, criterion, device, cfg, optimizer=None, desc="")
    metrics = compute_metrics(eval_res["preds"], eval_res["targets"], NUM_CLASSES)
    analyzer = LevelAttributionAnalyzer(num_classes=NUM_CLASSES)
    analyzer.update(eval_res["preds"], eval_res["targets"], eval_res["levels"], patient_id=eval_res["studyids"])
    attr = analyzer.compute()
    print(f"metrics: macro_f1={metrics['macro_f1']:.3f} kappa={metrics['kappa']:.3f} acc={metrics['accuracy']:.3f}")
    print(f"worst_level_acc={attr['worst_level_accuracy']} pathology P/R="
          f"{attr['pathology_precision']:.3f}/{attr['pathology_recall']:.3f} "
          f"per_level_keys={list(attr['per_level'].keys())}")
    assert metrics["macro_f1"] > 0.0, "macro_f1 is zero"
    assert attr["n"] > 0, "analyzer recorded no findings"
    assert "worst_level_accuracy" in attr and "pathology_precision" in attr, "new metrics missing"

    # Checkpoint round-trip.
    ckpt_path = os.path.join(os.path.dirname(__file__), "_tmp_ckpt.pt")
    torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
    model2 = build_model(cfg)
    model2.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"])
    os.remove(ckpt_path)
    print("checkpoint save/load OK")

    print("\nTraining-loop integration test PASSED.")


if __name__ == "__main__":
    main()
