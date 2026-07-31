"""Offline forward-pass verification (instructions.md Phase 2, Step 11).

Builds a synthetic batch (no dataset needed) and runs a full forward pass through all
3 x 3 = 9 tokenizer x positional-encoding combinations using the offline MockBackbone.
Prints shapes at every stage and asserts output shapes are correct.

Run:  python -m tests.test_forward     (or)     python tests/test_forward.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models import build_model

IMAGE_SIZE = 224
NUM_CLASSES = 3


def make_synthetic_batch(batch_size=3, image_size=IMAGE_SIZE, seed=0):
    """Synthetic batch mimicking the collate output.

    Mixes vertebra (type 0) and disc (type 1) tokens so head-filtering is exercised.
    Per sample: a few disc levels plus one vertebra token.
    """
    rng = np.random.RandomState(seed)
    images = torch.randn(batch_size, 3, image_size, image_size)

    boxes, level_indices, level_types, targets, num_levels = [], [], [], [], []
    for bi in range(batch_size):
        k_disc = rng.randint(3, 6)  # 3-5 disc levels
        types = [1] * k_disc + [0]  # add one vertebra token
        k = len(types)
        for j in range(k):
            cx, cy = rng.uniform(40, image_size - 40, size=2)
            half = 20
            boxes.append([float(bi), cx - half, cy - half, cx + half, cy + half])
        level_indices.extend(range(k))
        level_types.extend(types)
        # discs get a class label; vertebra gets ignore (-1)
        targets.extend(list(rng.randint(0, NUM_CLASSES, size=k_disc)) + [-1])
        num_levels.append(k)

    return {
        "images": images,
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "level_indices": torch.tensor(level_indices, dtype=torch.long),
        "level_types": torch.tensor(level_types, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "num_levels": num_levels,
        "study_ids": list(range(batch_size)),
    }


def base_config():
    return {
        "backbone": "mock",  # offline; no DINOv2 download
        "backbone_dim": 384,
        "patch_size": 14,
        "freeze_backbone": True,
        "embed_dim": 256,
        "encoder_layers": 2,
        "encoder_heads": 4,
        "dropout": 0.1,
        "max_levels": 24,
        "image_size": IMAGE_SIZE,
        "roi_output_size": 7,
        "num_stenosis_classes": NUM_CLASSES,
        "num_pfirrmann_classes": 5,
        "task": "stenosis",
    }


def main():
    batch = make_synthetic_batch()
    n_total = int(sum(batch["num_levels"]))
    n_disc = int((batch["level_types"] == 1).sum())
    print(f"Synthetic batch: B={len(batch['num_levels'])} N_total={n_total} N_disc={n_disc}")
    print(f"  images {tuple(batch['images'].shape)}  boxes {tuple(batch['boxes'].shape)}")

    failures = 0
    for tok in ("anatomy", "strips", "patches"):
        for pe in ("ordinal", "learned", "none"):
            config = base_config()
            config["tokenizer"] = tok
            config["pos_encoding"] = pe
            model = build_model(config)
            model.eval()

            with torch.no_grad():
                out = model(batch)
            logits = out["logits"]
            n_params = model.count_trainable_params()

            ok = logits.shape == (n_disc, NUM_CLASSES)
            status = "OK " if ok else "FAIL"
            print(
                f"[{status}] tokenizer={tok:8s} pos={pe:8s} | "
                f"logits {tuple(logits.shape)} encoded {tuple(out['encoded_tokens'].shape)} "
                f"| trainable {n_params/1e6:.3f}M"
            )
            if not ok:
                failures += 1

            # attention path (used by evaluate.py)
            with torch.no_grad():
                _, attn = model.forward_with_attention(batch)
            assert len(attn) == config["encoder_layers"], "attention layer count mismatch"

    if failures:
        print(f"\n{failures} combination(s) FAILED")
        sys.exit(1)
    print("\nAll 9 tokenizer x pos-encoding combinations passed. Trainable params in 1-3M range expected.")


if __name__ == "__main__":
    main()
