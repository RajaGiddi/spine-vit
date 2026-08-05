import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models import build_model

IMAGE_SIZE = 224
NUM_CLASSES = 3


def make_synthetic_batch(batch_size=3, image_size=IMAGE_SIZE, seed=0):
    random_state = np.random.RandomState(seed)
    images = torch.randn(batch_size, 3, image_size, image_size)

    boxes, level_indices, level_types, targets, num_levels = [], [], [], [], []
    for bi in range(batch_size):
        k_disc = random_state.randint(3, 6)
        types = [1] * k_disc + [0]
        k = len(types)
        for j in range(k):
            center_x, center_y = random_state.uniform(40, image_size - 40, size=2)
            half = 20
            boxes.append([float(bi), center_x - half, center_y - half, center_x + half, center_y + half])
        level_indices.extend(range(k))
        level_types.extend(types)
        targets.extend(list(random_state.randint(0, NUM_CLASSES, size=k_disc)) + [-1])
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
        "backbone": "mock",
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

            passed = logits.shape == (n_disc, NUM_CLASSES)
            status = "OK " if passed else "FAIL"
            print(
                f"[{status}] tokenizer={tok:8s} pos={pe:8s} | "
                f"logits {tuple(logits.shape)} encoded {tuple(out['encoded_tokens'].shape)} "
                f"| trainable {n_params/1e6:.3f}M"
            )
            if not passed:
                failures += 1

            with torch.no_grad():
                _, attn = model.forward_with_attention(batch)
            assert len(attn) == config["encoder_layers"], "attention layer count mismatch"

    if failures:
        print(f"\n{failures} combination(s) FAILED")
        sys.exit(1)
    print("\nAll 9 tokenizer x pos-encoding combinations passed. Trainable params in 1-3M range expected.")


if __name__ == "__main__":
    main()
