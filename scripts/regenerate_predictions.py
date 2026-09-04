"""Regenerate test_predictions.json for runs that only saved scalar metrics.

The anatomy, patches and strips runs predate train.py writing per-study
predictions, so only `worst_level_accuracy` as a single number survives. That is
enough for argmax scoring but not for re-scoring with fractional tie credit,
which needs the per-study prediction vectors.

Their checkpoints exist, so this is inference, not retraining: rebuild the split
from the saved config (the seed makes it deterministic), load best_model.pt, and
run the same evaluate_split path train.py used.

Nothing is overwritten. test_predictions.json is written only where it is
missing; test_results.json, history.json and best_model.pt are left alone. Every
regenerated run is checked against its stored metrics, and a run whose numbers
do not reproduce is reported rather than written.

    python scripts/regenerate_predictions.py --runs rsna_anatomy_ordinal_256_2_s42 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import build_model  # noqa: E402
from train import (  # noqa: E402
    build_criterion,
    compute_class_weights,
    evaluate_split,
    get_dataloaders,
    resolve_device,
    set_seed,
)

TOL = 1e-4


def regenerate(run_dir: Path, data_dir: str, device_arg: str, dry_run: bool = False) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    config["data_dir"] = data_dir
    stored = json.loads((run_dir / "test_results.json").read_text())

    # The split is a pure function of the seed, so this reproduces the exact
    # test studies the run was scored on.
    set_seed(config.get("seed", 42))
    device = resolve_device(device_arg)

    train_ds, _, _, test_loader = get_dataloaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    weights = compute_class_weights(train_ds, config["num_classes"],
                                    scheme=config.get("class_weight", "inverse")).to(device)
    criterion, predict_fn = build_criterion(config, train_ds, weights, device)

    res = evaluate_split(model, test_loader, criterion, device, config,
                         desc="test", predict_fn=predict_fn)

    got_k = res["metrics"].get("kappa")
    got_w = res["attribution"].get("worst_level_accuracy")
    want_k = (stored.get("metrics") or {}).get("kappa")
    want_w = (stored.get("attribution") or {}).get("worst_level_accuracy")

    ok = (want_k is None or abs(got_k - want_k) < TOL) and \
         (want_w is None or abs(got_w - want_w) < TOL)

    out = {"run": run_dir.name, "kappa": got_k, "stored_kappa": want_k,
           "worst_lvl": got_w, "stored_worst_lvl": want_w, "reproduced": ok}

    if ok and not dry_run:
        predictions = {k: [int(v) for v in res[k]]
                       for k in ("studyids", "levels", "predictions", "targets")}
        (run_dir / "test_predictions.json").write_text(json.dumps(predictions))
        out["written"] = True
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments_dir", default="outputs_modal")
    parser.add_argument("--data_dir", default="data/rsna")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="regenerate even if test_predictions.json exists")
    args = parser.parse_args()

    results = []
    for name in args.runs:
        run_dir = Path(args.experiments_dir) / name
        if not (run_dir / "best_model.pt").exists():
            print(f"  {name}: no best_model.pt, skipping")
            continue
        if (run_dir / "test_predictions.json").exists() and not args.force:
            print(f"  {name}: already has predictions, skipping")
            continue
        r = regenerate(run_dir, args.data_dir, args.device, args.dry_run)
        results.append(r)
        flag = "OK " if r["reproduced"] else "MISMATCH"
        print(f"  {flag} {name}: kappa {r['kappa']:.4f} (stored {r['stored_kappa']:.4f})  "
              f"worst_lvl {r['worst_lvl']:.4f} (stored {r['stored_worst_lvl']:.4f})", flush=True)

    bad = [r for r in results if not r["reproduced"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} reproduced their stored metrics")
    if bad:
        print("NOT written (metrics did not reproduce):", [r["run"] for r in bad])


if __name__ == "__main__":
    main()
