import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from utils.metrics import compute_metrics, compute_class_weights, LevelAttributionAnalyzer
from utils.metrics import RSNA_LEVEL_NAMES

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")


def load_config(overrides):
    with open(CONFIG_PATH) as config_file:
        config = yaml.safe_load(config_file)

    for key in overrides:
        if overrides[key] is not None:
            config[key] = overrides[key]

    if config["task"] == "pfirrmann":
        config["num_classes"] = config["num_pfirrmann_classes"]
    else:
        config["num_classes"] = config["num_stenosis_classes"]

    if config.get("box_source") == "detected":
        path = config.get("detected_centers_path")
        if not path:
            path = os.path.join(config["output_dir"], "detector", "detected_centers.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"--box_source detected needs {path}. Run train_detector.py --export first."
            )
        with open(path) as centers_file:
            config["detected_centers"] = json.load(centers_file)
        print("loaded detected centers for", len(config["detected_centers"]), "studies")

    return config


def resolve_device(name):
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "mps" and has_mps:
        return torch.device("mps")

    if name == "cuda" or name == "mps":
        print("device", name, "is not available, using cpu instead")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experiment_name(config):
    name = f"{config['dataset']}_{config['tokenizer']}_{config['pos_encoding']}"
    name = f"{name}_{config['embed_dim']}_{config['encoder_layers']}"

    if config.get("box_source", "oracle") == "detected":
        name = f"{name}_det"

    if not config.get("freeze_backbone", True):
        name = f"{name}_ft"

    box_size = config.get("box_size", 32)
    if box_size != 32:
        name = f"{name}_b{box_size}"

    if config.get("head", "ce") == "coral":
        name = f"{name}_coral"

    return f"{name}_s{config['seed']}"


def format_metric(value):
    if isinstance(value, int) or isinstance(value, float):
        return format(value, ".3f")
    return "n/a"


def get_dataloaders(config):
    if config["dataset"] == "rsna":
        from data.rsna_dataset import make_rsna_splits, rsna_collate_fn

        train_ds, val_ds, test_ds = make_rsna_splits(config["data_dir"], config)
        collate = rsna_collate_fn
    elif config["dataset"] == "spider":
        from data.spider_dataset import make_spider_splits, spider_collate_fn

        train_ds, val_ds, test_ds = make_spider_splits(config["data_dir"], config)
        collate = spider_collate_fn
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    limit = config.get("limit_samples")
    if limit:
        small = max(2, limit // 2)
        train_ds.samples = train_ds.samples[:limit]
        val_ds.samples = val_ds.samples[:small]
        test_ds.samples = test_ds.samples[:small]

    batch_size = config["batch_size"]
    num_workers = config.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate, num_workers=num_workers)
    return train_ds, train_loader, val_loader, test_loader


def move_batch(batch, device):
    tensor_keys = [
        "images", "boxes", "level_indices", "level_types", "targets",
        "axial_images", "axial_boxes", "axial_level_indices", "axial_slot",
        "sag_multi_images",
    ]
    moved = dict(batch)
    for key in tensor_keys:
        if key in batch and torch.is_tensor(batch[key]):
            moved[key] = batch[key].to(device)
    return moved


def argmax_predict(logits):
    return logits.argmax(-1)


def join_arrays(arrays):
    if len(arrays) == 0:
        return np.array([])
    return np.concatenate(arrays)


def run_epoch(model, loader, criterion, device, config, optimizer=None, desc="", predict_fn=None):
    is_train = optimizer is not None
    if predict_fn is None:
        predict_fn = argmax_predict
    model.train(is_train)

    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    all_levels = []
    all_study_ids = []

    for batch in tqdm(loader, desc=desc, leave=False):
        study_ids = np.repeat(np.asarray(batch["study_ids"]), batch["num_levels"])
        batch = move_batch(batch, device)

        with torch.set_grad_enabled(is_train):
            out = model(batch)
            logits = out["logits"]
            disc_mask = out["disc_mask"]
            targets = batch["targets"][disc_mask]

            if (targets != -1).sum() == 0:
                continue

            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                trainable = []
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        trainable.append(parameter)
                torch.nn.utils.clip_grad_norm_(trainable, config.get("grad_clip", 1.0))
                optimizer.step()

        total_loss = total_loss + float(loss.item())
        num_batches = num_batches + 1

        keep = disc_mask.detach().cpu().numpy()
        all_preds.append(predict_fn(logits).detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_levels.append(out["disc_level_indices"].detach().cpu().numpy())
        all_study_ids.append(study_ids[keep])

    return {
        "loss": total_loss / max(1, num_batches),
        "preds": join_arrays(all_preds),
        "targets": join_arrays(all_targets),
        "levels": join_arrays(all_levels),
        "studyids": join_arrays(all_study_ids),
    }


def evaluate_split(model, loader, criterion, device, config, desc="val", predict_fn=None):
    result = run_epoch(model, loader, criterion, device, config, None, desc, predict_fn)

    if config["dataset"] == "rsna":
        level_names = RSNA_LEVEL_NAMES
    else:
        level_names = None

    analyzer = LevelAttributionAnalyzer(level_names=level_names, num_classes=config["num_classes"])
    analyzer.update(result["preds"], result["targets"], result["levels"],
                    patient_id=result["studyids"])

    result["metrics"] = compute_metrics(result["preds"], result["targets"], config["num_classes"])
    result["attribution"] = analyzer.compute()
    return result


def build_criterion(config, train_dataset, class_weights, device):
    if config.get("head", "ce") == "coral":
        from utils.metrics import coral_loss, coral_predict, coral_pos_weights

        pos_weight = coral_pos_weights(train_dataset, config["num_classes"]).to(device)

        def criterion(logits, targets):
            return coral_loss(logits, targets, pos_weight)

        print("head: CORAL ordinal, pos_weight", [round(weight, 2) for weight in pos_weight.tolist()])
        return criterion, coral_predict

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    return criterion, None


def main():
    args = parse_args()
    config = load_config(vars(args))
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    name = config.get("experiment_name")
    if not name:
        name = experiment_name(config)
    out_dir = os.path.join(config["output_dir"], name)
    os.makedirs(out_dir, exist_ok=True)

    if config.get("skip_if_done") and os.path.exists(os.path.join(out_dir, "test_results.json")):
        print(out_dir, "is already complete")
        return

    with open(os.path.join(out_dir, "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=2)
    print("experiment ->", out_dir)
    print("device", device, "dataset", config["dataset"], "task", config["task"])
    print("tokenizer", config["tokenizer"], "pos_encoding", config["pos_encoding"])

    train_ds, train_loader, val_loader, test_loader = get_dataloaders(config)

    model = build_model(config).to(device)
    print("trainable params: %.3fM" % (model.count_trainable_params() / 1e6))

    weight_scheme = config.get("class_weight", "inverse")
    class_weights = compute_class_weights(train_ds, config["num_classes"], scheme=weight_scheme)
    class_weights = class_weights.to(device)
    print("class weights (%s):" % weight_scheme, [round(weight, 3) for weight in class_weights.tolist()])

    criterion, predict_fn = build_criterion(config, train_ds, class_weights, device)

    trainable = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable.append(parameter)
    optimizer = torch.optim.AdamW(trainable, lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"],
                                                           eta_min=1e-6)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_kappa": [],
        "val_balanced_acc": [],
        "val_worst_lvl": [],
        "val_path_prec": [],
        "val_path_rec": [],
    }

    select_metric = config.get("select_metric", "kappa")
    select_window = max(1, config.get("select_window", 3))
    val_scores = []
    best_score = -1e9
    best_epoch = -1
    epochs_since_best = 0

    for epoch in range(1, config["epochs"] + 1):
        train_res = run_epoch(model, train_loader, criterion, device, config, optimizer,
                              desc=f"train {epoch}", predict_fn=predict_fn)
        train_metrics = compute_metrics(train_res["preds"], train_res["targets"],
                                        config["num_classes"])
        val_res = evaluate_split(model, val_loader, criterion, device, config,
                                 desc=f"val {epoch}", predict_fn=predict_fn)
        scheduler.step()

        val_metrics = val_res["metrics"]
        attribution = val_res["attribution"]
        history["train_loss"].append(train_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_macro_f1"].append(val_metrics.get("macro_f1", 0.0))
        history["val_kappa"].append(val_metrics.get("kappa", 0.0))
        history["val_balanced_acc"].append(val_metrics.get("balanced_acc", 0.0))
        history["val_worst_lvl"].append(attribution.get("worst_level_accuracy"))
        history["val_path_prec"].append(attribution.get("pathology_precision"))
        history["val_path_rec"].append(attribution.get("pathology_recall"))

        print("epoch %3d | train_loss %.4f (f1 %.3f) | val_loss %.4f f1 %.3f kappa %.3f" % (
            epoch, train_res["loss"], train_metrics.get("macro_f1", 0),
            val_res["loss"], val_metrics.get("macro_f1", 0), val_metrics.get("kappa", 0)))
        print("          worst_lvl", format_metric(attribution.get("worst_level_accuracy")),
              "path P/R", format_metric(attribution.get("pathology_precision")),
              "/", format_metric(attribution.get("pathology_recall")))

        with open(os.path.join(out_dir, "history.json"), "w") as history_file:
            json.dump(history, history_file, indent=2)

        val_scores.append(val_metrics.get(select_metric, 0.0))
        recent = val_scores[-select_window:]
        smoothed = float(np.mean(recent))

        if smoothed > best_score:
            best_score = smoothed
            best_epoch = epoch
            epochs_since_best = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "smoothed_score": smoothed,
            }
            torch.save(checkpoint, os.path.join(out_dir, "best_model.pt"))
        else:
            epochs_since_best = epochs_since_best + 1
            if epochs_since_best >= config["patience"]:
                print("early stopping at epoch", epoch,
                      "- best smoothed val", select_metric, "%.3f" % best_score,
                      "at epoch", best_epoch)
                break

    checkpoint = torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_res = evaluate_split(model, test_loader, criterion, device, config, desc="test",
                              predict_fn=predict_fn)

    test_out = {
        "metrics": test_res["metrics"],
        "attribution": test_res["attribution"],
        "best_epoch": best_epoch,
    }
    with open(os.path.join(out_dir, "test_results.json"), "w") as results_file:
        json.dump(test_out, results_file, indent=2)

    predictions = {}
    for key in ["studyids", "levels", "preds", "targets"]:
        values = []
        for value in test_res[key]:
            values.append(int(value))
        predictions[key] = values
    with open(os.path.join(out_dir, "test_predictions.json"), "w") as predictions_file:
        json.dump(predictions, predictions_file)

    test_metrics = test_res["metrics"]
    test_attribution = test_res["attribution"]
    print("TEST macro_f1 %.3f kappa %.3f" % (test_metrics.get("macro_f1", 0),
                                                    test_metrics.get("kappa", 0)))
    print("worst_lvl_acc", format_metric(test_attribution.get("worst_level_accuracy")),
          "on", test_attribution.get("n_studies_with_pathology", 0), "studies")
    print("pathology P/R", format_metric(test_attribution.get("pathology_precision")),
          "/", format_metric(test_attribution.get("pathology_recall")),
          "fp_rate", format_metric(test_attribution.get("pathology_fp_rate")))
    print("results saved to", out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Spine-ViT")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=["rsna", "spider"])
    parser.add_argument("--tokenizer", type=str, default=None,
                        choices=["anatomy", "strips", "patches", "cast_crop"])
    parser.add_argument("--pos_encoding", type=str, default=None,
                        choices=["ordinal", "learned", "none"])
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--encoder_layers", type=int, default=None)
    parser.add_argument("--encoder_heads", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--class_weight", type=str, default=None,
                        choices=["inverse", "sqrt_inverse", "none"])
    parser.add_argument("--select_metric", type=str, default=None,
                        choices=["kappa", "macro_f1", "balanced_acc"])
    parser.add_argument("--select_window", type=int, default=None,
                        help="number of epochs in the validation moving average")
    parser.add_argument("--skip_if_done", action="store_true", default=None,
                        help="skip the run if test_results.json already exists")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--task", type=str, default=None, choices=["stenosis", "pfirrmann"])
    parser.add_argument("--box_size", type=int, default=None,
                        help="ROI size in resized pixels (default 32)")
    parser.add_argument("--head", type=str, default=None, choices=["ce", "coral"])
    parser.add_argument("--box_source", type=str, default=None, choices=["oracle", "detected"])
    parser.add_argument("--detected_centers_path", type=str, default=None)
    parser.add_argument("--use_oracle", dest="use_oracle_regions", action="store_true",
                        default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--limit_samples", type=int, default=None,
                        help="train on a small subset for a quick check")
    parser.add_argument("--experiment_name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
