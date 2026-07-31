"""Run the Spine-ViT ablation matrix on Modal — every (config, seed) in parallel.

Reuses train.py unchanged. Frozen configs run on L4; the fine-tuned-backbone config
runs on A100 (full DINOv2 backprop). All results are written to a persistent Volume so
`--skip_if_done` makes the whole sweep resumable, and you download them for evaluate.py.

Running everything (including anatomy+ordinal) here means all reported numbers come from
identical CUDA hardware — cleaner than mixing the local MPS runs with CUDA baselines.

-------------------------------------------------------------------------------
One-time setup (local):
    pip install modal
    modal setup                                            # authenticate
    modal volume create spine-vit-data
    modal volume put spine-vit-data data/rsna /rsna        # upload the 500-study set (~400MB)

Launch the sweep (18 runs in parallel):
    modal run modal_run.py

Pull results back and aggregate locally:
    modal volume get spine-vit-outputs / ./outputs_modal
    ./.venv/bin/python evaluate.py --experiments_dir outputs_modal --from_saved
-------------------------------------------------------------------------------
"""

import modal

app = modal.App("spine-vit-ablations")

TORCH_HOME = "/root/.torch"
SEEDS = [42, 43, 44]
EPOCHS = 40


def _cache_dinov2():
    """Bake the DINOv2 weights into the image so 18 containers don't each re-download."""
    import torch

    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "numpy", "pandas", "scikit-learn", "pyyaml",
        "pydicom", "pillow", "scipy", "matplotlib", "seaborn", "tqdm",
    )
    .env({"TORCH_HOME": TORCH_HOME})
    .run_function(_cache_dinov2)
    # add the repo code, but NOT the dataset or heavy local dirs.
    # NOTE: exclude the dataset DIRECTORY "data/rsna" exactly — do NOT use "data/rsna_*",
    # which also matches the source file data/rsna_dataset.py (that was the bug).
    .add_local_dir(
        ".", "/root/spine-vit",
        ignore=[
            "data/rsna", "data/rsna/**",   # dataset dir (arrives via the Volume); keeps rsna_dataset.py
            "outputs", "outputs/**", "outputs_archive", "outputs_archive/**",
            "outputs_modal", "outputs_modal/**",
            ".venv", ".venv/**", ".git", ".git/**", "**/__pycache__", "**/*.pyc", "*.zip",
        ],
    )
)

data_vol = modal.Volume.from_name("spine-vit-data", create_if_missing=True)
out_vol = modal.Volume.from_name("spine-vit-outputs", create_if_missing=True)


@app.function(image=image, gpu="L4", volumes={"/data": data_vol, "/outputs": out_vol}, timeout=2 * 60 * 60)
def train_one(extra_args):
    """Run one train.py invocation. GPU is L4 by default; the driver overrides to A100
    for the fine-tuned config via .with_options()."""
    import os, glob, subprocess

    os.chdir("/root/spine-vit")
    # auto-detect the data dir regardless of how the volume upload nested it
    hits = glob.glob("/data/**/train_label_coordinates.csv", recursive=True)
    data_dir = os.path.dirname(hits[0]) if hits else "/data/rsna"

    cmd = [
        "python", "train.py",
        "--data_dir", data_dir, "--dataset", "rsna",
        "--output_dir", "/outputs", "--device", "cuda",
        "--epochs", str(EPOCHS), "--skip_if_done",
    ] + extra_args
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    out_vol.commit()  # persist results to the volume


@app.function(image=image, gpu="L4", volumes={"/data": data_vol, "/outputs": out_vol}, timeout=3 * 60 * 60)
def train_detector_fn(epochs: int = 60):
    """Train the disc detector and export detected_centers.json to the outputs volume."""
    import os, glob, subprocess

    os.chdir("/root/spine-vit")
    hits = glob.glob("/data/**/train_label_coordinates.csv", recursive=True)
    data_dir = os.path.dirname(hits[0]) if hits else "/data/rsna"
    cmd = ["python", "train_detector.py", "--data_dir", data_dir, "--output_dir", "/outputs",
           "--device", "cuda", "--epochs", str(epochs), "--export"]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    out_vol.commit()   # persist checkpoint + detected_centers.json + localization_report.json


@app.function(image=image, gpu="L4", volumes={"/data": data_vol, "/outputs": out_vol}, timeout=2 * 60 * 60)
def train_fusion_one(extra_args):
    """One train_fusion.py invocation (two-view sagittal+axial grader). Requires the AXIAL
    DICOM series on the data volume (re-upload train_images with the axial subset first)."""
    import os, glob, subprocess

    os.chdir("/root/spine-vit")
    hits = glob.glob("/data/**/train_label_coordinates.csv", recursive=True)
    data_dir = os.path.dirname(hits[0]) if hits else "/data/rsna"
    cmd = ["python", "train_fusion.py", "--data_dir", data_dir, "--dataset", "rsna",
           "--output_dir", "/outputs", "--device", "cuda", "--epochs", str(EPOCHS), "--skip_if_done"] + extra_args
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    out_vol.commit()


@app.local_entrypoint()
def fusion():
    """v2 Task 2: two-view fusion ablation + axial box-size dose-response.

    Ablation (headline, axial_box=32 to pixel-match the sagittal headline):
        sag-only (control) | axial-only | fusion-A (concat) | fusion-B (attn)
    Box-size dose-response (robustness, NOT headline-selection): fusion-B at axial_box 16/24
    (box 32 already covered by the ablation). Each run reports canal metrics on the FULL
    test set AND the axial-available subset. 3 seeds each.
    """
    ablation = [
        ["--views", "sag"],                              # sagittal-only control (matched)
        ["--views", "axial"],                            # axial-only
        ["--views", "both", "--fusion", "concat"],       # fusion-A
        ["--views", "both", "--fusion", "attn"],         # fusion-B
    ]
    sweep = [
        ["--views", "both", "--fusion", "attn", "--axial_box_size", "16"],
        ["--views", "both", "--fusion", "attn", "--axial_box_size", "24"],
    ]
    handles = []
    for cfg in ablation + sweep:
        for s in SEEDS:
            handles.append(train_fusion_one.spawn(cfg + ["--seed", str(s)]))
    print(f"spawned {len(handles)} runs ({len(ablation)} ablation + {len(sweep)} sweep x {len(SEEDS)} seeds)...")
    failed = 0
    for h in handles:
        try:
            h.get()
        except Exception as e:
            failed += 1
            print("  fusion run FAILED:", e)
    print(f"done — {len(handles) - failed}/{len(handles)} succeeded.")
    print("download: modal volume get spine-vit-outputs / ./outputs_modal --force")
    print("aggregate: ./.venv/bin/python evaluate.py --experiments_dir outputs_modal --from_saved  # (full + axial-subset)")


@app.local_entrypoint()
def fusion_aug():
    """4-way ablation x 3 seeds, per-view augmented (sag hflip off, axial on). Tagged `_aug`."""
    ablation = [
        ["--views", "sag"],
        ["--views", "axial"],
        ["--views", "both", "--fusion", "concat"],
        ["--views", "both", "--fusion", "attn"],
    ]
    handles = []
    for cfg in ablation:
        for s in SEEDS:
            handles.append(train_fusion_one.spawn(cfg + ["--augment", "--seed", str(s)]))
    print(f"spawned {len(handles)} augmented runs ({len(ablation)} configs x {len(SEEDS)} seeds)...")
    failed = 0
    for h in handles:
        try:
            h.get()
        except Exception as e:
            failed += 1
            print("  aug fusion run FAILED:", e)
    print(f"done — {len(handles) - failed}/{len(handles)} succeeded.")
    print("download: modal volume get spine-vit-outputs / ./outputs_modal --force")
    print("aggregate: ./.venv/bin/python scripts/aggregate_fusion.py --experiments_dir outputs_modal --aug")


@app.local_entrypoint()
def fusion_budget():
    """One fan-out: seeds 45,46 for sag + axial (4 runs) and the 5-slice parasagittal budget
    control at seeds 42-44 (3 runs). Extends the ablation to 5 seeds and adds the budget control."""
    jobs = []
    for s in (45, 46):                                  # extend key pair to 5 seeds
        jobs.append((["--views", "sag"], s))
        jobs.append((["--views", "axial"], s))
    for s in SEEDS + [45, 46]:                          # 5-slice sag budget control at 5 seeds
        jobs.append((["--views", "sag", "--sag_slices", "5"], s))
    handles = [train_fusion_one.spawn(cfg + ["--augment", "--seed", str(s)]) for cfg, s in jobs]
    print(f"spawned {len(handles)} runs (4 seed-extension + 3 budget-control)...")
    failed = 0
    for h in handles:
        try:
            h.get()
        except Exception as e:
            failed += 1
            print("  budget/seed run FAILED:", e)
    print(f"done — {len(handles) - failed}/{len(handles)} succeeded.")
    print("aggregate: ./.venv/bin/python scripts/aggregate_fusion.py --experiments_dir outputs_modal --aug")


@app.local_entrypoint()
def fusion_fus_ext():
    """Extend the concat/attn fusion rows to 5 seeds (add 45,46; 4 runs)."""
    jobs = []
    for cfg in (["--views", "both", "--fusion", "concat"], ["--views", "both", "--fusion", "attn"]):
        for s in (45, 46):
            jobs.append(cfg + ["--augment", "--seed", str(s)])
    handles = [train_fusion_one.spawn(j) for j in jobs]
    print(f"spawned {len(handles)} fusion seed-extension runs (concat,attn x 45,46)...")
    failed = 0
    for h in handles:
        try:
            h.get()
        except Exception as e:
            failed += 1
            print("  fusion ext run FAILED:", e)
    print(f"done — {len(handles) - failed}/{len(handles)} succeeded.")
    print("aggregate: ./.venv/bin/python scripts/aggregate_fusion.py --experiments_dir outputs_modal --aug")


@app.local_entrypoint()
def detector_pipeline(epochs: int = 60):
    """Task 1 production: detector -> export -> 3-seed detected-box grading (MICCAI)."""
    print(f"[1/2] training detector ({epochs} epochs) + exporting centers ...")
    train_detector_fn.remote(epochs)   # blocking; commits detected_centers.json to the volume
    print("[2/2] detected-box grading, 3 seeds in parallel ...")
    extra = ["--tokenizer", "anatomy", "--pos_encoding", "ordinal",
             "--box_source", "detected",
             "--detected_centers_path", "/outputs/detector/detected_centers.json"]
    handles = [train_one.spawn(extra + ["--seed", str(s)]) for s in SEEDS]
    failed = 0
    for h in handles:
        try:
            h.get()
        except Exception as e:
            failed += 1
            print("  detected run FAILED:", e)
    print(f"done — detector + {len(handles) - failed}/{len(handles)} detected grading runs.")
    print("\nnext (local):")
    print("  modal volume get spine-vit-outputs / ./outputs_modal")
    print("  ./.venv/bin/python evaluate.py --experiments_dir outputs_modal --from_saved   # oracle vs detected table")
    print("  ./.venv/bin/python analyze_localization.py --experiments_dir outputs_modal    # grading-vs-detection figure")
    print("  cat outputs_modal/detector/localization_report.json                            # mm error, per level")


@app.local_entrypoint()
def coral():
    """Matched 3-seed, 40-epoch CORAL (anatomy+ordinal) vs the CE reference — to state an
    honest 'tried CORAL, no benefit' rather than concluding from one undertrained seed."""
    extra = ["--tokenizer", "anatomy", "--pos_encoding", "ordinal", "--head", "coral"]
    handles = [train_one.spawn(extra + ["--seed", str(s)]) for s in SEEDS]
    for h in handles:
        try:
            h.get()
        except Exception as e:
            print("  run FAILED:", e)
    print("done. evaluate.py --from_saved -> compare *_coral vs CE (kappa, kappa_linear, MAE, Moderate recall).")


@app.local_entrypoint()
def resolve_cast():
    """Add seeds 45,46 for anatomy+ordinal and cast_crop to resolve the borderline
    Δκ=0.052 (p=0.176 at 3 seeds) — the ROI-Align-vs-independent-crop architectural call."""
    configs = [
        ["--tokenizer", "anatomy",   "--pos_encoding", "ordinal"],
        ["--tokenizer", "cast_crop", "--pos_encoding", "ordinal"],
    ]
    handles = [train_one.spawn(c + ["--seed", str(s)]) for c in configs for s in (45, 46)]
    print(f"spawned {len(handles)} runs (2 configs x seeds 45,46)...")
    for h in handles:
        try:
            h.get()
        except Exception as e:
            print("  run FAILED:", e)
    print("done. re-run evaluate.py --from_saved -> anatomy & cast_crop now aggregate over 5 seeds.")


@app.local_entrypoint()
def box_size_sweep():
    """Dose-response: oracle & detected grading at shrinking box sizes toward the ~6.8mm
    detector error. As the box narrows to the error magnitude, the oracle-vs-detected gap
    should widen — locating where the robustness boundary is. Reuses the detector centers
    already in the volume (no detector retrain). box=32 already exists from the main sweep."""
    sizes = [16, 24, 48]
    handles = []
    for bs in sizes:
        for source in ("oracle", "detected"):
            extra = ["--tokenizer", "anatomy", "--pos_encoding", "ordinal", "--box_size", str(bs)]
            if source == "detected":
                extra += ["--box_source", "detected",
                          "--detected_centers_path", "/outputs/detector/detected_centers.json"]
            for s in SEEDS:
                handles.append(train_one.spawn(extra + ["--seed", str(s)]))
    print(f"spawned {len(handles)} runs ({len(sizes)} sizes x oracle/detected x {len(SEEDS)} seeds)...")
    for h in handles:
        try:
            h.get()
        except Exception as e:
            print("  run FAILED:", e)
    print("done. download --force, then: ./.venv/bin/python analyze_boxsize.py --experiments_dir outputs_modal")


@app.local_entrypoint()
def main():
    frozen = [
        ["--tokenizer", "strips",    "--pos_encoding", "ordinal"],  # baseline
        ["--tokenizer", "patches",   "--pos_encoding", "ordinal"],  # baseline
        ["--tokenizer", "cast_crop", "--pos_encoding", "ordinal"],  # CAST baseline (Task 2)
        ["--tokenizer", "anatomy",   "--pos_encoding", "learned"],  # pos-enc ablation
        ["--tokenizer", "anatomy",   "--pos_encoding", "none"],     # pos-enc ablation
        ["--tokenizer", "anatomy",   "--pos_encoding", "ordinal"],  # OURS
    ]
    ft = ["--tokenizer", "anatomy", "--pos_encoding", "ordinal", "--no-freeze_backbone", "--lr", "3e-5"]

    handles = []
    for cfg in frozen:
        for s in SEEDS:
            handles.append(train_one.spawn(cfg + ["--seed", str(s)]))
    for s in SEEDS:  # fine-tuned on A100
        handles.append(train_one.with_options(gpu="A100").spawn(ft + ["--seed", str(s)]))

    print(f"spawned {len(handles)} runs in parallel; waiting for all to finish...")
    failed = 0
    for h in handles:
        try:
            h.get()
        except Exception as e:  # one run failing shouldn't lose the rest
            failed += 1
            print("  run FAILED:", e)
    print(f"done — {len(handles) - failed}/{len(handles)} succeeded.")
    print("download: modal volume get spine-vit-outputs / ./outputs_modal")
    print("aggregate: ./.venv/bin/python evaluate.py --experiments_dir outputs_modal --from_saved")
