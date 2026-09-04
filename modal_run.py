import modal

app = modal.App("spine-vit-ablations")

TORCH_HOME = "/root/.torch"
SEEDS = [42, 43, 44]
EPOCHS = 40


def cache_dinov2():
    import torch

    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "numpy", "pandas", "scikit-learn", "pyyaml",
        "pydicom", "pillow", "scipy", "matplotlib", "seaborn", "tqdm",
        "SimpleITK",
    )
    .env({"TORCH_HOME": TORCH_HOME})
    .run_function(cache_dinov2)
    .add_local_dir(
        ".", "/root/spine-vit",
        ignore=[
            "data/rsna", "data/rsna/**",
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
    import os, glob, subprocess

    os.chdir("/root/spine-vit")
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
    out_vol.commit()


@app.function(image=image, gpu="L4", volumes={"/data": data_vol, "/outputs": out_vol}, timeout=3 * 60 * 60)
def train_detector_fn(epochs = 60):
    import os, glob, subprocess

    os.chdir("/root/spine-vit")
    hits = glob.glob("/data/**/train_label_coordinates.csv", recursive=True)
    data_dir = os.path.dirname(hits[0]) if hits else "/data/rsna"
    cmd = ["python", "train_detector.py", "--data_dir", data_dir, "--output_dir", "/outputs",
           "--device", "cuda", "--epochs", str(epochs), "--export"]
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    out_vol.commit()


@app.function(image=image, gpu="L4", volumes={"/data": data_vol, "/outputs": out_vol}, timeout=2 * 60 * 60)
def train_fusion_one(extra_args):
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
    ablation = [
        ["--views", "sag"],
        ["--views", "axial"],
        ["--views", "both", "--fusion", "concat"],
        ["--views", "both", "--fusion", "attn"],
    ]
    sweep = [
        ["--views", "both", "--fusion", "attn", "--axial_box_size", "16"],
        ["--views", "both", "--fusion", "attn", "--axial_box_size", "24"],
    ]
    handles = []
    for cfg in ablation + sweep:
        for seed in SEEDS:
            handles.append(train_fusion_one.spawn(cfg + ["--seed", str(seed)]))
    print(f"spawned {len(handles)} runs ({len(ablation)} ablation + {len(sweep)} sweep x {len(SEEDS)} seeds)...")
    failed = 0
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            failed += 1
            print("  fusion run FAILED:", e)
    print(f"done - {len(handles) - failed}/{len(handles)} succeeded.")
    print("download: modal volume get spine-vit-outputs / ./outputs_modal --force")
    print("aggregate: ./.venv/bin/python evaluate.py --experiments_dir outputs_modal --from_saved  # (full + axial-subset)")


@app.local_entrypoint()
def fusion_aug():
    ablation = [
        ["--views", "sag"],
        ["--views", "axial"],
        ["--views", "both", "--fusion", "concat"],
        ["--views", "both", "--fusion", "attn"],
    ]
    handles = []
    for cfg in ablation:
        for seed in SEEDS:
            handles.append(train_fusion_one.spawn(cfg + ["--augment", "--seed", str(seed)]))
    print(f"spawned {len(handles)} augmented runs ({len(ablation)} configs x {len(SEEDS)} seeds)...")
    failed = 0
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            failed += 1
            print("  aug fusion run FAILED:", e)
    print(f"done - {len(handles) - failed}/{len(handles)} succeeded.")
    print("download: modal volume get spine-vit-outputs / ./outputs_modal --force")
    print("aggregate: ./.venv/bin/python scripts/aggregate_fusion.py --experiments_dir outputs_modal --aug")


@app.local_entrypoint()
def fusion_budget():
    jobs = []
    for seed in (45, 46):
        jobs.append((["--views", "sag"], seed))
        jobs.append((["--views", "axial"], seed))
    for seed in SEEDS + [45, 46]:
        jobs.append((["--views", "sag", "--sag_slices", "5"], seed))
    handles = [train_fusion_one.spawn(cfg + ["--augment", "--seed", str(seed)]) for cfg, seed in jobs]
    print(f"spawned {len(handles)} runs (4 seed-extension + 3 budget-control)...")
    failed = 0
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            failed += 1
            print("  budget/seed run FAILED:", e)
    print(f"done - {len(handles) - failed}/{len(handles)} succeeded.")
    print("aggregate: ./.venv/bin/python scripts/aggregate_fusion.py --experiments_dir outputs_modal --aug")


@app.local_entrypoint()
def fusion_fus_ext():
    jobs = []
    for cfg in (["--views", "both", "--fusion", "concat"], ["--views", "both", "--fusion", "attn"]):
        for seed in (45, 46):
            jobs.append(cfg + ["--augment", "--seed", str(seed)])
    handles = [train_fusion_one.spawn(j) for j in jobs]
    print(f"spawned {len(handles)} fusion seed-extension runs (concat,attn x 45,46)...")
    failed = 0
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            failed += 1
            print("  fusion ext run FAILED:", e)
    print(f"done - {len(handles) - failed}/{len(handles)} succeeded.")
    print("aggregate: ./.venv/bin/python scripts/aggregate_fusion.py --experiments_dir outputs_modal --aug")


@app.local_entrypoint()
def detector_pipeline(epochs = 60):
    print(f"[1/2] training detector ({epochs} epochs) + exporting centers ...")
    train_detector_fn.remote(epochs)
    print("[2/2] detected-box grading, 3 seeds in parallel ...")
    extra = ["--tokenizer", "anatomy", "--pos_encoding", "ordinal",
             "--box_source", "detected",
             "--detected_centers_path", "/outputs/detector/detected_centers.json"]
    handles = [train_one.spawn(extra + ["--seed", str(seed)]) for seed in SEEDS]
    failed = 0
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            failed += 1
            print("  detected run FAILED:", e)
    print(f"done - detector + {len(handles) - failed}/{len(handles)} detected grading runs.")
    print("\nnext (local):")
    print("  modal volume get spine-vit-outputs / ./outputs_modal")
    print("  ./.venv/bin/python evaluate.py --experiments_dir outputs_modal --from_saved   # oracle vs detected table")
    print("  ./.venv/bin/python analyze_localization.py --experiments_dir outputs_modal    # grading-vs-detection figure")
    print("  cat outputs_modal/detector/localization_report.json                            # mm error, per level")


@app.local_entrypoint()
def coral():
    extra = ["--tokenizer", "anatomy", "--pos_encoding", "ordinal", "--head", "coral"]
    handles = [train_one.spawn(extra + ["--seed", str(seed)]) for seed in SEEDS]
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            print("  run FAILED:", e)
    print("done. evaluate.py --from_saved -> compare *_coral vs CE (kappa, kappa_linear, MAE, Moderate recall).")


@app.local_entrypoint()
def resolve_cast():
    configs = [
        ["--tokenizer", "anatomy",   "--pos_encoding", "ordinal"],
        ["--tokenizer", "cast_crop", "--pos_encoding", "ordinal"],
    ]
    handles = [train_one.spawn(config + ["--seed", str(seed)]) for config in configs for seed in (45, 46)]
    print(f"spawned {len(handles)} runs (2 configs x seeds 45,46)...")
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            print("  run FAILED:", e)
    print("done. re-run evaluate.py --from_saved -> anatomy & cast_crop now aggregate over 5 seeds.")


@app.local_entrypoint()
def resolve_patch_strips():
    configs = [
        ["--tokenizer", "patches", "--pos_encoding", "ordinal"],
        ["--tokenizer", "strips",  "--pos_encoding", "ordinal"],
    ]
    handles = [train_one.spawn(config + ["--seed", str(seed)]) for config in configs for seed in (45, 46)]
    print(f"spawned {len(handles)} runs (2 configs x seeds 45,46)...")
    failed = 0
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            failed += 1
            print("  run FAILED:", e)
    print(f"done - {len(handles) - failed}/{len(handles)} succeeded.")
    print("download: modal volume get spine-vit-outputs / ./outputs_modal --force")
    print("analyze:  ./.venv/bin/python scripts/tokenizer_stats.py")


@app.local_entrypoint()
def box_size_sweep():
    sizes = [16, 24, 48]
    handles = []
    for bs in sizes:
        for source in ("oracle", "detected"):
            extra = ["--tokenizer", "anatomy", "--pos_encoding", "ordinal", "--box_size", str(bs)]
            if source == "detected":
                extra += ["--box_source", "detected",
                          "--detected_centers_path", "/outputs/detector/detected_centers.json"]
            for seed in SEEDS:
                handles.append(train_one.spawn(extra + ["--seed", str(seed)]))
    print(f"spawned {len(handles)} runs ({len(sizes)} sizes x oracle/detected x {len(SEEDS)} seeds)...")
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            print("  run FAILED:", e)
    print("done. download --force, then: ./.venv/bin/python analyze_boxsize.py --experiments_dir outputs_modal")


@app.local_entrypoint()
def main():
    frozen = [
        ["--tokenizer", "strips",    "--pos_encoding", "ordinal"],
        ["--tokenizer", "patches",   "--pos_encoding", "ordinal"],
        ["--tokenizer", "cast_crop", "--pos_encoding", "ordinal"],
        ["--tokenizer", "anatomy",   "--pos_encoding", "learned"],
        ["--tokenizer", "anatomy",   "--pos_encoding", "none"],
        ["--tokenizer", "anatomy",   "--pos_encoding", "ordinal"],
    ]
    ft = ["--tokenizer", "anatomy", "--pos_encoding", "ordinal", "--no-freeze_backbone", "--lr", "3e-5"]

    handles = []
    for cfg in frozen:
        for seed in SEEDS:
            handles.append(train_one.spawn(cfg + ["--seed", str(seed)]))
    for seed in SEEDS:
        handles.append(train_one.with_options(gpu="A100").spawn(ft + ["--seed", str(seed)]))

    print(f"spawned {len(handles)} runs in parallel; waiting for all to finish...")
    failed = 0
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            failed += 1
            print("  run FAILED:", e)
    print(f"done - {len(handles) - failed}/{len(handles)} succeeded.")
    print("download: modal volume get spine-vit-outputs / ./outputs_modal")
    print("aggregate: ./.venv/bin/python evaluate.py --experiments_dir outputs_modal --from_saved")
