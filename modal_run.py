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
            # Credentials must never enter an image layer. .env holds the Kaggle
            # token; nothing that runs here needs it, and the downloader takes
            # it from a Modal Secret at runtime instead.
            ".env", "*.env", "**/.env", ".kaggle", ".kaggle/**", "kaggle.json",
        ],
    )
)

data_vol = modal.Volume.from_name("spine-vit-data", create_if_missing=True)
out_vol = modal.Volume.from_name("spine-vit-outputs", create_if_missing=True)


@app.function(image=image, gpu="L4", volumes={"/data": data_vol.read_only(), "/outputs": out_vol}, timeout=2 * 60 * 60)
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


@app.function(image=image, gpu="L4", volumes={"/data": data_vol.read_only(), "/outputs": out_vol}, timeout=3 * 60 * 60)
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


@app.function(image=image, gpu="L4", volumes={"/data": data_vol.read_only(), "/outputs": out_vol}, timeout=2 * 60 * 60)
def train_fusion_one(extra_args):
    import os, glob, subprocess

    os.chdir("/root/spine-vit")
    # An explicit --data_dir in extra_args wins (argparse takes the last one).
    # The glob is sorted so the fallback cannot silently switch corpora once
    # more than one exists on the volume.
    hits = sorted(glob.glob("/data/*/train_label_coordinates.csv"))
    data_dir = os.path.dirname(hits[0]) if hits else "/data/rsna"
    if "--data_dir" not in extra_args:
        print(f"data_dir (auto): {data_dir}", flush=True)
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


# Deliberately minimal: no torch, no repo, no GPU. The fewer things that run in
# the container holding a writable data volume and a live API token, the smaller
# the blast radius if any one of them is compromised.
fetch_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("kagglehub==1.0.2", "pandas==2.2.3", "scikit-learn==1.5.2")
)

COMPETITION = "rsna-2024-lumbar-spine-degenerative-classification"
ALLOWED_CSVS = ("train.csv", "train_label_coordinates.csv", "train_series_descriptions.csv")


def _safe_relpath(rel: str) -> str:
    """Reject anything that is not a plain file under the competition tree.

    kagglehub writes wherever the returned path says, so a traversal in a
    crafted path could land outside /data. Cheap to check, so check.
    """
    import posixpath
    if rel.startswith("/") or ".." in rel.split("/"):
        raise ValueError(f"refusing suspicious path: {rel!r}")
    normalised = posixpath.normpath(rel)
    if normalised.startswith(("/", "..")):
        raise ValueError(f"refusing suspicious path: {rel!r}")
    if not (normalised in ALLOWED_CSVS or normalised.startswith("train_images/")):
        raise ValueError(f"outside the competition tree: {rel!r}")
    return normalised


@app.function(
    image=fetch_image,
    volumes={"/data": data_vol},          # writable here, and only here
    secrets=[modal.Secret.from_name("kaggle")],
    timeout=6 * 60 * 60,
    cpu=4.0,                              # extraction is CPU-bound
)
def fetch_rsna(mode: str = "bulk", target: str = "rsna_full",
               n_studies: int = 1500, context_slices: int = 3,
               delay: float = 0.2):
    """Download the competition into /data/<target>, from inside Modal.

    Defaults to /data/rsna_full so the 500-study corpus at /data/rsna, and
    the splits every published result depends on, stay untouched.

    mode="bulk"  one archive, ~30-60 min. The right choice here: Kaggle
                 rate-limits per request, so 67k individual fetches take
                 6-10 h while a single bulk transfer is bandwidth-bound.
                 rsna_setup.py fetches per-file because it was written for a
                 laptop with limited disk; neither limit applies on Modal.
    mode="files" per-file top-up, for filling gaps without re-pulling 100 GB.

    Only this competition is ever requested, and credentials come from the
    Modal Secret - never written to disk, never logged.
    """
    import os, time, pathlib, shutil, kagglehub
    from kagglehub.exceptions import KaggleApiHTTPError, NotFoundError

    # kagglehub takes either an access token or a username/key pair. They are
    # not interchangeable: an access token supplied as KAGGLE_KEY authenticates
    # as nobody and Kaggle returns 401.
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    has_pair = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not (has_token or has_pair):
        raise SystemExit(
            "the 'kaggle' Modal Secret must set either\n"
            "  KAGGLE_API_TOKEN=<contents of ~/.kaggle/access_token>   (what you have), or\n"
            "  KAGGLE_USERNAME=<user> KAGGLE_KEY=<32-hex key from kaggle.json>")
    print(f"auth: {'access token' if has_token else 'username/key pair'}", flush=True)

    # Separate from /data/rsna: the 500-study CSVs there define the splits
    # every published run used, and the full CSVs would overwrite them.
    out = pathlib.Path("/data") / target
    out.mkdir(parents=True, exist_ok=True)
    # Stage everything on the volume, not the container's ephemeral disk - the
    # archive is ~90-100 GB and no ephemeral_disk request is needed if nothing
    # large is written locally. TMPDIR covers zip extraction, which would
    # otherwise land in /tmp.
    staging = pathlib.Path("/data/.kagglehub")
    staging.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(staging)
    os.environ["TMPDIR"] = str(staging)
    import tempfile
    tempfile.tempdir = str(staging)

    if mode == "check":
        # Dry run: exercise the secret, kagglehub auth, the path guard and the
        # volume write, using only the three CSVs (a few MB). Also reports the
        # cohort size, which is what sets the full-scale multiplier.
        t0 = time.time()
        for name in ALLOWED_CSVS:
            rel = _safe_relpath(name)
            if not (out / rel).exists():
                kagglehub.competition_download(COMPETITION, path=rel, output_dir=str(out))
            if not (out / rel).exists():
                raise SystemExit(f"could not fetch {name} - check the kaggle secret")
        data_vol.commit()

        import pandas as pd
        series = pd.read_csv(out / "train_series_descriptions.csv")
        coords = pd.read_csv(out / "train_label_coordinates.csv")
        labels = pd.read_csv(out / "train.csv")
        desc = series.series_description.fillna("")
        sag = set(series[desc == "Sagittal T2/STIR"].study_id)
        axi = set(series[desc == "Axial T2"].study_id)
        canal = coords[coords.condition == "Spinal Canal Stenosis"]
        # ["level"], not .level - attribute access collides with a pandas name.
        per_study = canal.groupby("study_id")["level"].nunique()
        all_five = set(per_study[per_study == 5].index)
        usable = sag & axi & all_five

        free = shutil.disk_usage("/data").free / 1e9
        out_str = {
            "auth": "ok",
            "studies_total": int(labels.study_id.nunique()),
            "with_sag_T2": len(sag),
            "with_axial_T2": len(axi),
            "with_both": len(sag & axi),
            "with_both_and_all_5_levels": len(usable),
            "current_corpus": 500,
            "multiplier_vs_current": round(len(usable) / 500, 2),
            "volume_free_gb": round(free, 1),
            "seconds": round(time.time() - t0, 1),
        }
        for k, v in out_str.items():
            print(f"  {k:30s} {v}")
        return out_str

    if mode == "bulk":
        t0 = time.time()
        print(f"bulk download of {COMPETITION} -> {out}", flush=True)
        path = kagglehub.competition_download(COMPETITION, output_dir=str(out))
        data_vol.commit()
        total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        n_dcm = sum(1 for _ in out.rglob("*.dcm"))
        mins = (time.time() - t0) / 60
        print(f"done in {mins:.1f} min: {total/1e9:.1f} GB, {n_dcm:,} DICOM files at {path}")
        shutil.rmtree("/data/.kagglehub", ignore_errors=True)
        data_vol.commit()
        return {"mode": "bulk", "gb": round(total / 1e9, 1), "dcm": n_dcm,
                "minutes": round(mins, 1)}

    if mode != "files":
        raise SystemExit(f"mode must be 'check', 'bulk' or 'files', got {mode!r}")

    import pandas as pd

    def grab(rel: str) -> bool:
        rel = _safe_relpath(rel)
        target = out / rel
        if target.exists():
            return True
        for attempt in range(5):
            try:
                kagglehub.competition_download(COMPETITION, path=rel, output_dir=str(out))
                return target.exists()
            except (KaggleApiHTTPError, NotFoundError) as err:
                status = getattr(getattr(err, "response", None), "status_code", None)
                if status in (429, 500, 502, 503, 504):
                    time.sleep(min(4 * 2 ** attempt, 90))
                    continue
                return False
        return False

    for name in ALLOWED_CSVS:
        if not grab(name):
            raise SystemExit(f"could not fetch {name}")
    data_vol.commit()

    series = pd.read_csv(out / "train_series_descriptions.csv")
    coords = pd.read_csv(out / "train_label_coordinates.csv")
    desc = series.series_description.fillna("")
    both = set(series[desc == "Sagittal T2/STIR"].study_id) & set(series[desc == "Axial T2"].study_id)
    studies = sorted(both)[:n_studies]
    axial = set(series[desc == "Axial T2"].series_id)

    wanted = set()
    for r in coords[coords.study_id.isin(studies)].itertuples():
        # Depth only matters for axial: the fixed-spacing rule only picks axial slices.
        ctx = context_slices if r.series_id in axial else 1
        for off in range(-ctx, ctx + 1):
            wanted.add((int(r.study_id), int(r.series_id), int(r.instance_number) + off))

    got = missing = 0
    for i, (sid, ser, inst) in enumerate(sorted(wanted), 1):
        if grab(f"train_images/{sid}/{ser}/{inst}.dcm"):
            got += 1
        else:
            missing += 1
        if i % 500 == 0:
            print(f"  {i}/{len(wanted)}  ok={got} missing={missing}", flush=True)
            data_vol.commit()
        if delay:
            time.sleep(delay)

    data_vol.commit()
    print(f"done: {got} files, {missing} unavailable")
    return {"mode": "files", "files": got, "missing": missing}


@app.local_entrypoint()
def fetch(mode: str = "bulk", target: str = "rsna_full",
          n_studies: int = 1500, context_slices: int = 3):
    print(fetch_rsna.remote(mode=mode, target=target, n_studies=n_studies,
                            context_slices=context_slices))


@app.local_entrypoint()
def fusion_fullscale(data_dir: str = "/data/rsna_full"):
    """Every Table 1 configuration plus the fixed-slice control, at full scale.

    Reads the corpus fetched by `fetch`, leaving /data/rsna and every published
    result untouched. Run names carry a _full suffix via --tag so nothing
    collides with the 500-study runs on the outputs volume.
    """
    configs = [
        ["--views", "sag"],
        ["--views", "sag", "--sag_slices", "5"],
        ["--views", "axial"],
        ["--views", "axial", "--axial_slice_selection", "fixed"],
        ["--views", "both", "--fusion", "concat"],
        ["--views", "both", "--fusion", "attn"],
    ]
    seeds = [42, 43, 44, 45, 46]
    handles = []
    for cfg in configs:
        for seed in seeds:
            handles.append(train_fusion_one.spawn(
                cfg + ["--data_dir", data_dir, "--augment", "--seed", str(seed)]))
    print(f"spawned {len(handles)} runs ({len(configs)} configs x {len(seeds)} seeds) "
          f"against {data_dir}")
    failed = 0
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            failed += 1
            print("  run FAILED:", e)
    print(f"done - {len(handles) - failed}/{len(handles)} succeeded.")


@app.local_entrypoint()
def fusion_fixedslice():
    """Axial with rule-based slice selection, as a control for expert choice.

    Identical to the existing axial configuration except that the slice is
    chosen by geometry rather than taken from the radiologist's annotation, so
    the comparison isolates slice selection. Five seeds, matching the axial
    configuration in Table 1. Writes to *_fixedslice_aug_s*, so nothing
    existing is overwritten.
    """
    seeds = [42, 43, 44, 45, 46]
    handles = [
        train_fusion_one.spawn(
            ["--views", "axial", "--axial_slice_selection", "fixed",
             "--augment", "--seed", str(seed)])
        for seed in seeds
    ]
    print(f"spawned {len(handles)} fixed-slice axial runs (seeds {seeds})...")
    failed = 0
    for handle in handles:
        try:
            handle.get()
        except Exception as e:
            failed += 1
            print("  fixed-slice run FAILED:", e)
    print(f"done - {len(handles) - failed}/{len(handles)} succeeded.")
    print("fetch predictions:")
    for seed in seeds:
        run = f"rsna_fusion_axial_ordinal_256_2_fixedslice_aug_s{seed}"
        print(f"  modal volume get spine-vit-outputs {run}/test_predictions.json "
              f"outputs_modal/{run}/test_predictions.json")
        print(f"  modal volume get spine-vit-outputs {run}/test_results.json "
              f"outputs_modal/{run}/test_results.json")
    print("then: ./.venv/bin/python scripts/bootstrap_worst_level.py --include_fixed")


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
