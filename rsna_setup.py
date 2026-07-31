"""
Download RSNA 2024 Lumbar Spine Degenerative Classification data from Kaggle.

Uses the official `kagglehub` library. Unlike the old `kaggle` CLI (one subprocess +
re-auth per file), kagglehub authenticates once in-process and caches downloads, so
fetching the many small DICOM files a subset needs is much faster.

Three modes:
    smoke  — 25 studies, file-level download. For pipeline verification.
    subset — N studies (default 500), stratified, file-level download. For experiments.
    full   — everything, downloaded directly to --out_dir. ~100+ GB. Only if you need it.

smoke and subset download ONLY the DICOM slices the loader actually reads (the
annotated sagittal-T2 slices plus their 2.5D neighbors), so a 500-study subset is a
few GB, not 100+. Only `full` pulls the whole competition.

Prerequisites
-------------
1. pip install kagglehub pandas scikit-learn        (already in .venv)
2. Kaggle API credentials (kagglehub 1.x accepts any of these, in priority order):
     - a single access token in ~/.kaggle/access_token, or the KAGGLE_API_TOKEN env
       var / .env entry (the modern "they just give you an API key" flow); or
     - KAGGLE_USERNAME + KAGGLE_KEY (shell export or .env, auto-loaded); or
     - ~/.kaggle/kaggle.json (chmod 600), if 'Create New Token' downloads one.
   Get a token at: https://www.kaggle.com/settings/api
3. Accept the competition rules at:
   https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/rules
   Downloads 403 until you do this in the browser.

Usage
-----
    # Step 1: always run this first (downloads the 3 small CSVs)
    python rsna_setup.py --mode csvs

    # Step 2: pick a mode
    python rsna_setup.py --mode smoke                  # 25 studies (quick check)
    python rsna_setup.py --mode subset --n 500         # 500 studies (experiments)
    python rsna_setup.py --mode full                   # everything (100+ GB)

    # Resume an interrupted download (skips what already exists)
    python rsna_setup.py --mode subset --n 500

Notes
-----
smoke and subset fetch files individually (only the annotated sagittal-T2 slices +
2.5D neighbors). Files download directly into --out_dir via kagglehub's output_dir,
bypassing the ~/.cache/kagglehub copy (no duplicated disk). The download is sequential,
so budget time for large N; it is resumable — files already on disk are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import kagglehub
import pandas as pd
from kagglehub.exceptions import KaggleApiHTTPError, NotFoundError

COMPETITION = "rsna-2024-lumbar-spine-degenerative-classification"
CSV_FILES = ["train.csv", "train_label_coordinates.csv", "train_series_descriptions.csv"]
LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
GRADE_MAP = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}


# ──────────────────────────────────────────────────────────────────────
# Preflight + kagglehub download helpers
# ──────────────────────────────────────────────────────────────────────

def load_dotenv(path: str = ".env") -> None:
    """Populate os.environ from a .env file (kagglehub does not read .env itself).

    Minimal `KEY=value` parser — no dependency. Existing environment variables are
    NOT overwritten, so a real shell export always wins.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def check_credentials() -> None:
    """Verify Kaggle credentials resolve, using kagglehub's OWN resolver.

    kagglehub 1.x accepts several methods, in priority order:
      1. a single access token — the KAGGLE_API_TOKEN env var, or a
         ~/.kaggle/access_token file (the modern "just an API key" flow);
      2. KAGGLE_USERNAME + KAGGLE_KEY;
      3. ~/.kaggle/kaggle.json.
    Delegating to get_kaggle_credentials() keeps this check in exact agreement with
    what the download will actually use.
    """
    try:
        from kagglehub.config import get_kaggle_credentials

        if get_kaggle_credentials() is not None:
            return
    except (ValueError, OSError) as err:
        sys.exit(f"ERROR: Kaggle credentials are present but invalid: {err}")

    sys.exit(
        "ERROR: No Kaggle credentials found. Provide ONE of:\n"
        "  - a token in ~/.kaggle/access_token  (or the KAGGLE_API_TOKEN env var)\n"
        "  - KAGGLE_USERNAME and KAGGLE_KEY     (shell export or .env)\n"
        "  - ~/.kaggle/kaggle.json              (chmod 600)\n"
        "  Get a token at: https://www.kaggle.com/settings/api"
    )


def _explain_http_error(err: Exception) -> None:
    """Exit with a targeted message for auth (401) / rules (403) failures.

    Returns without exiting for other statuses (e.g. 404) so per-file callers can treat
    a missing slice as "unavailable" rather than fatal. Uses the HTTP status code first
    (the response body text is unreliable — Kaggle's 401 body even mentions "rules").
    """
    resp = getattr(err, "response", None)
    status = getattr(resp, "status_code", None)
    text = str(err).lower()

    if status == 401 or "401" in text or "unauthenticated" in text:
        sys.exit(
            "ERROR: 401 Unauthenticated — Kaggle rejected your username/key.\n"
            "  - Check KAGGLE_USERNAME and KAGGLE_KEY (or ~/.kaggle/kaggle.json).\n"
            "  - KAGGLE_KEY must be the CURRENT 32-char key from 'Create New Token'\n"
            "    (creating a new token invalidates any older key).\n"
            "  Token: https://www.kaggle.com/settings -> API -> Create New Token"
        )
    if status == 403 or "403" in text or "forbidden" in text:
        sys.exit(
            "ERROR: 403 Forbidden — accept the competition rules, then re-run:\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}/rules"
        )


RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# download_file outcomes
OK, MISSING, RATE_LIMITED = "ok", "missing", "rate_limited"


def download_file(rel_path: str, out_dir: Path, force: bool = False, max_retries: int = 5) -> str:
    """Download one competition file directly to out_dir/rel_path via kagglehub.

    Uses kagglehub's output_dir so the file is written straight to its destination,
    bypassing the ~/.cache/kagglehub copy — no duplication. (kagglehub still drops a
    small 0-byte marker under out_dir/.complete/ for its own resume; harmless.)

    Returns one of:
      OK           - file is present on disk;
      MISSING      - genuinely absent (404, e.g. a neighbor slice past a boundary);
      RATE_LIMITED - transient throttling (429/5xx) survived all retries.
    A 401/403 exits with guidance. Distinguishing MISSING from RATE_LIMITED lets the
    caller stop cleanly on throttling instead of marking real files "unavailable".
    """
    target = out_dir / rel_path
    if target.exists() and not force:
        return OK

    delay = 4.0
    for attempt in range(1, max_retries + 1):
        try:
            kagglehub.competition_download(
                COMPETITION, path=rel_path, force_download=force, output_dir=str(out_dir)
            )
            return OK if target.exists() else MISSING
        except (KaggleApiHTTPError, NotFoundError) as err:
            status = getattr(getattr(err, "response", None), "status_code", None)
            if status in RETRYABLE_STATUS:
                if attempt < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 90.0)
                    continue
                return RATE_LIMITED
            if status == 404 or isinstance(err, NotFoundError):
                return MISSING
            _explain_http_error(err)  # 401/403 -> exit
            return MISSING


# ──────────────────────────────────────────────────────────────────────
# Mode: csvs
# ──────────────────────────────────────────────────────────────────────

def download_csvs(out_dir: Path) -> None:
    """Download the three annotation CSVs. A few MB total."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading annotation CSVs to {out_dir}/")

    for fname in CSV_FILES:
        target = out_dir / fname
        if target.exists():
            print(f"  {fname} — already present, skipping")
            continue
        print(f"  {fname} ...", end=" ", flush=True)
        status = download_file(fname, out_dir)
        print("ok" if status == OK else f"FAILED ({status})")

    missing = [f for f in CSV_FILES if not (out_dir / f).exists()]
    if missing:
        sys.exit(f"ERROR: failed to download {missing}. Check credentials and rules acceptance.")

    labels = pd.read_csv(out_dir / "train.csv")
    coords = pd.read_csv(out_dir / "train_label_coordinates.csv")
    print(f"\n  {len(labels)} studies in train.csv")
    print(f"  {len(coords)} coordinate annotations")
    print(f"  Conditions: {sorted(coords.condition.unique())}")


# ──────────────────────────────────────────────────────────────────────
# Study selection
# ──────────────────────────────────────────────────────────────────────

def select_studies(csv_dir: Path, n: int, seed: int = 42) -> list[int]:
    """
    Select study IDs that are usable for our task, stratified by severity.

    A study is usable if it has:
      - a sagittal T2 series
      - spinal canal stenosis coordinates at all 5 levels
      - complete (non-null) stenosis severity labels
    """
    labels = pd.read_csv(csv_dir / "train.csv")
    coords = pd.read_csv(csv_dir / "train_label_coordinates.csv")
    series = pd.read_csv(csv_dir / "train_series_descriptions.csv")

    desc = series.series_description.fillna("").str.lower()
    sag_t2 = set(series[desc.str.contains("sagittal") & desc.str.contains("t2")].study_id)

    canal = coords[coords.condition == "Spinal Canal Stenosis"]
    # Bracket access is required: `.level` collides with a pandas GroupBy attribute
    level_counts = canal.groupby("study_id")["level"].nunique()
    all_levels = set(level_counts[level_counts == len(LEVELS)].index)

    grade_cols = [
        f"spinal_canal_stenosis_{lv.lower().replace('/', '_')}" for lv in LEVELS
    ]
    grade_cols = [c for c in grade_cols if c in labels.columns]
    complete = set(labels[labels[grade_cols].notna().all(axis=1)].study_id)

    usable = sorted(sag_t2 & all_levels & complete)
    print(f"\nUsable studies: {len(usable)} of {len(labels)} total")
    print(f"  has sagittal T2:        {len(sag_t2)}")
    print(f"  has all 5 levels:       {len(all_levels)}")
    print(f"  has complete labels:    {len(complete)}")

    if not usable:
        sys.exit("ERROR: no usable studies found. Check the CSV files.")

    if n >= len(usable):
        selected = usable
    else:
        # Stratify on worst grade so rare Severe cases stay represented
        df = labels[labels.study_id.isin(usable)].copy()
        df["worst"] = df[grade_cols].apply(
            lambda row: max(GRADE_MAP.get(v, 0) for v in row if pd.notna(v)), axis=1
        )
        from sklearn.model_selection import train_test_split
        selected, _ = train_test_split(
            df.study_id.tolist(), train_size=n, stratify=df.worst, random_state=seed
        )
        selected = sorted(selected)

    report_distribution(labels, selected, grade_cols)
    return selected


def report_distribution(labels: pd.DataFrame, study_ids: list[int], grade_cols: list[str]) -> None:
    """Print the severity distribution of the selected studies."""
    df = labels[labels.study_id.isin(study_ids)]
    print(f"\nSelected {len(study_ids)} studies")

    print("\n  Per-level severity counts:")
    header = f"    {'Level':<8}"
    for name in GRADE_MAP:
        header += f"{name:>14}"
    print(header)

    totals = {k: 0 for k in GRADE_MAP}
    for col, lv in zip(grade_cols, LEVELS):
        counts = df[col].value_counts()
        row = f"    {lv:<8}"
        for name in GRADE_MAP:
            c = int(counts.get(name, 0))
            totals[name] += c
            row += f"{c:>14}"
        print(row)

    total = sum(totals.values())
    print(f"    {'-' * 50}")
    row = f"    {'Total':<8}"
    for name in GRADE_MAP:
        row += f"{totals[name]:>14}"
    print(row)
    if total:
        row = f"    {'':<8}"
        for name in GRADE_MAP:
            row += f"{totals[name] / total * 100:>13.1f}%"
        print(row)


# ──────────────────────────────────────────────────────────────────────
# Mode: smoke / subset (file-level download)
# ──────────────────────────────────────────────────────────────────────

def download_file_level(
    csv_dir: Path, out_dir: Path, n: int, context_slices: int = 1, tag: str = "subset",
    delay: float = 0.25, rl_break: int = 6,
) -> None:
    """
    Download N studies file by file — no full-archive download.

    For each selected study, fetches only the DICOM slices referenced by the Spinal
    Canal Stenosis coordinate annotations on its sagittal-T2 series, plus
    `context_slices` neighbors on each side (needed for the 2.5D input). These are
    exactly the slices `data/rsna_dataset.py` reads, so the download stays small.
    Missing neighbors at series boundaries are expected and ignored. Resumable:
    files already on disk are skipped.

    Kaggle rate-limits per-file downloads, so a large N may need several runs: a small
    inter-file `delay` paces requests, and after `rl_break` consecutive throttled files
    the run stops cleanly and tells you to resume later (re-running skips what's done).
    """
    study_ids = select_studies(csv_dir, n)
    coords = pd.read_csv(csv_dir / "train_label_coordinates.csv")
    series = pd.read_csv(csv_dir / "train_series_descriptions.csv")
    canal = coords[coords.condition == "Spinal Canal Stenosis"]

    # Restrict to sagittal-T2 series so we grab exactly the series the loader uses.
    desc = series.series_description.fillna("").str.lower()
    sagt2_series = set(series[desc.str.contains("sagittal") & desc.str.contains("t2")].series_id)

    # Build the set of files we need
    wanted: set[tuple[int, int, int]] = set()
    for sid in study_ids:
        rows = canal[(canal.study_id == sid) & (canal.series_id.isin(sagt2_series))]
        for _, r in rows.iterrows():
            for offset in range(-context_slices, context_slices + 1):
                wanted.add((int(sid), int(r.series_id), int(r.instance_number) + offset))

    print(f"\nDownloading up to {len(wanted)} DICOM files for {len(study_ids)} studies")
    print("(some neighbor slices won't exist at series boundaries — that's expected)\n")

    ok = missing = 0
    consecutive_rl = 0
    stopped = False
    for i, (sid, series_id, inst) in enumerate(sorted(wanted), 1):
        rel = f"train_images/{sid}/{series_id}/{inst}.dcm"
        status = download_file(rel, out_dir)
        if status == OK:
            ok += 1
            consecutive_rl = 0
        elif status == MISSING:
            missing += 1
            consecutive_rl = 0
        else:  # RATE_LIMITED
            consecutive_rl += 1
            if consecutive_rl >= rl_break:
                print(f"\n[stop] Hit sustained Kaggle rate-limiting after {ok} files this run.")
                print("       Wait ~30-60 min for the quota to reset, then re-run the SAME")
                print("       command to resume — files already on disk are skipped.")
                stopped = True
                break
        if i % 50 == 0:
            print(f"  {i}/{len(wanted)} attempted — {ok} ok, {missing} missing")
        if delay:
            time.sleep(delay)

    print(f"\nDone: {ok} files downloaded, {missing} genuinely unavailable (boundary slices).")
    if stopped:
        print("NOTE: run stopped early due to rate-limiting — re-run to fetch the rest.")

    # Filtered CSVs let the loader use whatever studies actually downloaded (it skips
    # studies whose image folder is absent), so writing them is safe even on a partial run.
    write_subset_csvs(csv_dir, out_dir, study_ids)
    write_id_list(out_dir, study_ids, tag)


# ──────────────────────────────────────────────────────────────────────
# Mode: full (whole competition, direct to out_dir)
# ──────────────────────────────────────────────────────────────────────

def download_full(out_dir: Path) -> Path:
    """Download the entire competition directly to out_dir (bypasses the cache)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading the full competition (100+ GB) directly to", out_dir)
    print("This is resumable — re-run if interrupted.\n")
    try:
        path = kagglehub.competition_download(COMPETITION, output_dir=str(out_dir))
    except KaggleApiHTTPError as err:
        _explain_http_error(err)
        raise
    print("Downloaded to", path)
    return Path(path)


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

def write_subset_csvs(csv_dir: Path, out_dir: Path, study_ids: list[int]) -> None:
    """
    Write CSVs filtered to the selected studies.

    This lets the rest of the pipeline run unchanged — it sees a self-consistent
    dataset where every study in train.csv actually has images on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    id_set = set(study_ids)

    for fname in CSV_FILES:
        src = csv_dir / fname
        if not src.exists():
            continue
        df = pd.read_csv(src)
        if "study_id" in df.columns:
            df = df[df.study_id.isin(id_set)]
        df.to_csv(out_dir / fname, index=False)

    print(f"  Wrote filtered CSVs to {out_dir}/")


def write_id_list(out_dir: Path, study_ids: list[int], tag: str) -> Path:
    """Save the selected study IDs for reproducibility."""
    path = out_dir / f"selected_ids_{tag}.csv"
    pd.DataFrame({"study_id": study_ids}).to_csv(path, index=False)
    print(f"  Wrote {path}")
    return path


def print_next_steps(out_dir: Path) -> None:
    """Point the user at the actual pipeline entry points."""
    print("\nNext steps:")
    print(f"  1. Inspect boxes:  python scripts/explore_rsna.py --data_dir {out_dir}")
    print(f"  2. Sanity train:   python train.py --data_dir {out_dir} --dataset rsna \\")
    print("                       --tokenizer anatomy --pos_encoding ordinal --epochs 5 --limit_samples 10")
    print(f"  3. Full train:     python train.py --data_dir {out_dir} --dataset rsna "
          "--tokenizer anatomy --pos_encoding ordinal")


def report_disk(out_dir: Path) -> None:
    """Report how much space the downloaded images take."""
    images = out_dir / "train_images"
    if not images.exists():
        return

    total = 0
    n_files = 0
    n_studies = 0
    for study in images.iterdir():
        if not study.is_dir():
            continue
        n_studies += 1
        for f in study.rglob("*.dcm"):
            total += f.stat().st_size
            n_files += 1

    print(f"\nOn disk: {n_studies} studies, {n_files} DICOM files, {total / 1e9:.2f} GB")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Download RSNA 2024 Lumbar Spine data from Kaggle (via kagglehub)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["csvs", "smoke", "subset", "full"],
        help="csvs: annotations only | smoke: 25 studies | subset: N studies | full: everything",
    )
    p.add_argument("--out_dir", default="data/rsna", help="where to put the data")
    p.add_argument("--csv_dir", default=None, help="where the CSVs live (default: same as out_dir)")
    p.add_argument("--n", type=int, default=None, help="number of studies (default: 25 smoke, 500 subset)")
    p.add_argument("--seed", type=int, default=42, help="random seed for study selection")
    p.add_argument("--context_slices", type=int, default=1,
                   help="neighbor slices per side to fetch (for 2.5D input)")
    p.add_argument("--delay", type=float, default=0.25,
                   help="seconds to pause between file downloads (paces Kaggle's rate limit)")
    args = p.parse_args()

    load_dotenv()  # pull KAGGLE_USERNAME/KAGGLE_KEY from .env if present
    check_credentials()

    out_dir = Path(args.out_dir)
    csv_dir = Path(args.csv_dir) if args.csv_dir else out_dir

    if args.mode == "csvs":
        download_csvs(out_dir)
        print("\nNext: python rsna_setup.py --mode subset --n 500")
        return

    # All image modes need the CSVs first
    if not all((csv_dir / f).exists() for f in CSV_FILES):
        print("CSVs not found — downloading them first.\n")
        download_csvs(csv_dir)
        print()

    if args.mode == "smoke":
        n = args.n or 25
        download_file_level(csv_dir, out_dir, n=n, context_slices=args.context_slices,
                            tag=f"smoke{n}", delay=args.delay)
        report_disk(out_dir)
        print_next_steps(out_dir)

    elif args.mode == "subset":
        # File-level download (no 100+ GB archive): fetch only the slices the loader
        # reads and write CSVs filtered to the selected studies.
        n = args.n or 500
        download_file_level(csv_dir, out_dir, n=n, context_slices=args.context_slices,
                            tag=f"subset{n}", delay=args.delay)
        report_disk(out_dir)
        print_next_steps(out_dir)

    elif args.mode == "full":
        download_full(out_dir)
        report_disk(out_dir)
        print_next_steps(out_dir)


if __name__ == "__main__":
    main()
