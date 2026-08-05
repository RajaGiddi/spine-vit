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


def load_dotenv(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def check_credentials():
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


def explain_http_error(err):
    resp = getattr(err, "response", None)
    status = getattr(resp, "status_code", None)
    text = str(err).lower()

    if status == 401 or "401" in text or "unauthenticated" in text:
        sys.exit(
            "ERROR: 401 Unauthenticated - Kaggle rejected your username/key.\n"
            "  - Check KAGGLE_USERNAME and KAGGLE_KEY (or ~/.kaggle/kaggle.json).\n"
            "  - KAGGLE_KEY must be the CURRENT 32-char key from 'Create New Token'\n"
            "    (creating a new token invalidates any older key).\n"
            "  Token: https://www.kaggle.com/settings -> API -> Create New Token"
        )
    if status == 403 or "403" in text or "forbidden" in text:
        sys.exit(
            "ERROR: 403 Forbidden - accept the competition rules, then re-run:\n"
            f"  https://www.kaggle.com/competitions/{COMPETITION}/rules"
        )


RETRYABLE_STATUS = {429, 500, 502, 503, 504}
OK, MISSING, RATE_LIMITED = "ok", "missing", "rate_limited"


def download_file(rel_path, out_dir, force=False, max_retries=5):
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
            explain_http_error(err)
            return MISSING


def download_csvs(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading annotation CSVs to {out_dir}/")

    for fname in CSV_FILES:
        target = out_dir / fname
        if target.exists():
            print(f"  {fname} - already present, skipping")
            continue
        print(f"  {fname} ...", end=" ", flush=True)
        status = download_file(fname, out_dir)
        print("ok" if status == OK else f"FAILED ({status})")

    missing = [name for name in CSV_FILES if not (out_dir / name).exists()]
    if missing:
        sys.exit(f"ERROR: failed to download {missing}. Check credentials and rules acceptance.")

    labels = pd.read_csv(out_dir / "train.csv")
    coords = pd.read_csv(out_dir / "train_label_coordinates.csv")
    print(f"\n  {len(labels)} studies in train.csv")
    print(f"  {len(coords)} coordinate annotations")
    print(f"  Conditions: {sorted(coords.condition.unique())}")


def select_studies(csv_dir, count, seed=42):
    labels = pd.read_csv(csv_dir / "train.csv")
    coords = pd.read_csv(csv_dir / "train_label_coordinates.csv")
    series = pd.read_csv(csv_dir / "train_series_descriptions.csv")

    desc = series.series_description.fillna("").str.lower()
    sag_t2 = set(series[desc.str.contains("sagittal") & desc.str.contains("t2")].study_id)

    canal = coords[coords.condition == "Spinal Canal Stenosis"]
    level_counts = canal.groupby("study_id")["level"].nunique()
    all_levels = set(level_counts[level_counts == len(LEVELS)].index)

    grade_cols = [
        f"spinal_canal_stenosis_{lv.lower().replace('/', '_')}" for lv in LEVELS
    ]
    grade_cols = [column for column in grade_cols if column in labels.columns]
    complete = set(labels[labels[grade_cols].notna().all(axis=1)].study_id)

    usable = sorted(sag_t2 & all_levels & complete)
    print(f"\nUsable studies: {len(usable)} of {len(labels)} total")
    print(f"  has sagittal T2:        {len(sag_t2)}")
    print(f"  has all 5 levels:       {len(all_levels)}")
    print(f"  has complete labels:    {len(complete)}")

    if not usable:
        sys.exit("ERROR: no usable studies found. Check the CSV files.")

    if count >= len(usable):
        selected = usable
    else:
        table = labels[labels.study_id.isin(usable)].copy()
        def worst_grade_in_row(row):
            grades = []
            for value in row:
                if pd.notna(value):
                    grades.append(GRADE_MAP.get(value, 0))
            return max(grades)

        table["worst"] = table[grade_cols].apply(worst_grade_in_row, axis=1)
        from sklearn.model_selection import train_test_split
        selected, _ = train_test_split(
            table.study_id.tolist(), train_size=count, stratify=table.worst, random_state=seed
        )
        selected = sorted(selected)

    report_distribution(labels, selected, grade_cols)
    return selected


def report_distribution(labels, study_ids, grade_cols):
    table = labels[labels.study_id.isin(study_ids)]
    print(f"\nSelected {len(study_ids)} studies")

    print("\n  Per-level severity counts:")
    header = f"    {'Level':<8}"
    for name in GRADE_MAP:
        header += f"{name:>14}"
    print(header)

    totals = {key: 0 for key in GRADE_MAP}
    for col, lv in zip(grade_cols, LEVELS):
        counts = table[col].value_counts()
        row = f"    {lv:<8}"
        for name in GRADE_MAP:
            count = int(counts.get(name, 0))
            totals[name] += count
            row += f"{count:>14}"
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


def download_file_level(
    csv_dir, out_dir, count, context_slices = 1, tag = "subset",
    delay = 0.25, rl_break = 6,
) :
    study_ids = select_studies(csv_dir, count)
    coords = pd.read_csv(csv_dir / "train_label_coordinates.csv")
    series = pd.read_csv(csv_dir / "train_series_descriptions.csv")
    canal = coords[coords.condition == "Spinal Canal Stenosis"]

    desc = series.series_description.fillna("").str.lower()
    sagt2_series = set(series[desc.str.contains("sagittal") & desc.str.contains("t2")].series_id)

    wanted = set()
    for sid in study_ids:
        rows = canal[(canal.study_id == sid) & (canal.series_id.isin(sagt2_series))]
        for _, response in rows.iterrows():
            for offset in range(-context_slices, context_slices + 1):
                wanted.add((int(sid), int(response.series_id), int(response.instance_number) + offset))

    print(f"\nDownloading up to {len(wanted)} DICOM files for {len(study_ids)} studies")
    print("(some neighbor slices won't exist at series boundaries - that's expected)\n")

    downloaded = missing = 0
    consecutive_rl = 0
    stopped = False
    for i, (sid, series_id, inst) in enumerate(sorted(wanted), 1):
        rel = f"train_images/{sid}/{series_id}/{inst}.dcm"
        status = download_file(rel, out_dir)
        if status == OK:
            downloaded += 1
            consecutive_rl = 0
        elif status == MISSING:
            missing += 1
            consecutive_rl = 0
        else:
            consecutive_rl += 1
            if consecutive_rl >= rl_break:
                print(f"\n[stop] Hit sustained Kaggle rate-limiting after {downloaded} files this run.")
                print("       Wait ~30-60 min for the quota to reset, then re-run the SAME")
                print("       command to resume - files already on disk are skipped.")
                stopped = True
                break
        if i % 50 == 0:
            print(f"  {i}/{len(wanted)} attempted - {downloaded} ok, {missing} missing")
        if delay:
            time.sleep(delay)

    print(f"\nDone: {downloaded} files downloaded, {missing} genuinely unavailable (boundary slices).")
    if stopped:
        print("NOTE: run stopped early due to rate-limiting - re-run to fetch the rest.")

    write_subset_csvs(csv_dir, out_dir, study_ids)
    write_id_list(out_dir, study_ids, tag)


def download_full(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading the full competition (100+ GB) directly to", out_dir)
    print("This is resumable - re-run if interrupted.\n")
    try:
        path = kagglehub.competition_download(COMPETITION, output_dir=str(out_dir))
    except KaggleApiHTTPError as err:
        explain_http_error(err)
        raise
    print("Downloaded to", path)
    return Path(path)


def write_subset_csvs(csv_dir, out_dir, study_ids):
    out_dir.mkdir(parents=True, exist_ok=True)
    id_set = set(study_ids)

    for fname in CSV_FILES:
        src = csv_dir / fname
        if not src.exists():
            continue
        table = pd.read_csv(src)
        if "study_id" in table.columns:
            table = table[table.study_id.isin(id_set)]
        table.to_csv(out_dir / fname, index=False)

    print(f"  Wrote filtered CSVs to {out_dir}/")


def write_id_list(out_dir, study_ids, tag):
    path = out_dir / f"selected_ids_{tag}.csv"
    pd.DataFrame({"study_id": study_ids}).to_csv(path, index=False)
    print(f"  Wrote {path}")
    return path


def print_next_steps(out_dir):
    print("\nNext steps:")
    print(f"  1. Inspect boxes:  python scripts/explore_rsna.py --data_dir {out_dir}")
    print(f"  2. Sanity train:   python train.py --data_dir {out_dir} --dataset rsna \\")
    print("                       --tokenizer anatomy --pos_encoding ordinal --epochs 5 --limit_samples 10")
    print(f"  3. Full train:     python train.py --data_dir {out_dir} --dataset rsna "
          "--tokenizer anatomy --pos_encoding ordinal")


def report_disk(out_dir):
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
        for dicom_file in study.rglob("*.dcm"):
            total += dicom_file.stat().st_size
            n_files += 1

    print(f"\nOn disk: {n_studies} studies, {n_files} DICOM files, {total / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Download RSNA 2024 Lumbar Spine data from Kaggle (via kagglehub)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["csvs", "smoke", "subset", "full"],
        help="csvs: annotations only | smoke: 25 studies | subset: N studies | full: everything",
    )
    parser.add_argument("--out_dir", default="data/rsna", help="where to put the data")
    parser.add_argument("--csv_dir", default=None, help="where the CSVs live (default: same as out_dir)")
    parser.add_argument("--n", type=int, default=None, help="number of studies (default: 25 smoke, 500 subset)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for study selection")
    parser.add_argument("--context_slices", type=int, default=1,
                   help="neighbor slices per side to fetch (for 2.5D input)")
    parser.add_argument("--delay", type=float, default=0.25,
                   help="seconds to pause between file downloads (paces Kaggle's rate limit)")
    args = parser.parse_args()

    load_dotenv()
    check_credentials()

    out_dir = Path(args.out_dir)
    csv_dir = Path(args.csv_dir) if args.csv_dir else out_dir

    if args.mode == "csvs":
        download_csvs(out_dir)
        print("\nNext: python rsna_setup.py --mode subset --n 500")
        return

    if not all((csv_dir / name).exists() for name in CSV_FILES):
        print("CSVs not found - downloading them first.\n")
        download_csvs(csv_dir)
        print()

    if args.mode == "smoke":
        count = args.n or 25
        download_file_level(csv_dir, out_dir, count=count, context_slices=args.context_slices,
                            tag=f"smoke{count}", delay=args.delay)
        report_disk(out_dir)
        print_next_steps(out_dir)

    elif args.mode == "subset":
        count = args.n or 500
        download_file_level(csv_dir, out_dir, count=count, context_slices=args.context_slices,
                            tag=f"subset{count}", delay=args.delay)
        report_disk(out_dir)
        print_next_steps(out_dir)

    elif args.mode == "full":
        download_full(out_dir)
        report_disk(out_dir)
        print_next_steps(out_dir)


if __name__ == "__main__":
    main()
