"""Select studies with both a sagittal T2 and an axial T2 series, and optionally
fetch the *complete* series for each.

The existing data/rsna tree holds only slices adjacent to a label coordinate
(the --context_slices 1 download pattern). Phase 0 operates on full volumes in
patient coordinates, so the paired series have to be pulled in full.

Instance numbers within an RSNA series are contiguous from 1, so --fetch walks
upward from 1 and stops after a run of consecutive 404s.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rsna_setup import MISSING, OK, RATE_LIMITED, download_file, load_dotenv

SAG_T2 = "Sagittal T2/STIR"
AXIAL_T2 = "Axial T2"


def _series_on_disk(images_dir, study_id, series_id):
    path = images_dir / str(study_id) / str(series_id)
    if not path.is_dir():
        return []
    return sorted(int(f.stem) for f in path.glob("*.dcm"))


def find_pairs(data_dir):
    """One (sagittal, axial) series pair per study.

    Where a study has several candidates, prefer the series carrying the most
    label coordinates - that is the one the graders read.
    """
    data_dir = Path(data_dir)
    series = pd.read_csv(data_dir / "train_series_descriptions.csv")
    coords = pd.read_csv(data_dir / "train_label_coordinates.csv")
    ann_counts = coords.groupby("series_id").size().to_dict()

    def pick(rows):
        if rows.empty:
            return None
        best = max(rows.series_id, key=lambda sid: (ann_counts.get(sid, 0), -sid))
        return int(best)

    pairs = []
    for study_id, rows in series.groupby("study_id"):
        sag = pick(rows[rows.series_description == SAG_T2])
        axi = pick(rows[rows.series_description == AXIAL_T2])
        if sag is not None and axi is not None:
            pairs.append({"study_id": int(study_id), "sag_series_id": sag, "ax_series_id": axi})
    return pd.DataFrame(pairs).sort_values("study_id").reset_index(drop=True)


def fetch_series(data_dir, study_id, series_id, miss_run=3, delay=0.25, max_instance=200):
    """Download every instance of one series. Returns (n_new, n_present, status)."""
    images_dir = Path(data_dir) / "train_images"
    known = _series_on_disk(images_dir, study_id, series_id)
    floor = (max(known) + miss_run) if known else miss_run

    new = present = 0
    misses = 0
    inst = 1
    while inst <= max_instance:
        target = images_dir / str(study_id) / str(series_id) / f"{inst}.dcm"
        if target.exists():
            present += 1
            misses = 0
        else:
            status = download_file(f"train_images/{study_id}/{series_id}/{inst}.dcm", Path(data_dir))
            if status == RATE_LIMITED:
                return new, present, RATE_LIMITED
            if status == OK:
                new += 1
                misses = 0
            else:
                misses += 1
                if misses >= miss_run and inst > floor:
                    break
            if delay:
                time.sleep(delay)
        inst += 1
    return new, present, OK


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data/rsna")
    parser.add_argument("--out", default="data/rsna/paired_studies.csv")
    parser.add_argument("--n", type=int, default=25, help="how many paired studies to select")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fetch", action="store_true", help="download the full series for each pair")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--miss_run", type=int, default=3,
                        help="consecutive 404s that mark the end of a series")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pairs = find_pairs(data_dir)
    print(f"Studies with both {SAG_T2} and {AXIAL_T2}: {len(pairs)}")

    if args.n < len(pairs):
        pairs = pairs.sample(n=args.n, random_state=args.seed).sort_values("study_id")
        pairs = pairs.reset_index(drop=True)
    print(f"Selected {len(pairs)} (seed={args.seed})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out, index=False)
    print(f"Wrote {out}")

    if not args.fetch:
        print("\nRe-run with --fetch to download the complete series.")
        return

    load_dotenv()
    log_path = out.with_name(out.stem + "_fetch_log.csv")
    with open(log_path, "w", newline="") as handle:
        log = csv.writer(handle)
        log.writerow(["study_id", "series_id", "plane", "n_slices", "n_new"])

        for row in pairs.itertuples():
            for plane, series_id in (("sagittal", row.sag_series_id), ("axial", row.ax_series_id)):
                new, present, status = fetch_series(
                    data_dir, row.study_id, series_id,
                    miss_run=args.miss_run, delay=args.delay,
                )
                total = len(_series_on_disk(data_dir / "train_images", row.study_id, series_id))
                print(f"  {row.study_id} {plane:9s} {series_id}: {total} slices (+{new} new)",
                      flush=True)
                log.writerow([row.study_id, series_id, plane, total, new])
                handle.flush()

                if status == RATE_LIMITED:
                    print("\n[stop] Kaggle rate-limited. Wait ~30-60 min and re-run the same "
                          "command - files already on disk are skipped.")
                    return

    print(f"\nWrote {log_path}")


if __name__ == "__main__":
    main()
