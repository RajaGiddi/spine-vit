"""Click-to-mark vertebral body centres for Experiment 0.

Two windows per study, sagittal then axial. Scroll or use the arrow keys to
change slice, click the centre of a vertebral body, press Enter to accept.
Mark the *same* vertebral body in both acquisitions - the body, not the disc,
and as close to its centre in all three axes as you can judge.

The RSNA label coordinates mark canal and subarticular points at disc levels,
not vertebral body centres, so they cannot stand in for this. They are drawn as
faint crosses for orientation only.

Usage:
    python scripts/mark_landmarks.py --n 20
    python scripts/mark_landmarks.py --study 109677683      # redo one
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from composition.volume import Volume  # noqa: E402

COLUMNS = ["study_id", "plane", "series_id", "instance_number", "row", "col"]


class SliceMarker:
    """One scrollable series with a single click-to-place mark."""

    def __init__(self, volume: Volume, title: str, hints=None):
        self.volume = volume
        self.hints = hints or {}
        self.k = len(volume) // 2
        self.mark = None
        self.accepted = False

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title(title)
        self.image = self.ax.imshow(volume.pixels[self.k], cmap="gray")
        self.dot, = self.ax.plot([], [], "o", color="#ff3b30", markersize=9,
                                 markeredgecolor="white", markeredgewidth=1.2)
        self.hint_marks, = self.ax.plot([], [], "+", color="#4da3ff", markersize=10,
                                        alpha=0.55, linestyle="none")
        self.ax.set_axis_off()
        self.title = title

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._redraw()

    def _redraw(self):
        g = self.volume.geometry[self.k]
        pixels = self.volume.pixels[self.k]
        self.image.set_data(pixels)
        self.image.set_clim(np.percentile(pixels, 1), np.percentile(pixels, 99.5))

        hint = self.hints.get(g.instance_number, [])
        if hint:
            xs, ys = zip(*hint)
            self.hint_marks.set_data(xs, ys)
        else:
            self.hint_marks.set_data([], [])

        state = "marked" if self.mark else "click the vertebral body centre"
        self.ax.set_title(
            f"{self.title}\nslice {self.k + 1}/{len(self.volume)} "
            f"(instance {g.instance_number})  -  {state}\n"
            "scroll/arrows: slice   click: mark   enter: accept   backspace: clear",
            fontsize=9,
        )
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        self._step(1 if event.button == "up" else -1)

    def _on_key(self, event):
        if event.key in ("up", "right"):
            self._step(1)
        elif event.key in ("down", "left"):
            self._step(-1)
        elif event.key == "backspace":
            self.mark = None
            self.dot.set_data([], [])
            self._redraw()
        elif event.key == "enter":
            if self.mark is None:
                print("  no mark placed yet")
                return
            self.accepted = True
            plt.close(self.fig)
        elif event.key == "escape":
            plt.close(self.fig)

    def _step(self, delta):
        self.k = int(np.clip(self.k + delta, 0, len(self.volume) - 1))
        self._redraw()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        col, row = int(round(event.xdata)), int(round(event.ydata))
        g = self.volume.geometry[self.k]
        if not (0 <= row < g.rows and 0 <= col < g.cols):
            return
        self.mark = (self.k, row, col)
        self.dot.set_data([col], [row])
        self._redraw()

    def run(self):
        plt.show()
        if not self.accepted or self.mark is None:
            return None
        k, row, col = self.mark
        return {
            "series_id": int(self.volume.series_id),
            "instance_number": int(self.volume.geometry[k].instance_number),
            "row": row,
            "col": col,
        }


def annotation_hints(data_dir: Path, series_id: int) -> dict:
    """{instance_number: [(x, y), ...]} from the RSNA label coordinates."""
    path = data_dir / "train_label_coordinates.csv"
    if not path.exists():
        return {}
    coords = pd.read_csv(path)
    rows = coords[coords.series_id == series_id]
    hints: dict[int, list] = {}
    for r in rows.itertuples():
        hints.setdefault(int(r.instance_number), []).append((float(r.x), float(r.y)))
    return hints


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data/rsna")
    parser.add_argument("--pairs", default="data/rsna/paired_studies.csv")
    parser.add_argument("--out", default="data/rsna/landmarks.csv")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--study", type=int, default=None, help="mark one study and exit")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images = data_dir / "train_images"
    pairs = pd.read_csv(args.pairs)
    if args.study:
        pairs = pairs[pairs.study_id == args.study]
        if pairs.empty:
            raise SystemExit(f"study {args.study} is not in {args.pairs}")

    out = Path(args.out)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=COLUMNS)
    complete = {sid for sid, g in done.groupby("study_id")
                if set(g.plane) >= {"sagittal", "axial"}}

    todo = [r for r in pairs.itertuples()
            if args.study or r.study_id not in complete][: args.n]
    if not todo:
        print(f"All studies in {args.pairs} already marked in {out}.")
        return

    print(f"{len(todo)} studies to mark. Close a window with Esc to stop.\n")
    records = []
    for i, r in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] study {r.study_id}")
        study_records = []
        for plane, series_id in (("sagittal", r.sag_series_id), ("axial", r.ax_series_id)):
            directory = images / str(r.study_id) / str(series_id)
            if not any(directory.glob("*.dcm")):
                print(f"  {plane}: no DICOMs on disk, skipping study")
                study_records = []
                break
            volume = Volume.from_dir(directory)
            marker = SliceMarker(volume, f"{r.study_id} - {plane}",
                                 annotation_hints(data_dir, int(series_id)))
            mark = marker.run()
            if mark is None:
                print("  aborted")
                study_records = []
                break
            mark.update(study_id=int(r.study_id), plane=plane)
            study_records.append(mark)

        if len(study_records) == 2:
            records.extend(study_records)
            fresh = pd.DataFrame(records)[COLUMNS]
            combined = pd.concat([done, fresh]).drop_duplicates(
                subset=["study_id", "plane"], keep="last")
            out.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(out, index=False)
            print(f"  saved -> {out}")
        else:
            break

    print(f"\nMarked {len(records) // 2} studies this session.")
    print(f"Next: python experiments/exp0_motion.py --landmarks {out}")


if __name__ == "__main__":
    main()
