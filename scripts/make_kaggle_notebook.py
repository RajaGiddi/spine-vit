"""Generate a standalone Kaggle notebook for Phase 0.

The notebook embeds the current composition/ and experiments/ sources, so there
is one source of truth: edit the repo, re-run this, re-upload. Nothing in the
notebook is hand-maintained.

Running on Kaggle removes the reason scripts/select_paired_studies.py --fetch
existed. The competition is mounted read-only at /kaggle/input with every slice
of every series present, so there is nothing to download and no rate limit.

    python scripts/make_kaggle_notebook.py
    -> notebooks/phase0_kaggle.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "phase0_kaggle.ipynb"

EMBED = [
    "composition/__init__.py",
    "composition/geometry.py",
    "composition/volume.py",
    "experiments/__init__.py",
    "experiments/exp0_motion.py",
    "experiments/exp0_retest.py",
]

COMPETITION = "rsna-2024-lumbar-spine-degenerative-classification"


def _bootstrap_cells():
    """Write the repo modules out as readable %%writefile cells.

    Not base64: an opaque blob makes the notebook unreviewable and its diffs
    meaningless. The source belongs where a reader can see what is running.
    """
    # %%writefile takes a literal path with no variable expansion, so the cells
    # below write relative to the working directory - which on Kaggle is
    # /kaggle/working, the one writable place.
    yield nbf.v4.new_code_cell('''import os, sys
from pathlib import Path

if Path("/kaggle/working").exists():
    os.chdir("/kaggle/working")

PKG = Path.cwd() / "phase0"
for sub in ("composition", "experiments"):
    (PKG / sub).mkdir(parents=True, exist_ok=True)
    (PKG / sub / "__init__.py").write_text("")
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
print("package root:", PKG)''')

    for rel in EMBED:
        if rel.endswith("__init__.py"):
            continue  # created above
        source = (ROOT / rel).read_text().rstrip("\n")
        yield nbf.v4.new_code_cell(f"%%writefile phase0/{rel}\n{source}")

    yield nbf.v4.new_code_cell('''import composition.geometry, composition.volume, experiments.exp0_motion  # noqa
print("imported OK from", PKG)''')


def _cells():
    yield nbf.v4.new_markdown_cell(
        "# Phase 0 - composition partitioning consistency\n\n"
        "Three experiments testing whether composition-based partitioning gives a "
        "consistent anatomy description across MRI acquisitions.\n\n"
        "**No training.** No labels, no gradients, no GPU. This is inference and "
        "measurement on full volumes in patient coordinates.\n\n"
        "Order of work, with reporting gates after steps 2 and 4:\n"
        "1. Geometry + verification tests - nothing proceeds until these pass\n"
        "2. Experiment 0, motion check, on 20 studies - **decision gate**\n"
        "3. Tissue / composition / partition\n"
        "4. Visual inspection on 5 studies\n"
        "5. Consistency measurement\n"
        "6. Experiments 1 and 2\n\n"
        "Attach the competition dataset to this notebook before running."
    )

    yield nbf.v4.new_markdown_cell(
        "## Setup\n\nThe Phase 0 modules, written out verbatim from the repo so the notebook is self-contained and still readable. Regenerate with `python scripts/make_kaggle_notebook.py` after any change."
    )
    yield from _bootstrap_cells()

    yield nbf.v4.new_code_cell('''from pathlib import Path
import numpy as np, pandas as pd

def find_data_dir():
    """Locate the competition wherever it got mounted.

    Attaching the competition gives /kaggle/input/<competition-slug>, but a
    community mirror lands under its own name and some are nested a level
    deeper. Search by content instead of assuming a path.

    Depth is capped deliberately: train_images holds ~2 million .dcm files, so
    an rglob would crawl for minutes.
    """
    roots = [Path("/kaggle/input"), Path("data/rsna"), Path("data"), Path(".")]
    patterns = ["train_series_descriptions.csv",
                "*/train_series_descriptions.csv",
                "*/*/train_series_descriptions.csv"]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            found += [h.parent for h in sorted(root.glob(pat))]
    # Prefer a directory that also carries the images.
    with_images = [d for d in found if (d / "train_images").is_dir()]
    return (with_images or found or [None])[0]

DATA_DIR = find_data_dir()
if DATA_DIR is None:
    attached = sorted(p.name for p in Path("/kaggle/input").glob("*")) \\
        if Path("/kaggle/input").exists() else []
    raise SystemExit(
        "Could not find train_series_descriptions.csv.\\n"
        f"  Attached inputs: {attached or 'none'}\\n"
        "  Add Input -> Competitions -> 'RSNA 2024 Lumbar Spine Degenerative "
        "Classification'.\\n"
        "  You must have accepted the competition rules for it to appear."
    )

IMAGES = DATA_DIR / "train_images"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("results")
WORK.mkdir(parents=True, exist_ok=True)

print("data :", DATA_DIR)
print("work :", WORK)
if not IMAGES.is_dir():
    print("\\nWARNING: no train_images/ beside the CSVs - the DICOMs are not attached.")
else:
    print("studies on disk:", sum(1 for p in IMAGES.iterdir() if p.is_dir()))''')

    yield nbf.v4.new_markdown_cell(
        "## 1. Paired studies\n\n"
        "Studies carrying both a sagittal T2/STIR and an axial T2 series. Where a "
        "study has several candidates, the one with the most label coordinates is "
        "taken - that is the series the graders read."
    )
    yield nbf.v4.new_code_cell('''SAG_T2, AXIAL_T2 = "Sagittal T2/STIR", "Axial T2"

series = pd.read_csv(DATA_DIR / "train_series_descriptions.csv")
coords = pd.read_csv(DATA_DIR / "train_label_coordinates.csv")
ann_counts = coords.groupby("series_id").size().to_dict()

def pick(rows):
    if rows.empty:
        return None
    return int(max(rows.series_id, key=lambda s: (ann_counts.get(s, 0), -s)))

pairs = []
for study_id, rows in series.groupby("study_id"):
    sag, axi = pick(rows[rows.series_description == SAG_T2]), pick(rows[rows.series_description == AXIAL_T2])
    if sag and axi:
        pairs.append({"study_id": int(study_id), "sag_series_id": sag, "ax_series_id": axi})

pairs = pd.DataFrame(pairs).sort_values("study_id").reset_index(drop=True)
print(f"studies with both a sagittal T2 and an axial T2: {len(pairs)}")

N_STUDIES = 25
selected = pairs.sample(n=min(N_STUDIES, len(pairs)), random_state=42).sort_values("study_id")
selected = selected.reset_index(drop=True)
selected.to_csv(WORK / "paired_studies.csv", index=False)
selected.head()''')

    yield nbf.v4.new_markdown_cell(
        "### Slice-count audit\n\n"
        "The reason for moving here. A local Kaggle-API download with "
        "`--context_slices 1` yields only slices adjacent to a label coordinate: "
        "**3 sagittal slices** per study (p25 = p50 = p75 = 3), and axial stacks "
        "with gaps in 583/593 series. A 3-slice sagittal slab is ~15 mm thick, so "
        "a vertebral body's left-right centroid inside it is truncation-biased and "
        "Exp 0 would confound motion with the download window.\n\n"
        "This cell confirms the mounted copy is complete."
    )
    yield nbf.v4.new_code_cell('''def slice_stats(df, col, label):
    counts, gapped = [], 0
    for r in df.itertuples():
        d = IMAGES / str(r.study_id) / str(getattr(r, col))
        inst = sorted(int(f.stem) for f in d.glob("*.dcm")) if d.is_dir() else []
        if not inst:
            continue
        counts.append(len(inst))
        gapped += (inst[-1] - inst[0] + 1) != len(inst)
    c = np.array(counts)
    print(f"{label:9s} n={len(c):3d}  slices min={c.min():3d} p25={np.percentile(c,25):5.1f} "
          f"median={np.median(c):5.1f} p75={np.percentile(c,75):5.1f} max={c.max():3d}   "
          f"non-contiguous={gapped}")
    return c

sag_counts = slice_stats(selected, "sag_series_id", "sagittal")
ax_counts  = slice_stats(selected, "ax_series_id",  "axial")

if np.median(sag_counts) <= 5:
    print("\\nWARNING: sagittal series still look truncated - Exp 0 would be unreliable.")
else:
    print(f"\\nFull series confirmed. Sagittal median {np.median(sag_counts):.0f} slices "
          f"(was 3 on the truncated local copy).")''')

    yield nbf.v4.new_markdown_cell(
        "## 2. Geometry verification\n\n"
        "The three checks the spec requires, run on real paired series. Nothing "
        "downstream is trustworthy until these pass.\n\n"
        "Geometry is per-slice, never one affine for the series: most axial lumbar "
        "stacks here are angled per disc level, so a single affine is wrong rather "
        "than merely imprecise."
    )
    yield nbf.v4.new_code_cell('''from composition.geometry import SeriesGeometry, overlap_box

RTOL, ATOL_MM = 1e-12, 1e-9
failures, rows = [], []

for r in selected.itertuples():
    sd, ad = IMAGES / str(r.study_id) / str(r.sag_series_id), IMAGES / str(r.study_id) / str(r.ax_series_id)
    if not (any(sd.glob("*.dcm")) and any(ad.glob("*.dcm"))):
        continue
    sag, axi = SeriesGeometry.from_dir(sd), SeriesGeometry.from_dir(ad)

    # check 1 - extent matches FOV x matrix size, corners coplanar
    for g in list(sag) + list(axi):
        c = g.corner_centres()
        if abs((c[2]-c[0]) @ g.col_cosine - (g.rows-1)*g.row_spacing) > RTOL * 1e3:
            failures.append((r.study_id, "extent-row"))
        if abs((c[1]-c[0]) @ g.row_cosine - (g.cols-1)*g.col_spacing) > RTOL * 1e3:
            failures.append((r.study_id, "extent-col"))
        if np.abs((c - g.position) @ g.normal).max() > ATOL_MM:
            failures.append((r.study_id, "coplanar"))

    # check 2 - one point resolved through two slices agrees
    for geom in (sag, axi):
        if len(geom) < 2:
            continue
        a, b = geom[0], geom[len(geom)-1]
        rng = np.random.default_rng(0)
        for _ in range(32):
            p = a.voxel_to_patient(rng.uniform(0, a.rows-1), rng.uniform(0, a.cols-1)) \\
                + rng.uniform(-8, 8) * a.normal
            ra, ca, oa = a.patient_to_voxel(p); rb, cb, ob = b.patient_to_voxel(p)
            Ra = a.voxel_to_patient(ra, ca) + oa * a.normal
            Rb = b.voxel_to_patient(rb, cb) + ob * b.normal
            if max(np.linalg.norm(Ra-p), np.linalg.norm(Rb-p), np.linalg.norm(Ra-Rb)) > ATOL_MM:
                failures.append((r.study_id, "two-slice"))
                break

    # check 3 - the two acquisitions overlap in patient space
    box = overlap_box(sag, axi)
    if box is None or np.any(box[1] - box[0] <= 10.0):
        failures.append((r.study_id, "overlap"))

    ss, aa = sag.summary(), axi.summary()
    ext = (box[1] - box[0]) if box else np.zeros(3)
    rows.append(dict(study=r.study_id, sag_n=ss["n_slices"], ax_n=aa["n_slices"],
                     sag_th=ss["thickness_mm"], ax_th=aa["thickness_mm"],
                     ax_angled=aa["angled"], ax_groups=aa["n_orientation_groups"],
                     ovl_x=round(ext[0],1), ovl_y=round(ext[1],1), ovl_z=round(ext[2],1)))

audit = pd.DataFrame(rows)
print(f"checked {len(audit)} studies -> {'ALL PASS' if not failures else f'{len(failures)} FAILURES'}")
if failures:
    print(sorted(set(failures)))
print(f"angled axial stacks: {int(audit.ax_angled.sum())}/{len(audit)}")
audit.head(10)''')

    yield nbf.v4.new_markdown_cell(
        "### Measurement floor\n\n"
        "One synthetic object imaged with two dissimilar geometries, zero motion by "
        "construction. Whatever discrepancy Exp 0 reports here is pure "
        "discretisation - the floor a real result has to clear."
    )
    yield nbf.v4.new_code_cell('''from composition.geometry import SliceGeometry
from composition.volume import Volume
from composition.geometry import SeriesGeometry
from experiments.exp0_motion import landmark_centroid

BLOB_CENTRE, BLOB_SIGMA = np.array([3.7, -41.3, -228.9]), 6.0

def phantom(plane, n, rs, cs, th, gap, rows=96, cols=96):
    if plane == "sagittal":
        rc, cc, stack = np.array([0.,1.,0.]), np.array([0.,0.,-1.]), np.array([1.,0.,0.])
    else:
        rc, cc, stack = np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])
    slices, pixels = [], []
    for k in range(n):
        origin = (BLOB_CENTRE + (k - (n-1)/2) * gap * stack
                  - cc * rs * (rows-1)/2 - rc * cs * (cols-1)/2)
        g = SliceGeometry(instance_number=k+1, position=origin, row_cosine=rc, col_cosine=cc,
                          row_spacing=rs, col_spacing=cs, thickness=th, rows=rows, cols=cols)
        rr, ccc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        d2 = ((g.voxel_to_patient(rr, ccc) - BLOB_CENTRE)**2).sum(axis=-1)
        slices.append(g); pixels.append(np.exp(-d2/(2*BLOB_SIGMA**2)).astype(np.float32))
    return Volume(SeriesGeometry(slices, 1, 2), pixels)

sag_p, ax_p = phantom("sagittal",17,.55,.55,4.0,4.4), phantom("axial",21,.31,.31,3.5,3.8)
for R in (8.0, 12.0, 16.0):
    out = {}
    for name, v in (("sag", sag_p), ("ax", ax_p)):
        g = v.geometry[len(v)//2]
        rr, cc, _ = g.patient_to_voxel(BLOB_CENTRE)
        out[name] = landmark_centroid(v, g.instance_number, int(round(rr)), int(round(cc)), R)["centroid_mm"]
    print(f"R={R:>5} mm   floor = {np.linalg.norm(out['sag']-out['ax']):.4f} mm")''')

    yield nbf.v4.new_markdown_cell(
        "## 3. Experiment 0 - landmark marking\n\n"
        "Mark the **same vertebral body** in both acquisitions - the body, not "
        "the disc, and near its centre in all three axes. L4 is a good default: "
        "mid-lumbar and well inside axial coverage in every study.\n\n"
        "**Use the fallback marker below, not this one.** Click marking needs "
        "ipympl, whose frontend extension (`jupyter-matplotlib`) does not load "
        "on Kaggle - you get *Failed to load model class MPLCanvasModel* and no "
        "images. `pip install ipympl` does not fix it; the JS extension is never "
        "registered with the frontend. This cell is kept only for running "
        "elsewhere, e.g. local Jupyter.\n\n"
        "The RSNA label coordinates cannot substitute for a manual mark: they "
        "sit on canal and subarticular points at *disc* levels, not body "
        "centres. They are drawn as faint blue crosses for orientation only."
    )
    yield nbf.v4.new_code_cell('''%matplotlib widget
import importlib.util

# Without ipympl the canvas model never registers and the frontend reports
# "Error displaying widget: model not found" - unhelpful, so say it plainly.
if importlib.util.find_spec("ipympl") is None:
    raise SystemExit(
        "ipympl is not installed, so click events cannot work.\\n"
        "  Either: !pip install ipympl   then Run > Restart & Clear, re-run from cell 2\\n"
        "  Or:     skip this cell and use the typing marker below (needs nothing extra)."
    )

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from composition.volume import Volume

LANDMARKS = WORK / "landmarks.csv"
COLUMNS = ["study_id", "plane", "series_id", "instance_number", "row", "col"]
N_TO_MARK = 20

def _hints(series_id):
    out = {}
    for r in coords[coords.series_id == series_id].itertuples():
        out.setdefault(int(r.instance_number), []).append((float(r.x), float(r.y)))
    return out

class Marker:
    """Slider + click marking for one series."""

    def __init__(self, volume, title, hints):
        self.v, self.hints, self.mark = volume, hints, None
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.fig.canvas.header_visible = False
        self.im = self.ax.imshow(volume.pixels[len(volume)//2], cmap="gray")
        self.dot, = self.ax.plot([], [], "o", color="#ff3b30", ms=9, mec="white", mew=1.2)
        self.crosses, = self.ax.plot([], [], "+", color="#4da3ff", ms=10, alpha=.55, ls="none")
        self.ax.set_axis_off(); self.title = title
        self.slider = widgets.IntSlider(value=len(volume)//2, min=0, max=len(volume)-1,
                                        description="slice", continuous_update=False)
        self.slider.observe(lambda c: self.show(c["new"]), names="value")
        self.fig.canvas.mpl_connect("button_press_event", self.click)
        self.show(self.slider.value)

    def show(self, k):
        self.k = k
        px, g = self.v.pixels[k], self.v.geometry[k]
        self.im.set_data(px); self.im.set_clim(np.percentile(px,1), np.percentile(px,99.5))
        h = self.hints.get(g.instance_number, [])
        self.crosses.set_data(*zip(*h)) if h else self.crosses.set_data([], [])
        self.ax.set_title(f"{self.title}\\nslice {k+1}/{len(self.v)} (instance {g.instance_number})"
                          f"  -  {'marked' if self.mark else 'click the vertebral body centre'}",
                          fontsize=9)
        self.fig.canvas.draw_idle()

    def click(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        g = self.v.geometry[self.k]
        r, c = int(round(e.ydata)), int(round(e.xdata))
        if 0 <= r < g.rows and 0 <= c < g.cols:
            self.mark = (self.k, r, c)
            self.dot.set_data([c], [r]); self.show(self.k)

    def result(self):
        if self.mark is None:
            return None
        k, r, c = self.mark
        return {"series_id": int(self.v.series_id),
                "instance_number": int(self.v.geometry[k].instance_number), "row": r, "col": c}

done = pd.read_csv(LANDMARKS) if LANDMARKS.exists() else pd.DataFrame(columns=COLUMNS)
complete = {s for s, g in done.groupby("study_id") if set(g.plane) >= {"sagittal", "axial"}}
todo = [r for r in selected.itertuples() if r.study_id not in complete][:N_TO_MARK]
print(f"{len(complete)} already marked, {len(todo)} to go")

state = {"i": 0, "markers": {}}
box = widgets.VBox([])
status = widgets.HTML()

def load(i):
    r = todo[i]
    state["markers"] = {}
    panels = []
    for plane, sid in (("sagittal", r.sag_series_id), ("axial", r.ax_series_id)):
        v = Volume.from_dir(IMAGES / str(r.study_id) / str(sid))
        m = Marker(v, f"{r.study_id} - {plane}", _hints(int(sid)))
        state["markers"][plane] = m
        panels.append(widgets.VBox([m.slider, widgets.Output()]))
        with panels[-1].children[1]:
            display(m.fig.canvas)   # ipympl renders the canvas, not the figure
    status.value = f"<b>[{i+1}/{len(todo)}] study {r.study_id}</b>"
    box.children = [status, widgets.HBox(panels), save_btn]

def save(_):
    global done
    got = {p: m.result() for p, m in state["markers"].items()}
    if any(v is None for v in got.values()):
        status.value += " &nbsp; <span style='color:#c00'>mark BOTH planes first</span>"
        return
    r = todo[state["i"]]
    fresh = pd.DataFrame([{**got[p], "study_id": int(r.study_id), "plane": p} for p in got])[COLUMNS]
    done = pd.concat([done, fresh]).drop_duplicates(subset=["study_id","plane"], keep="last")
    done.to_csv(LANDMARKS, index=False)
    plt.close("all")
    state["i"] += 1
    if state["i"] >= len(todo):
        box.children = [widgets.HTML(f"<b>Done - {len(done)//2} studies in {LANDMARKS}</b>")]
    else:
        load(state["i"])

save_btn = widgets.Button(description="Save & next", button_style="success")
save_btn.on_click(save)

if todo:
    load(0)
    display(box)
else:
    print("nothing to mark")''')

    yield nbf.v4.new_markdown_cell(
        "### Marker (use this one)\n\n"
        "Inline images, core ipywidgets only - nothing that needs ipympl.\n\n"
        "Set **slice** first, then type **row** / **col**; the crosshair follows as "
        "you type. The live readout under the panels converts both marks to patient "
        "coordinates and reports the gap, so a mismatch is visible before you save "
        "rather than after twenty studies.\n\n"
        "Reading the gap: one lumbar vertebra is about 35 mm tall, so a gap near "
        "that size almost always means the two panes are on **different vertebrae**. "
        "That is the error worth catching - it would swamp the motion signal Exp 0 "
        "is trying to measure."
    )
    yield nbf.v4.new_code_cell('''%matplotlib inline
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from composition.volume import Volume
from composition.geometry import SeriesGeometry, plane_trace
from experiments.exp0_motion import landmark_centroid

LANDMARKS = WORK / "landmarks.csv"
COLUMNS = ["study_id", "plane", "series_id", "instance_number", "row", "col"]
N_TO_MARK = 20
RADIUS_MM = 8.0          # must match the value cell 21 measures with

def _hints(series_id):
    out = {}
    for r in coords[coords.series_id == int(series_id)].itertuples():
        out.setdefault(int(r.instance_number), []).append((float(r.x), float(r.y)))
    return out

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

def disc_anchors(study_id, series_id):
    """World-space centre of each labelled disc level, from THIS series alone.

    Each acquisition is labelled by its own annotations. Pooling sagittal and
    axial points would assume they agree in space, which is the very thing
    Exp 0 is testing.
    """
    g = SeriesGeometry.from_dir(IMAGES / str(int(study_id)) / str(int(series_id)))
    by_inst = {sl.instance_number: sl for sl in g}
    rows = coords[(coords.study_id == int(study_id)) & (coords.series_id == int(series_id))]
    out = {}
    for lv, grp in rows.groupby("level"):
        pts = [by_inst[int(r.instance_number)].voxel_to_patient(r.y, r.x)
               for r in grp.itertuples() if int(r.instance_number) in by_inst]
        if pts:
            out[lv] = np.mean(pts, axis=0)
    return out

def level_of(point, anchors):
    """Name the level a point sits at, using the disc ladder along S-I."""
    have = [lv for lv in LEVELS if lv in anchors]
    if len(have) < 2:
        return "?"
    z = {lv: anchors[lv][2] for lv in have}
    ordered = sorted(have, key=lambda lv: -z[lv])
    pz = point[2]
    if pz > z[ordered[0]]:
        return "above " + ordered[0]
    if pz < z[ordered[-1]]:
        return "below " + ordered[-1]
    for a, b in zip(ordered, ordered[1:]):
        if z[b] <= pz <= z[a]:
            frac = (z[a] - pz) / max(z[a] - z[b], 1e-6)
            if frac <= 0.2:
                return a + " disc"
            if frac >= 0.8:
                return b + " disc"
            return b.split("/")[0] + " body"
    return "?"

def panel(volume, title, hints, on_change, overlay=None):
    n, g0 = len(volume), volume.geometry[0]
    sl = widgets.IntSlider(value=n // 2, min=0, max=n - 1, description="slice",
                           continuous_update=False)
    rw = widgets.IntText(value=g0.rows // 2, description="row")
    cl = widgets.IntText(value=g0.cols // 2, description="col")
    out = widgets.Output()

    def draw(*_):
        with out:
            clear_output(wait=True)
            k = sl.value
            px, g = volume.pixels[k], volume.geometry[k]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(px, cmap="gray",
                      vmin=np.percentile(px, 1), vmax=np.percentile(px, 99.5))
            h = hints.get(g.instance_number, [])
            if h:
                ax.plot(*zip(*h), "+", color="#4da3ff", ms=10, alpha=.55, ls="none")
            ax.axhline(rw.value, color="#ff3b30", lw=.7)
            ax.axvline(cl.value, color="#ff3b30", lw=.7)
            ax.plot([cl.value], [rw.value], "o", color="#ff3b30", ms=8, mec="white", mew=1.2)

            # Where the current axial slice cuts this image. Two planes meet in
            # a line, so this shows which vertebra the axial slice passes
            # through - a direct check that needs no distance threshold.
            ov = overlay() if overlay else None
            if ov is not None:
                tr = plane_trace(g, ov)
                if tr is not None:
                    (r0, c0), (r1, c1) = tr
                    ax.plot([c0, c1], [r0, r1], "-", color="#ffd60a", lw=1.6, alpha=.9)
                    ax.text(c1, r1, f" axial {ov.instance_number}", color="#ffd60a",
                            fontsize=7, va="center", ha="right")

            ax.set_xticks(np.arange(0, g.cols, 50)); ax.set_yticks(np.arange(0, g.rows, 50))
            ax.tick_params(labelsize=6); ax.grid(color="#4da3ff", alpha=.25, lw=.4)
            ax.set_title(f"{title}\\nslice {k+1}/{n} (instance {g.instance_number})", fontsize=9)
            plt.show()
            plt.close(fig)          # inline keeps every figure alive otherwise
        on_change()

    for w in (sl, rw, cl):
        w.observe(draw, names="value")
    draw()

    def get():
        return {"series_id": int(volume.series_id),
                "instance_number": int(volume.geometry[sl.value].instance_number),
                "row": int(rw.value), "col": int(cl.value)}

    def point_mm():
        # The Exp 0 quantity, not the raw voxel centre. A marked voxel is pinned
        # to its own slice plane, so comparing raw centres reports the gap
        # between the two planes rather than any real mismatch. The weighted
        # centroid averages over a sphere spanning several slices and escapes
        # that, which is exactly why Exp 0 uses it.
        return landmark_centroid(volume, volume.geometry[sl.value].instance_number,
                                 int(rw.value), int(cl.value), RADIUS_MM)["centroid_mm"]

    def current_slice():
        return volume.geometry[sl.value]

    return widgets.VBox([sl, rw, cl, out]), get, point_mm, draw, current_slice

done = pd.read_csv(LANDMARKS) if LANDMARKS.exists() else pd.DataFrame(columns=COLUMNS)
complete = {s for s, g in done.groupby("study_id") if set(g.plane) >= {"sagittal", "axial"}}
todo = [r for r in selected.itertuples() if r.study_id not in complete][:N_TO_MARK]
print(f"{len(complete)} already marked, {len(todo)} to go")

state = {"i": 0, "get": {}, "mm": {}, "anchors": {}}
shell, status, readout = widgets.VBox([]), widgets.HTML(), widgets.HTML()
save_btn = widgets.Button(description="Save & next", button_style="success")

def refresh(*_):
    if set(state["mm"]) < {"sagittal", "axial"}:
        return
    try:
        s, a = state["mm"]["sagittal"](), state["mm"]["axial"]()
    except Exception as err:
        readout.value = f"<span style='color:#c00'>centroid failed: {err}</span>"
        return
    d = s - a
    gap = float(np.linalg.norm(d))
    lv_s = level_of(s, state["anchors"].get("sagittal", {}))
    lv_a = level_of(a, state["anchors"].get("axial", {}))

    if lv_s != "?" and lv_a != "?" and lv_s != lv_a:
        colour, verdict = "#c00", f"DIFFERENT levels: {lv_s} vs {lv_a}"
    elif gap < 15:
        colour, verdict = "#1a7f37", f"good - both on {lv_s}"
    elif gap < 30:
        colour, verdict = "#9a6700", "same level, but drifting - recentre"
    else:
        colour, verdict = "#c00", "likely DIFFERENT vertebrae"

    sense = "inferior to" if d[2] < 0 else "superior to"
    readout.value = (
        f"<div style='font-family:monospace;font-size:13px'>"
        f"sagittal  {lv_s:<12s} ({s[0]:+7.1f}, {s[1]:+7.1f}, {s[2]:+7.1f}) mm<br>"
        f"axial     {lv_a:<12s} ({a[0]:+7.1f}, {a[1]:+7.1f}, {a[2]:+7.1f}) mm<br>"
        f"delta (sagittal - axial)  L {d[0]:+6.1f}   P {d[1]:+6.1f}   S {d[2]:+6.1f}<br>"
        f"<span style='color:#666'>sagittal mark is {abs(d[2]):.1f} mm {sense} the axial mark</span><br>"
        f"<b style='color:{colour}'>gap {gap:6.1f} mm - {verdict}</b>"
        f"<br><span style='color:#666'>weighted centroids, r={RADIUS_MM:.0f} mm - same measure cell 21 reports</span></div>")

def load(i):
    r = todo[i]
    # Sagittal is built first so it can trace whichever axial slice is current.
    vols = {p: Volume.from_dir(IMAGES / str(int(r.study_id)) / str(int(sid)))
            for p, sid in (("sagittal", r.sag_series_id), ("axial", r.ax_series_id))}
    sids = {"sagittal": r.sag_series_id, "axial": r.ax_series_id}

    w_s, get_s, mm_s, draw_s, _ = panel(
        vols["sagittal"], f"{r.study_id} - sagittal", _hints(int(sids["sagittal"])),
        refresh, overlay=lambda: state.get("ax_slice"))

    def axial_changed():
        state["ax_slice"] = state["ax_current"]() if state.get("ax_current") else None
        draw_s()                     # retrace the line on the sagittal pane
        refresh()

    w_a, get_a, mm_a, _, cur_a = panel(
        vols["axial"], f"{r.study_id} - axial", _hints(int(sids["axial"])), axial_changed)
    state["ax_current"] = cur_a
    state["ax_slice"] = cur_a()
    draw_s()

    panels = [w_s, w_a]
    state["get"] = {"sagittal": get_s, "axial": get_a}
    state["mm"] = {"sagittal": mm_s, "axial": mm_a}
    state["anchors"] = {p: disc_anchors(r.study_id, sids[p]) for p in sids}
    status.value = f"<b>[{i+1}/{len(todo)}] study {r.study_id}</b>"
    shell.children = [status, widgets.HBox(panels), readout, save_btn]
    refresh()

def save(_):
    global done
    r = todo[state["i"]]
    fresh = pd.DataFrame([{**state["get"][p](), "study_id": int(r.study_id), "plane": p}
                          for p in state["get"]])[COLUMNS]
    done = pd.concat([done, fresh]).drop_duplicates(subset=["study_id", "plane"], keep="last")
    done.to_csv(LANDMARKS, index=False)
    state["i"] += 1
    if state["i"] >= len(todo):
        shell.children = [widgets.HTML(f"<b>Done - {len(done)//2} studies in {LANDMARKS}</b>")]
    else:
        load(state["i"])

save_btn.on_click(save)
if todo:
    load(0); display(shell)
else:
    print("nothing left to mark")''')

    yield nbf.v4.new_markdown_cell(
        "## 4. Experiment 0 - measurement\n\n"
        "Intensity-weighted centroid of a sphere around each mark, computed in each "
        "acquisition's own voxels, converted to patient coordinates and compared.\n\n"
        "**Decision gate.** `phase0_experiment_protocol.md` is not in the repo, so "
        "absent the real rule this reports against *median discrepancy > 0.5 x the "
        "larger slice thickness => motion dominates, stop*. Replace `THRESHOLD` with "
        "the protocol's rule when available."
    )
    yield nbf.v4.new_code_cell('''from experiments.exp0_motion import measure_study, summarise, print_summary
import json

RADIUS_MM, THRESHOLD = 8.0, 0.5

if not LANDMARKS.exists():
    raise SystemExit(
        f"No landmarks saved yet at {LANDMARKS}.\\n"
        "  Run cell 17 - or cell 19 if the widget one does not render - and mark\\n"
        "  at least one study. The file is written when you press 'Save & next',\\n"
        "  so nothing exists until a study is completed in BOTH planes."
    )

marks = pd.read_csv(LANDMARKS)
ready = [s for s, g in marks.groupby("study_id") if set(g.plane) >= {"sagittal", "axial"}]
if not ready:
    raise SystemExit(
        f"{LANDMARKS} has rows but no study is marked in both planes.\\n"
        "  Exp 0 compares sagittal against axial, so each study needs both."
    )
print(f"{len(ready)} studies marked in both planes\\n")

results, skipped = [], []

for study_id, g in marks.groupby("study_id"):
    planes = {p: sub.iloc[0] for p, sub in g.groupby("plane")}
    if not {"sagittal", "axial"} <= planes.keys():
        skipped.append((study_id, "needs both planes")); continue
    s, a = planes["sagittal"], planes["axial"]
    try:
        out = measure_study(
            IMAGES / str(study_id) / str(int(s.series_id)),
            IMAGES / str(study_id) / str(int(a.series_id)),
            {"instance_number": s.instance_number, "row": s.row, "col": s.col},
            {"instance_number": a.instance_number, "row": a.row, "col": a.col},
            radius_mm=RADIUS_MM)
    except Exception as err:
        skipped.append((study_id, str(err))); continue
    out["study_id"] = int(study_id); results.append(out)
    print(f"  {study_id}: {out['discrepancy_mm']:6.2f} mm  ({out['ratio_to_thickest']:.2f} x thickest)")

if skipped:
    print("\\nskipped:", skipped)

table = pd.DataFrame(results)
table.to_csv(WORK / "exp0_motion.csv", index=False)
summary = summarise(table); summary["radius_mm"] = RADIUS_MM
print_summary(summary)

median_ratio = summary["ratio_to_thickest"]["median"]
verdict = "MOTION DOMINATES - stop and report" if median_ratio > THRESHOLD else "proceed to Stage 3"
summary["assumed_threshold"] = THRESHOLD
summary["verdict"] = verdict
print(f"\\n>>> median ratio {median_ratio:.2f} vs assumed threshold {THRESHOLD} -> {verdict}")

(WORK / "exp0_motion.summary.json").write_text(json.dumps(summary, indent=2))''')

    yield nbf.v4.new_markdown_cell(
        "## 5. Test-retest - is the landmark even repeatable?\n\n"
        "Exp 0's residual can only be called motion if the landmark can be found "
        "reliably. Measured sensitivity says otherwise: moving the mark 4 mm moves "
        "the centroid 3.8 mm (ratio 0.96), because vertebral marrow is near-uniform "
        "and an 8 mm sphere has no intensity structure to lock onto. Hand error "
        "passes through roughly 1:1.\n\n"
        "So mark everything a second time and measure the repeatability directly.\n\n"
        "**Blind means blind.** Do this on a different day, and do not open the "
        "first pass beforehand. This cell writes to a separate file, never loads "
        "the first one, and shuffles the study order so the sequence gives nothing "
        "away. Re-marking from memory measures recall, not the landmark.\n\n"
        "Run the marker cell above first - this reuses its `panel()`."
    )
    yield nbf.v4.new_code_cell('''RETEST = WORK / "landmarks_retest.csv"

done_rt = pd.read_csv(RETEST) if RETEST.exists() else pd.DataFrame(columns=COLUMNS)
complete_rt = {s for s, g in done_rt.groupby("study_id")
               if set(g.plane) >= {"sagittal", "axial"}}

# Shuffle: marking in the same order as pass one is its own memory cue.
order = selected.sample(frac=1.0, random_state=1234).reset_index(drop=True)
todo_rt = [r for r in order.itertuples() if r.study_id not in complete_rt][:N_TO_MARK]
print(f"{len(complete_rt)} re-marked, {len(todo_rt)} to go   (writing to {RETEST.name})")

state_rt = {"i": 0, "get": {}, "mm": {}, "anchors": {}}
shell_rt, status_rt, readout_rt = widgets.VBox([]), widgets.HTML(), widgets.HTML()
save_rt = widgets.Button(description="Save & next", button_style="warning")

def refresh_rt(*_):
    if set(state_rt["mm"]) < {"sagittal", "axial"}:
        return
    try:
        sp, ap = state_rt["mm"]["sagittal"](), state_rt["mm"]["axial"]()
    except Exception as err:
        readout_rt.value = f"<span style='color:#c00'>{err}</span>"; return
    d = sp - ap
    lv_s = level_of(sp, state_rt["anchors"].get("sagittal", {}))
    lv_a = level_of(ap, state_rt["anchors"].get("axial", {}))
    ok = (lv_s == lv_a) and lv_s != "?"
    readout_rt.value = (
        f"<div style='font-family:monospace;font-size:13px'>"
        f"sagittal {lv_s} &nbsp; axial {lv_a}<br>"
        f"<b style='color:{'#1a7f37' if ok else '#c00'}'>"
        f"gap {np.linalg.norm(d):.1f} mm</b></div>")

def load_rt(i):
    r = todo_rt[i]
    sids = {"sagittal": r.sag_series_id, "axial": r.ax_series_id}
    vols = {p: Volume.from_dir(IMAGES / str(int(r.study_id)) / str(int(sid)))
            for p, sid in sids.items()}
    w_s, get_s, mm_s, draw_s, _ = panel(
        vols["sagittal"], f"{r.study_id} - sagittal", _hints(int(sids["sagittal"])),
        refresh_rt, overlay=lambda: state_rt.get("ax_slice"))

    def ax_changed():
        state_rt["ax_slice"] = state_rt["ax_current"]() if state_rt.get("ax_current") else None
        draw_s(); refresh_rt()

    w_a, get_a, mm_a, _, cur_a = panel(
        vols["axial"], f"{r.study_id} - axial", _hints(int(sids["axial"])), ax_changed)
    state_rt["ax_current"] = cur_a; state_rt["ax_slice"] = cur_a(); draw_s()
    state_rt["get"] = {"sagittal": get_s, "axial": get_a}
    state_rt["mm"] = {"sagittal": mm_s, "axial": mm_a}
    state_rt["anchors"] = {p: disc_anchors(r.study_id, sid) for p, sid in sids.items()}
    status_rt.value = (f"<b>[RETEST {i+1}/{len(todo_rt)}] study {r.study_id}</b>")
    shell_rt.children = [status_rt, widgets.HBox([w_s, w_a]), readout_rt, save_rt]
    refresh_rt()

def save_rt_fn(_):
    global done_rt
    r = todo_rt[state_rt["i"]]
    fresh = pd.DataFrame([{**state_rt["get"][p](), "study_id": int(r.study_id), "plane": p}
                          for p in state_rt["get"]])[COLUMNS]
    done_rt = pd.concat([done_rt, fresh]).drop_duplicates(subset=["study_id","plane"], keep="last")
    done_rt.to_csv(RETEST, index=False)
    state_rt["i"] += 1
    if state_rt["i"] >= len(todo_rt):
        shell_rt.children = [widgets.HTML(f"<b>Re-marked {len(done_rt)//2} studies</b>")]
    else:
        load_rt(state_rt["i"])

save_rt.on_click(save_rt_fn)
if todo_rt:
    load_rt(0); display(shell_rt)
else:
    print("nothing left to re-mark")''')

    yield nbf.v4.new_markdown_cell(
        "### Test-retest analysis\n\n"
        "Per-axis scatter is the headline, not the magnitude. The two readings:\n\n"
        "- **scatter isotropic, Exp 0 offset still A-P** -> the offset is a real "
        "difference between how the landmark reads in profile and in cross-section\n"
        "- **scatter itself A-P heavy** -> A-P is simply the hard axis to judge, and "
        "the offset is marking bias too\n\n"
        "The predicted residual is not fitted: sagittal and axial marks are "
        "independent, so their per-axis variances add, and Exp 0's residual either "
        "matches that prediction or exceeds it."
    )
    yield nbf.v4.new_code_cell('''from experiments.exp0_retest import (
    mark_deltas, scatter_summary, predicted_between_plane, interpret, print_report)
import json

first, second = WORK / "landmarks.csv", WORK / "landmarks_retest.csv"
if not second.exists():
    raise SystemExit(f"No second pass yet at {second} - run the retest marker above.")

deltas = mark_deltas(pd.read_csv(first), pd.read_csv(second), IMAGES, RADIUS_MM)
if deltas.empty:
    raise SystemExit("no (study, plane) pairs appear in both passes")

summary = scatter_summary(deltas)
predicted = predicted_between_plane(summary)

exp0 = json.loads((WORK / "exp0_motion.summary.json").read_text()) \
    if (WORK / "exp0_motion.summary.json").exists() else {}
residual = (exp0.get("decomposition") or {}).get("residual_median_mm")

reading = interpret(summary, predicted, residual)
print_report(deltas, summary, predicted, reading)

deltas.to_csv(WORK / "exp0_retest.csv", index=False)
(WORK / "exp0_retest.json").write_text(json.dumps(
    {"summary": summary, "predicted": predicted, "reading": reading}, indent=2))
print()
print(f"Wrote {WORK/'exp0_retest.csv'} and {WORK/'exp0_retest.json'}")''')

    yield nbf.v4.new_markdown_cell(
        "---\n\n**Stop here and report Exp 0 before building Stage 3.** The spec makes "
        "this a gate: if motion dominates, the composition model cannot rescue it."
    )


def main():
    nb = nbf.v4.new_notebook(cells=list(_cells()))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(OUT))
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(nb.cells)} cells)")
    print(f"Embedded: {', '.join(EMBED)}")


if __name__ == "__main__":
    main()
