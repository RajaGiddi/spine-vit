import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.rsna_dataset import RSNADataset, build_rsna_index, LEVELS

ACCENT = "#1d4ed8"
RIGHT = "#16a34a"
WRONG = "#dc2626"
GT = "#f59e0b"
GRADES = ["Norm", "Mod", "Sev"]
FIGW = 3.3

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7,
    "axes.linewidth": 0.8, "lines.linewidth": 0.8,
    "pdf.fonttype": 42, "savefig.dpi": 200,
})


def save(fig, stem):
    fig.savefig(f"figures/{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"figures/{stem}.png", bbox_inches="tight", pad_inches=0.01, dpi=200)
    plt.close(fig)


def plain_axes(ax, img):
    lo, hi = np.percentile(img, [2, 98])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")


NEUTRAL = "#334155"


def label(ax, x, y, text, color, ha="right", alpha=0.85):
    ax.text(x, y, text, color=color, fontsize=5, ha=ha, va="center", zorder=5,
            bbox=dict(facecolor="white", alpha=alpha, pad=0.4, edgecolor="none"))


class Data:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        idx = build_rsna_index(data_dir, "stenosis")
        self.ds = RSNADataset(data_dir, samples=idx, augment=False, box_size=32,
                              image_size=224, use_25d=True, box_source="oracle")
        self.pos = {sample["study_id"]: i for i, sample in enumerate(self.ds.samples)}

    def slice_boxes(self, study):
        i = self.pos[study]
        item = self.ds[i]
        img = item["image"].numpy()[1]
        return img, item["boxes"].numpy(), item["level_indices"].numpy(), item["targets"].numpy(), self.ds.samples[i]

    def dims(self, samp):
        path = os.path.join(self.data_dir, "train_images", str(samp["study_id"]),
                         str(samp["series_id"]), f"{samp['instance_number']}.dcm")
        dicom = pydicom.dcmread(path, stop_before_pixels=True)
        ps = getattr(dicom, "PixelSpacing", None)
        rs, cs = (float(ps[0]), float(ps[1])) if ps is not None else (1.0, 1.0)
        return int(dicom.Rows), int(dicom.Columns), rs, cs


def figure1(data, log, study=None):
    if study is None:
        spread = []
        for sample in data.ds.samples:
            if len(sample["levels"]) == 5:
                x_values = [level["x"] for level in sample["levels"]]
                spread.append((max(x_values) - min(x_values), sample["study_id"]))
        spread.sort()
        study = spread[int(0.65 * len(spread))][1]
    img, boxes, lvls, _, _ = data.slice_boxes(study)
    width = img.shape[1]
    height = img.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=(FIGW, 1.35))
    plain_axes(ax[0], img)
    for box, li in zip(boxes, lvls):
        x1, y1, x2, y2 = box
        ax[0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=0.8, ec=ACCENT, fc="none"))
        label(ax[0], width - 2, (y1 + y2) / 2, LEVELS[li], ACCENT, alpha=0.5)
    ax[0].set_title("(a) Anatomy-aware", fontsize=7)

    plain_axes(ax[1], img)
    band_height = height / 5
    for k in range(6):
        y = min(max(k * band_height, 0.7), height - 0.7)
        ax[1].plot([0, width], [y, y], color=ACCENT, lw=0.7, ls=(0, (3, 2)))
    for k in range(5):
        label(ax[1], width - 2, (k + 0.5) * band_height, LEVELS[k], ACCENT, alpha=0.5)
    ax[1].set_title("(b) Uniform strips", fontsize=7)

    plain_axes(ax[2], img)
    for k in range(0, 225, 14):
        ax[2].axhline(k, color="0.7", lw=0.3, alpha=0.5)
        ax[2].axvline(k, color="0.7", lw=0.3, alpha=0.5)
    ax[2].set_title("(c) Patch grid", fontsize=7)

    fig.tight_layout(pad=0.2, w_pad=0.4)
    save(fig, "tokenization_comparison")
    log.append(f"Figure 1 study: {study}")


def load_predictions_by_study(pred_path):
    tp = json.load(open(pred_path))
    out = {}
    for sid, level, pr, tg in zip(tp["studyids"], tp["levels"], tp["preds"], tp["targets"]):
        out.setdefault(sid, {})[level] = (pr, tg)
    return out, list(dict.fromkeys(tp["studyids"]))


def grade_box(ax, cx, cy, color, half=16, lw=0.9):
    ax.add_patch(patches.Rectangle((cx - half, cy - half), 2 * half, 2 * half, lw=lw, ec=color, fc="none"))


def figure2(data, log, pred_path, det_path):
    byid, order = load_predictions_by_study(pred_path)
    det = json.load(open(det_path))

    def argworst(per_level, key):
        def grade_at_level(level):
            return per_level[level][key]

        return max(sorted(per_level), key=grade_at_level)

    succ = next((sample for sample in order if sample in data.pos and len(byid[sample]) == 5
                 and all(pred_grade == true_grade for pred_grade, true_grade in byid[sample].values())
                 and max(true_grade for _, true_grade in byid[sample].values()) >= 1), None)
    mis = next((sample for sample in order if sample in data.pos
                and max(true_grade for _, true_grade in byid[sample].values()) >= 1
                and max(pred_grade for pred_grade, _ in byid[sample].values()) >= 1
                and any(pred_grade >= 1 and true_grade >= 1 for pred_grade, true_grade in byid[sample].values())
                and argworst(byid[sample], 0) != argworst(byid[sample], 1)), None)

    worst_l12 = None
    for sample in order:
        if sample not in data.pos or str(sample) not in det:
            continue
        samp_i = data.pos[sample]
        gt = {level["level_idx"]: (level["x"], level["y"]) for level in data.ds.samples[samp_i]["levels"]}
        if 0 not in gt or "0" not in det[str(sample)]:
            continue
        _, _, rs, cs = data.dims(data.ds.samples[samp_i])
        dx = (det[str(sample)]["0"][0] - gt[0][0]) * cs
        dy = (det[str(sample)]["0"][1] - gt[0][1]) * rs
        mm = float(np.hypot(dx, dy))
        if worst_l12 is None or mm > worst_l12[1]:
            worst_l12 = (sample, mm)

    fig, ax = plt.subplots(1, 3, figsize=(FIGW, 1.5))

    img, boxes, lvls, tgts, _ = data.slice_boxes(succ)
    width = img.shape[1]
    plain_axes(ax[0], img)
    for box, li in zip(boxes, lvls):
        x1, y1, x2, y2 = box
        pred_grade, true_grade = byid[succ][li]
        grade_box(ax[0], (x1 + x2) / 2, (y1 + y2) / 2, RIGHT)
        txt = f"{LEVELS[li]}: {GRADES[pred_grade]}"
        if pred_grade != true_grade:
            txt = txt + f" ({GRADES[true_grade]})"
        label(ax[0], width - 2, (y1 + y2) / 2, txt, RIGHT)
    ax[0].set_title("(a) Correct", fontsize=7)

    img, boxes, lvls, tgts, _ = data.slice_boxes(mis)
    width = img.shape[1]
    plain_axes(ax[1], img)
    true_w, pred_w = argworst(byid[mis], 1), argworst(byid[mis], 0)
    for box, li in zip(boxes, lvls):
        x1, y1, x2, y2 = box
        pred_grade, true_grade = byid[mis][li]
        worst = li in (true_w, pred_w)
        box_c = RIGHT if li == true_w else (WRONG if li == pred_w else "0.6")
        txt_c = RIGHT if li == true_w else (WRONG if li == pred_w else NEUTRAL)
        grade_box(ax[1], (x1 + x2) / 2, (y1 + y2) / 2, box_c, lw=1.2 if worst else 0.7)
        txt = f"{LEVELS[li]}: {GRADES[pred_grade]}"
        if pred_grade != true_grade:
            txt = txt + f" ({GRADES[true_grade]})"
        label(ax[1], width - 2, (y1 + y2) / 2, txt, txt_c)
    ax[1].set_title("(b) Under Grading", fontsize=7)

    sample, mm = worst_l12
    img, _, _, _, samp = data.slice_boxes(sample)
    width = img.shape[1]
    h0, w0, _, _ = data.dims(samp)
    sx, sy = 224 / w0, 224 / h0
    plain_axes(ax[2], img)
    gt = {level["level_idx"]: (level["x"], level["y"]) for level in samp["levels"]}
    gx, gy = gt[0][0] * sx, gt[0][1] * sy
    dx, dy = det[str(sample)]["0"][0] * sx, det[str(sample)]["0"][1] * sy
    ax[2].plot([gx, dx], [gy, dy], color=WRONG, lw=0.8)
    ax[2].scatter([gx], [gy], sample=10, c=GT, marker="o", zorder=3, edgecolors="none")
    ax[2].scatter([dx], [dy], sample=10, c=ACCENT, marker="x", zorder=3, linewidths=0.9)
    label(ax[2], min(width - 2, max(gx, dx) + 4), (gy + dy) / 2, f"{mm:.0f} mm", WRONG, ha="left")
    ax[2].set_title("(c) Detector failure", fontsize=7)

    fig.tight_layout(pad=0.2, w_pad=0.4)
    save(fig, "qualitative")
    log += [f"Figure 2a (correct): {succ}", f"Figure 2b (misattribution): {mis}",
            f"Figure 2c (detector fail L1/L2): {sample}  ({mm:.1f} mm)"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/rsna")
    parser.add_argument("--pred", default="outputs_modal/rsna_anatomy_ordinal_256_2_b48_s42/test_predictions.json")
    parser.add_argument("--detector", default="outputs/detector/detected_centers.json")
    parser.add_argument("--fig1_study", type=int, default=None)
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    data = Data(args.data_dir)
    log = []
    figure1(data, log, study=args.fig1_study)
    figure2(data, log, args.pred, args.detector)
    print("\n".join(log))
    with open("figures/selection_log.txt", "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
