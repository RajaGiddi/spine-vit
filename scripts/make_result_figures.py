#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGW = 3.3

PALETTE = ["#0072B2", "#009E73", "#E69F00", "#CFCFCF"]
ACCENT = "#0072B2"
INK = "#222222"

TOKENIZERS = [
    ("Anatomy (ours)", "rsna_anatomy_ordinal_256_2"),
    ("CAST crop", "rsna_cast_crop_ordinal_256_2"),
    ("Patch-query", "rsna_patches_ordinal_256_2"),
    ("Uniform strips", "rsna_strips_ordinal_256_2"),
]
TOK_SEEDS = (42, 43, 44)

PLANE_SEEDS = (42, 43, 44, 45, 46)
SAG1 = "rsna_fusion_sag_ordinal_256_2_aug"
SAG5 = "rsna_fusion_sag_ordinal_256_2_sag5_aug"
AXIAL = "rsna_fusion_axial_ordinal_256_2_aug"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6,
    "axes.linewidth": 0.8, "lines.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 200,
})


PAD = 0.01


def save(fig, stem):
    for _ in range(4):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        err = FIGW - (bb.width + 2 * PAD)
        if abs(err) < 0.002:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + err, h)
    fig.savefig(f"figures/{stem}.pdf", bbox_inches="tight", pad_inches=PAD)
    fig.savefig(f"figures/{stem}.png", bbox_inches="tight", pad_inches=PAD, dpi=200)
    plt.close(fig)


def load_run(experiments_dir, prefix, seed):
    """(kappa, worst_lvl, path) for one run; handles both single- and two-view key schemas."""
    d = os.path.join(experiments_dir, f"{prefix}_s{seed}")
    p = os.path.join(d, "test_results.json")
    tr = json.load(open(p))
    m = tr.get("metrics_full") or tr.get("metrics") or {}
    att = tr.get("attribution_full") or tr.get("attribution") or {}
    return m.get("kappa"), att.get("worst_level_accuracy"), p


def figure3(experiments_dir, log):
    stats_by_tok, srcs = [], []
    for label, prefix in TOKENIZERS:
        ks, ws = [], []
        for s in TOK_SEEDS:
            k, w, p = load_run(experiments_dir, prefix, s)
            ks.append(k); ws.append(w); srcs.append(p)
        stats_by_tok.append((label, np.mean(ks), np.std(ks), np.mean(ws), np.std(ws)))

    fig, ax = plt.subplots(figsize=(FIGW, 2.1))
    n = len(stats_by_tok)
    width = 0.19
    centres = np.array([0.0, 1.12])
    offs = (np.arange(n) - (n - 1) / 2) * width

    for i, (label, km, ks, wm, ws) in enumerate(stats_by_tok):
        x = centres + offs[i]
        first = i == 0
        ax.bar(x, [km, wm], width=width * 0.92, yerr=[ks, ws], color=PALETTE[i],
               edgecolor=INK if first else "none", linewidth=0.7 if first else 0.0,
               error_kw=dict(elinewidth=0.7, capsize=1.4, capthick=0.7, ecolor=INK), zorder=2)

    lo, hi = centres[1] + offs[0] - width * 0.75, centres[1] + offs[-1] + width * 0.75
    ax.plot([lo, hi], [0.20, 0.20], ls=(0, (3, 2)), lw=0.8, color=INK, zorder=3)
    ax.text(lo - 0.03, 0.20, "chance", fontsize=6, ha="right", va="center", color=INK)

    ax.set_ylim(0, 0.75)
    ax.set_yticks(np.arange(0, 0.76, 0.15))
    ax.set_xlim(centres[0] + offs[0] - 0.19, centres[1] + offs[-1] + 0.19)
    ax.set_xticks(np.concatenate([centres[0] + offs, centres[1] + offs]))
    ax.set_xticklabels([t[0] for t in stats_by_tok] * 2, rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=6)
    for c, name in zip(centres, ["Quadratic-weighted $\\kappa$", "Worst-level accuracy"]):
        ax.text(c, -0.42, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.set_ylabel("Score")
    ax.tick_params(length=2.5, pad=1.0)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color="0.9", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    save(fig, "tokenizer_comparison")
    log.append("Figure 3 (figures/tokenizer_comparison.pdf) - seeds "
               f"{', '.join(map(str, TOK_SEEDS))}, mean +/- 1 SD (population):")
    for label, km, ks, wm, ws in stats_by_tok:
        log.append(f"    {label:<16} kappa {km:.3f}+/-{ks:.3f}   worst_lvl {wm:.3f}+/-{ws:.3f}")
    log += [f"    {p}" for p in srcs]


def figure4(experiments_dir, log):
    wl, srcs = {}, []
    for tag, prefix in (("sag1", SAG1), ("sag5", SAG5), ("axial", AXIAL)):
        for s in PLANE_SEEDS:
            _, w, p = load_run(experiments_dir, prefix, s)
            wl[(tag, s)] = w
            srcs.append(p)

    budget = np.array([wl[("sag5", s)] - wl[("sag1", s)] for s in PLANE_SEEDS])
    acquis = np.array([wl[("axial", s)] - wl[("sag5", s)] for s in PLANE_SEEDS])
    cols = [("Slice budget", budget), ("Axial acquisition", acquis)]
    pvals = [stats.ttest_rel(np.array([wl[(b, s)] for s in PLANE_SEEDS]),
                             np.array([wl[(a, s)] for s in PLANE_SEEDS])).pvalue
             for a, b in (("sag1", "sag5"), ("sag5", "axial"))]

    fig, ax = plt.subplots(figsize=(FIGW, 2.36))
    xs = np.array([0.0, 1.0])
    jit = np.linspace(-0.085, 0.085, len(PLANE_SEEDS))

    for j, s in enumerate(PLANE_SEEDS):
        px = xs + jit[j]
        py = [budget[j], acquis[j]]
        ax.plot(px, py, color="0.65", lw=0.7, zorder=1)
        ax.scatter(px, py, s=9, marker="o", color=ACCENT, edgecolors="white",
                   linewidths=0.4, zorder=5)

    ax.axhline(0.0, color=INK, lw=0.8, zorder=1)
    for i, (_, d) in enumerate(cols):
        ax.plot([xs[i] - 0.16, xs[i] + 0.16], [d.mean()] * 2, color=INK, lw=2.0,
                solid_capstyle="butt", zorder=4)

    top = max(budget.max(), acquis.max())
    for i, ((label, d), pv) in enumerate(zip(cols, pvals)):
        ax.text(xs[i], top + 0.028, f"{d.mean():+.3f}, p={pv:.3f}", fontsize=6.5,
                ha="center", va="bottom", color=INK)
    j45 = PLANE_SEEDS.index(45)
    ax.text(xs[0] + jit[j45] - 0.035, budget[j45], "seed 45", fontsize=6, color="0.45",
            ha="right", va="center")

    ax.set_ylim(-0.06, top + 0.075)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cols])
    ax.set_ylabel("$\\Delta$ worst-level accuracy")
    ax.tick_params(length=2.5, pad=1.5)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    save(fig, "plane_decomposition")
    log.append("")
    log.append("Figure 4 (figures/plane_decomposition.pdf) - paired per-seed deltas on worst_lvl, "
               f"seeds {', '.join(map(str, PLANE_SEEDS))}:")
    log.append("    budget (sag5 - sag1):  " + ", ".join(f"s{s}={v:+.3f}" for s, v in zip(PLANE_SEEDS, budget))
               + f"  | mean {budget.mean():+.3f}, p={pvals[0]:.3f}")
    log.append("    acquisition (axial - sag5): " + ", ".join(f"s{s}={v:+.3f}" for s, v in zip(PLANE_SEEDS, acquis))
               + f"  | mean {acquis.mean():+.3f}, p={pvals[1]:.3f}")

    log.append("    leave-one-seed-out (mean, p, 95% CI):")
    for j, s in enumerate(PLANE_SEEDS):
        keep = [i for i in range(len(PLANE_SEEDS)) if i != j]
        parts = []
        for nm, d in (("budget", budget), ("acquisition", acquis)):
            dk = d[keep]
            t = stats.ttest_1samp(dk, 0.0)
            ci = stats.t.ppf(0.975, len(dk) - 1) * dk.std(ddof=1) / np.sqrt(len(dk))
            parts.append(f"{nm} {dk.mean():+.3f} p={t.pvalue:.3f} [{dk.mean()-ci:+.3f}, {dk.mean()+ci:+.3f}]")
        log.append(f"      drop s{s}: " + "  |  ".join(parts))
    log += [f"    {p}" for p in srcs]


CURVE_PREFIX = "rsna_anatomy_ordinal_256_2"
SMOOTH = 3


def _smooth(y, w=SMOOTH):
    """Centred moving average, shrinking at the edges so the curve keeps its endpoints."""
    y = np.asarray(y, dtype=float)
    return np.array([y[max(0, i - w // 2):i + w // 2 + 1].mean() for i in range(len(y))])


def figure_curves(experiments_dir, log, seed=None):
    survey = {}
    for d in sorted(glob.glob(os.path.join(experiments_dir, f"{CURVE_PREFIX}_s*"))):
        cf = json.load(open(os.path.join(d, "config.json")))
        tr = json.load(open(os.path.join(d, "test_results.json")))
        h = json.load(open(os.path.join(d, "history.json")))
        m = tr.get("metrics_full") or tr.get("metrics") or {}
        survey[int(cf["seed"])] = dict(kappa=m["kappa"], best_epoch=tr["best_epoch"],
                                       n_epochs=len(h["train_loss"]), cap=cf["epochs"],
                                       hist=h, dir=d)
    mean_k = float(np.mean([v["kappa"] for v in survey.values()]))

    if seed is None:
        modal_n = max({v["n_epochs"] for v in survey.values()},
                      key=lambda n: sum(v["n_epochs"] == n for v in survey.values()))
        cands = [s for s, v in survey.items() if v["n_epochs"] == modal_n] or list(survey)
        seed = min(cands, key=lambda s: abs(survey[s]["kappa"] - mean_k))
    r = survey[seed]
    h, best, n = r["hist"], r["best_epoch"], r["n_epochs"]
    early = n < r["cap"]
    ep = np.arange(1, n + 1)
    tr_l, va_l = np.array(h["train_loss"], dtype=float), np.array(h["val_loss"], dtype=float)
    val_min = int(va_l.argmin()) + 1

    fig, ax = plt.subplots(figsize=(FIGW, 2.17))
    for y, c, ls, lab in ((tr_l, ACCENT, "-", "Train loss"), (va_l, "#E69F00", (0, (4, 2)), "Val loss")):
        ax.plot(ep, y, color=c, ls=ls, lw=0.8, alpha=0.22, zorder=2)
        ax.plot(ep, _smooth(y), color=c, ls=ls, lw=0.9, zorder=3, label=lab)

    lo_y, hi_y = min(tr_l.min(), va_l.min()), max(tr_l.max(), va_l.max())
    pad = 0.06 * (hi_y - lo_y)
    ax.set_ylim(lo_y - pad, hi_y + 2.2 * pad)

    ax.axvline(best, color=INK, ls=(0, (2, 2)), lw=0.7, zorder=4)
    ax.text(best - 0.6, hi_y + 1.5 * pad, "selected", fontsize=6, ha="right", va="center", color=INK)

    ax.set_xlim(1, n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.tick_params(length=2.5, pad=1.5)
    ax.yaxis.grid(True, color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, handletextpad=0.4,
              borderpad=0.0, borderaxespad=0.3)

    save(fig, "training_curves")
    log.append("")
    log.append(f"Figure 5 (figures/training_curves.pdf) - {os.path.basename(r['dir'])}, "
               f"seed {seed}; config {r['dir']}/config.json, curves {r['dir']}/history.json")
    log.append(f"    seed survey (5-seed mean test kappa {mean_k:.4f}):")
    for s in sorted(survey):
        v = survey[s]
        mark = "  <- picked" if s == seed else ""
        log.append(f"      s{s}: test_kappa {v['kappa']:.4f} ({v['kappa']-mean_k:+.4f} vs mean)  "
                   f"best_epoch {v['best_epoch']}  epochs {v['n_epochs']}/{v['cap']}  "
                   f"early_stop={'yes' if v['n_epochs'] < v['cap'] else 'no'}{mark}")
    log.append(f"    epochs are 1-indexed: train.py loops range(1, epochs+1), so best_epoch "
               f"in test_results.json is the 1-based epoch number (history index best_epoch-1)")
    log.append(f"    selected checkpoint epoch {best} (3-epoch smoothed val kappa, = config select_window)")
    log.append(f"    early stopping {'triggered' if early else 'did NOT trigger'}: "
               f"{n} of {r['cap']} epochs ran; final epoch {n-1} "
               f"(recorded here but deliberately NOT marked on the figure - the curves ending "
               f"says training stopped, and the marker was clutter at 84 mm)")
    log.append(f"    train loss {tr_l[0]:.4f} -> {tr_l[-1]:.4f} (min {tr_l.min():.4f}); "
               f"val loss {va_l[0]:.4f} -> {va_l[-1]:.4f}, minimum {va_l.min():.4f} at epoch {val_min}")
    log.append(f"    NOTE: val loss bottoms at epoch {val_min} then rises while train loss keeps falling - "
               f"the checkpoint at epoch {best} sits {best-val_min} epochs past the val-loss minimum")
    log.append(f"    SMOOTHING APPLIED: raw curves drawn faint (alpha 0.22) behind a centred "
               f"{SMOOTH}-epoch moving average; val loss is epoch-to-epoch noisy and the trend is "
               f"otherwise hard to read at 84 mm")
    log.append("    single panel used: train and val loss share a scale "
               f"({min(tr_l.min(), va_l.min()):.2f}-{max(tr_l.max(), va_l.max()):.2f}), so the "
               "two-panel variant was not needed (histories record no train kappa, only val)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_dir", default="outputs_modal")
    ap.add_argument("--curve_seed", type=int, default=None, help="override the training-curve seed")
    args = ap.parse_args()

    os.makedirs("figures", exist_ok=True)
    log = []
    figure3(args.experiments_dir, log)
    figure4(args.experiments_dir, log)
    figure_curves(args.experiments_dir, log, seed=args.curve_seed)
    print("\n".join(log))
    with open("figures/results_figure_log.txt", "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
