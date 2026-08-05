# Work log: results figures and tokenizer significance testing

Session of 2026-08-04. Covers three new paper figures, a paired-significance harness, and the
findings that came out of it. No paper text was changed.

## 1. Code

### `scripts/make_result_figures.py` (new, 313 lines)

Figures 3-5. Every number is read from the per-run `test_results.json` / `config.json` /
`history.json` tree; nothing is hardcoded, so the figures track the runs.

| Component | Notes |
|---|---|
| shared rcParams | sans-serif, 7 pt ticks, 8 pt labels, 0.8 pt lines, no top/right spines, `pdf.fonttype 42` |
| `PALETTE` | Okabe-Ito ordered dark to light so series separate in greyscale |
| `load_run()` | tolerates both JSON schemas: `metrics`/`attribution` (single-view) and `metrics_full`/`attribution_full` (two-view) |
| `save()` | iteratively fits the tight bbox to exactly 84 mm, then writes vector PDF + 200 dpi PNG |
| `figure3()` | tokenizer comparison: grouped bars, ±1 SD, chance line, direct per-bar labels |
| `figure4()` | plane decomposition: paired dot plot, per-seed lines, p-values computed at render time |
| `figure_curves()` | training curves: seed survey + auto-pick, 3-epoch smoothing over faint raw curves |
| log | `figures/results_figure_log.txt`: source paths, seed survey, smoothing disclosure, jackknife |

`save()` matters more than it looks: tight-cropping alone produced 76-84 mm PDFs, so each figure
would have been rescaled by a different factor at `\columnwidth` and the declared point sizes
would not have been the ones on the page.

### `scripts/tokenizer_stats.py` (new, 129 lines)

Paired-significance harness. Discovers seeds from `config.json` rather than assuming them, prints
per-tokenizer aggregates (κ, worst-level, macro F1), runs every comparison at both the 3-seed and
full shared-seed set, and reports which verdicts change between the two. Written for the 5-seed
rerun; not yet run against new data.

### `modal_run.py` (+23 lines)

Added the `resolve_patch_strips()` entrypoint (patches and strips at seeds 45, 46) mirroring the
existing `resolve_cast()`. Launched but did not execute: the Modal workspace is over its spend
limit. No compute ran and nothing was charged.

### Outputs

`figures/tokenizer_comparison.{pdf,png}`, `figures/plane_decomposition.{pdf,png}`,
`figures/training_curves.{pdf,png}`, `figures/results_figure_log.txt`. Separately reran the
existing `scripts/make_figures.py` after the `"(b) Under Grading"` title edit.

Reproduce:

```
./.venv/bin/python scripts/make_result_figures.py      # figures 3-5 + log
./.venv/bin/python scripts/tokenizer_stats.py          # all paired tests
./.venv/bin/python scripts/make_figures.py --data_dir data/rsna   # figures 1-2
```

## 2. Statistical findings

### 2.1 Seed 45 is conservative, not inflationary

The plane decomposition's acquisition p = 0.050 is determined by seed 45, but in the direction
that *weakens* the claim. Leave-one-seed-out on worst-level accuracy:

| dropped | budget | acquisition |
|---|---|---|
| none (all 5) | +0.105, p=0.005 | +0.078, p=0.050 |
| s42 | +0.104, p=0.024 | +0.069, p=0.138 |
| s43 | +0.113, p=0.015 | +0.079, p=0.118 |
| s45 | +0.088, p=0.002 | +0.104, p=0.002, CI [+0.071, +0.137] |
| s46 | +0.113, p=0.015 | +0.069, p=0.138 |

Dropping s45 moves acquisition from +0.078 (p=0.050) to +0.104 (p=0.002) and tightens the CI
from [+0.000, +0.155] to [+0.071, +0.137]. Four of five seeds agree; s45 is the lone dissenter.

Why s45 is atypical: the seed drives the split, and s45's test set carries 34 studies with
pathology (54 pathological levels) against 27 for s42/s43/s46 and 33 for s44, the most skewed of
the five.

Consequence for the draft: the `r = -0.84` anticorrelation figure in `task2_writeup.md` is a
one-seed artifact. Without s45, `r = +0.49`. The a-priori "anticorrelated by construction"
argument still holds; the quoted empirical number does not.

### 2.2 Tokenizer comparisons (3 seeds, paired)

| comparison | metric | mean Δ | 95% CI | p | verdict |
|---|---|---|---|---|---|
| anatomy - strips | κ | +0.3172 | [+0.173, +0.461] | 0.0110 | separable |
| anatomy - strips | worst-lvl | +0.2189 | [+0.126, +0.312] | 0.0096 | separable |
| anatomy - patch-query | κ | +0.2327 | [+0.014, +0.452] | 0.0447 | separable (marginal) |
| anatomy - patch-query | worst-lvl | +0.1841 | [-0.184, +0.552] | 0.1643 | not separable |
| anatomy - CAST (n=5) | κ | +0.0785 | [+0.025, +0.132] | 0.0153 | separable |
| anatomy - CAST (n=3) | κ | +0.0519 | [-0.013, +0.117] | 0.0756 | not separable |
| anatomy - CAST (n=5) | worst-lvl | +0.0319 | [-0.119, +0.183] | 0.5882 | not separable (signs 2/5) |
| patch-query - strips | κ | +0.0845 | [-0.164, +0.333] | 0.2809 | not separable |
| patch-query - strips | worst-lvl | +0.0348 | [-0.241, +0.311] | 0.6421 | not separable |

Notes:

- The paper's "strips underperformed by 0.084" does not survive. The 0.084 is real (it equals
  the mean paired delta exactly), but seed 42 reverses sign on both metrics. Soften to "did not
  outperform", while avoiding any implication of *equivalence*, since at n=3 the CI spans ±0.25
  κ, wider than the entire anatomy-to-strips gap. This is an underpowered null.
- The "gaps are 7-10x the seed SD" heuristic does not survive either. It uses per-config SD
  instead of the SD of paired deltas, and predicts neither outcome: anatomy vs CAST on κ passes
  at n=5 with a gap of only 0.078 (all five deltas positive, tight), while anatomy vs patch-query
  on worst-level fails with a gap of 0.184. Replace it with the tests.
- The "expected p ≈ 0.176" for anatomy vs CAST is the *unpaired* Welch test on 3-seed κ
  (p = 0.1786). The paired 5-seed test is the correct analysis and clears at p = 0.0153.

### 2.3 Patch-query variance

Per-seed κ 0.3210 / 0.5592 / 0.3679. The spread is real, but not independent noise:

| tokenizer | s42 | s43 | s44 | SD(n-1) | s43 vs own mean |
|---|---|---|---|---|---|
| anatomy | 0.6218 | 0.6923 | 0.6319 | 0.038 | +0.0436 |
| CAST-crop | 0.5521 | 0.6283 | 0.6101 | 0.040 | +0.0314 |
| patch-query | 0.3210 | 0.5592 | 0.3679 | 0.126 | +0.1432 |
| strips | 0.3491 | 0.3964 | 0.2491 | 0.075 | +0.0649 |

All four tokenizers peak at seed 43. The splits are shared within a seed, so this is common-mode
split difficulty that patch-query amplifies ~3x, which is why pairing helps and unpaired tests
run weaker. Patch-query's s43 run is independently suspect: its best *validation* κ was 0.359,
the lowest of its three runs, yet it produced the highest *test* κ (0.559). Split luck, not a
better model. The variance is numerically 3.3x anatomy's but not statistically established at
n=3 (F=10.95, p=0.167; Levene across all four W=0.569, p=0.651).

### 2.4 Seed-count audit

`rsna_anatomy_ordinal_256_2` has five seed dirs (42-46), all with complete `config.json` /
`history.json` / `test_results.json`, all differing from s42 only in `seed`. Any table listing
three seeds for anatomy undercounts. The real constraint is patch-query and strips, which
genuinely have only seeds 42-44, hence every comparison involving them sits at n=3 with
t_crit 4.30.

### 2.5 Training curves (Figure 5)

Seed 44 chosen: modal run shape (35 epochs, checkpoint at 25, shared with s42) and closest of any
seed to the 5-seed mean κ (0.6319 vs 0.6369). Seeds 45 (ran to the 40-epoch cap) and 46 (stopped
at 20) are the atypical runs.

The figure shows something stronger than the draft claims. Validation loss does not plateau; it
bottoms at 0.4526 at epoch 20, then rises to 0.5232 while train loss keeps falling to 0.2661.
The selected checkpoint at epoch 25 therefore sits 5 epochs past the validation-loss minimum,
because selection tracks smoothed val κ, not val loss. Seed 42 shows the same pattern more
starkly (loss minimum at epoch 11, checkpoint at 25).

Smoothing is applied and disclosed: raw curves at alpha 0.22 behind a centred 3-epoch moving
average, the same window `select_window=3` that checkpoint selection itself uses.

## 3. Open items

1. Modal spend limit. Four runs (patches, strips at seeds 45, 46) are staged behind
   `resolve_patch_strips()` and blocked. Raise the limit, then rerun; `tokenizer_stats.py` and a
   one-line `TOK_SEEDS` change in `make_result_figures.py` complete the 5-seed analysis.
2. Code-drift assumption. The existing runs' configs lack the `head` key that current
   `default.yaml` carries, so they predate the committed code. Every read site uses
   `config.get("head", "ce")`, so it is provably inert, but git cannot bound what *else* moved,
   because every commit carries the identical timestamp `2026-07-31 18:05`. Pooling new s45/s46
   runs with old s42-44 assumes equivalent code. A reproducibility control (rerun seed 42 under a
   distinct `--experiment_name`, confirm κ=0.3210) would verify it, at one extra run and no extra
   wall-clock.
3. Paper text edits, authorized but not applied: "plateaus" to "degrades" and the caption
   selection note. The draft is not in this repo (no `.tex` files, and no markdown contains the
   sentence). The only overfitting mention is `paper_notes.md:48`, a different claim about the
   fine-tuned backbone.
4. Values needing update once 5-seed numbers land: the tokenizer table, the "underperformed by
   0.084" claim, the "7-10x seed SD" justification, the `r = -0.84` caveat, and any seed-count
   column showing three for anatomy.
5. Figure 2b predicate. The panel now titled "Under Grading" still selects on *misattribution*
   (`argworst(preds) != argworst(targets)`), and its box colours encode predicted-worst vs
   true-worst. The rendered case does show under-grading, but title, predicate, and colour
   encoding have diverged.
