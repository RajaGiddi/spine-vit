# Spine-ViT — analysis plan & running notes

## Main result (anatomy + ordinal, ours)
RSNA, 500 studies (train 350 / val 75 / test 75), 3 seeds {42, 43, 44}, frozen DINOv2
ViT-S/14, sqrt-inverse class weights, 3-epoch-smoothed val-κ selection.

| metric | seed 42 | seed 43 | seed 44 | mean ± std |
|---|---|---|---|---|
| test κ (quadratic) | 0.596 | 0.669 | 0.682 | **0.649 ± 0.037** |
| worst_lvl_acc (n=27) | 0.593 | 0.593 | 0.545 | **0.577 ± 0.022** |
| pathology fp_rate | 0.034 | 0.062 | — | low (over-flagging resolved) |

`worst_lvl_acc = 0.577` is ~3× the chance rate (0.20) — the level-attribution signal is
real and stable across seeds.

## Pre-registered interpretation thresholds (DECIDED BEFORE SEEING ABLATION NUMBERS)
Seed-to-seed κ std ≈ 0.037 (3 seeds/config). For a two-config comparison at 3 seeds each:

- **Δ ≥ 0.09** → convincing; claim the difference.
- **0.06 ≤ Δ < 0.09** → suggestive; report as a trend, not a claim.
- **Δ < 0.05** → **write "no significant difference"** — do NOT reach for it.

Apply the same bands to `worst_lvl_acc` comparisons. This decision is fixed in advance
to avoid post-hoc rationalization of whatever gap appears.

## Sampling-CI caveat (report alongside the seed std)
The seed std (±0.022 for worst_lvl) captures *training* variance, not *test-set* sampling
error. Each point estimate is on 27 pathological test studies, so its binomial CI is
roughly **±0.19**. Report BOTH: "0.577 ± 0.022 across seeds; single-run 95% CI ≈ ±0.19 on
27 studies." Do not present the tight seed std as if it were the confidence on the value.

## LumbarDISC gap — reviewer's first question (draft paragraph)
> Our anatomy+ordinal model reaches quadratic-weighted κ = 0.649 ± 0.037 on held-out
> test, below the LumbarDISC framework's reported κ = 0.765. This gap is expected and
> stems from deliberate scope differences, not the tokenization method: (i) **data scale**
> — 500 studies vs their ~2,700; (ii) **input** — a 2.5D three-slice stack from a single
> sagittal-T2 sequence vs full 3D volumes with multi-sequence input; (iii) **backbone** —
> a frozen natural-image-pretrained DINOv2 ViT-S/14 with a ~1.8M-parameter trainable head
> vs an end-to-end-trained domain-specific network. Our contribution is orthogonal to raw
> grading accuracy: the anatomy-aware tokenizer targets **level attribution**
> (worst-affected-level accuracy = 0.577 ± 0.022, ~3× chance), a clinically decisive axis
> — which level to treat — that prior work does not report. We hold data scale, input
> dimensionality, and backbone fixed across all ablations to isolate the tokenizer's
> effect; we expect the κ gap to narrow with more data and fine-tuning.

## Launch order (finite quota)
Baselines (strips, patches) → pos-enc ablation (learned, none) → ours (auto-skips) →
fine-tuned backbone last (expendable; likely overfits on 350 studies). See
`scripts/run_ablations.sh`.

## Task 1 (MICCAI): learned detector — oracle vs detected
Deployability result: replace GT coordinates with a heatmap detector (frozen DINOv2 +
0.13M DSNT coordinate-regression head), fixed 32px boxes, same splits/seeds.

| Box source | κ | worst_lvl | macro_f1 | Localization (median mm) |
|---|---|---|---|---|
| Oracle    | 0.649 ± 0.031 | 0.542 ± 0.036 | 0.583 ± 0.038 | 0.0 |
| Detected  | 0.627 ± 0.030 | 0.456 ± 0.045 | 0.548 ± 0.053 | 6.84 |
| Gap       | −0.022 (n.s.) | −0.086 (suggestive) | −0.035 | — |

- **κ preserved** (−0.022 < 0.05 band): grading is deployable without oracle coordinates.
- **worst_lvl drops suggestively** (−0.086): level attribution is more localization-
  sensitive than grade prediction, but detected 0.456 is still ~2.3× chance.
- Detector test mm: median 6.84, mean 8.56, <5mm 31%, <10mm 71%. Per-level median:
  L1/L2 10.5 (hardest) · L2/L3 7.3 · L3/L4 6.8 · L4/L5 6.3 · L5/S1 5.5 (best).
- **L1/L2-hardest is a per-level effect, not edge cutoff**: across-level corr(y, err)=−0.97;
  within-level corr(err, edge-distance)≈0 (n.s.). Explanation: less anatomical context
  above the top level to anchor. (Refutes the edge-cutoff sub-hypothesis.)
- **Localization→grading is tolerance-then-threshold, not smooth**: corr(err, grading_acc)
  n.s. (Spearman −0.09); only the >12mm bin drops (0.79 vs ~0.87). With ~15–25mm boxes,
  sub-box localization error doesn't hurt grading — consistent with the tiny κ gap.
  (Caveat: per-study exact-accuracy is majority-Normal-saturated; coarse probe.)

## Task 2 (CAST baseline) + box-size dose-response
cast_crop (independent frozen ResNet-18 crops) vs anatomy (shared-map ROI-Align), oracle
box 32, param-matched (1.85M vs 1.82M), same boxes/splits/seeds/schedule.
- κ: anatomy 0.649±0.031 vs cast_crop 0.597±0.032, **Δ=0.052, p=0.176 (n.s. at 3 seeds)**.
- worst_lvl: 0.542±0.036 vs 0.535±0.117 — tied. Both ≫ strips (0.332) / patches (0.416).
- **Verdict: the extraction mechanism is not an established driver.** Contribution =
  anatomy-LEVEL localization + deployable detector (Task 1), not ROI-Align-vs-crops.
  ACTION: add seeds 45–46 (anatomy+ordinal, cast_crop) to formally resolve the 0.052 gap.

Box-size (oracle vs detected κ): gaps at box 16/24/32/48 = 0.023 / 0.052 / 0.022 / 0.000 —
small, non-monotonic, within noise. Even 16px ≈ 22mm ≫ 6.8mm error, so the disc stays in
the box; the robustness boundary is below ~5px (impractical). Report as "grading robust to
localization error across 16–48px because the receptive field exceeds the error." Mild
oracle optimum at box 24 (0.681); box 48 dilutes with background.

## v2 Task 1 (CORAL ordinal loss) — no benefit, retain CE (PENDING matched 3-seed)
- Reported "kappa" was quadratic-weighted (QWK) all along -> model SELECTION was already
  ordinal-aware, compressing CORAL's expected gain (only the loss was ordering-blind).
- First run had a WEIGHTING BUG (3-class per-sample weights instead of per-threshold
  pos_weight) — squeezed Moderate. FIXED to per-threshold pos_weight [7.06, 20.34].
- Fixed CORAL, seed 42, 20ep: QWK 0.580 (< CE seed-42 0.622), MAE 0.195, **Moderate still
  0/25**. Thresholds monotonic (b0=0.008 >= b1=-0.007) but band width 0.015 -> degenerate.
- CRITICAL CONTRAST: CE (softmax) Moderate recall = 63/131 = 0.48 over 5 seeds; CORAL = 0/25.
  So CORAL specifically collapses Moderate where CE does NOT -> it's a CORAL head property,
  not the data. (Earlier "CE also rarely predicts Moderate" was an UNVERIFIED, wrong claim.)
- DECISION: DROP CORAL. Paper one-liner only: "we also evaluated a CORAL ordinal head; it did
  not improve over cross-entropy with QWK-based selection." Do not formalize (no 3-seed spend),
  do not make a strong shared-weight claim from one seed. CE + QWK-selection is the reference.
- IMPORTANT (NOT a limitation): CE achieves ~0.48 Moderate recall -> the 3-class task is
  legitimate, NOT binary-in-disguise. Do NOT add that caveat; the comparisons are real 3-class.
- Parked (cheap, later): ordinal soft labels (Moderate -> ~[0.1,0.8,0.1]) — one line in the loss.

## v2 Task 2 (AXIAL FUSION) — built + locally verified, PENDING Modal sweep
Central canal stenosis is read on AXIAL T2 clinically; v1 uses sagittal only. Fusion adds
the matching axial ROI per level (canal center = L/R Subarticular midpoint; slice =
annotation instance_number). Shared frozen DINOv2 + shared anatomy ROI-Align tokenizer
encode BOTH views — the ONLY thing separating a sagittal from an axial token at the same
level is a learned view-type embedding (zero-init, must earn it).

VISUAL GATE — PASS (all 20 all-5 overlays + 2 provenance-flagged studies inspected):
- Derived canal center on the central canal in every panel; no anterior offset needed
  (posterior_offset=0 correct). Non-monotonic instance#s (178041181, 29931867, 109454808,
  109677683) are benign — each level's slice is from its OWN series and shows correct-level
  anatomy. Robust across noisy / tight-FOV / pathological studies.

BOX-MM READOUT (pixel parity != physical parity; report mm for both views):
- Sagittal 224-space px -> mm (median): 16->20.0, 24->30.0, 32->40.0.
- Axial   224-space px -> mm (median): 16->14.2, 24->21.3, 32->28.4.
- Axial has finer in-plane spacing/tighter FOV -> same px covers LESS mm. Each view swept
  at its own px; compared at matched PHYSICAL scale, not matched px.

ARCHITECTURE (models/spine_fusion.py, data/rsna_fusion.py, train_fusion.py):
- Modes: views=sag (control) | axial | both+concat (fusion-A) | both+attn (fusion-B).
- Fusion-A concat: per level proj([sag ; axial]) -> 1 fused token; missing axial ->
  learned placeholder. Params 1.952M.
- Fusion-B attn: sag+axial tokens share ONE transformer sequence; self-attn mixes across
  view AND level; per-level readout = sagittal position. Params 1.820M. Masks by ABSENCE
  (no axial token in seq) — no placeholder needed.
- Coverage-masked so sag-only studies/levels still train.

PRE-REGISTRATION (fixed before Modal numbers):
- Headline fusion at axial_box=32 (pixel-matched to sagittal headline). 4-way ablation x3 seeds.
- Axial box sweep {16,24,32} (fusion-B) is a dose-response ROBUSTNESS curve, NOT
  headline-selection. Sagittal box stays 32.
- Report canal metrics TWICE: full test set + axial-available subset (train_fusion.py does this).
- Same interpretation bands as v1 (Δ>=0.09 claim / 0.06-0.09 trend / <0.05 n.s.); 3 seeds min.
- Fusion runs are UN-augmented (both views must transform jointly; deferred) — matched across
  the whole fusion ablation, kept separate from the v1 augmented sagittal headline (0.649).

PRE-MODAL LOCAL VERIFICATION (scripts/fusion_smoke.py, all PASS):
- Forward shapes finite for all 4 modes; per-level logits (N,3); disc_mask all-True.
- Alignment invariant exact (aligned axial == source token at slot; max|err|=0).
- Coverage-masked training: loss falls (attn 1.19->0.29, concat 0.83->0.27); view-embedding
  gets gradient in BOTH modes (attn 2.14e-1, concat 1.95e-1); missing-axial placeholder gets
  gradient in concat (3.47e-2) and correctly NONE in attn (masks by absence).
- train_fusion.py runs end-to-end (train/val/test + dual metrics) for attn & concat.

RESULTS (18 runs, 3 seeds; axial subset = 367/375 tokens = 98% -> subset≈full, no coverage confound):
  ablation (axial_box=32):        κ full        worst_lvl      macro_f1
    sag-only (control, un-aug)   0.637±0.063   0.503±0.068    0.619
    axial-only                   0.662±0.077   0.520±0.029    0.592
    fusion-A (concat)            0.689±0.033   0.675±0.059    0.607
    fusion-B (attn)              0.642±0.069   0.530±0.018    0.599
  box sweep (attn) κ: 16px(14mm) 0.657 / 24px(21mm) 0.638 / 32px(28mm) 0.642 — flat, within noise.

PAIRED (same-seed) deltas vs sag-only control — the honest test (shared splits):
  - κ: NO fusion mode robustly beats sagittal-only. concat Δκ [+0.068,-0.010,+0.096] mean
    +0.051 but SIGN FLIPS on seed 43 -> not a claim. attn +0.004. axial +0.024 (small, consistent).
  - worst_lvl: fusion-A (concat) Δ [+0.296,+0.037,+0.182] mean +0.172, POSITIVE ON ALL 3 SEEDS,
    large on the two hard seeds (42,44). attn/axial sign-flip -> nothing. => the ONE directionally
    robust signal is CONCAT improving LEVEL ATTRIBUTION, not grade κ.

INTERPRETATION (preliminary, pending seeds 45,46):
  - Fusion does not improve grade κ over the sagittal anatomy tokenizer (already captures gradable
    signal). The gain, if any, is on level attribution (the paper's core axis) via SIMPLE concat.
  - SURPRISE 1: concat (fusion-A) >> attention (fusion-B) on worst_lvl. Mechanism hypothesis:
    self-attention mixes tokens across LEVELS, blurring the per-level distinctions worst-level ID
    needs; concat keeps each level's fused token independent. (State as hypothesis, not proven.)
  - SURPRISE 2: un-augmented fusion control has 2x the seed std of the v1 augmented headline
    (0.063 vs 0.031). This inflates all bands. Per-view augmentation would cut variance.

AUGMENTED RE-RUN (12 runs, per-view aug: sag-hflip OFF [A-P mirror invalid], axial-hflip ON
[L-R valid]; box verified to track image under flip). This OVERTURNS the un-augmented reading:
  ablation (aug, axial_box=32):   κ full        worst_lvl      macro_f1
    sag-only (control)           0.659±0.042   0.483±0.061    0.604
    axial-only                   0.698±0.075   0.677±0.033    0.651
    fusion-A (concat)            0.696±0.082   0.652±0.032    0.632
    fusion-B (attn)              0.647±0.033   0.550±0.049    0.594
  PAIRED Δworst_lvl vs sag control: axial [+0.222,+0.148,+0.212] mean +0.194 ALL POS & LARGE;
    concat [+0.222,+0.074,+0.212] +0.169 all pos; attn [+0.074,+0.037,+0.091] +0.067 all pos.
  PAIRED Δκ: axial +0.039, concat +0.038, attn -0.012 — all SIGN-FLIP -> n.s.

REATTRIBUTION (important, like the CORAL episode): the un-augmented sweep MIS-ATTRIBUTED the
worst_lvl gain to CONCAT FUSION. Cause: un-augmented axial-only was UNDERTRAINED (worst_lvl
0.520). With per-view aug, axial-ONLY reaches 0.677 -> the signal is the AXIAL VIEW, not fusion:
  - Axial-only improves LEVEL ATTRIBUTION +0.194 worst_lvl over sagittal-only, positive & large on
    all 3 seeds (3 splits) -> clears the pre-registered CLAIM band. Clinically grounded: canal is
    read on axial T2; axial gives 5 dedicated per-level slices vs sagittal's 1 shared slice.
  - FUSION IS A NULL: concat ties axial-only (0.652 vs 0.677); attn UNDERperforms (0.550) — cross-
    level attention dilutes per-level signal. Two-view fusion does NOT beat the single axial view.
  - Grade κ comparable across all (~0.66-0.70, n.s.) — axial advantage is on WHICH level, not severity.
  - BONUS: v1's invalid sagittal hflip was NOT load-bearing — no-hflip sag control κ 0.659 vs v1 0.649 (n.s.).

v2 Task 2 CONCLUSION (defensible): drop the "fusion helps" framing (unsupported). Headline = AXIAL
ROI tokenization improves clinical level attribution over sagittal (+0.194 worst_lvl, robust),
matching clinical practice; naive two-view fusion adds nothing. The 4-way ablation is what let us
isolate the axial VIEW (not fusion) as the driver — report it as the mechanism check.
CAVEATS: worst_lvl n=27 (wide single-run CI ~±0.19) but robust across 3 seeds; axial-only κ std
high (0.075, seed42 κ 0.598 low); axial "advantage" partly = more per-level pixels (5 vs 1 slice),
state as mechanism not confound.

OPTIONAL bulletproofing (not required — claim already clears bands at 3 seeds): add seeds 45,46 for
sag-only + axial-only (4 runs) to tighten the +0.194 worst_lvl estimate.

--- 5-SEED LOCK + BUDGET CONTROL (why the flip's lesson demanded a NEW experiment) -----------------
The augmented flip was NOT seed noise — it was an UNCONTROLLED PROTOCOL DIFFERENCE (un-augmented
axial undertrained; axial-only lost more to it than concat, which still had the trained sag stream).
Seeds can't catch that; fixing the protocol did. Lesson: a protocol asymmetry silently reassigns the
effect to the wrong config. One asymmetry remained under the headline: AXIAL reads 5 dedicated per-
level slices, SAGITTAL reads 1. Same class of confound -> must control BEFORE publishing a 5-seed #.

AXIAL-ONLY LEVEL-ATTRIBUTION CLAIM, LOCKED AT 5 SEEDS (seeds 42-46, paired vs sag-only control):
  Δworst_lvl = [0.222, 0.148, 0.212, 0.147, 0.185]  mean +0.183  ALL 5 POSITIVE, all in CLAIM band.
  Δκ         = [-0.056, 0.067, 0.106, 0.143, 0.041] mean +0.060  4/5 pos -> TREND only.
  => axial advantage is on WHICH level (attribution), NOT grade severity. 5-seed means:
     sag-only κ 0.618±0.069 wl 0.471±0.057 ; axial-only κ 0.678±0.063 wl 0.654±0.042.

BUDGET CONTROL RESULT (sag_slices=5, parasagittal, mean-pooled; matches axial's 5-slice budget with
PLANE held constant). Matched seeds 42-44, PAIRED (no seed-set confound):
  sag 1-slice wl 0.483 ; sag 5-slice wl 0.575 ; axial wl 0.677.
  BUDGET effect       (sag5-sag1)   = +0.092  [0.111,0.074,0.091] ALL POS  -> 47% of total
  ACQUISITION residual (axial-sag5) = +0.102  [0.111,0.074,0.121] ALL POS  -> 53% of total
  TOTAL               (axial-sag1)  = +0.194
  => ~50/50. NEITHER clean threshold fired. The axial level-attribution advantage DECOMPOSES:
     ~half is INPUT BUDGET (more slices), ~half is AXIAL ACQUISITION (orientation + level-specific
     slice selection; see NAMING). [3-SEED PRELIM — SUPERSEDED by the 5-seed PAIRED-T 2x2 below:
     budget +0.105 p=0.005, acquisition +0.078 p=0.050 (both SIG); anchor total +0.183 p<0.001.]
  CORRECTED CLAIM: not "axial plane +0.19" (47% would be an uncontrolled slice-count artifact) but
  "axial ACQUISITION +0.19, decomposing into budget +0.09 and a slice-matched acquisition residual
  +0.10 (all seeds pos)." BONUS: 5-slice parasagittal sag is a free +0.092 wl, no axial needed.

  NAMING (honest): call the residual factor "axial ACQUISITION", NOT "axial plane". The residual
  bundles orientation WITH annotator-supplied level-specific slice selection (the per-level axial
  instance is human-chosen); we cannot separate orientation from that in this data. Naming the unit
  as the protocol preempts the obvious reviewer question.

  TWO-FACTOR TABLE (central Task 2 artifact; DEFINITIVE = PAIRED t-test, 5 seeds 42-46, UNROUNDED
  deltas. Report p-values NOT the sign count — the sign test gets STRICTER with n and discards
  magnitude, understating the effect):
                        | slice budget (sag5-sag1)      | axial acquisition (axial-sag5)
    severity (kappa)    | +0.037 t=1.17 p=0.31 -> n.s.   | +0.023 t=1.12 p=0.32 -> n.s.
    level attribution   | +0.105 t=5.53 p=0.005 -> SIG   | +0.078 t=2.78 p=0.050 -> SIG
    ANCHOR total (axial-sag1): wl +0.183 t=11.7 p<0.001 ; kappa +0.060 t=1.78 p=0.15 (n.s.)
    (means: sag1 wl 0.471|sag5 0.576|axial 0.654 ; sag1 κ 0.618|sag5 0.656|axial 0.678)
  FRAMING: anchor on the TOTAL (axial-sag1 CANCELS the sag5 term -> cleanest quantity): axial improves
  LEVEL ATTRIBUTION +0.183 (p<0.001), NO effect on severity (κ total n.s.). That total PARTITIONS into
  two SIGNIFICANT components — budget +0.105 (p=0.005) and acquisition +0.078 (p=0.050) — not two
  fragile independent claims. budget capturable in a SAGITTAL-ONLY pipeline (parasagittal), no axial.
  SHARED-TERM ANTICORRELATION (state in paper): budget=sag5-sag1, acquisition=axial-sag5 share sag5
  with OPPOSITE sign -> corr(budget,acq)=-0.84. One strong sag5 seed (45) mechanically inflates budget
  AND deflates acquisition — the seed-45 "acquisition negative" is ONE lucky sag5 seed shown twice,
  not independent evidence. So component-level seed-robustness UNDERSTATES the partition's stability.
  STATS NOTE: paired t is the right test (deltas ~symmetric, no outliers). Wilcoxon floor at n=5 is
  p=0.062 even at 5/5, so it can't reach 0.05 regardless — don't cite its non-significance as evidence.
  CAUTION: aggregator's unpaired κ split is a SEED-SET ARTIFACT (mixes seed sets); ignore it.

FUSION ROWS NOW 5-SEED (whole ablation uniform; the null is load-bearing vs the fusion camp
DeepSPINE/M-SCAN/Park/Shi, so it must not be the sole under-powered cell). 5-seed:
  concat κ 0.697±0.067 wl 0.673±0.036 ; attn κ 0.626±0.055 wl 0.519±0.057.
FUSION NULL STATED WITH POWER (paired vs axial-only, 5 seeds):
  concat-axial: wl +0.019 [-0.067,+0.106] p=0.57 ; κ +0.018 [-0.044,+0.081] p=0.46 -> TIE (powered null).
  attn -axial: wl -0.135 [-0.157,-0.113] p<0.001 -> attention fusion SIGNIFICANTLY WORSE (not "mild").
  => neither fusion beats the better single view: concat ties it (well-bounded null), attn degrades it.
  concat ties axial-only => sagittal token adds nothing UNDER CONCAT FUSION AT THIS DATA SCALE (scope
  the claim; a different mechanism or more data could change it — don't say "redundant" unqualified).
  PARAM COUNTS (refute the capacity hypothesis for attn — compute, don't assume): trainable params
  sag/axial/ATTN all = 1,820,168 (attn adds ZERO over single-view); concat = 1,952,008 (+131,840 via
  concat_proj). So attn underperforms at IDENTICAL capacity to axial-only -> NOT over-parameterization;
  concat ADDS params yet does not degrade. Report counts, rule out capacity, do NOT assert mechanism.

DEEPSPINE (verified from full text) — Lu, Pedemonte, Bizzo, Doyle, Andriole, Michalski, Gonzalez,
Pomerantz. "DeepSPINE: ..." MLHC 2018, PMLR 85:403-419, arXiv:1807.10215 (spotlight).
  Table 2 (class-average ACCURACY, mean±std): Canal Stenosis Axial 78.6±2.7 | Sagittal 78.6±2.4 |
  Both 80.4±1.6.  Foraminal: Ax 76.6±2.5 | Sag 74.3±1.7 | Both 78.1±0.4. Metric = accuracy, NOT κ.
  THREE FULL-TEXT POINTS (matter more than the numbers):
   1. NOT budget-matched: axial 360x360x8 (8 slices, 2D branch); sagittal 160x320x25 (25 slices, 3D
      branch) "because there were more slices in the sagittal disc volumes". Their canal tie = 25-slice
      sag 3D vs 8-slice axial 2D. => their tie is PREDICTED by our decomposition (sag's 3x budget
      offsets axial's acquisition edge -> parity). Our MATCHED-budget control reveals what theirs couldn't.
   2. Both inputs LEVEL-SPECIFIC (per-disc oblique reformats, planes perpendicular to spine-curve
      tangent) -> they can't isolate the acquisition/level-specificity effect either. Supports our
      "acquisition" naming caution.
   3. They anticipated our MODERATE finding: models more accurate on normal+severe than mild+moderate,
      attributed to "higher inter-reader variability for the intermediate grades." => DIRECT support
      for the CEILING/label-saturation argument AND for the Task-1 Moderate-recall discussion. Cite there.
  DRAFT positioning rewritten with exact numbers + the budget-mismatch-supports-decomposition logic.

  DEVICE-MOVE BUG AUDIT (clean): buggy key sag_multi_images read ONLY under `sag_slices>1` guard
  (spine_fusion.py:83) -> only the 3 sag5 rows could touch it; other 16 augmented rows structurally
  immune. Bug was NON-SILENT: pre-fix sag5 runs exited 1 and wrote NO test_results.json (observed),
  so no degraded-but-completed survivor exists. All 3 current sag5 results are from the post-fix 7/7
  rerun. => entire v2 aug table is post-fix (sag5) or bug-immune (rest). No pre-fix number survives.

METHODS BUG CAUGHT (record so it never recurs): CPU smoke MASKS device-move bugs. New batch tensor
`sag_multi_images` was not in train.move_batch()'s fixed key list -> stayed on CPU while the CUDA model
ran -> all 3 sag5 runs died (exit 1) on Modal though the --device cpu smoke passed. Fix: add every new
batch tensor key to move_batch. RULE: any new batch[...] tensor MUST be added to move_batch, and the
local smoke should assert move_batch covers all model batch-tensor reads (added that check).

## OPEN VERIFICATION (before paper submission)
- **LumbarDISC κ=0.765 weighting unknown.** We report QUADRATIC-weighted κ. If their 0.765 is
  linear-weighted or unweighted, our comparison table is biased in our favor — a reviewer will
  catch it. Verify from their paper; if they report linear, either recompute ours as linear for
  that row or footnote the difference. Do NOT ship the table until this is resolved.

## Known limitations to state
- Fixed-extent ROI in resized-pixel space (consistent receptive field, matches strips);
  residual physical-FOV variability from `PixelSpacing` (0.325–1.176 mm/px) is a
  limitations sentence, not a confound. Physical resampling deferred.
- Frozen backbone; single sequence (sagittal T2); spinal-canal-stenosis only.
