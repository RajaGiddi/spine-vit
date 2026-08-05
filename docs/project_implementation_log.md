# Spine-ViT implementation log, start to present

What was built, in the order it was built, and what each piece is for. Numbers here are the
committed ones from `paper_notes.md` / `task2_writeup.md`, verified against the run tree.
The most recent work (results figures, paired significance testing) is in
[worklog_figures_and_stats.md](worklog_figures_and_stats.md).

Core idea: replace ViT patch tokenization with ROI-Align pooling from vertebral-level regions,
so each token *is* an anatomical level (L1/L2 through L5/S1). The payoff metric is
worst-affected-level accuracy, i.e. which level to treat, which prior work does not report.

## 1. Data acquisition and preprocessing

### `rsna_setup.py` (534 lines)

Kaggle download harness for RSNA 2024 / LumbarDISC. Credential loading from `.env`, HTTP error
explanation, retrying file-level download, and the part that matters scientifically,
`select_studies()`, which picks a 500-study subset at a fixed seed and reports its grade
distribution so the subset isn't silently skewed. Writes subset CSVs and an ID list.

### `data/rsna_dataset.py` (376 lines), the preprocessing core

| Stage | Implementation |
|---|---|
| DICOM to array | `load_dicom_slice()`: rescale slope/intercept, then DICOM window/level if present, else min-max. Handles `MultiValue` window tags. |
| Study to sample | `build_rsna_index()`: parses three CSVs, filters to sagittal T2, picks the series with the most canal annotations, then the most-annotated instance as the representative slice. |
| Partial-download robustness | Skips studies whose image folder is absent *and* whose exact representative slice is missing, so a rate-limited partial download degrades gracefully instead of crashing in `__getitem__`. |
| 2.5D input | Annotated slice plus its two neighbours stacked as pseudo-RGB; falls back to repeating the centre slice when neighbours are absent or shape-mismatched. |
| Point to box | `coord_to_box()`: fixed-extent box centred on the annotation. Resize first, then place a fixed 32-px box in resized space, so every ROI has an identical model receptive field, removing the variable-extent confound from RSNA's ~3.6x spread in pixel spacing. |
| Normalization | Per-image z-score, applied *after* augmentation. |
| Splits | `make_rsna_splits()`: patient-level 70/15/15 by `study_id`, seeded, so no study leaks across splits. |
| Collate | `rsna_collate_fn()`: variable level counts to `boxes (N_total,5)` with a leading batch-index column for ROI-Align, plus `num_levels` list. |

Grading targets come from `train.csv` (`Normal/Mild`/`Moderate`/`Severe` to 0/1/2), with
`IGNORE_INDEX = -1` for missing labels. All RSNA tokens are discs (`level_type == 1`).

### `data/transforms.py` (130 lines)

`SpineAugmentation` with one hard invariant: every geometric transform applied to the image is
applied identically to the boxes. Vertical shift, multiplicative intensity jitter, horizontal
flip (mirrors box x), random crop-and-resize (offsets then rescales boxes), additive Gaussian
noise. Colour jitter deliberately omitted, meaningless for grayscale MRI. Boxes are clamped and
re-ordered at the end so `x1<=x2`, `y1<=y2`.

### `data/spider_dataset.py` (398 lines)

Secondary dataset: SPIDER NIfTI/`.mha` volumes with GT segmentation masks, 5-class Pfirrmann
grading, interleaved vertebra + disc tokens (this is why `level_type` exists at all).

## 2. Model

### `models/backbone.py`, feature extraction

`DINOv2Backbone` (ViT-S/14, 384-dim, patch 14) returns a spatial feature map for ROI-Align.
Frozen by default: parameters `requires_grad=False`, `train()` overridden to keep the module in
eval mode regardless of the parent's mode, and forward wrapped in `no_grad`. `MockBackbone` is a
tiny random CNN with the identical output contract, so the whole stack can be tested offline
with no weights download and no network.

### `models/tokenizer.py`, the ablation centrepiece

Four tokenizers, one shared contract
(`forward(feature_map, boxes, level_indices, num_levels) -> (N_total, embed_dim)`), so they are
drop-in interchangeable and everything downstream is held fixed:

| Tokenizer | Mechanism | What it isolates |
|---|---|---|
| AnatomyTokenizer (ours) | ROI-Align on the precise per-level boxes from a shared DINOv2 map | n/a |
| UniformStripTokenizer | ROI-Align on K equal full-width horizontal strips, ignoring box x/y | does *precise* localization matter, or just ordering? |
| PatchTokenizer | learned per-level query cross-attends over all patch tokens | does structured localization help at all? Keeps per-level outputs so attribution stays comparable |
| CASTCropTokenizer | crops each ROI from the original image, encodes each independently with a frozen ImageNet ResNet-18 | is the *shared feature map* the novelty, or just the boxes? |

All four end in the same pool -> Linear -> LayerNorm -> GELU projection, so token dimensionality
and downstream capacity match.

### `models/encoder.py`, positional encodings and the transformer

Three positional-encoding variants sharing a module shape:

- `OrdinalPositionalEncoding`: learned embedding over ordinal level position, `std=0.02` init
  (a smooth ordered ramp).
- `LearnedIdentityEncoding`: structurally identical, `std=0.10` init so levels start as distinct
  identities rather than an order. The CAST-style framing baseline.
- `NoPositionalEncoding`: zeros.

`AnatomyEncoder` adds positional + type embedding (0=vertebra, 1=disc), packs the flat
`(N_total, D)` stream into padded `(B, max_K, D)` with a key-padding mask, runs a pre-norm
transformer (`norm_first=True`), and unpacks back to flat.

`forward_with_attention()` reproduces the pre-norm layer math by hand (`norm1`, self-attention
with `need_weights=True`, residual, `norm2`, FFN, residual) because the built-in
`TransformerEncoderLayer` forces `need_weights` off. Returns per-layer head-averaged
level x level attention for the overlay figures.

### `models/heads.py`

`GradingHeads` filters to disc tokens, then applies a task-specific MLP. Two output layers:
standard `Linear` for cross-entropy, or `CoralLayer`, a single shared weight vector with K-1
independent thresholds, so thresholds stay monotonic and ordinal predictions are coherent.

### `models/spine_grader.py` / `models/spine_fusion.py`

`SpineGrader` assembles backbone, tokenizer, encoder, heads for the single-view case.

`SpineFusionGrader` extends it to two views with a shared backbone and a shared anatomy
tokenizer, so the only thing distinguishing a sagittal from an axial token at the same level is
a learned view-type embedding, zero-initialized, so it starts as a no-op and must earn its
contribution. Four modes:

- `views="sag"`: sagittal control.
- `views="axial"`: axial only, missing levels get a learned placeholder.
- `both` + `concat` (fusion-A): per level, `proj([sag ; axial])` gives one fused token.
  1.952 M params.
- `both` + `attn` (fusion-B): sag and axial tokens share one transformer sequence;
  self-attention mixes across view *and* level; per-level readout from the sagittal position.
  1.820 M params, zero over single-view.

Fusion is coverage-masked: axial coverage is partial (~94% any level, ~90% all-5), so a level
with no axial slice contributes a placeholder (concat) or simply no token (attn), and every
study still trains on its sagittal tokens. Output contract matches `SpineGrader` exactly, so
`train.py`'s epoch loop is reused unchanged.

`_sag_tokens_multi()` implements the parasagittal budget control: mean-pool the per-level token
over K slices, producing the same `(N_total, D)` output so only slice budget changes, not
architecture.

### `models/detector.py`

`DiscHeatmapDetector`: the same frozen DINOv2, plus a ~0.13 M-param decoder (1x1 reduce,
bilinear upsample to 56x56, two convs) predicting one heatmap per disc level. Bilinear
interpolate + conv rather than transposed conv, to avoid checkerboard artifacts. Emits raw
logits; decoding is spatial-softmax + soft-argmax, which gives sub-pixel centres and a
differentiable coordinate loss.

## 3. Two-view data plumbing

### `data/rsna_axial.py`

RSNA annotates Left/Right Subarticular Stenosis on axial series. The canal centre is derived as
the midpoint of the two subarticular points, and the axial slice is that annotation's
`instance_number`, so each level comes from its own angled stack.

The module surfaces its own risks rather than hiding them: `posterior_offset` for
anterior/posterior placement (default 0), `axial_monotonicity_flags()` to catch levels whose
instance numbers are non-monotonic within a shared series, and a single-sided fallback flag.
`axial_box_mm()` converts a 224-space box to physical mm, because pixel parity is not physical
parity across views.

A visual gate was run before trusting any of it: all 20 all-5-level overlays plus the 2
provenance-flagged studies were inspected. Derived centre landed on the canal in every panel,
`posterior_offset=0` confirmed correct, non-monotonic cases confirmed benign.

Box mm readout (median): sagittal 16/24/32 px gives 20.0/30.0/40.0 mm; axial gives
14.2/21.3/28.4 mm. Each view was swept at its own pixel size and compared at matched *physical*
scale.

### `data/rsna_fusion.py`

`RSNAFusionDataset` extends the sagittal dataset with a per-level axial ROI and an explicit
alignment contract: `axial_slot` maps each sagittal token, in flat order, to its axial token's
row in the batched stack, or -1. Per-view augmentation is a correctness point, not a tuning
knob: sagittal hflip is disabled (mirroring anterior-posterior is anatomically invalid), axial
hflip is enabled (left-right is valid).

## 4. Training

`train.py` (355) / `train_fusion.py` (215) / `train_detector.py` (209).

- Loss: weighted cross-entropy with `ignore_index=-1`, class weights from training targets
  (`sqrt_inverse` default). CORAL path swaps in per-threshold `pos_weight` and the CORAL decode.
- Optimizer: AdamW on trainable params only, cosine-annealed to 1e-6.
- Checkpoint selection: a 3-epoch trailing moving average of validation κ, not the raw value, so
  a single lucky epoch cannot win. At this validation size that is what makes selection
  defensible. Early stopping on patience 10, 40-epoch cap.
- Epoch indexing: the loop is `range(1, epochs+1)`, so `best_epoch` in `test_results.json` is
  1-based (history index `best_epoch - 1`).
- Modal (`modal_run.py`, 362): every (config, seed) as a parallel container on L4 (A100 for the
  fine-tuned variant), results to a persistent volume, `--skip_if_done` making sweeps resumable.
  Entry points: `main`, `fusion`, `fusion_aug`, `fusion_budget`, `fusion_fus_ext`,
  `detector_pipeline`, `coral`, `resolve_cast`, `box_size_sweep`, `resolve_patch_strips`.

### `tests/`

`test_forward.py` (9 forward-pass combinations, shapes + param counts) and `test_train_loop.py`
(synthetic loss-decreases + metrics sanity), both on `MockBackbone` so they run offline with no
data. `scripts/fusion_smoke.py` (170) additionally verifies the alignment invariant exactly
(`max|err| = 0`), that loss falls in both fusion modes, that the view embedding receives
gradient in both, and that the missing-axial placeholder receives gradient in concat and
correctly none in attn.

## 5. Metrics

`utils/metrics.py` (278): macro-F1, Cohen's κ (quadratic-weighted), balanced accuracy,
per-class F1, class weights, CORAL loss/decode, and `LevelAttributionAnalyzer`.

Worst-level accuracy is the signature metric: per study, does `argmax` over predicted grades
land on a truly worst-affected level? Unlike pathology recall it cannot be inflated by flagging
pathology everywhere, since over-flagging makes the argmax arbitrary. Pathology detection is
therefore reported as precision *and* recall *and* FP rate.

`utils/detector_metrics.py` (119): Gaussian heatmap targets, spatial softmax, soft-argmax,
coordinate loss, mm-space localization error using per-study `PixelSpacing`.

`evaluate.py` (348): re-runs checkpoints or aggregates saved results, builds comparison tables,
confusion matrices, attention heatmaps/overlays, and aggregates over seeds.

## 6. Experiments and findings

### v1: tokenizer and positional-encoding ablation (RSNA, 500 studies, frozen DINOv2)

Anatomy + ordinal: κ 0.649 ± 0.031, worst-level 0.542 ± 0.036 (~2.7x the 0.20 chance rate).
Against LumbarDISC's κ 0.765 the gap is attributed to scope, not method: 500 studies vs ~2,700,
2.5D single-sequence vs full 3D multi-sequence, frozen ViT-S/14 with a 1.8 M trainable head vs
end-to-end domain-specific training.

Interpretation bands were pre-registered before seeing the numbers (Δ>=0.09 claim / 0.06-0.09
trend / <0.05 n.s.) specifically to prevent post-hoc rationalization.

### Task 1: learned detector, oracle vs detected boxes

| box source | κ | worst-lvl | median localization |
|---|---|---|---|
| oracle | 0.649 ± 0.031 | 0.542 ± 0.036 | 0.0 mm |
| detected | 0.627 ± 0.030 | 0.456 ± 0.045 | 6.84 mm |

Grading is deployable without oracle coordinates (κ -0.022, n.s.); level attribution is more
localization-sensitive (-0.086). L1/L2 is hardest (10.5 mm), and this was shown to be a
per-level effect, not edge cutoff: across-level corr(y, err) = -0.97 while within-level
corr(err, edge-distance) is ~0. Localization to grading is tolerance-then-threshold rather than
smooth: only the >12 mm bin degrades.

### CORAL ordinal head: tried, rejected

First run had a real weighting bug (3-class per-sample weights instead of per-threshold
`pos_weight`), fixed to `[7.06, 20.34]`. Even fixed: QWK 0.580 vs CE 0.622, and Moderate recall
0/25 where CE achieves 0.48 across five seeds. Thresholds were monotonic but the band width
collapsed to 0.015. Dropped, with a one-line paper mention. The contrast also settled a separate
question: the 3-class task is legitimate, not binary-in-disguise.

### Task 2: does the axial view help? (the most instructive sequence)

Three successive reattributions, each caused by controlling something that had been
uncontrolled:

1. The un-augmented sweep attributed a worst-level gain to *concat fusion*.
2. Per-view augmentation overturned it. Un-augmented axial-only had been *undertrained* (0.520
   to 0.677 with augmentation). The driver was the axial view, not fusion. Seeds could never
   have caught this; only fixing the protocol asymmetry did.
3. Budget control then addressed the remaining asymmetry, that axial reads 5 dedicated per-level
   slices and sagittal reads 1, by adding a 5-slice mean-pooled parasagittal control that holds
   plane constant and changes only slice budget.

Final 5-seed picture (paired, seeds 42-46):

| config | κ | worst-lvl |
|---|---|---|
| sagittal, 1 slice | 0.618 ± 0.069 | 0.471 ± 0.057 |
| sagittal, 5 parasagittal | 0.656 ± 0.045 | 0.576 ± 0.071 |
| axial | 0.678 ± 0.063 | 0.654 ± 0.042 |
| fusion concat | 0.697 ± 0.067 | 0.673 ± 0.036 |
| fusion attn | 0.626 ± 0.055 | 0.519 ± 0.057 |

Two-factor decomposition of worst-level accuracy:

| component | Δ | t | p |
|---|---|---|---|
| total (axial - sag1) | +0.183 | 11.7 | <0.001 |
| budget (sag5 - sag1) | +0.105 | 5.53 | 0.005 |
| acquisition (axial - sag5) | +0.078 | 2.78 | 0.050 |

On severity neither component is significant. Fusion is a powered null: concat ties axial-only
(+0.019, p=0.57), attn is significantly *worse* (-0.135, p<0.001), and the capacity hypothesis
was ruled out by counting parameters rather than assuming, since attn adds zero params over
single-view while concat adds 131,840 and does not degrade.

The residual factor is deliberately named "axial acquisition", not "axial plane", because it
bundles orientation with annotator-chosen level-specific slice selection, which this data cannot
separate.

DeepSPINE (MLHC 2018) was read in full text and turns out to *support* the decomposition: their
axial/sagittal canal tie is 8-slice axial vs 25-slice sagittal, not budget-matched, so their
parity is what our partition predicts.

### Bugs found and recorded

- Device-move bug: the new `sag_multi_images` batch tensor wasn't in `move_batch()`'s key list,
  so it stayed on CPU while the model ran on CUDA. A `--device cpu` smoke test passed and hid it.
  Non-silent on Modal (exit 1, no `test_results.json`), so no degraded run survived. Rule
  adopted: any new `batch[...]` tensor must be added to `move_batch`.
- Modal image glob: excluding `data/rsna_*` also matched `data/rsna_dataset.py`, deleting source
  from the image. Fixed to exclude the directory exactly.

## 7. Figures, statistics, and current state

Covered in detail in [worklog_figures_and_stats.md](worklog_figures_and_stats.md):
`scripts/make_figures.py` (tokenization comparison, qualitative cases),
`scripts/make_result_figures.py` (tokenizer comparison, plane decomposition, training curves),
`scripts/tokenizer_stats.py` (paired significance harness).

Headline findings from that work: seed 45 is the *conservative* seed, not the load-bearing one
(dropping it moves acquisition to +0.104, p=0.002); the `r = -0.84` anticorrelation is a
one-seed artifact (+0.49 without it); "strips underperformed by 0.084" does not survive a paired
test; and the "7-10x seed SD" heuristic predicts neither which gaps separate nor which don't.

Open: four runs (patches, strips at seeds 45, 46) staged behind `resolve_patch_strips()` and
blocked on the Modal spend limit; two authorized paper-text edits pending because the draft is
not in this repo.
