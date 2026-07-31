# Spine-ViT: Anatomy-Aware Vision Transformer for Level-Specific Spine Pathology Grading

Replaces standard ViT patch tokenization with **ROI-Align pooling from vertebral-level
regions**, so each token corresponds to a specific anatomical level (L1/L2, L2/L3, …)
rather than an arbitrary patch. The goal is to reduce level-misattribution errors and to
report a **worst-affected-level accuracy** metric (which level to treat) that prior work doesn't.

Target venue: NeurIPS 2026 VLM4RWD Workshop (8-page paper).

- **Primary dataset — RSNA 2024 / LumbarDISC**: DICOM, point-coordinate annotations,
  3-class spinal-canal-stenosis grading on sagittal T2.
- **Secondary dataset — SPIDER**: NIfTI/`.mha` volumes, GT segmentation masks,
  5-class Pfirrmann grading (interleaved vertebra + disc tokens).

---

## Project layout

```
configs/default.yaml      # hyperparameters; CLI flags override
data/
  rsna_dataset.py         # sagittal DICOM loading, coord→box, patient splits, collate
  rsna_axial.py           # axial series indexing (per-level slice selection)
  rsna_fusion.py          # two-view dataset (sagittal + axial), parasagittal budget control
  rsna_detector.py        # detector dataset (learned box centers)
  spider_dataset.py       # SPIDER NIfTI loading
  transforms.py           # box-consistent augmentations
models/
  backbone.py             # DINOv2 ViT-S/14 (+ offline MockBackbone)
  tokenizer.py            # AnatomyTokenizer / UniformStrip / Patch
  encoder.py              # ordinal / learned / none pos-encodings + transformer
  heads.py                # per-level stenosis / Pfirrmann heads
  spine_grader.py         # single-view assembly + build_model()
  spine_fusion.py         # two-view grader (sag/axial/concat/attn) + build_fusion_model()
  detector.py             # disc-center detector
utils/                    # metrics.py, detector_metrics.py, visualization.py
scripts/                  # EDA, aggregate_fusion.py, export_results.py, box_mm_readout.py, analyze_*.py
tests/                    # offline forward + train-loop tests (MockBackbone)
train.py / train_fusion.py / train_detector.py    # training entry points
evaluate.py               # eval + figures + comparison table
modal_run.py              # Modal entry points for the GPU ablation sweeps
docs/                     # task2_writeup.md, paper_notes.md, instructions.md
notebooks/reproduce_results.ipynb   # reproduces the Task 2 tables from results/
results/fusion_results.csv          # committed per-run metrics (source for the notebook)
```

---

## Setup

> **Python version note.** Your system default is Python 3.14, which PyTorch does not
> ship wheels for yet. A local **Python 3.12** virtualenv has already been created at
> `.venv/` with all dependencies installed. Use it, or recreate with 3.11/3.12.

```bash
# Use the prepared environment
source .venv/bin/activate           # or: ./.venv/bin/python ...

# Or create fresh (Python 3.11 or 3.12)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Verify the install (no dataset needed)

```bash
python tests/test_forward.py        # 9 forward-pass combinations, shapes + param counts
python tests/test_train_loop.py     # synthetic loss-decreases + metrics sanity
```

Both use an offline `MockBackbone`, so they run without downloading DINOv2 or any data.

---

## Data

### RSNA 2024 / LumbarDISC
Download from [Kaggle](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data)
or the [AWS Open Data Registry](https://registry.opendata.aws/rsna-lumbar-spine-degenerative-classification-dataset/).
Expected layout:

```
rsna-2024/
  train.csv
  train_label_coordinates.csv
  train_series_descriptions.csv
  train_images/{study_id}/{series_id}/{instance_number}.dcm
```

### SPIDER
Download from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.8009679).
Expected layout (adjust in `data/spider_dataset.py` if yours differs):

```
spider/
  images/*.mha
  masks/*.mha
  radiological_gradings.csv   (or overview.csv)
```

### Sanity-check the data once it's in place

```bash
python scripts/explore_rsna.py   --data_dir /path/to/rsna-2024   # writes outputs/eda/
python scripts/explore_spider.py --data_dir /path/to/spider
```

These print shapes + grade distributions and save slices with boxes overlaid — confirm
the boxes land on the discs before training.

---

## Training

```bash
# Main model (ours): anatomy tokenizer + ordinal encoding on RSNA
python train.py --data_dir /path/to/rsna-2024 --dataset rsna \
    --tokenizer anatomy --pos_encoding ordinal

# Quick 5-epoch sanity check on 10 samples
python train.py --data_dir /path/to/rsna-2024 --dataset rsna \
    --tokenizer anatomy --pos_encoding ordinal --epochs 5 --limit_samples 10
```

Each run writes to `outputs/{dataset}_{tokenizer}_{pos_encoding}_{embed_dim}_{layers}/`:
`config.json`, `best_model.pt`, `history.json`, `test_results.json`.

Key flags: `--tokenizer {anatomy,strips,patches}`, `--pos_encoding {ordinal,learned,none}`,
`--task {stenosis,pfirrmann}`, `--no-freeze_backbone`, `--use_oracle` (SPIDER),
`--device {cuda,mps,cpu}`. See `configs/default.yaml` for everything else.

### Full ablation study

```bash
RSNA_DIR=/path/to/rsna-2024 SPIDER_DIR=/path/to/spider bash scripts/run_ablations.sh
```

Runs all 7 experiments then `evaluate.py`.

---

## Evaluation

```bash
# Re-run each best checkpoint on the test set + generate every figure
python evaluate.py --experiments_dir outputs --data_dir /path/to/rsna-2024 --generate_figures

# Aggregate already-saved test_results.json without re-running (no data/GPU needed)
python evaluate.py --experiments_dir outputs --from_saved --generate_figures
```

Artifacts land in `outputs/evaluation/`: `comparison_table.md`, `comparison.json`, per-model
confusion matrices and training curves, attention heatmaps/overlays, ablation bar charts, and
the cross-model level-attribution heatmap.

### Main results table (to populate)

| Model | Tokenizer | Pos Enc | Macro F1 | κ | Bal Acc | Worst-Level Acc | Path. P | Path. R |
|---|---|---|---|---|---|---|---|---|
| Ours | Anatomy | Ordinal | ? | ? | ? | ? | ? | ? |
| Ours (CAST-style) | Anatomy | Learned | ? | ? | ? | ? | ? | ? |
| Ours (no encoding) | Anatomy | None | ? | ? | ? | ? | ? | ? |
| Uniform strips | Strips | Ordinal | ? | ? | ? | ? | ? | ? |
| Patch tokens | Patches | Ordinal | ? | ? | ? | ? | ? | ? |
| Ours (fine-tuned) | Anatomy | Ordinal | ? | ? | ? | ? | ? | ? |
| LumbarDISC framework (ref) | Cuboid | Context | 0.783 | 0.765 | — | — | — | — |

Targets to match/beat: **κ ≈ 0.765, macro-F1 ≈ 0.783**.

**Signature metric — Worst-Level Accuracy.** Per study, does `argmax` over the levels of
the *predicted* grade land on a truly worst-affected level? This is the clinical question
("which level do you operate on?") and, unlike pathology *recall* alone, it **cannot be
inflated by flagging pathology everywhere** — over-flagging makes the prediction argmax
arbitrary. Pathology detection is reported as **precision AND recall** (+ FP rate) for the
same reason: recall alone is dishonest when the model over-predicts the rare grades.

---

## Two-view study (Task 2)

Does adding the axial view help canal-stenosis grading? Trained on Modal
(`modal_run.py`); metrics in `results/fusion_results.csv`; full analysis in
`docs/task2_writeup.md` and `notebooks/reproduce_results.ipynb` (paired, 5 seeds).

| config | κ | worst-level acc |
|---|---|---|
| sagittal (1 slice) | 0.618 | 0.471 |
| sagittal (5 parasagittal) | 0.656 | 0.576 |
| axial | 0.678 | 0.654 |
| fusion (concat) | 0.697 | 0.673 |
| fusion (attn) | 0.626 | 0.519 |

The axial advantage is on level attribution, not severity, and decomposes into slice
**budget** (+0.105, p=0.005) and **axial acquisition** (+0.078, p=0.050); fusion does not
beat the better single view. Reproduce:

```bash
python scripts/export_results.py                 # outputs_modal/ → results/fusion_results.csv
jupyter notebook notebooks/reproduce_results.ipynb
```

---

## Design notes

- **Batch contract** (both datasets' collate): `images (B,3,H,W)`, `boxes (N_total,5)` as
  `[batch_idx,x1,y1,x2,y2]` for ROI-Align, `level_indices/level_types/targets (N_total,)`,
  and `num_levels` (list). Discs are `level_type==1`; grading applies to disc tokens only.
- **Patch tokenizer.** Implemented as a learned per-level query that cross-attends over all
  patch tokens (no spatial-localization prior) — a fair baseline that keeps the pipeline
  uniform and still yields per-level predictions, so level-attribution stays comparable.
- **Frozen backbone.** DINOv2 is frozen by default (kept in eval mode, run under `no_grad`);
  only tokenizer/encoder/heads train (~1.8–2.1M params).
- **Attention extraction.** `SpineGrader.forward_with_attention` reproduces the pre-norm
  encoder-layer math manually to request `need_weights=True` (the built-in layer forces it
  off), returning per-layer level×level attention for the overlay/heatmap figures.
- **Reproducibility.** `seed=42` throughout; patient-level splits so no study leaks across
  train/val/test.
```
