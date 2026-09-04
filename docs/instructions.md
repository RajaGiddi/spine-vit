# Spine-ViT: Implementation Specification

## Project Overview

Build an **anatomy-aware vision transformer for level-specific spine pathology grading**. The core idea: replace standard ViT patch tokenization with ROI-Align pooling from vertebral-level regions, so each token corresponds to a specific anatomical level (L1-L2, L2-L3, etc.) rather than an arbitrary image patch. Compare against baselines to show this reduces level-misattribution errors.

**Target venue:** NeurIPS 2026 VLM4RWD Workshop (deadline Aug 31, 2026). 8-page paper.

**Two datasets:**
- **Primary: RSNA 2024 / LumbarDISC** (~2,697 patients, DICOM, point-coordinate annotations, 3-class stenosis grading)
- **Secondary: SPIDER** (~218 patients, NIfTI, segmentation masks, Pfirrmann 5-class grading)

---

## Project Structure

```
spine-vit/
├── configs/
│   └── default.yaml
├── data/
│   ├── __init__.py
│   ├── rsna_dataset.py          # LumbarDISC / RSNA 2024 data loader
│   ├── spider_dataset.py        # SPIDER data loader
│   └── transforms.py            # Shared augmentations
├── models/
│   ├── __init__.py
│   ├── backbone.py              # DINOv2 feature extraction
│   ├── tokenizer.py             # ROI-Align, uniform strips, patch tokenizers
│   ├── encoder.py               # Transformer encoder + positional encoding variants
│   ├── heads.py                 # Per-level classification heads
│   └── spine_grader.py          # Full model assembly + build_model()
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # F1, kappa, level-attribution analysis
│   └── visualization.py         # Attention maps, confusion heatmaps, overlay plots
├── scripts/
│   ├── explore_rsna.py          # EDA notebook-style script for RSNA data
│   ├── explore_spider.py        # EDA notebook-style script for SPIDER data
│   └── run_ablations.sh         # Full ablation study launcher
├── train.py                     # Training loop
├── evaluate.py                  # Standalone evaluation + generate all figures
└── requirements.txt
```

---

## Step 1: Environment & Dependencies

Create `requirements.txt`:
```
torch>=2.0
torchvision>=0.15
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
pyyaml
SimpleITK
nibabel
pydicom
pillow
```

Python 3.10+. Single GPU (A100 or V100 preferred, T4 minimum).

---

## Step 2: Data, RSNA 2024 / LumbarDISC

### 2.1 Source

Download from Kaggle: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data

Or AWS Open Data Registry: https://registry.opendata.aws/rsna-lumbar-spine-degenerative-classification-dataset/

### 2.2 File structure after download

```
rsna-2024/
├── train.csv                        # study_id + 25 severity columns
├── train_label_coordinates.csv      # study_id, series_id, instance_number, condition, level, x, y
├── train_series_descriptions.csv    # study_id, series_id, series_description
├── train_images/
│   └── {study_id}/
│       └── {series_id}/
│           └── {instance_number}.dcm
```

### 2.3 Key CSV schemas

**train.csv** columns:
- `study_id` (int)
- 25 severity columns named as `{condition}_{level}` where:
  - conditions: `spinal_canal_stenosis`, `left_neural_foraminal_narrowing`, `right_neural_foraminal_narrowing`, `left_subarticular_stenosis`, `right_subarticular_stenosis`
  - levels: `l1_l2`, `l2_l3`, `l3_l4`, `l4_l5`, `l5_s1`
- Values: `Normal/Mild`, `Moderate`, `Severe` (encode as 0, 1, 2)

**train_label_coordinates.csv** columns:
- `study_id`, `series_id`, `instance_number`, `condition`, `level`, `x`, `y`
- The (x, y) are pixel coordinates on the specific DICOM instance indicating where the condition is localized
- condition values: `Spinal Canal Stenosis`, `Left Neural Foraminal Narrowing`, `Right Neural Foraminal Narrowing`, `Left Subarticular Stenosis`, `Right Subarticular Stenosis`
- level values: `L1/L2`, `L2/L3`, `L3/L4`, `L4/L5`, `L5/S1`

**train_series_descriptions.csv** columns:
- `study_id`, `series_id`, `series_description`
- Filter for sagittal T2 series: `series_description` contains `Sagittal T2` (case-insensitive). This is the primary imaging sequence for spinal canal stenosis assessment.

### 2.4 RSNA Dataset class (`data/rsna_dataset.py`)

**Loading pipeline per sample:**

1. From `train_label_coordinates.csv`, for a given study_id, find all rows for sagittal T2 series
2. Group by level (L1/L2 through L5/S1). Each level has one or more (x, y) coordinates (one per condition)
3. Load the DICOM instance using `pydicom.dcmread(path)`. Extract pixel array. Normalize to float32 [0, 1] using window/level from DICOM tags or simple min-max.
4. For each level, compute a bounding box centered on the (x, y) coordinate:
   - Use the spinal canal stenosis coordinate as the disc center (it's the most central anatomically)
   - Box size: 64×64 pixels (tunable). If the image is large, scale proportionally.
   - Clamp to image boundaries
   - Format as [x1, y1, x2, y2]
5. Assign ordinal level indices: L1/L2=0, L2/L3=1, L3/L4=2, L4/L5=3, L5/S1=4
6. All levels are disc levels (level_type=1 for all)
7. Look up severity grades from `train.csv` for spinal canal stenosis at each level. Encode as 0/1/2.

**Important details:**
- Some studies have multiple sagittal T2 series. Pick the first one, or the one with the most coordinate annotations.
- Some levels may have missing coordinates. Skip those levels for that sample.
- Some studies have missing severity labels (NaN). Use -1 as ignore index.
- The DICOM pixel arrays may need orientation correction. Check `ImageOrientationPatient` and `ImagePositionPatient` tags. Many RSNA solutions just read raw pixels and resize; start with that.

**Image preprocessing:**
- Load the specific DICOM slice indicated by `instance_number`
- If doing 2D: use that single slice
- If doing 2.5D (recommended for more context): load the target slice plus 2 adjacent slices (instance_number-1, instance_number, instance_number+1), stack as 3 channels. This gives pseudo-RGB input matching DINOv2's expected 3-channel input.
- Resize to 224×224 (or 384×384 for DINOv2 ViT-S/14)
- Z-score normalization per image

**Train/val/test split:**
- Patient-level split (no study_id in multiple splits)
- 70/15/15 or use an existing k-fold split from Kaggle solutions
- Set random seed=42 for reproducibility

**Collate function:**
- Variable number of levels per sample (usually 5, but some may have fewer)
- Boxes formatted for ROI-Align: (N_total, 5) with [batch_idx, x1, y1, x2, y2]
- Level indices: (N_total,) long tensor
- Level types: (N_total,) long tensor (all 1 for discs in RSNA)
- Targets: (N_total,) long tensor of severity grades
- num_levels: list of ints (one per sample in batch)

### 2.5 Handling multiple conditions

RSNA has 5 conditions. Start with **spinal canal stenosis only** (sagittal T2, most comparable to prior work). This keeps the problem clean and matches what SPIDER also has.

If time permits, add left/right neural foraminal narrowing (sagittal T1) and subarticular stenosis (axial T2) as additional tasks. But for the workshop paper, spinal canal stenosis on sagittal T2 is sufficient.

---

## Step 3: Data, SPIDER

### 3.1 Source

Download from Zenodo: https://zenodo.org/doi/10.5281/zenodo.8009679
Or HuggingFace: `load_dataset("cdoswald/SPIDER")`

### 3.2 SPIDER Dataset class (`data/spider_dataset.py`)

**Key differences from RSNA:**
- NIfTI format (.mha or .nii.gz), not DICOM
- 3D volumes: extract mid-sagittal slice
- Ground-truth segmentation masks available (label scheme: vertebrae 1,2,3...; discs 201,202,...; canal 100)
- Pfirrmann grading (5-class: I-V, encode as 0-4) and stenosis
- Both vertebral body AND disc tokens (interleaved), not just disc tokens
- Bottom-up labeling (label 1 = lowest vertebra)

**Loading pipeline per sample:**

1. Load 3D volume with SimpleITK: `sitk.ReadImage(path)` → `sitk.GetArrayFromImage(img)` → (D, H, W) float32
2. Load corresponding segmentation mask
3. Extract mid-sagittal slice: `volume[D//2]` for both image and mask
4. From the 2D mask, find unique labels (exclude 0=background, 100=canal)
5. For each label, compute bounding box from mask pixels (min/max x, min/max y, with 2px padding)
6. Interleave vertebrae and discs bottom-up: vertebra_1, disc_201, vertebra_2, disc_202, ...
7. Assign ordinal indices sequentially: 0, 1, 2, ...
8. level_type: 0 for vertebrae, 1 for discs
9. Look up Pfirrmann grades from the overview CSV for disc levels
10. Preprocess image: z-score normalize, resize to 224×224, repeat to 3 channels

**Oracle vs. heuristic regions:**
- `use_oracle_regions=True` (default): use ground-truth segmentation masks for bounding boxes
- `use_oracle_regions=False`: use the intensity-profile heuristic (see below)

**Intensity-profile heuristic (for when masks are unavailable):**
1. Extract vertical intensity profile along image midline (average a 5-pixel-wide strip)
2. Smooth with Gaussian filter (sigma=5-10 pixels)
3. Find peaks (vertebral bodies = bright on T2) using `scipy.signal.find_peaks`
4. Find valleys between peaks (discs = dark)
5. Assign labels bottom-up
6. Create bounding boxes centered on each peak/valley

---

## Step 4: Shared Transforms (`data/transforms.py`)

Augmentations applied during training only:

```python
class SpineAugmentation:
    """
    Medical-image-appropriate augmentations.
    No color jitter (meaningless for grayscale MRI).
    """
    def __init__(self, image_size=224):
        self.image_size = image_size

    def __call__(self, image, boxes):
        """
        Args:
            image: (H, W) numpy array
            boxes: (K, 4) numpy array [x1, y1, x2, y2]
        Returns:
            image, boxes (both augmented consistently)
        """
        # 1. Random vertical shift (±10%): shift image and boxes together
        # 2. Intensity jitter (±5%): multiply pixel values by random factor in [0.95, 1.05]
        # 3. Horizontal flip (50% probability): flip image, mirror box x-coordinates
        # 4. Random crop-and-resize: crop [85-100%] of image area, resize back to image_size
        # 5. Gaussian noise: add N(0, 0.01) noise
        #
        # IMPORTANT: when augmenting, transform boxes consistently with the image.
        # Vertical shift → shift box y-coordinates by the same amount.
        # Horizontal flip → flip box x-coordinates: new_x1 = W - old_x2, new_x2 = W - old_x1
        # Crop → adjust box coordinates relative to crop, clamp to boundaries.
        pass
```

Note: boxes must be transformed consistently with the image. This is critical: if you shift the image down 10px, shift all box y-coordinates down 10px too.

---

## Step 5: Model Architecture

### 5.1 Backbone (`models/backbone.py`)

**DINOv2 ViT-S/14** pretrained from Meta.

```python
class DINOv2Backbone(nn.Module):
    def __init__(self, model_name="dinov2_vits14", freeze=True):
        # Load: torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        # embed_dim = 384 for ViT-S
        # patch_size = 14
        # spatial_scale = 1/14 (for ROI-Align)

    def forward(self, x):
        # x: (B, 3, H, W) where H, W divisible by 14
        # Use get_intermediate_layers(x, n=1)[0] to get patch tokens
        # Returns (B, N_patches, 384) where N_patches = (H/14) * (W/14)
        # Reshape to spatial: (B, 384, H/14, W/14) for ROI-Align
        # If freeze=True, run with torch.no_grad() and keep backbone in eval mode
```

### 5.2 Tokenizer (`models/tokenizer.py`)

Three variants for controlled comparison:

**AnatomyTokenizer (our method):**
```python
class AnatomyTokenizer(nn.Module):
    def __init__(self, backbone_dim=384, embed_dim=256, roi_output_size=7, spatial_scale=1/14):
        # ROI-Align output → AdaptiveAvgPool2d(1) → Linear(backbone_dim, embed_dim) + LayerNorm + GELU

    def forward(self, feature_map, boxes):
        # feature_map: (B, 384, H', W') from backbone
        # boxes: (N_total, 5) with [batch_idx, x1, y1, x2, y2] in INPUT image coordinates
        # torchvision.ops.roi_align(feature_map, boxes, output_size=7, spatial_scale=1/14, aligned=True)
        # → (N_total, 384, 7, 7) → pool → (N_total, 384) → project → (N_total, 256)
```

**UniformStripTokenizer (baseline):**
```python
class UniformStripTokenizer(nn.Module):
    # Divides image into K equal horizontal strips (K=5 for 5 disc levels)
    # Uses ROI-Align on strip bounding boxes
    # Same downstream architecture, tests whether precise localization matters
```

**PatchTokenizer (baseline):**
```python
class PatchTokenizer(nn.Module):
    # Uses standard ViT patch tokens directly
    # Projects from backbone_dim to embed_dim
    # For per-level prediction: needs a learned mapping from patches to levels
    # OR: global average pool for per-image (worst-grade) prediction only
```

### 5.3 Encoder (`models/encoder.py`)

**Three positional encoding variants (ablation centerpiece):**

```python
class OrdinalPositionalEncoding(nn.Module):
    # nn.Embedding(max_levels, embed_dim)
    # Indexed by ordinal position: L1/L2=0, L2/L3=1, ..., L5/S1=4
    # Encodes sequential ordering explicitly

class LearnedIdentityEncoding(nn.Module):
    # nn.Embedding(max_levels, embed_dim)
    # Same structure but conceptually different:
    # treated as level IDENTITY, not ordinal position
    # (In practice, same implementation but initialized differently
    # and the distinction matters for the paper's framing)

class NoPositionalEncoding(nn.Module):
    # Returns 0.0 (no positional info)
```

**AnatomyEncoder:**
```python
class AnatomyEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1,
                 max_levels=12, pos_encoding="ordinal"):
        # pos_encoder: one of the three above based on pos_encoding arg
        # type_embedding: nn.Embedding(2, embed_dim) for vertebra(0) vs disc(1)
        # encoder: nn.TransformerEncoder with pre-norm (norm_first=True)
        # norm: final LayerNorm

    def forward(self, tokens, level_indices, level_types, num_levels):
        # 1. tokens = tokens + pos_encoder(level_indices) + type_embedding(level_types)
        # 2. Pack variable-length sequences into padded batch
        #    - Create (B, max_K, embed_dim) padded tensor
        #    - Create (B, max_K) key_padding_mask (True = ignore)
        # 3. Run transformer encoder with padding mask
        # 4. Unpack back to flat (N_total, embed_dim)
        # Return encoded tokens
```

### 5.4 Classification Heads (`models/heads.py`)

```python
class GradingHeads(nn.Module):
    def __init__(self, embed_dim=256, num_stenosis_classes=3, num_pfirrmann_classes=5):
        # stenosis_head: Linear → GELU → Dropout(0.1) → Linear(embed_dim → num_stenosis_classes)
        # pfirrmann_head: same structure → num_pfirrmann_classes
        # Only one head is active at a time depending on dataset

    def forward(self, encoded_tokens, level_types, task="stenosis"):
        # Filter to disc-level tokens only (level_types == 1)
        # Apply appropriate head based on task
        # Return logits + disc_mask
```

### 5.5 Full Model (`models/spine_grader.py`)

```python
class SpineGrader(nn.Module):
    def __init__(self, config):
        # Assemble: backbone → tokenizer → encoder → heads
        # tokenizer_type from config: "anatomy", "strips", "patches"
        # pos_encoding from config: "ordinal", "learned", "none"

    def forward(self, batch):
        # 1. feature_map = backbone(images)
        # 2. tokens = tokenizer(feature_map, boxes)
        # 3. encoded = encoder(tokens, level_indices, level_types, num_levels)
        # 4. logits, disc_mask = heads(encoded, level_types, task)
        # Return dict: logits, disc_mask, encoded_tokens

def build_model(config) -> SpineGrader:
    # Factory function from config dict
```

---

## Step 6: Training (`train.py`)

### 6.1 Loss

```python
# Weighted cross-entropy with class imbalance handling
# For RSNA (3-class): compute inverse-frequency weights from training set
# For SPIDER (5-class): same
# Use ignore_index=-1 for missing labels
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
```

### 6.2 Optimizer

```python
# AdamW, only trainable parameters (backbone is frozen)
optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=1e-2
)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
```

### 6.3 Training loop

```
For each epoch:
    1. Train on all batches:
       - Forward pass → get logits
       - Compute CE loss on valid targets only
       - Backward, clip gradients (max_norm=1.0), step
    2. Evaluate on validation set:
       - Compute loss, macro F1, Cohen's kappa, balanced accuracy
       - Run LevelAttributionAnalyzer
    3. Learning rate scheduler step
    4. Early stopping on validation macro F1 (patience=15)
    5. Save best model checkpoint
    6. Log all metrics to history.json
```

### 6.4 CLI arguments

```
--data_dir          Path to dataset root
--dataset           "rsna" or "spider"
--tokenizer         "anatomy" | "strips" | "patches"
--pos_encoding      "ordinal" | "learned" | "none"
--backbone          "dinov2_vits14" (default)
--freeze_backbone   Flag (default True)
--embed_dim         256
--encoder_layers    2
--encoder_heads     4
--image_size        224
--batch_size        16
--lr                1e-4
--epochs            100
--patience          15
--device            "cuda" or "cpu"
--output_dir        "outputs"
--seed              42
--task              "stenosis" | "pfirrmann" (which head to use)
```

### 6.5 Experiment naming

Output directory: `outputs/{dataset}_{tokenizer}_{pos_encoding}_{embed_dim}_{encoder_layers}/`

Each experiment saves:
- `config.json`: full config
- `best_model.pt`: best checkpoint
- `history.json`: per-epoch metrics
- `test_results.json`: final test metrics

---

## Step 7: Evaluation (`evaluate.py`)

### 7.1 Metrics (`utils/metrics.py`)

**Primary metrics (for main results table):**

```python
def compute_metrics(predictions, targets, num_classes, task_name=""):
    # Filter invalid targets (== -1)
    # macro_f1: sklearn.metrics.f1_score(average="macro")
    # kappa: sklearn.metrics.cohen_kappa_score
    # balanced_acc: sklearn.metrics.balanced_accuracy_score
    # accuracy: simple (predictions == targets).mean()
    # Per-class F1 for each grade
    # Return dict of all metrics
```

**Level-attribution analysis (signature experiment):**

```python
class LevelAttributionAnalyzer:
    """
    For each pathological finding (grade >= threshold),
    check if the model correctly identified pathology at that level.

    Produces:
    - Level-attribution accuracy (% of pathological levels correctly flagged)
    - Per-level accuracy (is L4/L5 harder than L2/L3?)
    - Grade confusion matrix
    - Miss rate (how often pathology is missed entirely)
    """
    def update(self, prediction_grades, true_grades, level_indices, patient_id):
        # Record each prediction for later analysis

    def compute(self, pathology_threshold=1):
        # Filter to pathological findings
        # Compute attribution accuracy, per-level accuracy
        # Build grade confusion matrix
        # Analyze misses
        # Return dict of results
```

### 7.2 Visualization (`utils/visualization.py`)

Generate these figures for the paper:

```python
def plot_grade_confusion_matrix(true, prediction, class_names, title, save_path):
    """Heatmap of predicted vs true grades. Use seaborn heatmap."""

def plot_level_attribution_heatmap(analyzer_results, save_path):
    """Heatmap showing per-level detection accuracy across models."""

def plot_attention_weights(attention_matrix, level_labels, save_path):
    """
    Visualize transformer attention weights as a heatmap.
    Rows/cols are vertebral levels (L1, L1-L2 disc, L2, ...).
    Shows which levels attend to which.
    """

def plot_attention_overlay(image, boxes, attention_weights, level_labels, save_path):
    """
    Overlay attention weights on the original MRI slice.
    Color each bounding box by how much attention it receives.
    """

def plot_training_curves(history, save_path):
    """Train/val loss and metric curves over epochs."""

def plot_ablation_comparison(results_dict, metric_name, save_path):
    """
    Bar chart comparing a metric across all ablation variants.
    results_dict: {"anatomy_ordinal": 0.78, "strips_ordinal": 0.65, ...}
    """
```

### 7.3 Evaluation script

```python
# evaluate.py
# 1. Load best model checkpoint for each experiment variant
# 2. Run on test set
# 3. Compute all metrics
# 4. Run LevelAttributionAnalyzer
# 5. Generate all figures
# 6. Print comparison table
# 7. Save everything to outputs/evaluation/
```

---

## Step 8: Ablation Study (`scripts/run_ablations.sh`)

Run these 7 experiments sequentially:

```bash
# === Main model ===
# 1. Anatomy tokenizer + ordinal encoding (OURS)
python train.py --dataset rsna --tokenizer anatomy --pos_encoding ordinal

# === Tokenizer ablation ===
# 2. Uniform strips + ordinal encoding
python train.py --dataset rsna --tokenizer strips --pos_encoding ordinal

# 3. Patch tokens (global classification baseline)
python train.py --dataset rsna --tokenizer patches --pos_encoding ordinal

# === Positional encoding ablation ===
# 4. Anatomy tokenizer + CAST-style learned encoding
python train.py --dataset rsna --tokenizer anatomy --pos_encoding learned

# 5. Anatomy tokenizer + NO positional encoding
python train.py --dataset rsna --tokenizer anatomy --pos_encoding none

# === Additional ===
# 6. Anatomy tokenizer + ordinal + fine-tuned backbone
python train.py --dataset rsna --tokenizer anatomy --pos_encoding ordinal --no-freeze_backbone

# === Cross-dataset (SPIDER with oracle masks) ===
# 7. Anatomy tokenizer + ordinal on SPIDER (oracle regions + Pfirrmann task)
python train.py --dataset spider --tokenizer anatomy --pos_encoding ordinal --task pfirrmann --use_oracle
```

After all experiments:
```bash
python evaluate.py --experiments_dir outputs/ --generate_figures
```

---

## Step 9: Expected Results Table

The paper's main table should look like this:

```
| Model                          | Tokenizer | Pos Enc  | Macro F1 | κ     | Bal Acc | Level Attr Acc |
|--------------------------------|-----------|----------|----------|-------|---------|----------------|
| Ours                           | Anatomy   | Ordinal  |   ?      |  ?    |   ?     |      ?         |
| Ours (CAST-style encoding)     | Anatomy   | Learned  |   ?      |  ?    |   ?     |      ?         |
| Ours (no encoding)             | Anatomy   | None     |   ?      |  ?    |   ?     |      ?         |
| Uniform strips baseline        | Strips    | Ordinal  |   ?      |  ?    |   ?     |      ?         |
| Patch tokens baseline          | Patches   | N/A      |   ?      |  ?    |   ?     |      N/A       |
| Ours (fine-tuned backbone)     | Anatomy   | Ordinal  |   ?      |  ?    |   ?     |      ?         |
|                                |           |          |          |       |         |                |
| LumbarDISC framework (ref)     | Cuboid    | Context  | 0.783    | 0.765 | -       |      -         |
```

Numbers to beat or match: **κ ≈ 0.765, macro-F1 ≈ 0.783** from the LumbarDISC framework paper.

The "Level Attr Acc" column is our signature metric that prior work doesn't report.

---

## Step 10: Key Implementation Notes

### 10.1 ROI-Align coordinate system

`torchvision.ops.roi_align` expects boxes in **input image coordinate space**, not feature map space. The `spatial_scale` parameter handles the conversion internally. So if your image is 224×224 and your feature map is 16×16, pass boxes in 224×224 coordinates and set `spatial_scale=1/14`.

Box format: `[batch_index, x1, y1, x2, y2]` where (x1, y1) is top-left and (x2, y2) is bottom-right.

### 10.2 Variable-length sequences

Different patients have different numbers of visible levels. The collate function must:
1. Concatenate all boxes across the batch with a batch-index column
2. Keep track of how many levels each sample has (`num_levels` list)
3. The transformer encoder pads to max_K in the batch and uses `key_padding_mask`

### 10.3 Frozen backbone

DINOv2 is frozen by default. This means:
- Only the tokenizer projection, encoder, and heads have trainable parameters
- Total trainable params should be ~1-3M (very lightweight)
- Training should be fast (minutes per epoch on a single GPU)
- Override `train()` on the backbone module to keep it in eval mode

### 10.4 Extracting attention weights

To visualize attention, you need to extract attention weights from the transformer encoder. PyTorch's `TransformerEncoderLayer` doesn't return attention by default. Two options:
1. Register a forward hook on the `self_attn` module inside each layer
2. Use `need_weights=True` in `MultiheadAttention.forward()`, which requires modifying the encoder layer

Implement option 1 (hook-based) in the evaluation script. Don't modify the training code for this.

### 10.5 DICOM loading for RSNA

```python
import pydicom

def load_dicom_slice(path):
    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply windowing if available
    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
        center = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else float(ds.WindowCenter[0])
        width = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else float(ds.WindowWidth[0])
        lower = center - width / 2
        upper = center + width / 2
        pixel_array = np.clip(pixel_array, lower, upper)
        pixel_array = (pixel_array - lower) / (upper - lower)
    else:
        # Simple min-max normalization
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)

    return pixel_array
```

### 10.6 Bounding box derivation from RSNA coordinates

The RSNA coordinates are point localizers (centroids). Convert to boxes:

```python
def coord_to_box(x, y, box_size=64, img_h=None, img_w=None):
    """Convert center (x, y) to bounding box [x1, y1, x2, y2]."""
    half = box_size / 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(img_w, x + half) if img_w else x + half
    y2 = min(img_h, y + half) if img_h else y + half
    return [x1, y1, x2, y2]
```

The box_size should be tunable. Start with 64 pixels on the original DICOM image, then scale when you resize to 224×224.

### 10.7 Handling the 2.5D input

For DINOv2 (expects 3-channel input):
- Option A (simple): repeat the grayscale slice to 3 channels. `image.repeat(3, 1, 1)`
- Option B (better, recommended): load 3 adjacent DICOM slices as 3 channels. The center slice is the annotated one; adjacent slices provide through-plane context. This is standard practice in medical imaging competitions.

For SPIDER (3D NIfTI volumes): load slices [mid-1, mid, mid+1] as 3 channels.

---

## Step 11: Execution Order

### Phase 1: Data (do first)
1. Implement `data/rsna_dataset.py` with DICOM loading, coordinate parsing, box derivation
2. Implement `data/spider_dataset.py` with NIfTI loading, mask-to-box extraction
3. Implement `data/transforms.py`
4. Write `scripts/explore_rsna.py`: load 5 samples, print shapes, visualize slices with boxes overlaid, print grade distributions
5. Write `scripts/explore_spider.py`: same for SPIDER

### Phase 2: Model (do second)
1. Implement `models/backbone.py`
2. Implement `models/tokenizer.py` (all 3 variants)
3. Implement `models/encoder.py` (all 3 pos encoding variants)
4. Implement `models/heads.py`
5. Implement `models/spine_grader.py` with `build_model()`
6. **Verify**: write a test that creates a random batch and runs a full forward pass through each tokenizer/encoding combination. Print shapes at every stage. This must pass before proceeding.

### Phase 3: Training (do third)
1. Implement `train.py`
2. Implement `utils/metrics.py`
3. Run a quick sanity check: train for 5 epochs on a tiny subset (10 samples), verify loss decreases and metrics are computed
4. Run the full training for the main model (anatomy + ordinal on RSNA)

### Phase 4: Evaluation (do fourth)
1. Implement `utils/visualization.py`
2. Implement `evaluate.py`
3. Run `scripts/run_ablations.sh`
4. Generate all figures and tables

### Phase 5: Frontier VLM evaluation (optional, for the paper)
1. Write a script that sends SPIDER/RSNA sagittal MRI images to GPT-4o and Gemini APIs
2. Prompt: "This is a sagittal T2 lumbar spine MRI. For each intervertebral disc level from L1/L2 to L5/S1, grade the severity of spinal canal stenosis as Normal/Mild, Moderate, or Severe."
3. Parse structured responses
4. Compute the same metrics (F1, kappa, level-attribution accuracy)
5. Add to the comparison table

---

## Step 12: Critical Success Criteria

Before proceeding to each next phase, verify:

- [ ] Phase 1: Can load and visualize 5 RSNA samples with correct boxes overlaid on the MRI slice
- [ ] Phase 1: Can load and visualize 5 SPIDER samples with segmentation-derived boxes overlaid
- [ ] Phase 1: Grade distributions are printed and reasonable (no all-zeros, no all-NaN)
- [ ] Phase 2: Forward pass through all 3×3 = 9 combinations of tokenizer × encoding produces correct output shapes
- [ ] Phase 2: Total trainable parameters is ~1-3M (backbone frozen)
- [ ] Phase 3: Loss decreases over 5 epochs on a tiny subset
- [ ] Phase 3: Metrics are non-zero after 5 epochs
- [ ] Phase 4: All figures generate without errors
- [ ] Phase 4: Ablation comparison table is populated with numbers for all 7 variants