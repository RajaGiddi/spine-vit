# Phase 0 log

## Stage 1 — geometry

Per-slice geometry, not one affine per series: 18/25 axial stacks are angled,
median 4 orientation groups (one per disc level). A single series affine is
wrong for most studies, not merely imprecise.

`patient_to_voxel` inverts the basis rather than projecting with dot products.
Header direction cosines are orthogonal only to ~1e-4, and projection turns
that into round-trip error growing with distance from the origin. Worst-case
error over all 50 paired series: 2.3e-13 mm.

## Data

`FrameOfReferenceUID` differs between the sagittal and axial series in **25/25**
studies. This is an anonymisation artifact, not a geometry problem — verified by
annotating the same anatomy in each series independently and comparing:

| | |
|---|---|
| median distance | 4.9 mm |
| p90 | 9.3 mm (excluding one bad annotation at 126 mm, 1/116) |
| per-axis median offset | (−0.2, +2.6, +0.9) mm |

Unrelated frames would give scattered distances and a large constant offset.
Confirmed visually: axial slice planes traced onto the sagittal image land on
the annotated disc levels.

## Experiment 0 — motion check, 25 studies

Raw: median discrepancy **6.49 mm**, 1.62 × the larger slice thickness.

**The first verdict was wrong.** `direction_consistency` compared the resultant
length against a `3/√n` rule of thumb — 0.56 vs 0.60 — and reported "no
consistent direction". The Rayleigh test (3nR̄² ~ χ²₃) gives **p = 3×10⁻⁵**. The
per-axis medians say the same thing unaided: x +0.05, z −0.03, **y −4.04**.
Replaced with the Rayleigh test.

Decomposition (fitted to the two summary statistics, then measured directly by
test-retest):

- shared **~4 mm anterior** offset — identical across 25 patients, so not motion
- **~5.3 mm** isotropic residual

### The instrument does not resolve motion

Measured sensitivity of the centroid to the mark:

| mark moved | centroid moved | ratio |
|---|---|---|
| 2 mm | 1.75 mm | 0.88 |
| 4 mm | 3.84 mm | 0.96 |
| 6 mm | 5.75 mm | 0.96 |

Vertebral marrow is near-uniform, so an 8 mm sphere has no intensity structure
to lock onto and hand error passes through ~1:1. Enlarging the sphere does not
help (0.65 at 20 mm; worse at 25 mm as it spills into neighbouring structures).
A hand scatter of σ≈3.5 mm/axis reproduces both observed statistics exactly.

The 0.13 mm phantom floor was computed for a perfect landmark and never
accounted for the observer. That is a design error in Exp 0, not a marking
error.

**"The instrument cannot resolve motion" is a legitimate Phase 0 result.** It
bounds motion below the measurement precision, which is what Experiment 1 needs
to know. It is not the result the protocol anticipated.

### Test-retest

Second blind pass, different day, shuffled order, written to a separate file
that the first pass is never loaded into. Reports **per-axis** scatter, which is
what distinguishes the two readings:

- scatter isotropic, offset still A-P → the offset is a real difference between
  how the landmark reads in profile vs cross-section
- scatter itself A-P heavy → A-P is the hard axis to judge, and the offset is
  marking bias too

The predicted Exp 0 residual is not fitted. Sagittal and axial marks are
independent, so their per-axis variances add; the observed residual either
matches that prediction or exceeds it.

### First retest pass was contaminated

Reported sagittal SD per axis (0.31, 6.36, 8.00) with a median re-mark distance
of only 3.96 mm. Those are inconsistent: for a Gaussian, those SDs imply a
median of 8.43 mm. A single re-mark landing one vertebra away (35 mm) gives an
SD of 7.83 at n=20 — essentially the reported 8.00.

So the SD described one mistake, not the landmark. Consequences:

- `scatter_summary` now reports MAD-based robust SD alongside the plain SD,
  flags re-marks beyond 4 robust MADs, and names the offending studies
- the drift check uses the median, not the mean — one level error was being
  reported as a drift of the whole pass
- `interpret` surfaces contamination *before* any other reading

The A-P anisotropy conclusion can move in either direction under contamination:
in a worked case a z-axis outlier **hid** an A-P effect in sagittal (0.52 plain
vs 2.38 robust) while **exaggerating** it in axial (3.20 plain vs 2.35 robust).
Neither the anisotropy reading nor the "marking noise explains the residual"
verdict can be banked until the robust figures are in — the predicted residual
falls sharply once the outlier is excluded, which can flip the verdict.

## Open

**Self-locating landmark — reintroduces segmentation.** The fix for the
sensitivity problem is a landmark anchored by the vertebral body's own boundary
rather than the seed, so the mark only has to land *inside* the body. That means
segmenting to establish geometry, which is what the design set out to avoid.
Not disqualifying — this is measurement infrastructure, not the representation
under test — but the distinction must stay explicit. Nothing in `composition/`
may depend on it.

**Protocol document absent.** `phase0_experiment_protocol.md` is not in the
repo. Exp 0 is reported against an assumed rule (median > 0.5 × the larger slice
thickness ⇒ motion dominates), flagged as an assumption, not the protocol's.

**Tissue model.** `MagneticFieldStrength`, `EchoTime`, `RepetitionTime` are
absent from every sampled series (0/60), so Stage 3 cannot select relaxation
values by field strength. Decision: assume 3T, keep both tables, expose an
override. `PatientSex`/`PatientAge`/`PatientSize`/`PatientWeight` are also
absent, so Exp 2 supports random pairing only — matched pairing has no metadata
to match on.
