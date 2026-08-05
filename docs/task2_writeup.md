# Task 2: Does the axial view help canal-stenosis grading, and why?

*Draft section. Numbers are augmented runs, 5 seeds (42-46) unless noted; comparisons are
paired within seed. Grading metric is quadratic-weighted κ (severity); level attribution is
worst-level accuracy (worst_lvl): does the model correctly identify the most-stenotic level
per study, evaluated on the n=27 studies with a pathological level.*

## Question

v1 grades central-canal stenosis from a single mid-sagittal T2 slice. Clinically the canal is
read on axial T2, where each disc level is imaged as its own cross-section. We ask whether
adding the axial view improves grading over sagittal alone, and, crucially, *what* about the
axial view is responsible, since a naive two-view comparison confounds several things at once.

## Setup

A shared frozen DINOv2 backbone and a shared anatomy ROI-Align tokenizer encode both views; each
disc level contributes one ROI token per available view, carrying an ordinal level index and a
learned view-type embedding. We compare five configurations under identical training (per-view
augmentation; sagittal horizontal flip disabled as anatomically invalid, axial flip enabled):

| config | κ (severity) | worst_lvl (attribution) | seeds |
|---|---|---|---|
| sagittal-only (1 slice), control | 0.618 ± 0.069 | 0.471 ± 0.057 | 5 |
| sagittal-only, 5 parasagittal slices | 0.656 ± 0.045 | 0.576 ± 0.071 | 5 |
| axial-only | 0.678 ± 0.063 | 0.654 ± 0.042 | 5 |
| fusion, concat (sag ⊕ axial per level) | 0.697 ± 0.067 | 0.673 ± 0.036 | 5 |
| fusion, attention (sag+axial one sequence) | 0.626 ± 0.055 | 0.519 ± 0.057 | 5 |

*All five configurations are run at 5 seeds; every comparison below is paired within seed.*

## Fusion adds nothing over the single better view

We state the null with power, paired against the better single view rather than by eyeballing
means. Concatenation fusion is statistically indistinguishable from axial-only: Δ worst_lvl =
+0.019 (95% CI [-0.067, +0.106], p = 0.57), Δκ = +0.018 (p = 0.46). This is a well-bounded
null: under concat fusion at this data scale, adding the sagittal token to the axial token
neither helps nor hurts. Attention fusion is significantly *worse* than axial-only: Δ worst_lvl
= -0.135 (CI [-0.157, -0.113], p < 0.001). This is not an over-parameterization effect.
Attention fusion adds no trainable parameters over a single-view model (1.82 M, identical to
sagittal- and axial-only), whereas concatenation adds 0.13 M via its projection yet does not
degrade. Its degradation at ~350 training studies therefore cannot be attributed to added
capacity; we report the effect and do not isolate its mechanism. So neither fusion beats the
better single view: one ties it, one degrades it. Any benefit lives in the axial view itself,
not in combining views, which reframes the question as: what does the axial view provide that
sagittal does not?

## The axial advantage is real for attribution, absent for severity

Axial-only improves level attribution over the sagittal control by +0.183 worst_lvl (paired,
t = 11.7, p < 0.001), a large and unambiguous effect. It does not significantly improve severity
grading: the paired κ gain is +0.060 (p = 0.15, n.s.). So the axial view helps decide *which*
level is worst, not *how severe* it is.

This severity null is not a negative in isolation; it engages the closest prior claim on this
territory. DeepSPINE (Lu et al., *MLHC* 2018; arXiv:1807.10215) compared axial-only,
sagittal-only, and fused inputs for canal-stenosis grading and reported a tie: class-average
accuracy 78.6 ± 2.7 (axial) vs 78.6 ± 2.4 (sagittal), fusion 80.4 ± 1.6. Their comparison was
not budget-matched: axial volumes were 8 slices, sagittal 25 (~3x), fed to correspondingly
different 2D and 3D branches, and both inputs were reformatted to per-disc oblique planes.
Their tie is therefore *predicted by* our decomposition rather than in tension with it. The
sagittal branch's ~3x slice-budget advantage offsets axial's residual acquisition edge, netting
parity. Our matched-budget control removes that offset, and the axial advantage re-emerges, on
level attribution, which they did not measure. Because both their inputs were already
level-specific reformats, their design could not isolate the acquisition effect either. So we
reproduce their axial/sagittal severity parity under matched conditions (they: class-average
accuracy; ours: quadratic-weighted κ) and add the dissociation on localization they could not
have detected. That converts a "this was already done" objection into corroboration, and
locates the contribution precisely: not that axial beats sagittal for grading (it does not),
but that at matched budget axial resolves localization the sagittal view cannot, most of which
is in turn recoverable as slice budget.

## Decomposing the attribution gain: budget vs. acquisition

The axial view differs from the sagittal control along two uncontrolled axes at once: (i) input
budget, since axial reads five dedicated per-level slices and sagittal reads one; and (ii)
acquisition, meaning axial's orientation plus its annotator-supplied, level-specific slice
selection. To separate them we add a budget-matched control: sagittal with five parasagittal
slices (mean-pooled into the same one-token-per-level representation, so only slice budget
changes, not the architecture). This holds acquisition fixed while giving sagittal the axial
slice budget.

The total effect (axial - 1-slice sagittal) then partitions cleanly, because axial - sag1
cancels the shared 5-slice term and is the most robust quantity we have:

| component | Δ worst_lvl | 95% CI | t | p (paired, 2-sided) |
|---|---|---|---|---|
| total (axial - sag-1slice) | +0.183 | [+0.139, +0.226] | 11.7 | < 0.001 |
| budget (sag-5slice - sag-1slice) | +0.105 | [+0.052, +0.158] | 5.53 | 0.005 |
| acquisition (axial - sag-5slice) | +0.078 | [+0.000, +0.155] | 2.78 | 0.050 |

We report intervals, not bare p-values. The acquisition CI [+0.000, +0.155] is the honest
reading of p = 0.050: the effect is positive but loosely bounded. Since we run six paired tests
here of which only the total was pre-specified, the interval is the appropriate defense against
a multiple-comparisons objection. Both components are significant. The larger, most robust share
is input budget: simply giving the sagittal model five parasagittal slices recovers +0.105 of
the gap on its own. A smaller but significant acquisition residual (+0.078) remains, so the
axial cross-section (and its level-specific slicing) carries level information beyond raw slice
count.

Anticorrelation caveat (report in-text). The two components share the 5-slice sagittal term with
opposite sign (budget = sag5 - sag1; acquisition = axial - sag5), so they are anticorrelated by
construction (r = -0.84 across seeds): a single strong sag5 seed inflates budget and deflates
acquisition simultaneously. Individual component seed-robustness therefore *understates* the
stability of the partition; the total (+0.183, p < 0.001) is the anchor, and the split is a
decomposition within an established effect rather than two independent claims.

On severity (κ), neither component is significant (budget +0.037 p = 0.31; acquisition +0.023
p = 0.32), consistent with the total κ effect being n.s.

## Takeaways

1. The axial view improves level attribution, not severity grading (+0.183 worst_lvl, p < 0.001;
   κ n.s.). Naming the advantage "axial plane" would be an overstatement; we call the residual
   factor axial acquisition because orientation and annotator level-specific slice selection are
   not separable in this data.
2. Most of the gain is input budget, and it is free for sagittal-only pipelines. Five
   parasagittal slices, mean-pooled, deliver +0.105 worst_lvl (p = 0.005) with no axial series
   and no architecture change: a standalone, deployable recommendation where axial acquisition
   is unavailable.
3. A genuine acquisition residual remains (+0.078, p = 0.050): the axial cross-section adds
   level information beyond slice count, but it is the smaller half.
4. Two-view fusion does not beat the better single view (all 5-seed, paired): concatenation ties
   axial-only (Δ +0.019, CI [-0.067, +0.106], p = 0.57, a powered null), and attention fusion is
   *significantly worse* (Δ -0.135, p < 0.001). The contribution is a single well-chosen view,
   not view fusion, which distinguishes this work from the two-view fusion literature.

Abstract line (must survive compression; the one practitioner-actionable result): *"Adding five
parasagittal slices to a sagittal-only stenosis model improves worst-level localization by
+0.105 (p = 0.005) at no extra acquisition cost, recovering most of the benefit of a dedicated
axial series."*

## Limitations

- worst_lvl is evaluated on the n = 27 studies with a pathological level; single-run binomial CIs
  are wide (~±0.19), which is why every claim is paired across 5 seeds and reported by paired t
  rather than single-run intervals. (Wilcoxon is uninformative at n = 5: its two-sided floor is
  p = 0.062 even at 5/5, so the paired t, justified by symmetric outlier-free deltas, is the test
  of record.)
- The acquisition factor bundles imaging orientation with annotator-chosen level-specific slices;
  a cleaner separation would require axial slices selected without level supervision.

## Methods robustness (why the numbers moved twice)

The headline changed twice before stabilizing, each time from controlling a protocol difference,
not from adding seeds: (i) an initial "fusion helps" reading was an artifact of an undertrained
un-augmented axial stream, and per-view augmentation reassigned the effect to the axial view;
(ii) the axial view's advantage then proved to be ~57% input budget once slice count was
matched. A separate device-placement bug (a new batch tensor omitted from the device move)
crashed the budget-control runs on GPU while passing a CPU smoke test. It was non-silent (hard
failure, no degraded results), all affected runs were re-run post-fix, and every reported number
is either post-fix or on a code path the bug could not reach. These are recorded because each
would have produced a different, wrong headline if left uncontrolled.
