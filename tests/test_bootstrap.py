"""Bootstrap over studies: does it reproduce the published numbers, and does
the pairing actually survive?

The pairing test is the one that matters. Bootstrapping a difference directly
is only worth doing if it gives a different - tighter - answer than
bootstrapping the two accuracies separately. If those agree, the pairing is not
being preserved and the CIs are wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.bootstrap_worst_level import (  # noqa: E402
    CONFIGS,
    SEEDS,
    ci,
    per_study_worst_level,
    seed_ci,
    worst_level_from_studies,
)

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs_modal"
PUBLISHED = {"sag-1slice": 0.471, "sag-5slice": 0.576, "axial": 0.654,
             "fusion-concat": 0.673, "fusion-attn": 0.519}

have_preds = all((OUTPUTS / p.format(s) / "test_predictions.json").exists()
                 for p in CONFIGS.values() for s in SEEDS)
needs_preds = pytest.mark.skipif(not have_preds, reason="per-study predictions not present")


# --------------------------------------------------------------------------
# metric replication
# --------------------------------------------------------------------------


def test_tie_for_worst_counts_as_correct():
    """Two levels tied at the worst grade - naming either one is correct."""
    pred = {"studyids": [1, 1, 1], "levels": [0, 1, 2],
            "targets": [2, 2, 0], "preds": [0, 5, 0]}
    assert per_study_worst_level(pred) == {1: 1}
    pred["preds"] = [5, 0, 0]
    assert per_study_worst_level(pred) == {1: 1}
    pred["preds"] = [0, 0, 5]
    assert per_study_worst_level(pred) == {1: 0}


def test_studies_with_no_pathology_are_excluded():
    pred = {"studyids": [1, 1, 2, 2], "levels": [0, 1, 0, 1],
            "targets": [0, 0, 0, 2], "preds": [1, 0, 0, 3]}
    out = per_study_worst_level(pred)
    assert 1 not in out, "a study with nothing pathological has no worst level"
    assert out == {2: 1}


@needs_preds
def test_reproduces_published_table_1():
    for name, pattern in CONFIGS.items():
        accs = []
        for s in SEEDS:
            import json
            pred = json.loads((OUTPUTS / pattern.format(s) / "test_predictions.json").read_text())
            flags = per_study_worst_level(pred)
            accs.append(worst_level_from_studies(flags, sorted(flags)))
        assert np.mean(accs) == pytest.approx(PUBLISHED[name], abs=0.001), name


@needs_preds
def test_split_size_varies_by_seed():
    """The draft says n=27; it is 27-34. The bootstrap must not assume otherwise."""
    import json
    sizes = []
    for s in SEEDS:
        pred = json.loads(
            (OUTPUTS / CONFIGS["sag-1slice"].format(s) / "test_predictions.json").read_text())
        sizes.append(len(per_study_worst_level(pred)))
    assert sizes == [27, 27, 33, 34, 27]


# --------------------------------------------------------------------------
# the pairing
# --------------------------------------------------------------------------


def _correlated_pair(n=30, seed=0, shared=0.8):
    """Two configurations that mostly agree study by study, as real ones do."""
    rng = np.random.default_rng(seed)
    base = rng.random(n) < 0.55
    flip_a = rng.random(n) < (1 - shared)
    flip_b = rng.random(n) < (1 - shared)
    a = {i: int(base[i] ^ flip_a[i]) for i in range(n)}
    b = {i: int(base[i] ^ flip_b[i]) for i in range(n)}
    return a, b


def test_paired_bootstrap_is_tighter_than_unpaired():
    """If the pairing is preserved, the difference CI must be narrower than
    what you get by resampling the two configurations independently.

    Agreement 0.9 is the realistic setting: the configurations being compared
    here differ by 0.019-0.135 in accuracy, so they agree on most studies.
    """
    a, b = _correlated_pair(shared=0.9)
    studies = np.array(sorted(a))
    rng = np.random.default_rng(1)
    iters = 4000

    paired, unpaired = np.empty(iters), np.empty(iters)
    for i in range(iters):
        draw = studies[rng.integers(0, len(studies), len(studies))]
        paired[i] = (worst_level_from_studies(a, draw)
                     - worst_level_from_studies(b, draw))
        d1 = studies[rng.integers(0, len(studies), len(studies))]
        d2 = studies[rng.integers(0, len(studies), len(studies))]
        unpaired[i] = (worst_level_from_studies(a, d1)
                       - worst_level_from_studies(b, d2))

    plo, phi = ci(paired)
    ulo, uhi = ci(unpaired)
    assert (phi - plo) < 0.7 * (uhi - ulo), (
        f"paired width {phi - plo:.3f} vs unpaired {uhi - ulo:.3f} - "
        "pairing is not being preserved")


def test_identical_configs_have_zero_paired_difference():
    a, _ = _correlated_pair()
    studies = np.array(sorted(a))
    rng = np.random.default_rng(2)
    diffs = []
    for _ in range(500):
        draw = studies[rng.integers(0, len(studies), len(studies))]
        diffs.append(worst_level_from_studies(a, draw) - worst_level_from_studies(a, draw))
    assert np.allclose(diffs, 0.0), "a configuration against itself must be exactly zero"


# --------------------------------------------------------------------------
# interval mechanics
# --------------------------------------------------------------------------


def test_ci_is_a_percentile_interval():
    x = np.arange(10001, dtype=float)
    lo, hi = ci(x)
    assert lo == pytest.approx(250.0, abs=1.0)
    assert hi == pytest.approx(9750.0, abs=1.0)


def test_seed_ci_uses_t_not_normal():
    """n=5 means 4 df; the t multiplier is 2.776, not 1.96."""
    lo, hi = seed_ci(0.5, 0.1, n=5)
    half = (hi - lo) / 2
    assert half == pytest.approx(2.776 * 0.1 / np.sqrt(5), rel=1e-3)


def test_seed_ci_narrows_with_more_seeds():
    w5 = np.diff(seed_ci(0.5, 0.1, n=5))[0]
    w20 = np.diff(seed_ci(0.5, 0.1, n=20))[0]
    assert w20 < w5



def test_paired_advantage_grows_with_agreement():
    """The tighter two configurations agree, the more the pairing buys."""
    widths = {}
    for shared in (0.6, 0.9):
        a, b = _correlated_pair(shared=shared)
        studies = np.array(sorted(a))
        rng = np.random.default_rng(1)
        diffs = []
        for _ in range(3000):
            draw = studies[rng.integers(0, len(studies), len(studies))]
            diffs.append(worst_level_from_studies(a, draw)
                         - worst_level_from_studies(b, draw))
        lo, hi = ci(diffs)
        widths[shared] = hi - lo
    assert widths[0.9] < widths[0.6], widths


# --------------------------------------------------------------------------
# the prediction files disagree on one key name
# --------------------------------------------------------------------------


def test_reads_either_prediction_key():
    """Existing runs wrote "preds"; the current tree writes "predictions"."""
    from scripts.bootstrap_worst_level import prediction_array

    body = {"studyids": [1, 1], "levels": [0, 1], "targets": [0, 2]}
    old = {**body, "preds": [0, 3]}
    new = {**body, "predictions": [0, 3]}
    assert np.array_equal(prediction_array(old), prediction_array(new))
    assert per_study_worst_level(old) == per_study_worst_level(new) == {1: 1}


def test_missing_prediction_key_is_reported_clearly():
    from scripts.bootstrap_worst_level import prediction_array

    with pytest.raises(SystemExit, match="none of"):
        prediction_array({"studyids": [1], "levels": [0], "targets": [0]})
