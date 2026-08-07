"""Tests for the epsilon-calibration shared utilities.

Covers the pure-arithmetic pieces of `src.sensitivity_common` added for the
epsilon-calibration experiment (no simulation anywhere):

* `sample_feasible_dvs` — uniform-on-feasible rejection sampling against the
  formal Borg constraints (feasibility, bounds, determinism, the draw cap);
* `apply_operator_rows` — vectorized §2 unit operators must reproduce the
  scalar operators exactly, including the non-finite sentinel semantics;
* `epsilon_nondominated` — Borg-convention ε-box archive filter (same-box
  corner tie-break, cross-box dominance, monotone shrink under coarser
  epsilons, non-finite exclusion, input validation);
* `ceil_to_clean_step` — epsilon rounding to clean native-unit steps.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.objectives_ensemble import (
    FailureFrequencyOp,
    PooledMeanOp,
    PooledPercentileOp,
)
from src.sensitivity_common import (
    apply_operator_rows,
    ceil_to_clean_step,
    epsilon_nondominated,
    sample_feasible_dvs,
)
from src.formulations import get_bounds
from src.simulation import compute_constraint_violations


###############################################################################
# sample_feasible_dvs
###############################################################################

def test_sample_feasible_dvs_returns_feasible_in_bounds():
    dvs, info = sample_feasible_dvs("ffmp", seed=123, n_samples=4)
    lows, highs = get_bounds("ffmp")
    assert dvs.shape == (4, len(lows))
    assert (dvs >= lows).all() and (dvs <= highs).all()
    for dv in dvs:
        assert compute_constraint_violations(dv, "ffmp") == [0.0]
    assert info["n_draws"] >= 4
    assert 0.0 < info["acceptance_rate"] <= 1.0


def test_sample_feasible_dvs_is_deterministic():
    a, info_a = sample_feasible_dvs("ffmp", seed=7, n_samples=3)
    b, info_b = sample_feasible_dvs("ffmp", seed=7, n_samples=3)
    np.testing.assert_array_equal(a, b)
    assert info_a == info_b


def test_sample_feasible_dvs_draw_cap_raises():
    with pytest.raises(RuntimeError, match="feasible"):
        # A cap of one small chunk cannot yield 1000 feasible vectors at ~1%
        # acceptance.
        sample_feasible_dvs("ffmp", seed=1, n_samples=1000,
                            chunk=512, max_draws=512)


###############################################################################
# apply_operator_rows — vectorized ops == scalar ops (incl. NaN sentinels)
###############################################################################

@pytest.mark.parametrize("op", [
    FailureFrequencyOp(k=1),
    FailureFrequencyOp(k=3),
    PooledPercentileOp(99.0, worst_value=100.0),
    PooledPercentileOp(1.0, worst_value=0.0),
    PooledMeanOp(worst_value=366.0),
])
def test_apply_operator_rows_matches_scalar(op):
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.0, 10.0, size=(20, 30))
    arr[rng.random(arr.shape) < 0.1] = np.nan  # sentinel path exercised
    vec = apply_operator_rows(op, arr)
    ref = np.array([op(row) for row in arr])
    np.testing.assert_allclose(vec, ref, rtol=0, atol=1e-12)


def test_apply_operator_rows_fallback_for_unknown_callable():
    arr = np.arange(12.0).reshape(3, 4)
    vec = apply_operator_rows(lambda u: float(np.max(u)), arr)
    np.testing.assert_array_equal(vec, [3.0, 7.0, 11.0])


def test_apply_operator_rows_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        apply_operator_rows(PooledMeanOp(0.0), np.zeros(5))


###############################################################################
# epsilon_nondominated
###############################################################################

def test_same_box_keeps_corner_closest():
    F = np.array([[0.1, 0.9], [0.15, 0.95], [0.5, 0.5]])
    kept = epsilon_nondominated(F, [1.0, 1.0])
    # All share box (0, 0); [0.5, 0.5] is closest to the corner (dist 0.5
    # vs 0.82 and 0.925).
    np.testing.assert_array_equal(kept, [2])


def test_cross_box_dominance():
    F = np.array([[0.1, 0.1], [0.9, 0.9]])
    kept = epsilon_nondominated(F, [0.2, 0.2])
    np.testing.assert_array_equal(kept, [0])  # box (0,0) dominates (4,4)


def test_tradeoff_boxes_all_kept():
    F = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
    kept = epsilon_nondominated(F, [0.2, 0.2])
    np.testing.assert_array_equal(kept, [0, 1, 2])  # boxes (0,4),(4,0),(2,2)


def test_coarser_epsilon_never_grows_archive():
    rng = np.random.default_rng(3)
    F = rng.uniform(0.0, 1.0, size=(200, 3))
    fine = len(epsilon_nondominated(F, [0.01] * 3))
    coarse = len(epsilon_nondominated(F, [0.25] * 3))
    assert coarse <= fine


def test_nonfinite_rows_excluded():
    F = np.array([[0.5, 0.5], [np.nan, 0.1], [0.4, np.inf]])
    kept = epsilon_nondominated(F, [0.1, 0.1])
    np.testing.assert_array_equal(kept, [0])


def test_epsilon_validation():
    F = np.zeros((2, 2))
    with pytest.raises(ValueError, match="positive"):
        epsilon_nondominated(F, [0.1, 0.0])
    with pytest.raises(ValueError, match="mismatch"):
        epsilon_nondominated(F, [0.1])


###############################################################################
# ceil_to_clean_step
###############################################################################

@pytest.mark.parametrize("x, expected", [
    (0.0117, 0.015),
    (0.9, 1.0),
    (1.34, 1.5),
    (2.0, 2.0),       # already clean stays put
    (0.05, 0.05),
    (0.003, 0.005),   # skips 0.0025 (below), lands on 0.005
    (3.1, 5.0),
    (7.0, 10.0),      # falls through the mantissa list to the next decade
])
def test_ceil_to_clean_step(x, expected):
    assert ceil_to_clean_step(x) == pytest.approx(expected)


@pytest.mark.parametrize("x", [0.0, -1.0, float("nan"), float("inf")])
def test_ceil_to_clean_step_degenerate(x):
    assert np.isnan(ceil_to_clean_step(x))
