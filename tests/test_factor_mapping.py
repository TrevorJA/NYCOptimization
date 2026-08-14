"""
tests/test_factor_mapping.py - Success/failure surface classification.

The module's purpose is to recover a planted success/failure boundary and to
be HONEST about skill, so every test plants a known structure:

  1. ``success_labels`` applies a subset criterion vector per SOW (non-finite
     values unsatisfied).
  2. ``fit_classifier`` recovers a planted separable boundary with high
     cross-validated AUC, and reports the positive class's probability.
  3. Degenerate labels (one class) return the declared ``single_class``
     backend with a recorded reason instead of raising -- a near-saturated
     criterion set is data, not a crash.
  4. A tiny minority class reduces the CV fold count or skips CV with a
     recorded reason (never a silent NaN).
  5. ``probability_surface`` grids the fitted probability over the top-2 axes
     with off-plane axes at their median.
  6. ``theta`` alignment and reference-SOW markers behave as documented.

Run:
    venv/bin/python -m pytest tests/test_factor_mapping.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import src.factor_mapping as fm  # noqa: E402
import src.robustness as rob  # noqa: E402

RNG = np.random.default_rng(3)
N = 300
AXES = ["em", "r1", "r2"]


@pytest.fixture(scope="module")
def planted():
    """(X, y): success iff em > 0.9 -- a clean axis-aligned boundary."""
    X = np.column_stack([
        RNG.uniform(0.7, 1.3, N),      # em
        RNG.uniform(0.0, 1.0, N),      # r1 (noise)
        RNG.uniform(0.0, 1.0, N),      # r2 (noise)
    ])
    return X, X[:, 0] > 0.9


def test_matrix_success_labels_subset_and_nonfinite():
    values = np.array([[0.9, 5.0], [0.7, 5.0], [np.nan, 5.0]])
    thr = {"rel": 0.8, "deficit": np.inf}          # deficit non-binding
    kinds = {"rel": "ge", "deficit": "le"}
    ok = fm.matrix_success_labels(values, ["rel", "deficit"], thr, kinds)
    assert ok.tolist() == [True, False, False], \
        "non-finite values must be unsatisfied; non-binding axes must pass"


def test_success_labels_indexes_the_requested_solution():
    cube = np.zeros((2, 3, 1))
    cube[0, :, 0] = [1.0, 0.0, 1.0]
    cube[1, :, 0] = [0.0, 0.0, 0.0]
    raw = rob.RawCube(
        cube=cube, solution_ids=[10, 11], sow_labels=[0, 1, 2],
        obj_names=["rel"], thresholds={"rel": 0.5}, kinds={"rel": "ge"},
        directions={"rel": "maximize"}, is_ensemble=True,
        realizations_per_sow=1, meta={},
    )
    ok = fm.success_labels(raw, {"rel": 0.5}, solution_index=0)
    assert ok.tolist() == [True, False, True]


def test_classifier_recovers_planted_boundary(planted):
    X, y = planted
    fit = fm.fit_classifier(X, y, AXES, space="theta")
    assert fit.backend == "gradient_boosting"
    assert fit.axes[int(np.argmax(fit.importances))] == "em"
    assert fit.cv_auc > 0.9, "a clean separable boundary must CV near-perfectly"
    assert fit.cv_note == ""
    assert fit.n_pos + fit.n_neg == N
    # predict_proba reports P(success): deep-dry point ~0, wet point ~1.
    p = fit.predict_proba(np.array([[0.75, 0.5, 0.5], [1.25, 0.5, 0.5]]))
    assert p[0] < 0.2 and p[1] > 0.8


def test_single_class_labels_degrade_loudly(planted):
    X, _ = planted
    fit = fm.fit_classifier(X, np.ones(N, dtype=bool), AXES)
    assert fit.backend == "single_class"
    assert "one class" in fit.cv_note
    assert np.isnan(fit.cv_auc)
    assert np.all(fit.predict_proba(X[:5]) == 1.0)


def test_tiny_minority_class_reduces_or_skips_cv(planted):
    X, _ = planted
    y = np.zeros(N, dtype=bool)
    y[:3] = True                                    # 3 positives, 5 requested folds
    fit = fm.fit_classifier(X, y, AXES, cv=5)
    assert fit.backend == "gradient_boosting"
    assert np.isfinite(fit.cv_auc)                  # folds reduced to 3, not skipped

    y1 = np.zeros(N, dtype=bool)
    y1[0] = True                                    # 1 positive: CV impossible
    fit1 = fm.fit_classifier(X, y1, AXES, cv=5)
    assert np.isnan(fit1.cv_auc)
    assert "minority class" in fit1.cv_note


def test_probability_surface_grid_and_anchor(planted):
    X, y = planted
    fit = fm.fit_classifier(X, y, AXES)
    g1, g2, P = fm.probability_surface(fit, X, 0, 1, grid_res=25)
    assert g1.shape == (25,) and g2.shape == (25,) and P.shape == (25, 25)
    assert g1.min() == pytest.approx(X[:, 0].min())
    assert g1.max() == pytest.approx(X[:, 0].max())
    assert np.all((P >= 0) & (P <= 1))
    # The boundary is on axis 0, so P must rise along g1 (columns), averaged
    # over the other axis.
    col_means = P.mean(axis=0)
    assert col_means[-1] - col_means[0] > 0.5


def test_top_axes_orders_by_importance(planted):
    X, y = planted
    fit = fm.fit_classifier(X, y, AXES)
    top = fm.top_axes(fit, 2)
    assert top[0] == 0                              # em dominates
    assert len(top) == 2 and top[1] in (1, 2)


def test_theta_alignment_guard():
    raw = rob.RawCube(
        cube=np.zeros((1, 4, 1)), solution_ids=[0], sow_labels=[0, 1, 2, 3],
        obj_names=["rel"], thresholds={"rel": 0.5}, kinds={"rel": "ge"},
        directions={"rel": "maximize"}, is_ensemble=True,
        realizations_per_sow=1, meta={},
    )
    fm.assert_theta_alignment(np.zeros((4, 3)), raw)   # aligned: no raise
    with pytest.raises(ValueError):
        fm.assert_theta_alignment(np.zeros((5, 3)), raw)


def test_reference_sows_expected_and_dry(planted):
    X, _ = planted
    refs = fm.reference_sows(X, AXES)
    assert list(refs["role"]) == ["expected", "dry"]
    expected = refs[refs["role"] == "expected"].iloc[0]
    assert expected["em"] == pytest.approx(np.median(X[:, 0]))
    dry = refs[refs["role"] == "dry"].iloc[0]
    assert dry["em"] == pytest.approx(X[:, 0].min())
