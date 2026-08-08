"""tests/test_robustness_threshold_diagnostics.py - Threshold-diagnostic pure functions.

Covers the pure computation helpers of
``scripts/supplemental/robustness_threshold_figures.py`` with INDEPENDENT
hand-computed synthetic inputs (nothing routed through the real cube, pywrdrb,
or the live threshold registry):

  1. sweep_fractions: inclusive ge/le comparison, non-finite counts as
     unsatisfied in the denominator (the _satisfy rule);
  2. the CDF equivalence the SI presentation claims: for ge the sweep curve is
     the survival function of the sample, for le it is the ECDF;
  3. stringency_coordinate: strict-inequality quantile position, both kinds
     (the compare_designs.default_stringency convention);
  4. sweep_grid: contains the default and every candidate exactly;
  5. theta_for_sows: sow-id-keyed join (realization_id // R; shuffled theta
     rows still land), non-constant theta within a SOW raises, a SOW with no
     theta rows raises;
  6. candidate_menu: the NYC stakeholder floor and the external flood anchor
     are attached only to their own (ANNUAL-named) objectives;
  7. build_recommendation_table: NaN recommendation columns on pass 1, the
     headline flag fires on a large fraction delta and on degeneracy-exit,
     stays quiet for a small move, and appends the joint Starr row;
  8. wilson_ci: contains the point fraction, pins to [0, 1] at the edges,
     narrows with n;
  9. the conjunction helpers (satisfaction_matrix, joint_satisficing_stats,
     failure_combinations, sole_cofailure, failing_count_distribution):
     hand-computed toy matrices, non-finite counts as fail;
 10. critical_m, nearest_to_zero_theta: boundary recovery on a clean step,
     degenerate -> NaN, nearest-to-zero selection.

Run:
    venv/bin/python -m pytest tests/test_robustness_threshold_diagnostics.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def _load_module():
    path = (PROJECT_DIR / "scripts" / "supplemental"
            / "robustness_threshold_figures.py")
    spec = importlib.util.spec_from_file_location(
        "robustness_threshold_figures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rtd = _load_module()


# ---------------------------------------------------------------------------
# sweep_fractions
# ---------------------------------------------------------------------------

def test_sweep_fractions_ge_inclusive_and_nan_fails():
    v = [0.2, 0.5, 0.5, 0.9, np.nan]
    # at t=0.5: 0.5, 0.5, 0.9 satisfy (inclusive); NaN fails; denominator is 5
    assert rtd.sweep_fractions(v, [0.5], "ge")[0] == pytest.approx(3 / 5)
    # at t=0.95 nothing satisfies; at t=0.0 all finite values satisfy
    assert rtd.sweep_fractions(v, [0.95], "ge")[0] == 0.0
    assert rtd.sweep_fractions(v, [0.0], "ge")[0] == pytest.approx(4 / 5)


def test_sweep_fractions_le_inclusive_and_nan_fails():
    v = [1.0, 2.0, 2.0, 10.0, np.nan]
    assert rtd.sweep_fractions(v, [2.0], "le")[0] == pytest.approx(3 / 5)
    assert rtd.sweep_fractions(v, [0.5], "le")[0] == 0.0
    assert rtd.sweep_fractions(v, [10.0], "le")[0] == pytest.approx(4 / 5)


def test_sweep_fractions_unknown_kind_raises():
    with pytest.raises(ValueError):
        rtd.sweep_fractions([1.0], [1.0], "eq")


def test_sweep_curve_is_the_cdf():
    """ge: sat(t) = survival = 1 - ECDF(t-); le: sat(t) = ECDF(t)."""
    rng = np.random.default_rng(7)
    v = rng.normal(size=200)
    grid = np.linspace(-3, 3, 41)
    ecdf = np.array([(v <= t).mean() for t in grid])
    survival = np.array([(v >= t).mean() for t in grid])
    np.testing.assert_allclose(rtd.sweep_fractions(v, grid, "le"), ecdf)
    np.testing.assert_allclose(rtd.sweep_fractions(v, grid, "ge"), survival)


# ---------------------------------------------------------------------------
# stringency_coordinate / sweep_grid
# ---------------------------------------------------------------------------

def test_stringency_coordinate_hand_computed():
    v = [0.1, 0.2, 0.3, 0.4]
    # ge: fraction strictly BELOW the threshold fails marginally
    assert rtd.stringency_coordinate(v, 0.3, "ge") == pytest.approx(2 / 4)
    # le: fraction strictly ABOVE fails marginally
    assert rtd.stringency_coordinate(v, 0.3, "le") == pytest.approx(1 / 4)
    # NaN excluded from the denominator (it is not a quantile position)
    assert rtd.stringency_coordinate([0.1, np.nan, 0.9], 0.5, "ge") == \
        pytest.approx(1 / 2)


def test_sweep_grid_contains_default_and_candidates_exactly():
    v = np.linspace(0.0, 1.0, 50)
    extras = [0.9512345, 0.5, 1.17]  # incl. one outside the support
    grid = rtd.sweep_grid(v, extras, n_points=21)
    for e in extras:
        assert np.any(grid == e)
    assert grid.min() <= 0.0 and grid.max() >= 1.17
    assert np.all(np.diff(grid) > 0)  # sorted, unique


# ---------------------------------------------------------------------------
# theta_for_sows
# ---------------------------------------------------------------------------

def _synthetic_theta(n_sow=4, reals_per_sow=3):
    """Realization ids 0..11, SOW s = rid // 3, theta row = (s, 10s, 100s)."""
    rids = np.arange(n_sow * reals_per_sow)
    sow_ids = rids // reals_per_sow
    theta = np.column_stack([sow_ids, 10.0 * sow_ids, 100.0 * sow_ids]).astype(float)
    return theta, rids


def test_theta_for_sows_joins_by_sow_id_not_position():
    theta, rids = _synthetic_theta()
    perm = np.random.default_rng(3).permutation(len(rids))
    out = rtd.theta_for_sows(theta[perm], rids[perm], realizations_per_sow=3,
                             sow_labels=[0, 1, 2, 3])
    expected = np.column_stack([np.arange(4), 10.0 * np.arange(4),
                                100.0 * np.arange(4)]).astype(float)
    np.testing.assert_allclose(out, expected)


def test_theta_for_sows_nonconstant_block_raises():
    theta, rids = _synthetic_theta()
    theta[4, 0] += 1e-6  # perturb one realization of SOW 1
    with pytest.raises(ValueError, match="not constant"):
        rtd.theta_for_sows(theta, rids, realizations_per_sow=3,
                           sow_labels=[0, 1, 2, 3])


def test_theta_for_sows_missing_sow_raises():
    theta, rids = _synthetic_theta()
    # Drop every realization of SOW 3: the cube's label 3 has no theta rows.
    keep = rids // 3 != 3
    with pytest.raises(ValueError, match="missing"):
        rtd.theta_for_sows(theta[keep], rids[keep], realizations_per_sow=3,
                           sow_labels=[0, 1, 2, 3])


# ---------------------------------------------------------------------------
# candidate_menu
# ---------------------------------------------------------------------------

def test_candidate_menu_floor_and_anchor_routing():
    v = np.linspace(0.0, 1.0, 101)
    floors = {"nyc_delivery_reliability_annual": 0.5}
    # Single external flood anchor since 2026-08-07: the simulated baseline is
    # the anchor script's recomputed value, never a hardcoded number.
    flood = {"observed_2000_2023": 1.17}

    nyc = rtd.candidate_menu("nyc_delivery_reliability_annual", "ge", v, 0.95,
                             anchor_val=0.8, floors=floors, flood_anchors=flood,
                             quantiles=(0.5,))
    assert nyc["current"] == 0.95
    assert nyc["historic_anchor"] == 0.8
    assert nyc["stakeholder_floor"] == 0.5
    assert nyc["sow_p50"] == pytest.approx(0.5)
    assert not any(k.startswith("anchor_") for k in nyc)

    fl = rtd.candidate_menu("downstream_flood_exceedance_annual", "le", v, 1.0,
                            anchor_val=0.35, floors=floors, flood_anchors=flood,
                            quantiles=(0.5,))
    assert fl["anchor_observed_2000_2023"] == 1.17
    assert "anchor_simulated_baseline" not in fl
    assert "stakeholder_floor" not in fl

    other = rtd.candidate_menu("nj_delivery_reliability_annual", "ge", v, 0.95,
                               anchor_val=None, floors=floors,
                               flood_anchors=flood, quantiles=(0.5,))
    assert "historic_anchor" not in other  # None anchor is dropped
    assert "stakeholder_floor" not in other


def test_stakeholder_floor_keys_are_annual():
    """The translated floor map must key by the annual reporting names."""
    assert all(k in rtd.SHORT_LABELS for k in rtd.STAKEHOLDER_FLOORS)


# ---------------------------------------------------------------------------
# build_recommendation_table
# ---------------------------------------------------------------------------

def _one_obj_inputs(values):
    names = ["obj_a"]
    kinds = {"obj_a": "ge"}
    current = {"obj_a": 0.95}
    sow_vals = {"obj_a": np.asarray(values, dtype=float)}
    return names, kinds, current, sow_vals


def test_recommendation_table_pass1_is_nan_and_unflagged():
    names, kinds, current, sow_vals = _one_obj_inputs(np.linspace(0, 1, 100))
    df = rtd.build_recommendation_table(names, kinds, current, sow_vals,
                                        recommended={}, basis={},
                                        headline_delta=0.10)
    row = df.iloc[0]
    assert np.isnan(row["recommended_threshold"])
    assert np.isnan(row["frac_sow_at_recommended"])
    assert not row["headline_impact"]


def test_recommendation_headline_flags_large_delta_and_degeneracy_exit():
    # values uniform on [0,1]: frac at 0.95 ~ 0.06, at 0.5 ~ 0.51 -> flags
    names, kinds, current, sow_vals = _one_obj_inputs(np.linspace(0, 1, 101))
    df = rtd.build_recommendation_table(names, kinds, current, sow_vals,
                                        recommended={"obj_a": 0.5},
                                        basis={"obj_a": "test"},
                                        headline_delta=0.10)
    assert bool(df.iloc[0]["headline_impact"])

    # degenerate current (all fail at 2.0) moving to a discriminating placement
    names, kinds, current, sow_vals = _one_obj_inputs(np.linspace(0, 1, 101))
    current = {"obj_a": 2.0}
    df = rtd.build_recommendation_table(names, kinds, current, sow_vals,
                                        recommended={"obj_a": 0.5},
                                        basis={}, headline_delta=10.0)
    assert df.iloc[0]["frac_sow_at_current"] == 0.0
    assert bool(df.iloc[0]["headline_impact"])  # degeneracy-exit, delta gate off


def test_recommendation_small_move_not_flagged():
    names, kinds, current, sow_vals = _one_obj_inputs(np.linspace(0, 1, 101))
    current = {"obj_a": 0.50}
    df = rtd.build_recommendation_table(names, kinds, current, sow_vals,
                                        recommended={"obj_a": 0.52},
                                        basis={}, headline_delta=0.10)
    assert not bool(df.iloc[0]["headline_impact"])


def test_recommendation_appends_joint_starr_row():
    # Single objective: the joint conjunction equals the marginal fraction.
    names, kinds, current, sow_vals = _one_obj_inputs(np.linspace(0, 1, 101))
    df = rtd.build_recommendation_table(names, kinds, current, sow_vals,
                                        recommended={}, basis={},
                                        headline_delta=0.10)
    assert len(df) == 2
    joint = df.iloc[-1]
    assert joint["objective"] == "ALL__joint_starr"
    assert joint["frac_sow_at_current"] == pytest.approx(
        df.iloc[0]["frac_sow_at_current"])
    # Wilson CI columns bracket the point fraction
    assert (df.iloc[0]["frac_sow_at_current_ci_lo"]
            <= df.iloc[0]["frac_sow_at_current"]
            <= df.iloc[0]["frac_sow_at_current_ci_hi"])


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

def test_wilson_ci_brackets_and_edges():
    lo, hi = rtd.wilson_ci(0.5, 1000, conf=0.95)
    assert lo < 0.5 < hi
    # symmetric around 0.5 and roughly the +/-1.6pp convention wide
    assert (0.5 - lo) == pytest.approx(hi - 0.5)
    assert 0.02 < (hi - lo) < 0.08
    # p = 0 pins the lower bound to exactly 0; p = 1 the upper to exactly 1
    lo0, hi0 = rtd.wilson_ci(0.0, 100)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and hi0 > 0.0
    lo1, hi1 = rtd.wilson_ci(1.0, 100)
    assert hi1 == pytest.approx(1.0, abs=1e-12) and lo1 < 1.0


def test_wilson_ci_narrows_with_n():
    lo_s, hi_s = rtd.wilson_ci(0.3, 100)
    lo_l, hi_l = rtd.wilson_ci(0.3, 10000)
    assert (hi_l - lo_l) < (hi_s - lo_s)


# ---------------------------------------------------------------------------
# Conjunction helpers
# ---------------------------------------------------------------------------

#: 4 units x 2 criteria (A, B): rows are (pass, pass), (fail, pass),
#: (fail, pass), (fail, fail) -> marginals A 0.25, B 0.75; joint 0.25.
_TOY_SAT = np.array([
    [True, True],
    [False, True],
    [False, True],
    [False, False],
])


def test_satisfaction_matrix_nan_fails():
    vals = {"a": [1.0, np.nan], "b": [0.0, 5.0]}
    sat = rtd.satisfaction_matrix(vals, ["a", "b"],
                                  {"a": 0.5, "b": 1.0}, {"a": "ge", "b": "le"})
    np.testing.assert_array_equal(sat, [[True, True], [False, False]])


def test_joint_satisficing_stats_hand_computed():
    stats = rtd.joint_satisficing_stats(_TOY_SAT, ["A", "B"])
    assert stats["joint_frac"] == pytest.approx(1 / 4)
    assert stats["comonotone_benchmark"] == pytest.approx(1 / 4)   # min marginal
    assert stats["independence_benchmark"] == pytest.approx(0.25 * 0.75)
    assert stats["co_occurrence_gap"] == pytest.approx(1 / 4 - 0.1875)
    assert stats["binding_criterion"] == "A"
    assert stats["binding_marginal_frac"] == pytest.approx(1 / 4)


def test_failure_combinations_counts_and_pooling():
    df = rtd.failure_combinations(_TOY_SAT, ["A", "B"], top_k=1)
    top = df.iloc[0]
    assert top["failing_criteria"] == "A" and top["count"] == 2
    assert top["frac"] == pytest.approx(2 / 4)
    other = df[df["failing_criteria"] == "(other combinations)"].iloc[0]
    assert other["count"] == 1                                     # the A+B row
    none = df[df["failing_criteria"] == "(none)"].iloc[0]
    assert none["count"] == 1 and none["n_failing"] == 0
    assert df["count"].sum() == len(_TOY_SAT)


def test_sole_cofailure_and_count_distribution():
    sole, co = rtd.sole_cofailure(_TOY_SAT)
    np.testing.assert_allclose(sole, [2 / 4, 0.0])
    np.testing.assert_allclose(co, [1 / 4, 1 / 4])
    # sole + co reproduces the marginal failure fraction
    np.testing.assert_allclose(sole + co, (~_TOY_SAT).mean(axis=0))
    dist = rtd.failing_count_distribution(_TOY_SAT)
    np.testing.assert_allclose(dist, [1 / 4, 2 / 4, 1 / 4])


# ---------------------------------------------------------------------------
# critical_m / nearest_to_zero_theta
# ---------------------------------------------------------------------------

def test_critical_m_recovers_step_and_nan_when_degenerate():
    m = np.linspace(0.0, 1.0, 400)
    boundary = rtd.critical_m(m, m > 0.5, window=51)
    assert boundary == pytest.approx(0.5, abs=0.02)
    assert np.isnan(rtd.critical_m(m, np.ones_like(m, dtype=bool), window=51))
    with pytest.raises(ValueError, match="window"):
        rtd.critical_m(m[:10], m[:10] > 0.5, window=51)


def test_nearest_to_zero_theta():
    theta = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.0], [5.0, 5.0]])
    idx = rtd.nearest_to_zero_theta(theta, 2)
    assert set(idx.tolist()) == {0, 2}
