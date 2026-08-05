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
  5. theta_for_sows: id-keyed join (shuffled realization ids still land),
     non-constant theta within a SOW raises, missing realization raises;
  6. candidate_menu: the NYC stakeholder floor and flood anchors are attached
     only to their own objectives;
  7. build_recommendation_table: NaN recommendation columns on pass 1, the
     headline flag fires on a large fraction delta and on degeneracy-exit,
     and stays quiet for a small move.

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
    """Realization ids 0..11, SOW s owns [3s, 3s+2], theta row = (s, 10s, 100s)."""
    rids = np.arange(n_sow * reals_per_sow)
    sow_ids = rids // reals_per_sow
    theta = np.column_stack([sow_ids, 10.0 * sow_ids, 100.0 * sow_ids]).astype(float)
    return theta, rids, sow_ids


def test_theta_for_sows_joins_by_id_not_position():
    theta, rids, sow_ids = _synthetic_theta()
    perm = np.random.default_rng(3).permutation(len(rids))
    out = rtd.theta_for_sows(theta[perm], rids[perm], list(rids),
                             list(sow_ids), sow_labels=[0, 1, 2, 3])
    expected = np.column_stack([np.arange(4), 10.0 * np.arange(4),
                                100.0 * np.arange(4)]).astype(float)
    np.testing.assert_allclose(out, expected)


def test_theta_for_sows_nonconstant_block_raises():
    theta, rids, sow_ids = _synthetic_theta()
    theta[4, 0] += 1e-6  # perturb one realization of SOW 1
    with pytest.raises(ValueError, match="not constant"):
        rtd.theta_for_sows(theta, rids, list(rids), list(sow_ids),
                           sow_labels=[0, 1, 2, 3])


def test_theta_for_sows_missing_realization_raises():
    theta, rids, sow_ids = _synthetic_theta()
    with pytest.raises(ValueError, match="missing"):
        rtd.theta_for_sows(theta[:-1], rids[:-1], list(rids), list(sow_ids),
                           sow_labels=[0, 1, 2, 3])


# ---------------------------------------------------------------------------
# candidate_menu
# ---------------------------------------------------------------------------

def test_candidate_menu_floor_and_anchor_routing():
    v = np.linspace(0.0, 1.0, 101)
    floors = {"nyc_delivery_reliability_weekly": 0.5}
    flood = {"observed_2000_2023": 1.17, "simulated_baseline": 0.35}

    nyc = rtd.candidate_menu("nyc_delivery_reliability_weekly", "ge", v, 0.95,
                             anchor_val=0.8, floors=floors, flood_anchors=flood,
                             quantiles=(0.5,))
    assert nyc["current"] == 0.95
    assert nyc["historic_anchor"] == 0.8
    assert nyc["stakeholder_floor"] == 0.5
    assert nyc["sow_p50"] == pytest.approx(0.5)
    assert not any(k.startswith("anchor_") for k in nyc)

    fl = rtd.candidate_menu("downstream_flood_exceedance_minor", "le", v, 1.0,
                            anchor_val=0.35, floors=floors, flood_anchors=flood,
                            quantiles=(0.5,))
    assert fl["anchor_observed_2000_2023"] == 1.17
    assert fl["anchor_simulated_baseline"] == 0.35
    assert "stakeholder_floor" not in fl

    other = rtd.candidate_menu("nj_delivery_reliability_weekly", "ge", v, 0.95,
                               anchor_val=None, floors=floors,
                               flood_anchors=flood, quantiles=(0.5,))
    assert "historic_anchor" not in other  # None anchor is dropped
    assert "stakeholder_floor" not in other


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
