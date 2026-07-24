"""Tests for the formal Borg constraint functions.

Covers `compute_constraint_violations` (pure DV arithmetic, no simulation)
and its registry wiring (`get_n_constrs`, `make_constraint_function`):
baseline feasibility, hand-computed directional violations, the tolerance
floor, and the clamp-equivalence property — each violation is positive
exactly when the corresponding apply-time clamp in `dvs_to_config` fires.

Zone-curve crossings are deliberately clamp-only (no constraint): the
monotonicity clamp resolves them at apply time and the clamped geometry is
the intended policy.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.formulations import (
    CONSTRAINT_NAMES,
    get_baseline_values,
    get_bounds,
    get_constraint_names,
    get_n_constrs,
    get_var_names,
    make_constraint_function,
)
from src.simulation import (
    _CFS_TO_MGD,
    _get_cached_defaults,
    compute_constraint_violations,
    dvs_to_config,
)

RESERVOIRS = ["cannonsville", "pepacton", "neversink"]


def _dv(formulation="ffmp", **overrides):
    names = get_var_names(formulation)
    dv = get_baseline_values(formulation).copy()
    for name, value in overrides.items():
        dv[names.index(name)] = value
    return dv


def _flood_date_cols(cfg):
    return [c for c in cfg.mrf_factors_daily_df.columns
            if c not in ("doy", "profile", "type")]


###############################################################################
# Registry wiring
###############################################################################

def test_registry():
    assert get_n_constrs() == 2
    assert get_constraint_names() == CONSTRAINT_NAMES == [
        "delivery_monotonicity", "flood_zone_ordering",
    ]
    fn = make_constraint_function("ffmp")
    cons = fn(list(get_baseline_values("ffmp")))
    assert isinstance(cons, list) and len(cons) == 2
    assert all(isinstance(c, float) for c in cons)


###############################################################################
# Baseline feasibility (exact zeros — Borg treats any nonzero as infeasible)
###############################################################################

@pytest.mark.parametrize("formulation", ["ffmp", "ffmp_10"])
def test_baseline_is_exactly_feasible(formulation):
    cons = compute_constraint_violations(
        get_baseline_values(formulation), formulation
    )
    assert cons == [0.0, 0.0]


def test_zone_crossings_are_not_constrained():
    # Crossing-inducing shifts are clamp-only: still constraint-feasible.
    cons = compute_constraint_violations(
        _dv(zone_vshift_level1c_c0=0.05, zone_vshift_level1b_c0=-0.10,
            zone_vshift_level1b_c1=-0.10, zone_tshift_level1c_c0=30.0),
        "ffmp",
    )
    assert cons == [0.0, 0.0]


###############################################################################
# Hand-computed directional violations
###############################################################################

def test_delivery_violation_value():
    # NYC: L4 (0.95) > L3 (0.60) and NJ: L5 (1.0) > L4 (0.80).
    cons = compute_constraint_violations(
        _dv(nyc_drought_factor_L3=0.60, nyc_drought_factor_L4=0.95,
            nyc_drought_factor_L5=0.90,
            nj_drought_factor_L4=0.80, nj_drought_factor_L5=1.0),
        "ffmp",
    )
    assert cons[0] == pytest.approx((0.95 - 0.60) + (1.0 - 0.80))
    assert cons[1] == 0.0


def test_flood_violation_value():
    # 2.0 x L1b vs 0.5 x L1a violates even outside the equal-rate window
    # (2.0*600 > 0.5*1500 at Cannonsville).
    cons = compute_constraint_violations(
        _dv(flood_release_scale_l1a_cannonsville=0.5,
            flood_release_scale_l1b_cannonsville=2.0),
        "ffmp",
    )
    cfg = _get_cached_defaults()
    date_cols = _flood_date_cols(cfg)
    baseline = float(cfg.constants["mrf_baseline_cannonsville"])
    cap = float(cfg.constants["flood_max_release_cannonsville_cfs"]) * _CFS_TO_MGD
    f_a = cfg.mrf_factors_daily_df.loc[
        "level1a_factor_mrf_cannonsville", date_cols].values.astype(float)
    f_b = cfg.mrf_factors_daily_df.loc[
        "level1b_factor_mrf_cannonsville", date_cols].values.astype(float)
    eff_a = np.minimum(f_a * baseline * 0.5, cap)
    eff_b = np.minimum(f_b * baseline * 2.0, cap)
    expected = max(0.0, float((eff_b - eff_a).max())) / baseline
    assert expected > 0.0
    assert cons[1] == pytest.approx(expected)
    assert cons[0] == 0.0


def test_flood_equal_multipliers_feasible():
    cons = compute_constraint_violations(
        _dv(flood_release_scale_l1a_pepacton=1.2,
            flood_release_scale_l1b_pepacton=1.2),
        "ffmp",
    )
    assert cons[1] == 0.0


###############################################################################
# Tolerance floor
###############################################################################

def test_tiny_violation_floors_to_exact_zero():
    dv = _dv(nyc_drought_factor_L3=0.85,
             nyc_drought_factor_L4=0.85 + 1e-12,
             nyc_drought_factor_L5=0.65)
    cons = compute_constraint_violations(dv, "ffmp")
    assert cons[0] == 0.0


###############################################################################
# Clamp equivalence: c_i > 0 iff the corresponding apply-time clamp fires
###############################################################################

@pytest.mark.parametrize("formulation", ["ffmp"])
def test_clamp_equivalence_on_random_vectors(formulation):
    names = get_var_names(formulation)
    lower, upper = get_bounds(formulation)
    rng = np.random.default_rng(42)

    for _ in range(40):
        dv = lower + rng.uniform(size=lower.size) * (upper - lower)
        params = dict(zip(names, dv))
        cons = compute_constraint_violations(dv, formulation)
        cfg = dvs_to_config(dv, formulation)

        # c1 <-> delivery minimum.accumulate clamp changed the factor arrays
        # (leading L1a-L2/L3 defaults never bind under the audited bounds).
        nyc_pre = [params["nyc_drought_factor_L3"],
                   params["nyc_drought_factor_L4"],
                   params["nyc_drought_factor_L5"]]
        nj_pre = [params["nj_drought_factor_L4"],
                  params["nj_drought_factor_L5"]]
        nyc_post = [float(cfg.constants[f"level{i}_factor_delivery_nyc"])
                    for i in (3, 4, 5)]
        nj_post = [float(cfg.constants[f"level{i}_factor_delivery_nj"])
                   for i in (4, 5)]
        clamped = not (np.allclose(nyc_pre, nyc_post, rtol=0, atol=1e-12)
                       and np.allclose(nj_pre, nj_post, rtol=0, atol=1e-12))
        assert (cons[0] > 0.0) == clamped, dv

        # c2 <-> the L1b <= L1a clamp changed an applied flood factor row.
        date_cols = _flood_date_cols(cfg)
        flood_clamped = False
        defaults = _get_cached_defaults()
        for res in RESERVOIRS:
            baseline = float(defaults.constants[f"mrf_baseline_{res}"])
            cap = (float(defaults.constants[f"flood_max_release_{res}_cfs"])
                   * _CFS_TO_MGD)
            f_b = defaults.mrf_factors_daily_df.loc[
                f"level1b_factor_mrf_{res}", date_cols].values.astype(float)
            mult = float(params[f"flood_release_scale_l1b_{res}"])
            preclamp_row = np.minimum(f_b * baseline * mult, cap) / baseline
            post_row = cfg.mrf_factors_daily_df.loc[
                f"level1b_factor_mrf_{res}", date_cols].values.astype(float)
            if not np.allclose(post_row, preclamp_row, rtol=0, atol=1e-12):
                flood_clamped = True
        assert (cons[1] > 0.0) == flood_clamped, dv
