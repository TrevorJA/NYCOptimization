"""Tests for the DV-space formal Borg constraint functions.

Covers `compute_constraint_violations` (pure DV arithmetic, no simulation)
and its registry wiring (`get_n_constrs`, `make_constraint_function`):
baseline feasibility, hand-computed directional violations, the tolerance
floor, and the clamp-equivalence property — the flood-zone violation is
positive exactly when the corresponding apply-time clamp in `dvs_to_config`
fires.

Delivery-stage monotonicity is structural, not constrained: the
allocation-reduction DVs are non-negative stage increments, so the decoded
delivery-factor arrays are non-increasing for every in-bounds vector
(covered by `test_delivery_factors_structurally_monotone`).

The post-simulation `nyc_reliability_floor` constraint is covered in
tests/test_reliability_floor_constraint.py.

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
    DV_CONSTRAINT_NAMES,
    POST_SIM_CONSTRAINT_NAMES,
    get_baseline_values,
    get_bounds,
    get_constraint_names,
    get_n_constrs,
    get_var_names,
    make_constraint_function,
)
from src.simulation import (
    _CFS_TO_MGD,
    _config_levels,
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
        "flood_zone_ordering", "nyc_reliability_floor",
    ]
    assert CONSTRAINT_NAMES == DV_CONSTRAINT_NAMES + POST_SIM_CONSTRAINT_NAMES
    # make_constraint_function is DV-space ONLY: one value, no simulation.
    fn = make_constraint_function("ffmp")
    cons = fn(list(get_baseline_values("ffmp")))
    assert isinstance(cons, list) and len(cons) == len(DV_CONSTRAINT_NAMES) == 1
    assert all(isinstance(c, float) for c in cons)


###############################################################################
# Baseline feasibility (exact zeros — Borg treats any nonzero as infeasible)
###############################################################################

@pytest.mark.parametrize("formulation", ["ffmp", "ffmp_10"])
def test_baseline_is_exactly_feasible(formulation):
    cons = compute_constraint_violations(
        get_baseline_values(formulation), formulation
    )
    assert cons == [0.0]


def test_zone_crossings_are_not_constrained():
    # Crossing-inducing shifts are clamp-only: still constraint-feasible.
    cons = compute_constraint_violations(
        _dv(zone_vshift_level1c_lower=0.05,
            zone_vshift_level1b_lower=-0.10, zone_tshift_level1c=30.0),
        "ffmp",
    )
    assert cons == [0.0]


###############################################################################
# Structural delivery monotonicity (no constraint, no clamp — by construction)
###############################################################################

@pytest.mark.parametrize("formulation", ["ffmp", "ffmp_10"])
def test_delivery_factors_structurally_monotone(formulation):
    # Non-negative allocation-reduction DVs decode to non-increasing
    # delivery-factor arrays for EVERY in-bounds vector — monotonicity needs
    # no feasibility signal.
    lower, upper = get_bounds(formulation)
    rng = np.random.default_rng(7)
    for _ in range(25):
        dv = lower + rng.uniform(size=lower.size) * (upper - lower)
        cfg = dvs_to_config(dv, formulation)
        drought_levels, _ = _config_levels(cfg)
        for party in ("nyc", "nj"):
            factors = np.array([
                float(cfg.constants[f"{lvl}_factor_delivery_{party}"])
                for lvl in drought_levels
            ])
            constrained = factors[factors <= 1.0]
            assert constrained.size > 0
            assert np.all(np.diff(constrained) <= 1e-12), dv


def test_delivery_extreme_reductions_stay_in_envelope():
    # Max reductions bottom out exactly at the audited depth envelope.
    dv = _dv(nyc_allocation_reduction_L3=0.20, nyc_allocation_reduction_L4=0.20,
             nyc_allocation_reduction_L5=0.10,
             nj_allocation_reduction_L4=0.20, nj_allocation_reduction_L5=0.15)
    cfg = dvs_to_config(dv, "ffmp")
    assert float(cfg.constants["level5_factor_delivery_nyc"]) == pytest.approx(0.50)
    assert float(cfg.constants["level5_factor_delivery_nj"]) == pytest.approx(0.65)
    # And the flood/reliability machinery still sees a feasible vector.
    assert compute_constraint_violations(dv, "ffmp") == [0.0]


###############################################################################
# Hand-computed directional violations
###############################################################################

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
    assert cons[0] == pytest.approx(expected)


def test_flood_equal_multipliers_feasible():
    cons = compute_constraint_violations(
        _dv(flood_release_scale_l1a_pepacton=1.2,
            flood_release_scale_l1b_pepacton=1.2),
        "ffmp",
    )
    assert cons[0] == 0.0


###############################################################################
# Tolerance floor
###############################################################################

def test_tiny_violation_floors_to_exact_zero():
    # A vanishingly small L1b > L1a excess (equal multipliers plus 1e-13)
    # must floor to exact 0.0 rather than leak into the feasibility signal.
    dv = _dv(flood_release_scale_l1a_pepacton=1.2,
             flood_release_scale_l1b_pepacton=1.2 + 1e-13)
    cons = compute_constraint_violations(dv, "ffmp")
    assert cons[0] == 0.0


###############################################################################
# Clamp equivalence: c > 0 iff the flood apply-time clamp fires
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

        # c1 <-> the L1b <= L1a clamp changed an applied flood factor row.
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
        assert (cons[0] > 0.0) == flood_clamped, dv
