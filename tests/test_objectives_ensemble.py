"""
tests/test_objectives_ensemble.py - Unit tests for the annual-unit ensemble
objective framework in src.objectives_ensemble (objective_definitions.md §2).

Covers:
  1. Water-year unit splitting: warm-up drop, leap-year stray day, trailing
     partial years, and the L-1 metric-bearing-unit rule.
  2. Stage-(ii) unit operators on synthetic pools: failure frequency (with k),
     pooled P99 / P01 percentiles, pooled mean — including the non-finite
     policy (failure-year for frequency; worst-value sentinel otherwise).
  3. Stage-(i) annual metrics on synthetic data dicts (delivery failing-week
     counts incl. the 0.99 factor and demand cap, flood days, storage minimum).
  4. NYCOPT_FAILURE_K and NYCOPT_SAT_THRESHOLDS env overrides (JSON, no CLI).
  5. Registry / ObjectiveSet wiring: names, directions, base-name resolution,
     Borg sign convention via compute_for_borg_ensemble, and the batched-path
     equivalence via compute_for_borg_from_units.

Run:
    venv/Scripts/python.exe -m pytest tests/test_objectives_ensemble.py -v
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    NYC_RESERVOIRS,
    NYC_TOTAL_CAPACITY,
    WARMUP_DAYS,
)
from src.objectives import OBJECTIVES, ObjectiveSet
import src.objectives_ensemble as obj_ens
from src.objectives_ensemble import (
    AnnualUnitObjective,
    ENSEMBLE_OBJECTIVES,
    FailureFrequencyOp,
    PooledMeanOp,
    PooledPercentileOp,
    SatisficingAgg,
    build_ensemble_objective_set,
    water_year_unit_slices,
)


def _wy_index(start_year: int, n_years: int) -> pd.DatetimeIndex:
    """Daily index spanning n whole water years from Oct 1 of start_year."""
    return pd.date_range(
        f"{start_year}-10-01", f"{start_year + n_years}-09-30", freq="D",
    )


# ---------------------------------------------------------------------------
# 1. Water-year unit splitting
# ---------------------------------------------------------------------------

def test_unit_slices_yield_L_minus_1_whole_water_years():
    """A 5-water-year realization yields 4 unit-years, each Oct 1 - Sep 30."""
    idx = _wy_index(1945, 5)
    slices = water_year_unit_slices(idx)
    assert len(slices) == 4
    for sl in slices:
        unit = idx[sl]
        assert (unit[0].month, unit[0].day) == (10, 1)
        assert (unit[-1].month, unit[-1].day) == (9, 30)
        assert len(unit) in (365, 366)
    # WY1946 (Oct 1945 - Sep 1946) has 365 days, so the warm-up consumes it
    # exactly and the first unit starts at position 365 (1946-10-01).
    assert slices[0].start == WARMUP_DAYS
    assert idx[slices[0].start] == pd.Timestamp("1946-10-01")
    # Units are contiguous.
    for a, b in zip(slices, slices[1:]):
        assert a.stop == b.start


def test_unit_slices_drop_leap_year_stray_day():
    """When the warm-up water year has 366 days (leap), dropping exactly 365
    days leaves one stray day of that year; it is not a whole water year and
    must be discarded, so L=3 still yields 2 units."""
    idx = _wy_index(1947, 3)  # WY1948 contains 1948-02-29 -> 366 days
    slices = water_year_unit_slices(idx)
    assert len(slices) == 2
    # The stray day (1948-09-30) at position 365 is skipped.
    assert slices[0].start == WARMUP_DAYS + 1
    assert idx[slices[0].start] == pd.Timestamp("1948-10-01")


def test_unit_slices_drop_trailing_partial_year():
    idx = pd.date_range("1945-10-01", "1950-12-31", freq="D")
    slices = water_year_unit_slices(idx)
    assert len(slices) == 4  # WY1951 fragment (Oct-Dec 1950) discarded
    assert idx[slices[-1].stop - 1] == pd.Timestamp("1950-09-30")


def test_unit_slices_empty_for_warmup_only_trace():
    idx = pd.date_range("1945-10-01", periods=WARMUP_DAYS, freq="D")
    assert water_year_unit_slices(idx) == []


# ---------------------------------------------------------------------------
# 2. Unit operators
# ---------------------------------------------------------------------------

def test_failure_frequency_default_k1():
    op = FailureFrequencyOp(k=1)
    # Counts [0, 0, 1, 2]: non-failure years are those with < 1 failing week.
    assert op([0.0, 0.0, 1.0, 2.0]) == pytest.approx(0.5)


def test_failure_frequency_k3():
    op = FailureFrequencyOp(k=3)
    assert op([0.0, 1.0, 2.0, 3.0, 4.0]) == pytest.approx(3.0 / 5.0)


def test_failure_frequency_nan_is_failure_year():
    op = FailureFrequencyOp(k=1)
    assert op([0.0, float("nan"), 0.0]) == pytest.approx(2.0 / 3.0)


def test_failure_frequency_empty_returns_zero():
    assert FailureFrequencyOp(k=1)([]) == 0.0


def test_failure_frequency_rejects_bad_k():
    with pytest.raises(ValueError, match="k must be >= 1"):
        FailureFrequencyOp(k=0)


def test_pooled_percentile_basic():
    op = PooledPercentileOp(q=99.0, worst_value=100.0)
    assert op([7.0, 7.0, 7.0]) == pytest.approx(7.0)


def test_pooled_percentile_nan_uses_worst_sentinel():
    op = PooledPercentileOp(q=99.0, worst_value=100.0)
    # NaN -> 100 (worst for a minimize deficit %), dragging P99 up.
    assert op([0.0, float("nan")]) == pytest.approx(99.0)  # P99 of [0, 100]
    assert op([]) == pytest.approx(100.0)


def test_pooled_percentile_p01_maximize_sentinel():
    op = PooledPercentileOp(q=1.0, worst_value=0.0)
    # NaN -> 0 (worst for a maximize storage %), dragging P01 down.
    vals = [50.0, 60.0, float("nan")]
    expected = float(np.percentile([50.0, 60.0, 0.0], 1.0))
    assert op(vals) == pytest.approx(expected)
    assert op([]) == pytest.approx(0.0)


def test_pooled_percentile_rejects_bad_q():
    with pytest.raises(ValueError, match="q must be"):
        PooledPercentileOp(q=101.0, worst_value=0.0)


def test_pooled_mean_basic_and_nan_sentinel():
    op = PooledMeanOp(worst_value=366.0)
    assert op([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert op([1.0, float("nan")]) == pytest.approx((1.0 + 366.0) / 2.0)
    assert op([]) == pytest.approx(366.0)


# ---------------------------------------------------------------------------
# 3. Stage-(i) annual metrics on synthetic data
# ---------------------------------------------------------------------------

def _delivery_data(idx: pd.DatetimeIndex, demand: pd.Series,
                   delivery: pd.Series) -> dict:
    return {
        "ibt_demands": pd.DataFrame({"demand_nyc": demand}, index=idx),
        "ibt_diversions": pd.DataFrame({"delivery_nyc": delivery}, index=idx),
    }


def test_delivery_failure_weeks_annual_counts_shortfall_block():
    idx = _wy_index(1945, 3)  # warm-up WY1946 + units WY1947, WY1948
    demand = pd.Series(500.0, index=idx)
    delivery = pd.Series(500.0, index=idx)
    # 14-day full shortfall inside the SECOND unit-year (WY1948).
    delivery.loc["1948-01-05":"1948-01-18"] = 0.0
    units = obj_ens._nyc_delivery_failure_weeks_annual(
        _delivery_data(idx, demand, delivery))
    assert units.shape == (2,)
    assert units[0] == 0.0
    # A 14-day zero-delivery block overlaps 2-3 weekly bins, all failing.
    assert units[1] in (2.0, 3.0)


def test_delivery_failure_weeks_annual_tolerates_1pct_shortfall():
    """delivery = 99.5% of demand is within the 0.99 factor -> no failures."""
    idx = _wy_index(1945, 3)
    demand = pd.Series(500.0, index=idx)
    units = obj_ens._nyc_delivery_failure_weeks_annual(
        _delivery_data(idx, demand, 0.995 * demand))
    assert np.all(units == 0.0)


def test_delivery_failure_weeks_annual_caps_demand_at_decree_right():
    """Demand above the 800 MGD cap is clipped: delivering 99% of the CAP
    satisfies the week even when raw demand is higher."""
    idx = _wy_index(1945, 3)
    demand = pd.Series(900.0, index=idx)       # above the Decree cap
    delivery = pd.Series(795.0, index=idx)     # >= 0.99 * 800 = 792
    units = obj_ens._nyc_delivery_failure_weeks_annual(
        _delivery_data(idx, demand, delivery))
    assert np.all(units == 0.0)


def test_delivery_deficit_cvar90_annual_full_year_shortfall():
    """A whole-unit-year total shortfall gives CVaR90 = 100 * 500/800 = 62.5%
    in that unit-year (every weekly-mean deficit identical) and 0 elsewhere."""
    idx = _wy_index(1945, 3)
    demand = pd.Series(500.0, index=idx)
    delivery = pd.Series(500.0, index=idx)
    delivery.loc["1947-10-01":"1948-09-30"] = 0.0  # entire second unit-year
    units = obj_ens._nyc_delivery_deficit_cvar90_annual(
        _delivery_data(idx, demand, delivery))
    assert units == pytest.approx([0.0, 62.5])


def test_flow_failure_weeks_annual_counts_low_flow_weeks():
    from config import MONTAGUE_DECREE_TARGET_MGD

    idx = _wy_index(1945, 3)
    flow = pd.Series(MONTAGUE_DECREE_TARGET_MGD + 500.0, index=idx)
    # 14-day zero-flow block inside the FIRST unit-year (WY1947).
    flow.loc["1947-01-05":"1947-01-18"] = 0.0
    units = obj_ens._montague_failure_weeks_annual(
        {"major_flow": pd.DataFrame({"delMontague": flow}, index=idx)})
    assert units.shape == (2,)
    # Weekly-MEAN basis: bins fully inside the block fail; boundary bins fail
    # only if enough block days dilute the mean below the target.
    assert units[0] in (2.0, 3.0)
    assert units[1] == 0.0


def test_flood_days_annual_counts_days_per_unit_year():
    from pywrdrb.flood_thresholds import flood_stage_thresholds
    from src.objectives import _DOWNSTREAM_GAUGES

    idx = _wy_index(1945, 3)
    below = {g: flood_stage_thresholds[g]["minor"] - 1.0
             for g in _DOWNSTREAM_GAUGES}
    stage = pd.DataFrame({g: np.full(len(idx), v) for g, v in below.items()},
                         index=idx)
    # One gauge floods on 3 days of the FIRST unit-year (WY1947).
    g0 = _DOWNSTREAM_GAUGES[0]
    stage.loc["1947-04-01":"1947-04-03", g0] = (
        flood_stage_thresholds[g0]["minor"] + 0.5
    )
    units = obj_ens._flood_days_minor_annual({"flood_stage": stage})
    assert units.tolist() == [3.0, 0.0]


def test_storage_min_annual_per_unit_year():
    idx = _wy_index(1945, 3)
    per_res = 0.8 * NYC_TOTAL_CAPACITY / len(NYC_RESERVOIRS)
    storage = pd.DataFrame(
        {r: np.full(len(idx), per_res) for r in NYC_RESERVOIRS}, index=idx,
    )
    # One-day dip to 40% total in the first unit-year.
    storage.loc["1947-08-15", :] = 0.4 * NYC_TOTAL_CAPACITY / len(NYC_RESERVOIRS)
    units = obj_ens._nyc_storage_min_annual({"res_storage": storage})
    assert units == pytest.approx([40.0, 80.0])


# ---------------------------------------------------------------------------
# 4. Env overrides
# ---------------------------------------------------------------------------

def test_env_failure_k_override(monkeypatch):
    monkeypatch.setenv(
        "NYCOPT_FAILURE_K",
        json.dumps({"nyc_delivery_reliability_annual": 3}),
    )
    importlib.reload(obj_ens)
    try:
        obj = obj_ens.ENSEMBLE_OBJECTIVES["nyc_delivery_reliability_annual"]
        assert isinstance(obj.unit_operator, obj_ens.FailureFrequencyOp)
        assert obj.unit_operator.k == 3
        # Other frequency objectives keep the default k = 1.
        other = obj_ens.ENSEMBLE_OBJECTIVES["trenton_flow_reliability_annual"]
        assert other.unit_operator.k == 1
    finally:
        monkeypatch.delenv("NYCOPT_FAILURE_K", raising=False)
        importlib.reload(obj_ens)


def test_env_failure_k_rejects_unknown(monkeypatch):
    monkeypatch.setenv(
        "NYCOPT_FAILURE_K", json.dumps({"not_a_real_objective": 2}),
    )
    with pytest.raises(KeyError, match="not_a_real_objective"):
        importlib.reload(obj_ens)
    monkeypatch.delenv("NYCOPT_FAILURE_K", raising=False)
    importlib.reload(obj_ens)


def test_env_threshold_override(monkeypatch):
    """The re-eval satisficing layer keeps its NYCOPT_SAT_THRESHOLDS override."""
    monkeypatch.setenv(
        "NYCOPT_SAT_THRESHOLDS",
        json.dumps({"nyc_delivery_reliability_weekly__sat95": 0.80}),
    )
    importlib.reload(obj_ens)
    try:
        agg = obj_ens.ENSEMBLE_OBJECTIVES[
            "nyc_delivery_reliability_annual"].aggregator
        assert isinstance(agg, obj_ens.SatisficingAgg)
        assert agg.threshold == pytest.approx(0.80)
    finally:
        monkeypatch.delenv("NYCOPT_SAT_THRESHOLDS", raising=False)
        importlib.reload(obj_ens)


def test_env_threshold_override_rejects_unknown(monkeypatch):
    monkeypatch.setenv(
        "NYCOPT_SAT_THRESHOLDS",
        json.dumps({"not_a_real_objective": 0.5}),
    )
    with pytest.raises(KeyError, match="not_a_real_objective"):
        importlib.reload(obj_ens)
    monkeypatch.delenv("NYCOPT_SAT_THRESHOLDS", raising=False)
    importlib.reload(obj_ens)


# ---------------------------------------------------------------------------
# 5. Registry / ObjectiveSet wiring
# ---------------------------------------------------------------------------

ANNUAL_NAMES = [
    "nyc_delivery_reliability_annual",
    "nyc_delivery_deficit_p99_pct",
    "montague_flow_reliability_annual",
    "montague_flow_deficit_p99_pct",
    "trenton_flow_reliability_annual",
    "downstream_flood_days_annual",
    "downstream_flood_days_annual_p99",
    "nyc_storage_min_p01_pct",
    "nj_delivery_reliability_annual",
]

# The §1 base names config.ACTIVE_OBJECTIVES uses (default 7-objective set).
ACTIVE_BASE_NAMES = [
    "nyc_delivery_reliability_weekly",
    "nyc_delivery_deficit_cvar90_pct",
    "montague_flow_reliability_weekly",
    "montague_flow_deficit_cvar90_pct",
    "trenton_flow_reliability_weekly",
    "downstream_flood_days_minor",
    "nyc_storage_p5_pct",
]


def test_registry_matches_expected_names():
    assert set(ENSEMBLE_OBJECTIVES) == set(ANNUAL_NAMES)


def test_base_names_resolve_to_active_annual_set():
    """config.ACTIVE_OBJECTIVES lists §1 base names; they must resolve to the
    annual objectives with the §2 directions."""
    obj_set = build_ensemble_objective_set(ACTIVE_BASE_NAMES)
    assert isinstance(obj_set, ObjectiveSet)
    assert obj_set.names == [
        "nyc_delivery_reliability_annual",
        "nyc_delivery_deficit_p99_pct",
        "montague_flow_reliability_annual",
        "montague_flow_deficit_p99_pct",
        "trenton_flow_reliability_annual",
        "downstream_flood_days_annual",
        "nyc_storage_min_p01_pct",
    ]
    assert obj_set.directions == [1, -1, 1, -1, 1, -1, 1]
    # The diagnostic P99 flood variant is NOT reachable via base names.
    assert "downstream_flood_days_annual_p99" not in obj_set.names
    # Every objective carries the re-eval layer (base + aggregator). Compare
    # against the live module attribute: earlier env-override tests reload
    # obj_ens, so the top-level class import may be a stale identity.
    for o in obj_set:
        assert o.base.name in ACTIVE_BASE_NAMES
        assert isinstance(o.aggregator, obj_ens.SatisficingAgg)


def test_build_ensemble_set_rejects_unknown_name():
    with pytest.raises(KeyError, match="Unknown ensemble objective"):
        build_ensemble_objective_set(["salt_front_intrusion_max_rm"])


def _fake_annual_objective(direction: str, unit_operator) -> AnnualUnitObjective:
    """AnnualUnitObjective whose annual metric reads data['units'] directly."""
    return AnnualUnitObjective(
        name=f"fake_{direction}",
        direction=direction,
        epsilon=0.01,
        description="synthetic",
        annual_metric=lambda data: np.asarray(data["units"], dtype=float),
        unit_operator=unit_operator,
        base=OBJECTIVES["nyc_delivery_reliability_weekly"],
        aggregator=SatisficingAgg(threshold=0.95, kind="ge"),
    )


def test_compute_pools_units_across_realizations():
    obj = _fake_annual_objective("maximize", FailureFrequencyOp(k=1))
    data_per_real = [{"units": [0.0, 1.0]}, {"units": [1.0, 1.0]}]
    # Pooled counts [0, 1, 1, 1] -> 1 of 4 unit-years without failure.
    assert obj.compute(data_per_real) == pytest.approx(0.25)
    assert obj.compute_for_borg(data_per_real) == pytest.approx(-0.25)


def test_compute_for_borg_sign_convention():
    max_obj = _fake_annual_objective("maximize", FailureFrequencyOp(k=1))
    min_obj = _fake_annual_objective("minimize", PooledMeanOp(worst_value=10.0))
    data_per_real = [{"units": [1.0, 2.0]}, {"units": [3.0, 4.0]}]
    obj_set = ObjectiveSet([max_obj, min_obj])
    borg = obj_set.compute_for_borg_ensemble(data_per_real)
    # maximize: frequency 0.0 negated -> -0.0; minimize: mean 2.5 kept raw.
    assert borg[0] == pytest.approx(0.0)
    assert borg[1] == pytest.approx(2.5)
    assert min_obj.compute(data_per_real) == pytest.approx(2.5)


def test_compute_for_borg_from_units_matches_compute_for_borg():
    obj = _fake_annual_objective("maximize", FailureFrequencyOp(k=2))
    data_per_real = [{"units": [0.0, 1.0]}, {"units": [2.0, 3.0]}]
    pooled = np.concatenate([obj.annual_units(d) for d in data_per_real])
    assert obj.compute_for_borg_from_units(pooled) == pytest.approx(
        obj.compute_for_borg(data_per_real))


def test_annual_unit_objective_rejects_bad_direction():
    with pytest.raises(ValueError, match="direction"):
        _fake_annual_objective("maximise", FailureFrequencyOp(k=1))


def test_registry_frequency_objectives_are_fractions():
    """Frequency objectives report 0-1 fractions (maximize)."""
    for name in ("nyc_delivery_reliability_annual",
                 "montague_flow_reliability_annual",
                 "trenton_flow_reliability_annual",
                 "nj_delivery_reliability_annual"):
        obj = ENSEMBLE_OBJECTIVES[name]
        assert obj.direction == "maximize"
        assert isinstance(obj.unit_operator, FailureFrequencyOp)
        val = obj.unit_operator([0.0, 5.0, float("nan")])
        assert 0.0 <= val <= 1.0
