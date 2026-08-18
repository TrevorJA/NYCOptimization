"""
tests/test_objectives_ensemble.py - Unit tests for the annual-unit ensemble
objective framework in src.objectives_ensemble (objective_definitions.md §2).

Covers:
  1. FFMP-year unit splitting: the date-based 6-month metric-exclusion cut,
     leap-year windows, trailing partial years, and the L-1
     metric-bearing-unit rule.
  2. Stage-(ii) unit operators on synthetic pools: failure frequency (with k),
     pooled P99 / P01 percentiles, pooled mean — including the non-finite
     policy (failure-year for frequency; worst-value sentinel otherwise).
  3. Stage-(i) annual metrics on synthetic data dicts (delivery failing-week
     counts incl. the 0.99 factor and the running-average entitlement, flood
     days, storage minimum).
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
    METRIC_EXCLUSION_MONTHS,
    NYC_RESERVOIRS,
    NYC_TOTAL_CAPACITY,
)
from src.objectives import OBJECTIVES, ObjectiveSet
import src.objectives_ensemble as obj_ens
from src.objectives_ensemble import (
    AnnualUnitObjective,
    ENSEMBLE_OBJECTIVES,
    FailureFrequencyOp,
    PooledMeanOp,
    PooledPercentileOp,
    build_ensemble_objective_set,
    ffmp_year_unit_slices,
)


def _dec_index(start_year: int, n_years: int) -> pd.DatetimeIndex:
    """Daily index spanning n whole years from Dec 1 of start_year (the epoch shape)."""
    return pd.date_range(
        f"{start_year}-12-01",
        pd.Timestamp(f"{start_year}-12-01") + pd.DateOffset(years=n_years) - pd.Timedelta(days=1),
        freq="D",
    )


# ---------------------------------------------------------------------------
# 1. FFMP-year unit splitting
# ---------------------------------------------------------------------------

def _exclusion_cutoff(idx: pd.DatetimeIndex) -> pd.Timestamp:
    """First timestamp inside the metric window of a realization index."""
    return idx[0] + pd.DateOffset(months=METRIC_EXCLUSION_MONTHS)


def test_unit_slices_yield_L_minus_1_whole_ffmp_years():
    """A 5-year December-start realization yields 4 unit-years, each Jun 1 - May 31."""
    idx = _dec_index(1945, 5)
    slices = ffmp_year_unit_slices(idx)
    assert len(slices) == 4
    for sl in slices:
        unit = idx[sl]
        assert (unit[0].month, unit[0].day) == (6, 1)
        assert (unit[-1].month, unit[-1].day) == (5, 31)
        assert len(unit) in (365, 366)
    # On the December epoch the 6-month exclusion ends exactly on the FFMP
    # operating-year boundary: the first unit opens at the cutoff itself.
    assert _exclusion_cutoff(idx) == pd.Timestamp("1946-06-01")
    assert idx[slices[0].start] == pd.Timestamp("1946-06-01")
    # The trailing Jun-Nov partial of the final year is discarded.
    assert idx[slices[-1].stop - 1] == pd.Timestamp("1950-05-31")
    # Units are contiguous.
    for a, b in zip(slices, slices[1:]):
        assert a.stop == b.start


def test_unit_slices_are_date_based_across_a_leap_unit():
    """A leap February inside a unit needs no special case: the cut is by
    date, and the Jun 1947 - May 1948 unit simply carries 366 days."""
    idx = _dec_index(1946, 3)  # unit 1 spans 1948-02-29
    slices = ffmp_year_unit_slices(idx)
    assert len(slices) == 2
    assert _exclusion_cutoff(idx) == pd.Timestamp("1947-06-01")
    assert idx[slices[0].start] == pd.Timestamp("1947-06-01")
    assert len(idx[slices[0]]) == 366
    assert len(idx[slices[1]]) == 365


def test_unit_slices_drop_trailing_partial_year():
    idx = pd.date_range("1945-12-01", "1951-02-28", freq="D")
    slices = ffmp_year_unit_slices(idx)
    assert len(slices) == 4  # the Jun 1950 - Feb 1951 fragment is discarded
    assert idx[slices[-1].stop - 1] == pd.Timestamp("1950-05-31")


def test_unit_slices_empty_for_exclusion_window_only_trace():
    """A trace no longer than the 6-month exclusion window has no units."""
    idx = pd.date_range("1945-12-01", "1946-05-31", freq="D")
    assert idx[-1] < _exclusion_cutoff(idx)
    assert ffmp_year_unit_slices(idx) == []


def test_unit_slices_december_epoch_yields_L_minus_1_units():
    """The staged stamping convention: an L-year December-start realization
    (config.ENSEMBLE_START_DATE epoch) yields exactly L - 1 whole FFMP-year
    units spanning Jun 1 year 1 - May 31 year L, opening exactly at the
    6-month cutoff (Jun 1, the FFMP operating-year boundary)."""
    L = 10
    idx = pd.date_range("1945-12-01", "1955-11-30", freq="D")
    slices = ffmp_year_unit_slices(idx)
    assert len(slices) == L - 1
    assert _exclusion_cutoff(idx) == pd.Timestamp("1946-06-01")
    assert idx[slices[0].start] == pd.Timestamp("1946-06-01")
    assert idx[slices[-1].stop - 1] == pd.Timestamp("1955-05-31")
    for sl in slices:
        unit = idx[sl]
        assert (unit[0].month, unit[0].day) == (6, 1)
        assert (unit[-1].month, unit[-1].day) == (5, 31)


def test_unit_slices_are_a_pure_date_rule_for_any_start():
    """A non-epoch (October) start still yields complete Jun-May units by the
    same pure-date rule -- no start-day assumption anywhere."""
    idx = pd.date_range("1945-10-01", "1955-09-30", freq="D")
    slices = ffmp_year_unit_slices(idx)
    assert len(slices) == 9
    assert idx[slices[0].start] == pd.Timestamp("1946-06-01")
    assert idx[slices[-1].stop - 1] == pd.Timestamp("1955-05-31")


def test_unit_slices_raise_on_gappy_unit():
    """A daily index with a missing day inside a unit raises instead of
    silently mis-scoring the unit."""
    idx = _dec_index(1945, 3)
    gappy = idx.delete(400)  # a day inside the first unit
    with pytest.raises(ValueError, match="gaps or duplicates"):
        ffmp_year_unit_slices(gappy)


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
    idx = _dec_index(1945, 3)  # units Jun 1946 - May 1947, Jun 1947 - May 1948
    demand = pd.Series(500.0, index=idx)
    delivery = pd.Series(500.0, index=idx)
    # 14-day full shortfall inside the SECOND unit-year (Jun 1947 - May 1948).
    delivery.loc["1948-01-05":"1948-01-18"] = 0.0
    units = obj_ens._nyc_delivery_failure_weeks_annual(
        _delivery_data(idx, demand, delivery))
    assert units.shape == (2,)
    assert units[0] == 0.0
    # A 14-day zero-delivery block overlaps 2-3 weekly bins, all failing.
    assert units[1] in (2.0, 3.0)


def test_delivery_failure_weeks_annual_tolerates_1pct_shortfall():
    """delivery = 99.5% of demand is within the 0.99 factor -> no failures."""
    idx = _dec_index(1945, 3)
    demand = pd.Series(500.0, index=idx)
    units = obj_ens._nyc_delivery_failure_weeks_annual(
        _delivery_data(idx, demand, 0.995 * demand))
    assert np.all(units == 0.0)


def test_delivery_failure_weeks_annual_uses_running_avg_entitlement():
    """Daily demand above the 800 MGD baseline is NOT clipped. Because prior
    under-delivery (795 < 800) banks running-average allowance, the entitlement
    tracks the higher demand, so delivering only 795 against a demand of 900
    fails nearly every week — the opposite of the old static-daily-cap behavior,
    which scored this as zero failures (795 >= 0.99 * 800)."""
    idx = _dec_index(1945, 3)
    demand = pd.Series(900.0, index=idx)       # above the flat daily baseline
    delivery = pd.Series(795.0, index=idx)     # under 800 -> allowance bank grows
    units = obj_ens._nyc_delivery_failure_weeks_annual(
        _delivery_data(idx, demand, delivery))
    assert units.shape == (2,)
    assert np.all(units >= 45.0)               # ~all weeks fail (was 0 under the old clip)


def test_delivery_failure_weeks_annual_caps_at_running_avg_allowance():
    """Demand far above the running-average right is not owed: delivering the
    full 800 MGD allowance satisfies every week even when demand is absurdly
    high, because the entitlement is capped at the banked allowance, not at
    demand. With delivery == 800 the bank holds steady at 800, so entitlement =
    min(5000, 800) = 800 == delivery every week."""
    idx = _dec_index(1945, 3)
    demand = pd.Series(5000.0, index=idx)      # absurd sustained demand
    delivery = pd.Series(800.0, index=idx)     # exactly the running-avg right
    units = obj_ens._nyc_delivery_failure_weeks_annual(
        _delivery_data(idx, demand, delivery))
    assert np.all(units == 0.0)


def test_delivery_deficit_cvar90_annual_full_year_shortfall():
    """A whole-unit-year total shortfall gives CVaR90 = 100 * 500/800 = 62.5%
    in that unit-year (every weekly-mean deficit identical) and 0 elsewhere."""
    idx = _dec_index(1945, 3)
    demand = pd.Series(500.0, index=idx)
    delivery = pd.Series(500.0, index=idx)
    delivery.loc["1947-06-01":"1948-05-31"] = 0.0  # entire second unit-year
    units = obj_ens._nyc_delivery_deficit_cvar90_annual(
        _delivery_data(idx, demand, delivery))
    assert units == pytest.approx([0.0, 62.5])


def test_flow_failure_weeks_annual_counts_low_flow_weeks():
    from config import MONTAGUE_DECREE_TARGET_MGD

    idx = _dec_index(1945, 3)
    flow = pd.Series(MONTAGUE_DECREE_TARGET_MGD + 500.0, index=idx)
    # 14-day zero-flow block inside the FIRST unit-year (Jun 1946 - May 1947).
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

    idx = _dec_index(1945, 3)
    below = {g: flood_stage_thresholds[g]["minor"] - 1.0
             for g in _DOWNSTREAM_GAUGES}
    stage = pd.DataFrame({g: np.full(len(idx), v) for g, v in below.items()},
                         index=idx)
    # One gauge floods on 3 days of the FIRST unit-year (Jun 1946 - May 1947).
    g0 = _DOWNSTREAM_GAUGES[0]
    stage.loc["1947-04-01":"1947-04-03", g0] = (
        flood_stage_thresholds[g0]["minor"] + 0.5
    )
    units = obj_ens._flood_days_minor_annual({"flood_stage": stage})
    assert units.tolist() == [3.0, 0.0]


def test_flood_exceedance_annual_integrates_worst_gauge_exceedance():
    from pywrdrb.flood_thresholds import flood_stage_thresholds
    from src.objectives import _DOWNSTREAM_GAUGES

    idx = _dec_index(1945, 3)
    below = {g: flood_stage_thresholds[g]["minor"] - 1.0
             for g in _DOWNSTREAM_GAUGES}
    stage = pd.DataFrame({g: np.full(len(idx), v) for g, v in below.items()},
                         index=idx)
    # One gauge 0.5 ft over flood stage on 3 days of the FIRST unit-year;
    # a second gauge 0.2 ft over on ONE of those days — the max-gauge basis
    # takes the worst exceedance per day, never the sum.
    g0, g1 = _DOWNSTREAM_GAUGES[0], _DOWNSTREAM_GAUGES[1]
    stage.loc["1947-04-01":"1947-04-03", g0] = (
        flood_stage_thresholds[g0]["minor"] + 0.5
    )
    stage.loc["1947-04-02", g1] = flood_stage_thresholds[g1]["minor"] + 0.2
    units = obj_ens._flood_exceedance_minor_annual({"flood_stage": stage})
    assert units == pytest.approx([1.5, 0.0])


def test_storage_min_annual_per_unit_year():
    idx = _dec_index(1945, 3)
    per_res = 0.8 * NYC_TOTAL_CAPACITY / len(NYC_RESERVOIRS)
    storage = pd.DataFrame(
        {r: np.full(len(idx), per_res) for r in NYC_RESERVOIRS}, index=idx,
    )
    # One-day dip to 40% total in the first unit-year.
    storage.loc["1946-08-15", :] = 0.4 * NYC_TOTAL_CAPACITY / len(NYC_RESERVOIRS)
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
        json.dumps({"nyc_delivery_reliability_annual__sat65": 0.80}),
    )
    importlib.reload(obj_ens)
    try:
        obj = obj_ens.ENSEMBLE_OBJECTIVES["nyc_delivery_reliability_annual"]
        assert obj.sat_threshold == pytest.approx(0.80)
        assert obj.sat_kind == "ge"      # follows the direction, not the override
        # Other objectives keep their registry defaults.
        other = obj_ens.ENSEMBLE_OBJECTIVES["nyc_storage_min_p01_pct"]
        assert other.sat_threshold == pytest.approx(
            obj_ens._DEFAULT_THRESHOLDS["nyc_storage_min_p01_pct__sat26"])
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
    "downstream_flood_exceedance_annual",
    "downstream_flood_days_annual",
    "downstream_flood_days_annual_p99",
    "nyc_storage_min_p01_pct",
    "nj_delivery_reliability_annual",
]

# The §1 base names config.ACTIVE_OBJECTIVES uses (default 8-objective set;
# NJ delivery activated 2026-07-30).
ACTIVE_BASE_NAMES = [
    "nyc_delivery_reliability_weekly",
    "nyc_delivery_deficit_cvar90_pct",
    "montague_flow_reliability_weekly",
    "montague_flow_deficit_cvar90_pct",
    "trenton_flow_reliability_weekly",
    "downstream_flood_exceedance_minor",
    "nyc_storage_p5_pct",
    "nj_delivery_reliability_weekly",
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
        "downstream_flood_exceedance_annual",
        "nyc_storage_min_p01_pct",
        "nj_delivery_reliability_annual",
    ]
    assert obj_set.directions == [1, -1, 1, -1, 1, -1, 1, 1]
    # The diagnostic P99 flood variant is NOT reachable via base names.
    assert "downstream_flood_days_annual_p99" not in obj_set.names
    # Every ACTIVE objective carries the re-eval satisficing criterion (a
    # non-None threshold; its kind derived from the objective's own direction).
    for o in obj_set:
        assert o.base.name in ACTIVE_BASE_NAMES
        assert o.sat_threshold is not None
        assert o.sat_kind == ("ge" if o.direction == "maximize" else "le")


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
        sat_threshold=0.95,
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


# ---------------------------------------------------------------------------
# 6. Re-eval satisficing criterion (sat_threshold / sat_kind)
# ---------------------------------------------------------------------------

def test_registry_thresholds_resolve_from_the_default_table():
    """Each objective's sat_threshold is its _DEFAULT_THRESHOLDS entry, resolved
    through the `<annual name>__sat<thr>` label."""
    for name, label in obj_ens._SAT_LABELS.items():
        obj = ENSEMBLE_OBJECTIVES[name]
        assert obj.sat_threshold == pytest.approx(
            obj_ens._DEFAULT_THRESHOLDS[label]), name


def test_sat_kind_follows_the_objectives_own_direction():
    """maximize -> 'ge', minimize -> 'le'; the criterion direction is DERIVED,
    never stored, so it cannot disagree with the objective."""
    for obj in ENSEMBLE_OBJECTIVES.values():
        assert obj.sat_kind == ("ge" if obj.direction == "maximize" else "le")


def test_every_active_objective_carries_a_satisficing_threshold():
    """The multivariate criterion is a conjunction over the ACTIVE set, so a
    missing threshold would silently zero the primary robustness metric."""
    obj_set = build_ensemble_objective_set(ACTIVE_BASE_NAMES)
    for o in obj_set:
        assert o.sat_threshold is not None, o.name


def test_diagnostic_p99_flood_variant_has_no_satisficing_role():
    assert ENSEMBLE_OBJECTIVES["downstream_flood_days_annual_p99"].sat_threshold \
        is None


def test_sat_threshold_defaults_to_none_on_hand_built_objectives():
    obj = AnnualUnitObjective(
        name="fake", direction="minimize", epsilon=0.1, description="synthetic",
        annual_metric=lambda data: np.asarray(data["units"], dtype=float),
        unit_operator=PooledMeanOp(worst_value=1.0),
        base=OBJECTIVES["nyc_delivery_reliability_weekly"],
    )
    assert obj.sat_threshold is None
    assert obj.sat_kind == "le"
