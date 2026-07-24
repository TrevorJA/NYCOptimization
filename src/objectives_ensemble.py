"""
objectives_ensemble.py - Ensemble (multi-realization) objective framework.

Implements the **two-layer annual-unit scheme** of
`docs/notes/methods/objective_definitions.md` §2 for all ensemble
(multi-realization) evaluations. The single-trace historic path is untouched:
it keeps the §1 temporal metrics in `src.objectives` on the full trace.

Two-layer scheme (Hamilton et al. 2022 vocabulary)
--------------------------------------------------
Stage (i) — **annual metric** per (realization × water-year) unit. Every
realization starts on a water-year boundary (config ``START_DATE``, Oct 1) and
spans L whole water years. The first ``WARMUP_DAYS`` (365) days are model
warm-up and are dropped; the remainder is split into whole water-year units
(Oct 1 – Sep 30). Any leading partial year — the single stray day left when
the warm-up year is a 366-day leap water year — and any trailing partial year
are discarded, so an L-year realization yields exactly **L − 1 metric-bearing
unit-years** (see :func:`water_year_unit_slices`).

Stage (ii) — **unit operator** over the POOLED unit-years of the whole
ensemble (all realizations' units concatenated):

- *Failure frequency* (reliability objectives): fraction of unit-years WITHOUT
  failure, where a unit-year fails when it has >= k failing weeks (k
  configurable via ``_DEFAULT_FAILURE_K`` / ``NYCOPT_FAILURE_K``). Maximize;
  0-1 fraction.
- *Pooled percentile*: P99 of the annual metric for the tail-deficit
  objectives ("worst-1st-percentile unit-year", minimize) and P01 for the
  annual-minimum-storage objective (maximize).
- *Pooled mean*: expected annual flood days (minimize); a P99 variant is
  registered as an inactive diagnostic pending the sensitivity experiment.

Non-finite annual metrics: a non-finite unit-year counts as a **failure-year**
for the frequency objectives; for the mean/percentile objectives it is
replaced by the objective's orientation-aware worst-possible sentinel
(``worst_value``: 100% for the bounded deficit percentages, 366 days for
annual flood days, 0% for annual minimum storage) before aggregation, so a
degenerate unit pushes the objective toward failure instead of being dropped.

Metric reuse: all weekly accounting (weekly sums for delivery, weekly means
for flows, the 0.99 satisfaction factor, deficit-% normalization, the
running-average delivery entitlement via ``_delivery_entitlement``, CVaR90 via
``_cvar_worst_mean``) is imported from `src.objectives` windowed-series cores,
so §1 and §2 share one formula per quantity. Deficit-% and storage-% metrics
are 0-100 scales matching §1; frequency objectives are 0-1 fractions.

Epsilons are PLACEHOLDERS in native metric units, pending the ensemble
objective-sensitivity experiment
(`docs/notes/methods/ensemble_objective_sensitivity_experiment.md`).

Re-evaluation satisficing layer (retained)
------------------------------------------
The re-eval robustness pipeline (`src.reeval_core`, `src.robustness`) persists
the PER-REALIZATION §1 base-metric matrix and scores satisficing/regret from
it offline. Each :class:`AnnualUnitObjective` therefore also carries:

- ``base`` — the §1 single-trace ``Objective`` whose per-realization values
  populate the persisted re-eval matrix (``src.simulation.evaluate_raw``);
- ``aggregator`` — the per-realization :class:`SatisficingAgg` used by
  ``reeval_core`` for the derived ``objectives_summary.csv`` and the
  threshold/kind metadata of the robustness scorecard.

The search path never touches these two; they exist so re-evaluation semantics
are byte-identical to the pre-annual-unit pipeline. Thresholds are labelled by
the historical ``<base>__sat<thr>`` keys and remain overridable via
``NYCOPT_SAT_THRESHOLDS`` (JSON name→threshold). No CLI flags.

Env overrides (JSON objects; pattern-matched, no CLI flags):
    NYCOPT_FAILURE_K       {"<annual objective name>": <k>, ...}
    NYCOPT_SAT_THRESHOLDS  {"<threshold label>": <threshold>, ...}
"""

from __future__ import annotations

import json
import os
from typing import Callable, Literal

import numpy as np
import pandas as pd

from config import (
    MONTAGUE_DECREE_TARGET_MGD,
    NJ_DELIVERY_CAP_MGD,
    NYC_DECREE_DIVERSION_CAP_MGD,
    TRENTON_DECREE_TARGET_MGD,
    WARMUP_DAYS,
)
from src.objectives import (
    OBJECTIVES,
    Objective,
    ObjectiveSet,
    _cvar_worst_mean,
    _delivery_entitlement,
    _DOWNSTREAM_GAUGES,
    _flood_over_stage_daily,
    _nyc_storage_pct_daily,
    _weekly_delivery_deficit_pct,
    _weekly_delivery_ok,
    _weekly_flow_deficit_pct,
    _weekly_flow_ok,
)


###############################################################################
# Stage (i) — water-year unit splitting
###############################################################################

def water_year_unit_slices(index: pd.DatetimeIndex) -> list[slice]:
    """Positional slices of the metric-bearing water-year units of a trace.

    Unit rule (objective_definitions.md §2): realizations start on a
    water-year boundary (Oct 1) and are daily-contiguous. The first
    ``WARMUP_DAYS`` (365) days are warm-up and are dropped; the remaining days
    are grouped by water year (a date with month >= 10 belongs to water year
    ``year + 1``), and only COMPLETE water years — first day Oct 1, last day
    Sep 30 — are kept. This discards the single stray day left after warm-up
    when the first water year is a 366-day leap year, and any trailing partial
    year, so an L-water-year realization yields exactly L − 1 unit-years.

    Args:
        index: Daily DatetimeIndex of the realization's full window.

    Returns:
        List of positional ``slice`` objects into ``index`` (usable with
        ``.iloc``), one per metric-bearing water-year unit, in time order.
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) <= WARMUP_DAYS:
        return []
    sub = idx[WARMUP_DAYS:]
    wy = np.asarray(sub.year) + (np.asarray(sub.month) >= 10).astype(int)
    change = np.flatnonzero(np.diff(wy)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(sub)]))
    slices = []
    for s, e in zip(starts, stops):
        first, last = sub[s], sub[e - 1]
        if (first.month, first.day) == (10, 1) and (last.month, last.day) == (9, 30):
            slices.append(slice(WARMUP_DAYS + int(s), WARMUP_DAYS + int(e)))
    return slices


###############################################################################
# Stage (ii) — unit operators over the pooled unit-years
###############################################################################

class FailureFrequencyOp:
    """Fraction of pooled unit-years WITHOUT failure (maximize; 0-1 fraction).

    The annual metric of a frequency objective is the unit-year's FAILING-WEEK
    COUNT; a unit-year is a failure-year when that count is >= ``k`` (so k = 1
    reproduces the §2 table's "any failing week" indicator). A non-finite
    annual metric counts as a failure-year — a degenerate unit cannot
    masquerade as a success. An empty pool returns 0.0 (worst), since Borg
    needs a finite vector.
    """

    def __init__(self, k: int = 1):
        if int(k) < 1:
            raise ValueError(f"failure-year threshold k must be >= 1, got {k}")
        self.k = int(k)

    def __call__(self, units) -> float:
        arr = np.asarray(units, dtype=float).ravel()
        if arr.size == 0:
            return 0.0
        ok = np.isfinite(arr) & (arr < self.k)
        return float(ok.sum()) / float(arr.size)


class PooledPercentileOp:
    """q-th percentile of the pooled unit-year metrics.

    Non-finite unit metrics are replaced by ``worst_value`` — the metric's
    orientation-aware worst bound (documented per registry entry) — before the
    percentile, so degenerate units drag the tail toward failure instead of
    being silently dropped. An empty pool returns ``worst_value``.
    """

    def __init__(self, q: float, worst_value: float):
        if not 0.0 <= float(q) <= 100.0:
            raise ValueError(f"percentile q must be in [0, 100], got {q}")
        self.q = float(q)
        self.worst_value = float(worst_value)

    def __call__(self, units) -> float:
        arr = np.asarray(units, dtype=float).ravel()
        if arr.size == 0:
            return self.worst_value
        arr = np.where(np.isfinite(arr), arr, self.worst_value)
        return float(np.percentile(arr, self.q))


class PooledMeanOp:
    """Mean of the pooled unit-year metrics (expected annual value).

    Same non-finite policy as :class:`PooledPercentileOp`: non-finite units
    are replaced by ``worst_value`` before the mean; an empty pool returns
    ``worst_value``.
    """

    def __init__(self, worst_value: float):
        self.worst_value = float(worst_value)

    def __call__(self, units) -> float:
        arr = np.asarray(units, dtype=float).ravel()
        if arr.size == 0:
            return self.worst_value
        arr = np.where(np.isfinite(arr), arr, self.worst_value)
        return float(arr.mean())


###############################################################################
# Re-evaluation satisficing aggregator (per-realization §1 metrics)
###############################################################################

class SatisficingAgg:
    """Fraction of finite per-realization values that meet the threshold.

    `kind="ge"` ⇒ raw >= threshold (use for maximize-base metrics).
    `kind="le"` ⇒ raw <= threshold (use for minimize-base metrics).

    NaN / non-finite values count as **unsatisfied** so a degenerate
    realization can't masquerade as satisficing. If every value is non-finite
    the result is 0.0 rather than NaN.

    Used by the re-evaluation pipeline (`src.reeval_core` summary derivation +
    robustness threshold/kind metadata) and the ensemble objective-sensitivity
    diagnostic — NOT by the search path, which uses the annual-unit operators
    above.
    """

    def __init__(self, threshold: float, kind: Literal["ge", "le"]):
        if kind not in ("ge", "le"):
            raise ValueError(f"kind must be 'ge' or 'le', got '{kind}'")
        self.threshold = float(threshold)
        self.kind = kind

    def __call__(self, values) -> float:
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return 0.0
        finite = np.isfinite(arr)
        if self.kind == "ge":
            sat = finite & (arr >= self.threshold)
        else:
            sat = finite & (arr <= self.threshold)
        return float(sat.sum()) / float(arr.size)


###############################################################################
# AnnualUnitObjective
###############################################################################

class AnnualUnitObjective:
    """A §2 two-layer ensemble objective (annual metric + pooled unit operator).

    Implements the same ``compute(...)`` / ``compute_for_borg(...)`` interface
    as ``Objective`` over a LIST of per-realization data dicts, so an
    ``ObjectiveSet`` of these works with
    ``ObjectiveSet.compute_for_borg_ensemble(data_per_real)``.

    Attributes:
        name: Registry name of the annual objective.
        direction: "maximize" or "minimize" (of the unit-operator output).
        epsilon: Borg epsilon in native metric units (PLACEHOLDER pending the
            ensemble objective-sensitivity experiment).
        description: Human-readable description.
        annual_metric: Callable ``data -> np.ndarray`` returning one annual
            value per metric-bearing water-year unit of the realization
            (stage i).
        unit_operator: Callable ``pooled_units -> float`` collapsing the
            pooled unit-years of the whole ensemble (stage ii).
        base: The §1 single-trace ``Objective`` this annualizes. Used ONLY by
            the re-evaluation path (``evaluate_raw`` per-realization matrix).
        aggregator: Per-realization ``SatisficingAgg`` consumed ONLY by the
            re-evaluation pipeline (summary derivation + threshold metadata).
    """

    def __init__(self, name: str, direction: str, epsilon: float,
                 description: str, annual_metric: Callable,
                 unit_operator: Callable, base: Objective,
                 aggregator: Callable):
        if direction not in ("maximize", "minimize"):
            raise ValueError(
                f"direction must be 'maximize' or 'minimize', got '{direction}'"
            )
        self.name = name
        self.direction = direction
        self.epsilon = float(epsilon)
        self.description = description
        self.annual_metric = annual_metric
        self.unit_operator = unit_operator
        self.base = base
        self.aggregator = aggregator

    @property
    def sign(self) -> int:
        """Return 1 for maximize, -1 for minimize."""
        return 1 if self.direction == "maximize" else -1

    def annual_units(self, data: dict) -> np.ndarray:
        """Stage (i): per-unit-year annual metrics for ONE realization."""
        return np.asarray(self.annual_metric(data), dtype=float).ravel()

    def compute(self, data_per_real: list) -> float:
        """Pool all realizations' unit-years and apply the unit operator."""
        units = [self.annual_units(d) for d in data_per_real]
        pooled = np.concatenate(units) if units else np.array([], dtype=float)
        return self.unit_operator(pooled)

    def compute_for_borg(self, data_per_real: list) -> float:
        """Borg minimizes, so negate maximize objectives."""
        raw = self.compute(data_per_real)
        return -raw if self.direction == "maximize" else raw

    def compute_for_borg_from_units(self, pooled_units) -> float:
        """Borg-format objective from precomputed pooled unit-year metrics.

        Equivalent to :meth:`compute_for_borg` but consumes the concatenated
        stage-(i) annual metrics directly. This lets the memory-batched
        ensemble path in ``src.simulation`` reduce each batch's realizations
        to their per-unit annual-metric vectors (freeing the timeseries), and
        aggregate once over the whole pooled ensemble at the end.
        """
        raw = self.unit_operator(np.asarray(pooled_units, dtype=float).ravel())
        return -raw if self.direction == "maximize" else raw


###############################################################################
# Stage (i) annual-metric functions
###############################################################################
# Each returns a float ndarray with one value per metric-bearing water-year
# unit (see water_year_unit_slices). Weekly accounting reuses the §1
# windowed-series cores from src.objectives, restricted to each unit-year's
# weeks (weekly bins are formed within the unit-year slice).


def _delivery_failure_weeks_annual(data: dict, demand_key: str,
                                   delivery_key: str, cap: float,
                                   reset: str) -> np.ndarray:
    """Failing-week count per unit-year for a delivery objective.

    A week fails when weekly-total delivery < 99% of the weekly-total
    running-average entitlement (min(demand, banked allowance); same weekly-sum
    basis as the §1 reliability metric). The entitlement bank is path-dependent,
    so it is reconstructed on the full realization series before water-year
    slicing.
    """
    demand = data["ibt_demands"][demand_key]
    delivery = data["ibt_diversions"][delivery_key]
    target = _delivery_entitlement(demand, delivery, cap, reset)
    return np.asarray([
        float((~_weekly_delivery_ok(target.iloc[sl], delivery.iloc[sl])).sum())
        for sl in water_year_unit_slices(demand.index)
    ], dtype=float)


def _nyc_delivery_failure_weeks_annual(data: dict) -> np.ndarray:
    """NYC failing-week count per unit-year (running-avg right, 800 MGD)."""
    return _delivery_failure_weeks_annual(
        data, "demand_nyc", "delivery_nyc", NYC_DECREE_DIVERSION_CAP_MGD,
        reset="annual",
    )


def _nj_delivery_failure_weeks_annual(data: dict) -> np.ndarray:
    """NJ failing-week count per unit-year (running-avg baseline, 100 MGD)."""
    return _delivery_failure_weeks_annual(
        data, "demand_nj", "delivery_nj", NJ_DELIVERY_CAP_MGD,
        reset="monthly",
    )


def _nyc_delivery_deficit_cvar90_annual(data: dict) -> np.ndarray:
    """CVaR90 of weekly NYC delivery deficit % within each unit-year. [0, 100]."""
    demand = data["ibt_demands"]["demand_nyc"]
    delivery = data["ibt_diversions"]["delivery_nyc"]
    target = _delivery_entitlement(
        demand, delivery, NYC_DECREE_DIVERSION_CAP_MGD, reset="annual")
    return np.asarray([
        _cvar_worst_mean(
            _weekly_delivery_deficit_pct(
                target.iloc[sl], delivery.iloc[sl], NYC_DECREE_DIVERSION_CAP_MGD,
            ).values
        )
        for sl in water_year_unit_slices(demand.index)
    ], dtype=float)


def _flow_failure_weeks_annual(flow: pd.Series, target: float) -> np.ndarray:
    """Failing-week count per unit-year: weekly-mean flow < the Decree target."""
    return np.asarray([
        float((~_weekly_flow_ok(flow.iloc[sl], target)).sum())
        for sl in water_year_unit_slices(flow.index)
    ], dtype=float)


def _montague_failure_weeks_annual(data: dict) -> np.ndarray:
    """Montague failing-week count per unit-year (target 1131.05 MGD)."""
    return _flow_failure_weeks_annual(
        data["major_flow"]["delMontague"], MONTAGUE_DECREE_TARGET_MGD,
    )


def _trenton_failure_weeks_annual(data: dict) -> np.ndarray:
    """Trenton failing-week count per unit-year (target 1938.95 MGD)."""
    return _flow_failure_weeks_annual(
        data["major_flow"]["delTrenton"], TRENTON_DECREE_TARGET_MGD,
    )


def _montague_deficit_cvar90_annual(data: dict) -> np.ndarray:
    """CVaR90 of weekly Montague flow deficit % within each unit-year. [0, 100]."""
    flow = data["major_flow"]["delMontague"]
    return np.asarray([
        _cvar_worst_mean(
            _weekly_flow_deficit_pct(flow.iloc[sl], MONTAGUE_DECREE_TARGET_MGD).values
        )
        for sl in water_year_unit_slices(flow.index)
    ], dtype=float)


def _flood_days_minor_annual(data: dict) -> np.ndarray:
    """Days per unit-year any tail gauge >= its NWS minor flood stage."""
    over = _flood_over_stage_daily(
        data["flood_stage"][_DOWNSTREAM_GAUGES], "minor",
    )
    return np.asarray([
        float(over.iloc[sl].sum())
        for sl in water_year_unit_slices(over.index)
    ], dtype=float)


def _nyc_storage_min_annual(data: dict) -> np.ndarray:
    """Annual minimum of daily aggregate NYC storage % per unit-year. [0, 100]."""
    storage_pct = _nyc_storage_pct_daily(data)
    return np.asarray([
        float(storage_pct.iloc[sl].min())
        for sl in water_year_unit_slices(storage_pct.index)
    ], dtype=float)


###############################################################################
# Failure-year week-count thresholds (k) & env override
###############################################################################
# k = minimum number of failing weeks that marks a water-year unit as a
# failure-year for the frequency (reliability) objectives (objective_definitions.md
# §2). NYC and Montague use k = 3: a failure year is a ~month-scale shortfall, not
# an isolated off week. This materially raises Montague reliability (its failing
# weeks are graded, so k reclassifies the 1-2-week years); NYC is nearly
# threshold-insensitive (its shortfalls are whole-season curtailments), so there
# the choice is mainly definitional. Trenton and NJ stay at k = 1 — at k = 3
# Trenton saturates toward 1.0, compressing the metric. Placeholder pending the
# ensemble objective-sensitivity experiment; overridable via NYCOPT_FAILURE_K.
_DEFAULT_FAILURE_K: dict[str, int] = {
    "nyc_delivery_reliability_annual":   3,
    "montague_flow_reliability_annual":  3,
    "trenton_flow_reliability_annual":   1,
    "nj_delivery_reliability_annual":    1,
}


def _resolve_failure_k() -> dict[str, int]:
    """Apply the NYCOPT_FAILURE_K JSON env override on top of the defaults.

    ``NYCOPT_FAILURE_K`` is a JSON object ``{"<annual objective name>": k}``.
    Unknown names raise KeyError so a typo cannot silently leave a default in
    place.
    """
    failure_k = dict(_DEFAULT_FAILURE_K)
    raw = os.environ.get("NYCOPT_FAILURE_K", "").strip()
    if raw:
        overrides = json.loads(raw)
        for name, k in overrides.items():
            if name not in failure_k:
                raise KeyError(
                    f"NYCOPT_FAILURE_K: unknown frequency objective '{name}'. "
                    f"Available: {sorted(failure_k)}"
                )
            failure_k[name] = int(k)
    return failure_k


###############################################################################
# Re-evaluation satisficing thresholds & env override
###############################################################################
# Per-BASE-objective satisficing levels applied to the PER-REALIZATION §1
# metrics of the persisted re-eval matrix (reeval_core summary derivation and
# robustness threshold/kind metadata). Labels keep the historical
# `<base>__sat<thr>` form; they are threshold labels, not objective names.
# Placeholder values pending the sensitivity experiment; override via
# NYCOPT_SAT_THRESHOLDS.

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "nyc_delivery_reliability_weekly__sat95":     0.95,
    "nyc_delivery_deficit_cvar90_pct__sat10":     10.0,
    "montague_flow_reliability_weekly__sat85":    0.85,
    "montague_flow_deficit_cvar90_pct__sat25":    25.0,
    "trenton_flow_reliability_weekly__sat85":     0.85,
    "nj_delivery_reliability_weekly__sat95":      0.95,
    "downstream_flood_days_minor__sat10":         10.0,
    "nyc_storage_p5_pct__sat25":                  25.0,
}


def _resolve_thresholds() -> dict[str, float]:
    """Apply NYCOPT_SAT_THRESHOLDS JSON env override on top of defaults."""
    thresholds = dict(_DEFAULT_THRESHOLDS)
    raw = os.environ.get("NYCOPT_SAT_THRESHOLDS", "").strip()
    if raw:
        overrides = json.loads(raw)
        for k, v in overrides.items():
            if k not in thresholds:
                raise KeyError(
                    f"NYCOPT_SAT_THRESHOLDS: unknown threshold label '{k}'. "
                    f"Available: {sorted(thresholds)}"
                )
            thresholds[k] = float(v)
    return thresholds


# (base_objective_name, threshold_label, kind, legacy_epsilon) — the re-eval
# satisficing layer. `kind` is the satisficing direction relative to the BASE
# objective (maximize-base -> "ge", minimize-base -> "le"). The 4th slot is
# the legacy satisficing-fraction epsilon, retained for the ensemble
# objective-sensitivity diagnostic
# (scripts/supplemental/ensemble_objective_sensitivity_figures.py), which
# consumes this spec directly.
_REGISTRY_SPEC: list[tuple[str, str, Literal["ge", "le"], float]] = [
    ("nyc_delivery_reliability_weekly",
     "nyc_delivery_reliability_weekly__sat95",   "ge", 0.02),
    ("nyc_delivery_deficit_cvar90_pct",
     "nyc_delivery_deficit_cvar90_pct__sat10",   "le", 0.02),
    ("montague_flow_reliability_weekly",
     "montague_flow_reliability_weekly__sat85",  "ge", 0.02),
    ("montague_flow_deficit_cvar90_pct",
     "montague_flow_deficit_cvar90_pct__sat25",  "le", 0.02),
    ("trenton_flow_reliability_weekly",
     "trenton_flow_reliability_weekly__sat85",   "ge", 0.02),
    ("nj_delivery_reliability_weekly",
     "nj_delivery_reliability_weekly__sat95",    "ge", 0.02),
    ("downstream_flood_days_minor",
     "downstream_flood_days_minor__sat10",       "le", 0.02),
    ("nyc_storage_p5_pct",
     "nyc_storage_p5_pct__sat25",                "ge", 0.02),
]


###############################################################################
# Annual-unit objective registry
###############################################################################
# One entry per §2 objective:
#   (name, base_name, direction, epsilon, annual_metric, operator, description)
# `operator` is either the string "frequency" (built with the resolved
# per-objective k) or a stage-(ii) operator instance whose `worst_value` is
# the metric's orientation-aware non-finite sentinel. The ACTIVE/default
# objectives' epsilons are in native units, calibrated as ~IQR/10 (Reed et al.
# 2013) of each objective's spread across 24 random-DV policies on the historic
# reference trace scored as N=1 over its 76 water-year units
# (objective_sensitivity_run.py, seed 42, 2026-07-15), rounded to clean steps
# and floored at the 1/76 frequency granularity; to be reconciled against the
# larger-NL ensemble sensitivity experiment
# (ensemble_objective_sensitivity_experiment.md). The optional NJ objective and
# the diagnostic P99 flood variant (both absent from the default set) keep
# PLACEHOLDER epsilons pending their own inclusion/calibration.

_ANNUAL_REGISTRY_SPEC: list[tuple] = [
    ("nyc_delivery_reliability_annual",
     "nyc_delivery_reliability_weekly", "maximize", 0.01,
     _nyc_delivery_failure_weeks_annual, "frequency",
     "Frac of pooled unit-years with < k weeks of NYC delivery "
     "< 99% of capped demand (800 MGD Decree cap)"),
    ("nyc_delivery_deficit_p99_pct",
     "nyc_delivery_deficit_cvar90_pct", "minimize", 1.0,
     _nyc_delivery_deficit_cvar90_annual, PooledPercentileOp(99.0, worst_value=100.0),
     "P99 across pooled unit-years of within-year CVaR90 weekly NYC "
     "delivery deficit, % of Decree cap [0-100]"),
    ("montague_flow_reliability_annual",
     "montague_flow_reliability_weekly", "maximize", 0.05,
     _montague_failure_weeks_annual, "frequency",
     "Frac of pooled unit-years with < k weeks of weekly-mean Montague "
     "flow < 1131.05 MGD Decree target"),
    ("montague_flow_deficit_p99_pct",
     "montague_flow_deficit_cvar90_pct", "minimize", 1.5,
     _montague_deficit_cvar90_annual, PooledPercentileOp(99.0, worst_value=100.0),
     "P99 across pooled unit-years of within-year CVaR90 weekly Montague "
     "flow deficit, % of Decree target [0-100]"),
    ("trenton_flow_reliability_annual",
     "trenton_flow_reliability_weekly", "maximize", 0.01,
     _trenton_failure_weeks_annual, "frequency",
     "Frac of pooled unit-years with < k weeks of weekly-mean Trenton "
     "flow < 1938.95 MGD Decree target"),
    ("downstream_flood_days_annual",
     "downstream_flood_days_minor", "minimize", 0.005,
     _flood_days_minor_annual, PooledMeanOp(worst_value=366.0),
     "Mean across pooled unit-years of days any tail gauge >= NWS minor "
     "flood stage (expected annual flood days)"),
    ("downstream_flood_days_annual_p99",
     "downstream_flood_days_minor", "minimize", 3.0,
     _flood_days_minor_annual, PooledPercentileOp(99.0, worst_value=366.0),
     "DIAGNOSTIC: P99 across pooled unit-years of annual minor-flood days "
     "(expectation can mask floods — Quinn et al. 2017)"),
    ("nyc_storage_min_p01_pct",
     "nyc_storage_p5_pct", "maximize", 2.0,
     _nyc_storage_min_annual, PooledPercentileOp(1.0, worst_value=0.0),
     "P01 across pooled unit-years of the annual minimum daily aggregate "
     "NYC storage, % of capacity [0-100]"),
    ("nj_delivery_reliability_annual",
     "nj_delivery_reliability_weekly", "maximize", 0.01,
     _nj_delivery_failure_weeks_annual, "frequency",
     "Frac of pooled unit-years with < k weeks of NJ diversion "
     "< 99% of capped demand (100 MGD baseline); pending redundancy screen"),
]

# Base-objective-name -> annual-objective-name, so config.ACTIVE_OBJECTIVES
# (which lists BASE §1 objective names) can drive the ensemble search /
# re-eval path unchanged. The diagnostic downstream_flood_days_annual_p99
# variant is deliberately absent (it shares a base with the mean form and is
# resolvable only by its own name).
_BASE_TO_ENSEMBLE: dict[str, str] = {
    "nyc_delivery_reliability_weekly":  "nyc_delivery_reliability_annual",
    "nyc_delivery_deficit_cvar90_pct":  "nyc_delivery_deficit_p99_pct",
    "montague_flow_reliability_weekly": "montague_flow_reliability_annual",
    "montague_flow_deficit_cvar90_pct": "montague_flow_deficit_p99_pct",
    "trenton_flow_reliability_weekly":  "trenton_flow_reliability_annual",
    "downstream_flood_days_minor":      "downstream_flood_days_annual",
    "nyc_storage_p5_pct":               "nyc_storage_min_p01_pct",
    "nj_delivery_reliability_weekly":   "nj_delivery_reliability_annual",
}


def _build_registry() -> dict[str, AnnualUnitObjective]:
    """Build the annual-unit registry, resolving k and re-eval thresholds."""
    failure_k = _resolve_failure_k()
    thresholds = _resolve_thresholds()
    sat_by_base = {
        base: (thresholds[label], kind)
        for base, label, kind, _eps in _REGISTRY_SPEC
    }
    registry: dict[str, AnnualUnitObjective] = {}
    for name, base_name, direction, eps, metric, op, desc in _ANNUAL_REGISTRY_SPEC:
        if base_name not in OBJECTIVES:
            raise KeyError(
                f"annual registry references unknown base objective '{base_name}'"
            )
        if base_name not in sat_by_base:
            raise KeyError(
                f"annual registry base '{base_name}' missing from the re-eval "
                f"satisficing layer (_REGISTRY_SPEC)"
            )
        if op == "frequency":
            op = FailureFrequencyOp(k=failure_k[name])
        thr, kind = sat_by_base[base_name]
        registry[name] = AnnualUnitObjective(
            name=name,
            direction=direction,
            epsilon=eps,
            description=desc,
            annual_metric=metric,
            unit_operator=op,
            base=OBJECTIVES[base_name],
            aggregator=SatisficingAgg(threshold=thr, kind=kind),
        )
    return registry


ENSEMBLE_OBJECTIVES: dict[str, AnnualUnitObjective] = _build_registry()

assert all(v in ENSEMBLE_OBJECTIVES for v in _BASE_TO_ENSEMBLE.values()), \
    "_BASE_TO_ENSEMBLE references unregistered annual objectives"


###############################################################################
# Assembler
###############################################################################

def build_ensemble_objective_set(items) -> ObjectiveSet:
    """Assemble an ObjectiveSet from a list of annual-unit objective names.

    Mirrors `src.objectives.build_objective_set` but resolves names against
    `ENSEMBLE_OBJECTIVES`. Items may be:
      - str: an annual objective name (e.g. ``nyc_delivery_reliability_annual``)
             OR the underlying §1 base objective name, which is resolved via
             `_BASE_TO_ENSEMBLE`. Accepting base names lets
             `config.ACTIVE_OBJECTIVES` drive the ensemble path directly.
      - AnnualUnitObjective: use directly.

    Returns:
        ObjectiveSet whose contained objectives all expose
        ``compute(data_per_real)``, ``compute_for_borg(data_per_real)``, and
        the batched-path methods ``annual_units(data)`` /
        ``compute_for_borg_from_units(pooled_units)``.
    """
    resolved = []
    for item in items:
        if isinstance(item, AnnualUnitObjective):
            resolved.append(item)
        elif isinstance(item, str):
            name = item
            if name not in ENSEMBLE_OBJECTIVES:
                name = _BASE_TO_ENSEMBLE.get(item)
                if name is None:
                    raise KeyError(
                        f"Unknown ensemble objective '{item}'. Pass an annual "
                        f"objective name or a base objective name. Available "
                        f"annual: {sorted(ENSEMBLE_OBJECTIVES)}; available base: "
                        f"{sorted(_BASE_TO_ENSEMBLE)}."
                    )
            resolved.append(ENSEMBLE_OBJECTIVES[name])
        else:
            raise TypeError(
                f"build_ensemble_objective_set items must be str or "
                f"AnnualUnitObjective; got {type(item).__name__}"
            )
    return ObjectiveSet(resolved)


def list_available_ensemble_objectives() -> str:
    """Return a formatted table of all registered annual-unit objectives."""
    lines = [f"Available ensemble objectives ({len(ENSEMBLE_OBJECTIVES)}):"]
    for name, obj in ENSEMBLE_OBJECTIVES.items():
        op = obj.unit_operator
        if isinstance(op, FailureFrequencyOp):
            op_str = f"failure-frequency(k={op.k})"
        elif isinstance(op, PooledPercentileOp):
            op_str = f"P{op.q:g}(sentinel={op.worst_value:g})"
        elif isinstance(op, PooledMeanOp):
            op_str = f"mean(sentinel={op.worst_value:g})"
        else:
            op_str = type(op).__name__
        lines.append(
            f"  {name}: base={obj.base.name}, {obj.direction}, "
            f"op={op_str}, eps={obj.epsilon} — {obj.description}"
        )
    return "\n".join(lines)
