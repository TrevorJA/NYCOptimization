"""
objectives_ensemble.py - Ensemble (multi-realization) objective framework.

Implements the **two-layer annual-unit scheme** of
`docs/notes/methods/objective_definitions.md` §2 for all ensemble
(multi-realization) evaluations. Every scenario design is scored through this
annual-unit registry — including `historic`, which is simply the N=1 case over
its 76 water-year units (see `src/formulations/__init__.py`); the §1 temporal
metrics in `src.objectives` supply the shared windowed-series cores and the
per-realization base metrics of the re-evaluation layer.

Two-layer scheme (Hamilton et al. 2022 vocabulary)
--------------------------------------------------
Stage (i) — **annual metric** per (realization × water-year) unit. Every
synthetic realization starts January 1 (config ``ENSEMBLE_START_DATE``; the
historic trace starts on its water-year boundary, config ``START_DATE``) and
spans L whole calendar years. The first ``METRIC_EXCLUSION_MONTHS`` (6)
calendar months are outside the metric window — the SSI-6 accumulation spin-up
the hazard-selection metrics also exclude — and are dropped by date; the
remainder, which begins Jul 1 of year 1, is split into whole water-year units
(Oct 1 – Sep 30). The leading partial (Jul 1 – Sep 30 of year 1) and the
trailing partial (Oct 1 – Dec 31 of year L) are discarded, so an L-year
realization yields exactly **L − 1 metric-bearing unit-years**, the first
starting Oct 1 of year 1 (see :func:`water_year_unit_slices`, which derives
the units from the dates and gives L − 1 units for October starts too).

Stage (ii) — **unit operator** over the POOLED unit-years of the whole
ensemble (all realizations' units concatenated):

- *Failure frequency* (reliability objectives): fraction of unit-years WITHOUT
  failure, where a unit-year fails when it has >= k failing weeks (k
  configurable via ``_DEFAULT_FAILURE_K`` / ``NYCOPT_FAILURE_K``). Maximize;
  0-1 fraction.
- *Pooled percentile*: P99 of the annual metric for the tail-deficit
  objectives ("worst-1st-percentile unit-year", minimize) and P01 for the
  annual-minimum-storage objective (maximize).
- *Pooled mean*: expected annual flood exceedance (minimize). The P99 unit
  operator is tie-degenerate at the campaign's 900 pooled unit-years and
  12-30x noisier under bootstrap; it stays registered as an inactive
  diagnostic.

Non-finite annual metrics: a non-finite unit-year counts as a **failure-year**
for the frequency objectives; for the mean/percentile objectives it is
replaced by the objective's orientation-aware worst-possible sentinel
(``worst_value``: 100% for the bounded deficit percentages, 5490 ft-days for
annual flood exceedance, 0% for annual minimum storage) before aggregation, so a
degenerate unit pushes the objective toward failure instead of being dropped.

Metric reuse: all weekly accounting (weekly sums for delivery, weekly means
for flows, the 0.99 satisfaction factor, deficit-% normalization, the
running-average delivery entitlement via ``_delivery_entitlement``, CVaR90 via
``_cvar_worst_mean``) is imported from `src.objectives` windowed-series cores,
so §1 and §2 share one formula per quantity. Deficit-% and storage-% metrics
are 0-100 scales matching §1; frequency objectives are 0-1 fractions.

The active set's epsilons are calibrated in native metric units by the
epsilon-calibration experiment (see the `_ANNUAL_REGISTRY_SPEC` comment and
`docs/notes/methods/epsilon_calibration_experiment.md`). NJ delivery is the
active 8th objective. The inactive-registry entries carry calibrated values
from the same experiment.

Re-evaluation layer (unified currency)
--------------------------------------
The re-eval robustness pipeline (`src.reeval_core`, `src.robustness`) scores
THE SAME annual-unit objectives, recomputed per deeply-uncertain state of the
world (SOW): each E_test SOW's R realizations contribute their unit-years to
one pooled sample, the §2 unit operator collapses that pool to the SOW's
objective value, and robustness/regret are transformations of those per-SOW
objective values (Herman et al. 2014, 2015; Trindade et al. 2017; McPhail et
al. 2018). There is ONE metric currency: search, robustness, and regret all
speak the annual-unit objective statistics.

Each :class:`AnnualUnitObjective` therefore also carries:

- ``base`` — the §1 single-trace ``Objective`` supplying the shared
  windowed-series cores (provenance; the re-eval path never computes its
  whole-trace scalar);
- ``sat_threshold`` / ``sat_kind`` — the satisficing criterion applied to the
  per-SOW objective values by the robustness layer (``sat_kind`` follows the
  objective's own direction: maximize → "ge", minimize → "le").

Thresholds are labelled by ``<annual name>__sat<thr>`` keys and remain
overridable via ``NYCOPT_SAT_THRESHOLDS`` (JSON label→threshold). No CLI flags.

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
    METRIC_EXCLUSION_MONTHS,
    MONTAGUE_DECREE_TARGET_MGD,
    NJ_DELIVERY_CAP_MGD,
    NYC_DECREE_DIVERSION_CAP_MGD,
    TRENTON_DECREE_TARGET_MGD,
)
from src.objectives import (
    OBJECTIVES,
    Objective,
    ObjectiveSet,
    _cvar_worst_mean,
    _delivery_entitlement,
    _DOWNSTREAM_GAUGES,
    _flood_over_stage_daily,
    _flood_exceedance_daily,
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

    Unit rule (objective_definitions.md §2): the rule is a pure function of
    the trace's dates, whatever day it starts. Days earlier than
    ``METRIC_EXCLUSION_MONTHS`` (6) calendar months after the first timestamp
    lie outside the metric window (the SSI-6 accumulation spin-up) and are
    dropped by date; the remaining days are grouped by water year (a date with
    month >= 10 belongs to water year ``year + 1``), and only COMPLETE water
    years — first day Oct 1, last day Sep 30 — are kept. On the January-start
    synthetic windows the remainder begins Jul 1 of year 1, so the leading
    partial (Jul 1 – Sep 30) and the trailing partial (Oct 1 – Dec 31 of year
    L) are discarded and an L-year realization yields exactly L − 1
    unit-years, the first starting Oct 1 of year 1. (An October-start trace,
    e.g. the historic record, likewise yields L − 1 units.)

    Args:
        index: Daily DatetimeIndex of the realization's full window.

    Returns:
        List of positional ``slice`` objects into ``index`` (usable with
        ``.iloc``), one per metric-bearing water-year unit, in time order.
    """
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return []
    cutoff = idx[0] + pd.DateOffset(months=METRIC_EXCLUSION_MONTHS)
    offset = int((idx < cutoff).sum())
    if offset >= len(idx):
        return []
    sub = idx[offset:]
    wy = np.asarray(sub.year) + (np.asarray(sub.month) >= 10).astype(int)
    change = np.flatnonzero(np.diff(wy)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(sub)]))
    slices = []
    for s, e in zip(starts, stops):
        first, last = sub[s], sub[e - 1]
        if (first.month, first.day) == (10, 1) and (last.month, last.day) == (9, 30):
            slices.append(slice(offset + int(s), offset + int(e)))
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
        epsilon: Borg epsilon in native metric units (calibrated for the
            active set; see the `_ANNUAL_REGISTRY_SPEC` comment).
        description: Human-readable description.
        annual_metric: Callable ``data -> np.ndarray`` returning one annual
            value per metric-bearing water-year unit of the realization
            (stage i).
        unit_operator: Callable ``pooled_units -> float`` collapsing the
            pooled unit-years of the whole ensemble (stage ii).
        base: The §1 single-trace ``Objective`` whose windowed-series cores the
            annual metric reuses (formula provenance; its whole-trace scalar is
            never computed by the re-evaluation path).
        sat_threshold: Satisficing level applied by the robustness layer to the
            per-SOW value of THIS objective (``None`` for objectives with no
            satisficing role, e.g. inactive diagnostics).
    """

    def __init__(self, name: str, direction: str, epsilon: float,
                 description: str, annual_metric: Callable,
                 unit_operator: Callable, base: Objective,
                 sat_threshold: float | None = None):
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
        self.sat_threshold = None if sat_threshold is None else float(sat_threshold)

    @property
    def sign(self) -> int:
        """Return 1 for maximize, -1 for minimize."""
        return 1 if self.direction == "maximize" else -1

    @property
    def sat_kind(self) -> Literal["ge", "le"]:
        """Satisficing direction, derived from the objective's own direction."""
        return "ge" if self.direction == "maximize" else "le"

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
    """Days per unit-year any tail gauge >= its NWS minor flood stage.

    Plain UNNORMALIZED within-year day count (a unit-year is already an annual
    window) — deliberately not the §1 base metric, which reports mean annual
    days/yr over its whole metric window.
    """
    over = _flood_over_stage_daily(
        data["flood_stage"][_DOWNSTREAM_GAUGES], "minor",
    )
    return np.asarray([
        float(over.iloc[sl].sum())
        for sl in water_year_unit_slices(over.index)
    ], dtype=float)


def _flood_exceedance_minor_annual(data: dict) -> np.ndarray:
    """Ft·days per unit-year above NWS minor flood stage at the worst gauge.

    Plain UNNORMALIZED within-year sum of the daily max-across-gauges
    exceedance (stage − minor)⁺ — the annual-unit counterpart of the §1
    ``downstream_flood_exceedance_minor`` metric, which reports mean annual
    ft·days/yr over its whole metric window.
    """
    sev = _flood_exceedance_daily(
        data["flood_stage"][_DOWNSTREAM_GAUGES], "minor",
    )
    return np.asarray([
        float(sev.iloc[sl].sum())
        for sl in water_year_unit_slices(sev.index)
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
# Trenton saturates toward 1.0, compressing the metric. The framing-convention
# k sweep (framing_convention_diagnostics.md §1) confirms no shipped k
# saturates in either ensemble composition and rankings are stable to k ± 1.
# Overridable via NYCOPT_FAILURE_K.
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
# Per-ANNUAL-objective satisficing levels applied by the robustness layer to
# the PER-SOW annual-unit objective values (the same statistics the search
# optimizes, pooled per deeply-uncertain state of the world at re-evaluation).
# Labels use the `<annual name>__sat<thr>` form; they are threshold labels,
# not objective names.
#
# STATUS: ADOPTED 2026-08-08, from the satisficing-threshold diagnostic re-run
# against the regenerated status-quo E_test re-eval cube (per-SOW annual-unit
# substrate, 1,000 SOWs x R=25; §0b rules applied — see
# docs/notes/methods/robustness_threshold_diagnostics.md and the measured
# basis in supplemental_config.RTD_RECOMMENDED_THRESHOLDS/_BASIS). The final
# vector equals the 2026-08-07 provisional values — every rule-1
# maintain-status-quo re-anchor (stricter-side rounding of the annual-unit
# historic anchors: nyc rel 0.6447 / nyc deficit P99 48.83 / montague rel
# 0.7895 / montague deficit P99 27.68 / trenton rel 0.8684 / nj rel 0.7368)
# and both rule-2 external goalposts survive the measured placements, so no
# label changed. Measured incumbent SOW pass fractions: nyc rel 0.005 /
# nyc deficit 0.980 / montague rel 0.000 / montague deficit 0.965 /
# trenton rel 0.054 / flood 0.443 / storage 0.014 / nj rel 0.020; joint
# Starr 0.000 (binding: montague rel, the one criterion whose anchor lies
# outside the incumbent's E_test support, max 0.746) — a motivation finding
# per §0, with ranking-criticality deferred to the OAT stringency sweep.
# STORAGE GOALPOST RESOLUTION: the carried 26% FFMP L5 floor is a criterion
# the incumbent itself fails on the historic trace (P01-of-annual-minima =
# 0%, the 1960s drought emptied storage), so §0b rule 1's re-anchor at
# maintain-status-quo would place the line at 0% — an all-pass, vacuous
# criterion. The diagnostic arbitrates FOR keeping the 26% rule-2 external
# goalpost: it is inside the E_test support (incumbent max 28.8%, 1.4% of
# SOWs pass), operationally meaningful (the FFMP drought-emergency
# boundary), and is adopted as a stringent ASPIRATIONAL criterion rather
# than an acceptable-performance line; the incumbent's historic failure of
# it is part of the motivation finding. Override via NYCOPT_SAT_THRESHOLDS.
# Already-persisted reeval_raw_meta.json files keep their snapshotted
# thresholds by design.

_DEFAULT_THRESHOLDS: dict[str, float] = {
    # Rule 1 re-anchors (stricter side of the annual-unit historic anchors).
    "nyc_delivery_reliability_annual__sat65":     0.65,
    "nyc_delivery_deficit_p99_pct__sat48":        48.0,
    "montague_flow_reliability_annual__sat79":    0.79,
    "montague_flow_deficit_p99_pct__sat27":       27.0,
    "trenton_flow_reliability_annual__sat87":     0.87,
    "nj_delivery_reliability_annual__sat74":      0.74,
    # Rule 2 external goalpost, CARRIED: the observed WY2001-2023 mean annual
    # exceedance in ft-days/yr — the §2 flood objective is the same quantity
    # (mean over unit-years of within-year ft-day sums), so the goalpost's
    # meaning survives the substrate change.
    "downstream_flood_exceedance_annual__sat1p17": 1.17,
    # DIAGNOSTIC counterpart in days/yr (inactive objective).
    "downstream_flood_days_annual__sat1":         1.0,
    # Rule 2 external goalpost, CARRIED: the FFMP drought-emergency (L5)
    # boundary's seasonal floor, 26% of combined capacity — now applied to the
    # per-SOW P01 of annual minimum storage ("the worst ~1% of unit-years stay
    # out of drought emergency"). See the status note above: the incumbent
    # fails this on the historic trace.
    "nyc_storage_min_p01_pct__sat26":             26.0,
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


# annual_objective_name -> threshold_label. The satisficing direction is NOT
# recorded here: it is the objective's own direction (maximize -> "ge",
# minimize -> "le"; ``AnnualUnitObjective.sat_kind``) — one fewer way for the
# criterion and the objective to disagree. The p99 flood-days diagnostic
# carries no satisficing role.
_SAT_LABELS: dict[str, str] = {
    "nyc_delivery_reliability_annual":
        "nyc_delivery_reliability_annual__sat65",
    "nyc_delivery_deficit_p99_pct":
        "nyc_delivery_deficit_p99_pct__sat48",
    "montague_flow_reliability_annual":
        "montague_flow_reliability_annual__sat79",
    "montague_flow_deficit_p99_pct":
        "montague_flow_deficit_p99_pct__sat27",
    "trenton_flow_reliability_annual":
        "trenton_flow_reliability_annual__sat87",
    "nj_delivery_reliability_annual":
        "nj_delivery_reliability_annual__sat74",
    "downstream_flood_exceedance_annual":
        "downstream_flood_exceedance_annual__sat1p17",
    "downstream_flood_days_annual":
        "downstream_flood_days_annual__sat1",
    "nyc_storage_min_p01_pct":
        "nyc_storage_min_p01_pct__sat26",
}


###############################################################################
# Annual-unit objective registry
###############################################################################
# One entry per §2 objective:
#   (name, base_name, direction, epsilon, annual_metric, operator, description)
# `operator` is either the string "frequency" (built with the resolved
# per-objective k) or a stage-(ii) operator instance whose `worst_value` is
# the metric's orientation-aware non-finite sentinel.
# Epsilons are the FINAL campaign vector in native metric units. Base
# calibration (2026-08-05): clean-ceil of max(signal IQR/10, bootstrap noise
# floor, frequency granularity) over the ensemble campaign designs. Revised
# 2026-08-12 from the grouped re-filter sweep on the 500k-NFE production
# fronts (ensemble-averaged objectives are far smoother than the single
# trace, so the 2026-08-05 vector under-resolved ensemble fronts): one
# shared epsilon per objective family — reliabilities 0.05, deficit-P99s
# 10.0 (the measured hazfill recommendation), flood and storage unchanged.
# Derivation, per-axis exceptions, and disclosures:
# docs/notes/methods/epsilon_calibration_experiment.md.

_ANNUAL_REGISTRY_SPEC: list[tuple] = [
    ("nyc_delivery_reliability_annual",
     "nyc_delivery_reliability_weekly", "maximize", 0.05,
     _nyc_delivery_failure_weeks_annual, "frequency",
     "Frac of pooled unit-years with < k weeks of NYC delivery "
     "< 99% of the running-average entitlement"),
    ("nyc_delivery_deficit_p99_pct",
     "nyc_delivery_deficit_cvar90_pct", "minimize", 10.0,
     _nyc_delivery_deficit_cvar90_annual, PooledPercentileOp(99.0, worst_value=100.0),
     "P99 across pooled unit-years of within-year CVaR90 weekly NYC "
     "delivery deficit, % of Decree cap [0-100]"),
    ("montague_flow_reliability_annual",
     "montague_flow_reliability_weekly", "maximize", 0.05,
     _montague_failure_weeks_annual, "frequency",
     "Frac of pooled unit-years with < k weeks of weekly-mean Montague "
     "flow < 1131.05 MGD Decree target"),
    ("montague_flow_deficit_p99_pct",
     "montague_flow_deficit_cvar90_pct", "minimize", 10.0,
     _montague_deficit_cvar90_annual, PooledPercentileOp(99.0, worst_value=100.0),
     "P99 across pooled unit-years of within-year CVaR90 weekly Montague "
     "flow deficit, % of Decree target [0-100]"),
    ("trenton_flow_reliability_annual",
     "trenton_flow_reliability_weekly", "maximize", 0.05,
     _trenton_failure_weeks_annual, "frequency",
     "Frac of pooled unit-years with < k weeks of weekly-mean Trenton "
     "flow < 1938.95 MGD Decree target"),
    ("downstream_flood_exceedance_annual",
     "downstream_flood_exceedance_minor", "minimize", 0.3,
     # worst_value: 366 days x ~15 ft, the largest per-day exceedance the
     # rating curves can produce before endpoint saturation (Bridgeville
     # 27.9 ft rated max - 13 ft minor).
     _flood_exceedance_minor_annual, PooledMeanOp(worst_value=5490.0),
     "Mean across pooled unit-years of ft-days above NWS minor flood stage "
     "at the worst-affected tail gauge (expected annual flood exceedance)"),
    ("downstream_flood_days_annual",
     "downstream_flood_days_minor", "minimize", 0.05,
     _flood_days_minor_annual, PooledMeanOp(worst_value=366.0),
     "DIAGNOSTIC (inactive; the day count is degenerate across policies): "
     "mean across pooled unit-years of days any tail gauge >= NWS minor "
     "flood stage"),
    ("downstream_flood_days_annual_p99",
     "downstream_flood_days_minor", "minimize", 1.5,
     _flood_days_minor_annual, PooledPercentileOp(99.0, worst_value=366.0),
     "DIAGNOSTIC (inactive; P99 is tie-degenerate at the campaign unit "
     "count): P99 across pooled unit-years of annual minor-flood days "
     "(expectation can mask floods — Quinn et al. 2017)"),
    ("nyc_storage_min_p01_pct",
     "nyc_storage_p5_pct", "maximize", 5.0,
     _nyc_storage_min_annual, PooledPercentileOp(1.0, worst_value=0.0),
     "P01 across pooled unit-years of the annual minimum daily aggregate "
     "NYC storage, % of capacity [0-100]"),
    ("nj_delivery_reliability_annual",
     "nj_delivery_reliability_weekly", "maximize", 0.05,
     _nj_delivery_failure_weeks_annual, "frequency",
     "Frac of pooled unit-years with < k weeks of NJ diversion "
     "< 99% of the running-average entitlement"),
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
    "downstream_flood_exceedance_minor":  "downstream_flood_exceedance_annual",
    "downstream_flood_days_minor":      "downstream_flood_days_annual",
    "nyc_storage_p5_pct":               "nyc_storage_min_p01_pct",
    "nj_delivery_reliability_weekly":   "nj_delivery_reliability_annual",
}


def _build_registry() -> dict[str, AnnualUnitObjective]:
    """Build the annual-unit registry, resolving k and satisficing thresholds."""
    failure_k = _resolve_failure_k()
    thresholds = _resolve_thresholds()
    registry: dict[str, AnnualUnitObjective] = {}
    for name, base_name, direction, eps, metric, op, desc in _ANNUAL_REGISTRY_SPEC:
        if base_name not in OBJECTIVES:
            raise KeyError(
                f"annual registry references unknown base objective '{base_name}'"
            )
        if op == "frequency":
            op = FailureFrequencyOp(k=failure_k[name])
        label = _SAT_LABELS.get(name)
        registry[name] = AnnualUnitObjective(
            name=name,
            direction=direction,
            epsilon=eps,
            description=desc,
            annual_metric=metric,
            unit_operator=op,
            base=OBJECTIVES[base_name],
            sat_threshold=None if label is None else thresholds[label],
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
