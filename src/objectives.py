"""
objectives.py - Whole-trace (§1) objective metric cores and registry.

Provides the `Objective` class, the `ObjectiveSet` container, the name-indexed
`OBJECTIVES` registry, and `build_objective_set()`. Runs select objectives by
listing registry names (`config.ACTIVE_OBJECTIVES` / `NYCOPT_OBJECTIVES`).

The ACTIVE search objectives are the annual-unit forms in
`src.objectives_ensemble`, which reuse the windowed-series cores defined here;
the whole-trace metrics in this module are single-trace diagnostics. Rationale
for the metric choices lives in docs/notes/methods/objective_definitions.md.

Naming convention: `{location}_{quantity}_{statistic}[_{unit}]`, e.g.
    nyc_delivery_reliability_weekly   -> NYC delivery, weekly reliability frequency
    nyc_delivery_deficit_cvar90_pct   -> NYC delivery, CVaR90 of weekly deficit, in %
    downstream_flood_exceedance_minor -> tail-gauge ft-days/yr above NWS minor stage
    nyc_storage_p5_pct                -> NYC storage, 5th percentile, in % of capacity

Contracts:
- Reliability = fraction of metric-window weeks a Decree threshold is met.
- Deficit CVaR90 = mean of the worst 10% of weekly deficits (max variants are
  diagnostics).
- Active flood metric = magnitude-weighted downstream flood exceedance
  (ft-days/yr at the worst tail gauge); day counts are diagnostics.
- Decree goalposts are the static 1954 quantities (NYC 800 MGD; Montague
  1131.05 MGD; Trenton 1938.95 MGD), never the live FFMP `mrf_target`.
- NYC/NJ delivery is scored against the running-average entitlement
  `min(demand, allowance)` (`_delivery_entitlement`), with the allowance bank
  accrued at the static baseline cap (a policy cannot lower its own goalpost).
- Salt-front and Lordville thermal metrics are diagnostic/deferred only.

Usage:
    from src.objectives import build_objective_set
    obj_set = build_objective_set(config.ACTIVE_OBJECTIVES)
    values = obj_set.compute(data)            # raw metric values
    borg_values = obj_set.compute_for_borg(data)  # all minimized
"""

import numpy as np
import pandas as pd

from pywrdrb.flood_thresholds import flood_stage_thresholds

from config import (
    NYC_RESERVOIRS,
    NYC_TOTAL_CAPACITY,
    METRIC_EXCLUSION_MONTHS,
    NYC_DECREE_DIVERSION_CAP_MGD,
    NJ_DELIVERY_CAP_MGD,
    MONTAGUE_DECREE_TARGET_MGD,
    TRENTON_DECREE_TARGET_MGD,
    LORDVILLE_THERMAL_THRESHOLD_C,
    SALT_FRONT_REFERENCE_RM,
)


# Reservoir-tail USGS gauges used by the downstream flood metrics (flooding
# here is attributable to NYC release decisions):
#   01426500 Hale Eddy   (below Cannonsville)
#   01421000 Fishs Eddy  (below Pepacton)
#   01436690 Bridgeville (below Neversink)
_DOWNSTREAM_GAUGES = ["01426500", "01421000", "01436690"]

# Tail fraction for the CVaR (Conditional Value-at-Risk) deficit metrics.
# 0.10 => CVaR90 => mean of the worst 10% of weekly deficits.
_CVAR_TAIL_FRAC = 0.10


###############################################################################
# Objective Class
###############################################################################

class Objective:
    """A single objective metric for the optimization problem.

    Attributes:
        name: Unique identifier for this objective.
        direction: "maximize" or "minimize".
        epsilon: Resolution for Borg epsilon-dominance archiving.
        description: Human-readable description.
        func: Callable(data: dict) -> float that computes the metric.
    """

    def __init__(self, name: str, direction: str, epsilon: float,
                 description: str, func):
        if direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got '{direction}'")
        self.name = name
        self.direction = direction
        self.epsilon = epsilon
        self.description = description
        self.func = func

    def compute(self, data: dict) -> float:
        """Compute the raw metric value from simulation data."""
        return self.func(data)

    def compute_for_borg(self, data: dict) -> float:
        """Compute value in Borg-compatible format (minimization).

        For maximize objectives, negates the value so Borg minimization
        is equivalent to maximization.
        """
        raw = self.compute(data)
        return -raw if self.direction == "maximize" else raw

    @property
    def sign(self) -> int:
        """Return 1 for maximize, -1 for minimize."""
        return 1 if self.direction == "maximize" else -1


###############################################################################
# ObjectiveSet — Ordered Collection of Objectives
###############################################################################

class ObjectiveSet:
    """An ordered collection of objectives for a specific optimization run.

    Provides the interface that Borg, diagnostics, and analysis scripts
    need: names, epsilons, directions, and batch compute methods.
    """

    def __init__(self, objectives: list):
        self._objectives = list(objectives)
        self._by_name = {obj.name: obj for obj in self._objectives}

    def __len__(self):
        return len(self._objectives)

    def __iter__(self):
        return iter(self._objectives)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._by_name[key]
        return self._objectives[key]

    @property
    def names(self) -> list:
        return [obj.name for obj in self._objectives]

    @property
    def epsilons(self) -> list:
        return [obj.epsilon for obj in self._objectives]

    @property
    def directions(self) -> list:
        """1 for maximize, -1 for minimize."""
        return [obj.sign for obj in self._objectives]

    @property
    def n_objs(self) -> int:
        return len(self._objectives)

    def compute(self, data: dict) -> list:
        """Compute all raw objective values from simulation data."""
        return [obj.compute(data) for obj in self._objectives]

    def compute_for_borg(self, data: dict) -> list:
        """Compute all objectives in Borg-compatible format (all minimized)."""
        return [obj.compute_for_borg(data) for obj in self._objectives]

    def compute_for_borg_ensemble(self, data_per_real: list) -> list:
        """Compute all objectives across realizations (Borg-minimized).

        Each contained objective must accept ``data_per_real`` (a list of
        per-realization data dicts) — i.e. it must be an
        ``AnnualUnitObjective`` from ``src.objectives_ensemble``. This is
        duck-typed: regular single-trace ``Objective`` instances will fail
        loudly when their metric function tries to subscript a list as a
        dict, which is the desired behavior (a single-trace ObjectiveSet
        should not be dispatched on the ensemble path).
        """
        return [obj.compute_for_borg(data_per_real) for obj in self._objectives]

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [f"ObjectiveSet ({self.n_objs} objectives):"]
        for obj in self._objectives:
            lines.append(
                f"  {obj.name}: {obj.direction} (eps={obj.epsilon}) — {obj.description}"
            )
        return "\n".join(lines)


###############################################################################
# Shared temporal-aggregation helpers
###############################################################################
# The `_weekly_*` / `_flood_*` / `_nyc_storage_pct_daily` cores operate on
# ALREADY-WINDOWED daily series: the §1 metrics apply `_metric_window` first,
# and the annual-unit metrics in `src.objectives_ensemble` apply FFMP-year unit
# slicing instead, so both paths share one weekly-accounting formula.


def _metric_window(obj):
    """Restrict a daily series to the metric window of its scenario.

    The window starts ``METRIC_EXCLUSION_MONTHS`` (6) calendar months after
    the first timestamp (the SSI-6 spin-up); on December-start traces the cut
    lands on June 1, the FFMP operating-year boundary. The cut is by date, so
    leap years need no special case.

    Args:
        obj: Daily-indexed pandas Series or DataFrame.

    Returns:
        The same type, with the pre-window rows removed. An empty input is
        returned unchanged.
    """
    idx = pd.DatetimeIndex(obj.index)
    if len(idx) == 0:
        return obj
    cutoff = idx[0] + pd.DateOffset(months=METRIC_EXCLUSION_MONTHS)
    return obj.loc[idx >= cutoff]


def _cvar_worst_mean(values, frac: float = _CVAR_TAIL_FRAC) -> float:
    """Mean of the worst (largest) ``frac`` fraction of finite values.

    For a series where larger = worse this is the CVaR at level ``(1 - frac)``,
    the mean of the worst ``ceil(frac * N)`` values. Non-finite entries are
    dropped; an empty series returns 0.0.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    k = max(1, int(np.ceil(frac * arr.size)))
    worst = np.sort(arr)[-k:]
    return float(worst.mean())


def _running_avg_budget(delivery: pd.Series, cap: float,
                        reset: str = "annual") -> pd.Series:
    """Reconstruct the FFMP running-average delivery allowance (the daily bank).

    Mirrors pywr-drb's ``FfmpNycRunningAvgParameter`` / ``FfmpNjRunningAvgParameter``
    recursion: ``budget[t] = max(0, budget[t-1] + cap - delivery[t-1])``, or
    ``cap`` on a reset day. The bank accrues at the STATIC ``cap``, never the
    model's drought-scaled allowance, so a policy cannot lower its own goalpost.

    Args:
        delivery: Daily delivery series (MGD) over the FULL realization window
            (the bank is path-dependent, so build it before any windowing).
        cap: Static running-average allowance (MGD); the daily accrual rate.
        reset: ``"annual"`` (NYC: reset on Jun 1) or ``"monthly"`` (NJ: reset on
            the 1st of each month). The NJ drought-factor reset and separate
            daily cap are not modeled.

    Returns:
        Daily allowance series aligned to ``delivery.index``.
    """
    idx = pd.DatetimeIndex(delivery.index)
    dlv = delivery.to_numpy(dtype=float)
    n = dlv.size
    budget = np.empty(n, dtype=float)
    if n == 0:
        return pd.Series(budget, index=idx)
    day = np.asarray(idx.day)
    if reset == "annual":
        is_reset = (np.asarray(idx.month) == 6) & (day == 1)
    elif reset == "monthly":
        is_reset = day == 1
    else:
        raise ValueError(f"reset must be 'annual' or 'monthly', got '{reset}'")
    budget[0] = cap  # model reset() sets the bank to cap at the series start
    for t in range(1, n):
        if is_reset[t]:
            budget[t] = cap
        else:
            b = budget[t - 1] + cap - dlv[t - 1]
            budget[t] = b if b > 0.0 else 0.0
    return pd.Series(budget, index=idx)


def _delivery_entitlement(demand: pd.Series, delivery: pd.Series, cap: float,
                          reset: str = "annual") -> pd.Series:
    """Daily realizable delivery entitlement = min(demand, running-avg allowance).

    Demand is not clipped at a flat daily ``cap``: a day's entitlement is the
    smaller of the demand and the allowance banked to that day
    (:func:`_running_avg_budget`), so demand beyond the banked right is not
    counted as owed.

    Args:
        demand: Daily demand series (MGD) over the full realization window.
        delivery: Daily delivery series (MGD) over the full realization window.
        cap: Static running-average allowance (MGD).
        reset: Budget-period reset cadence (see :func:`_running_avg_budget`).

    Returns:
        Daily entitlement series aligned to ``demand.index``.
    """
    budget = _running_avg_budget(delivery, cap, reset)
    target = np.minimum(demand.to_numpy(dtype=float),
                        budget.to_numpy(dtype=float))
    return pd.Series(target, index=demand.index)


def _weekly_delivery_deficit_pct(target: pd.Series, delivery: pd.Series,
                                 cap: float) -> pd.Series:
    """Weekly delivery deficit as % of a static Decree cap (windowed series).

    ``target`` is the daily realizable entitlement (:func:`_delivery_entitlement`).
    Normalized to the static ``cap`` so a fixed shortfall reads identically
    year-round.

    Args:
        target: Already-windowed daily entitlement series (MGD).
        delivery: Already-windowed daily delivery series (MGD).
        cap: Static Decree cap (MGD), used as the normalization denominator.

    Returns:
        Weekly deficit series in % of ``cap`` [0-100].
    """
    weekly_target = target.resample("W").mean()
    weekly_delivery = delivery.resample("W").mean()
    deficit = (weekly_target - weekly_delivery).clip(lower=0)
    return 100.0 * deficit / cap


def _nyc_weekly_delivery_deficit_pct(data: dict) -> pd.Series:
    """Post-exclusion weekly NYC delivery deficit, as % of the 800 MGD Decree cap."""
    delivery = data["ibt_diversions"]["delivery_nyc"]
    target = _delivery_entitlement(
        data["ibt_demands"]["demand_nyc"], delivery,
        NYC_DECREE_DIVERSION_CAP_MGD, reset="annual",
    )
    return _weekly_delivery_deficit_pct(
        _metric_window(target), _metric_window(delivery),
        NYC_DECREE_DIVERSION_CAP_MGD,
    )


def _weekly_flow_deficit_pct(flow: pd.Series, target: float) -> pd.Series:
    """Weekly flow deficit as % of a static Decree flow target (windowed series)."""
    weekly_flow = flow.resample("W").mean()
    deficit = (target - weekly_flow).clip(lower=0)
    return 100.0 * deficit / target


#: Numerical headroom (MGD) on the weekly Decree-target comparison so weeks
#: meeting the target exactly count as successes (rounding leaves weekly means
#: ~1e-12 MGD below target). 1e-6 is far below the 0.01-MGD scale of real
#: deficits.
_FLOW_TARGET_TOL_MGD = 1e-6


def _weekly_flow_ok(flow: pd.Series, target: float) -> pd.Series:
    """Weekly success indicators: weekly-mean flow >= a static Decree target.

    Operates on an already-windowed daily flow series. The comparison carries
    ``_FLOW_TARGET_TOL_MGD`` of numerical headroom so weeks where the model
    delivers the target exactly are successes. A week with a non-finite
    weekly mean compares False (a degenerate week is a failure week).
    """
    return flow.resample("W").mean() >= target - _FLOW_TARGET_TOL_MGD


def _flow_reliability_weekly(flow: pd.Series, target: float) -> float:
    """Fraction of metric-window weeks weekly-mean flow meets a Decree target."""
    ok = _weekly_flow_ok(_metric_window(flow), target)
    total = len(ok)
    if total == 0:
        return 0.0
    return float(ok.sum()) / total


def _weekly_delivery_ok(target: pd.Series, delivery: pd.Series) -> pd.Series:
    """Weekly success indicators: weekly-total delivery >= 99% of the entitlement.

    Operates on already-windowed daily series; weekly totals (sum basis) are the
    Decree accounting convention. ``target`` is the daily realizable entitlement
    (:func:`_delivery_entitlement`). A non-finite weekly comparison is False (a
    degenerate week is a failure week).
    """
    weekly_target = target.resample("W").sum()
    weekly_delivery = delivery.resample("W").sum()
    return weekly_delivery >= 0.99 * weekly_target


def _delivery_reliability_weekly(demand: pd.Series, delivery: pd.Series,
                                 cap: float, reset: str = "annual") -> float:
    """Fraction of metric-window weeks weekly delivery >= 99% of the entitlement.

    The entitlement is the running-average Decree right
    (:func:`_delivery_entitlement`), reconstructed on the FULL series before the
    metric window is taken, so the allowance bank carries the correct state.
    """
    target = _delivery_entitlement(demand, delivery, cap, reset)
    ok = _weekly_delivery_ok(_metric_window(target), _metric_window(delivery))
    total = len(ok)
    if total == 0:
        return 0.0
    return float(ok.sum()) / total


def _flood_over_stage_daily(stage: pd.DataFrame, level: str) -> pd.Series:
    """Daily indicators: is ANY tail gauge at/above the named NWS stage?

    Operates on an already-windowed daily stage DataFrame whose columns are the
    ``_DOWNSTREAM_GAUGES`` ids. ``level`` is one of "action", "minor",
    "moderate", "major" in ``pywrdrb.flood_thresholds.flood_stage_thresholds``.
    """
    thresh = pd.Series(
        {g: flood_stage_thresholds[g][level] for g in _DOWNSTREAM_GAUGES}
    )
    return stage.ge(thresh, axis=1).any(axis=1)


def _flood_days_anygauge(data: dict, level: str) -> float:
    """Mean annual days (days/yr) any tail gauge is at/above the named NWS stage.

    The metric-window day count is divided by the window length in years
    (days / 365.25), so the value is comparable across metric windows of
    different lengths (e.g. the ~76-yr historic trace vs 10-yr ensemble
    realizations). An empty metric window returns 0.0.
    """
    stage = _metric_window(data["flood_stage"][_DOWNSTREAM_GAUGES])
    n_days = len(stage)
    if n_days == 0:
        return 0.0
    count = float(_flood_over_stage_daily(stage, level).sum())
    return count / (n_days / 365.25)


def _flood_exceedance_daily(stage: pd.DataFrame, level: str) -> pd.Series:
    """Daily max-across-gauges positive exceedance above the named stage (ft).

    Operates on an already-windowed daily stage DataFrame whose columns are
    the ``_DOWNSTREAM_GAUGES`` ids. Each day contributes the largest
    exceedance across the three gauges (max-gauge basis avoids triple-counting
    a basin-wide event; flood_objective_diagnostics.md §0b).
    """
    thresh = pd.Series(
        {g: flood_stage_thresholds[g][level] for g in _DOWNSTREAM_GAUGES}
    )
    return stage.sub(thresh, axis=1).clip(lower=0.0).max(axis=1)


def _flood_exceedance_anygauge(data: dict, level: str) -> float:
    """Mean annual ft·days above the named NWS stage at the worst gauge.

    The magnitude-weighted counterpart of :func:`_flood_days_anygauge`:
    Σ over days of max-across-gauges (stage − threshold)⁺, divided by the
    metric-window length in years. [ft-days/yr]. An empty metric window
    returns 0.0.
    """
    stage = _metric_window(data["flood_stage"][_DOWNSTREAM_GAUGES])
    n_days = len(stage)
    if n_days == 0:
        return 0.0
    total = float(_flood_exceedance_daily(stage, level).sum())
    return total / (n_days / 365.25)


def _nyc_storage_pct_daily(data: dict) -> pd.Series:
    """Daily combined NYC storage as % of total system capacity (full window)."""
    storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1)
    return 100.0 * storage / NYC_TOTAL_CAPACITY


def _nyc_combined_storage_pct(data: dict) -> pd.Series:
    """Metric-window daily combined NYC storage as % of total system capacity."""
    return _metric_window(_nyc_storage_pct_daily(data))


###############################################################################
# Public aliases of the shared reductions
###############################################################################
# Post-processing that must agree with the objective values it annotates
# (e.g. the historic-timeseries figure) imports these instead of re-deriving
# the logic.

metric_window = _metric_window
weekly_flow_ok = _weekly_flow_ok
delivery_entitlement = _delivery_entitlement
nyc_storage_pct_daily = _nyc_storage_pct_daily
FLOW_TARGET_TOL_MGD = _FLOW_TARGET_TOL_MGD


###############################################################################
# Metric Functions — NYC water supply (1954 Decree right = 800 MGD)
###############################################################################


def _nyc_delivery_reliability_weekly(data: dict) -> float:
    """Fraction of weeks NYC delivery meets >= 99% of its running-average Decree right. [0, 1]."""
    return _delivery_reliability_weekly(
        data["ibt_demands"]["demand_nyc"],
        data["ibt_diversions"]["delivery_nyc"],
        NYC_DECREE_DIVERSION_CAP_MGD,
        reset="annual",
    )


def _nyc_delivery_deficit_cvar90_pct(data: dict) -> float:
    """CVaR90 of weekly NYC delivery deficit, as % of the 800 MGD Decree cap. [0, 100]."""
    return _cvar_worst_mean(_nyc_weekly_delivery_deficit_pct(data).values)


def _nyc_delivery_deficit_max_pct(data: dict) -> float:
    """DIAGNOSTIC: worst single-week NYC delivery deficit, % of Decree cap. [0, 100]."""
    s = _nyc_weekly_delivery_deficit_pct(data)
    return float(s.max()) if len(s) > 0 else 0.0


###############################################################################
# Metric Functions — New Jersey water supply (D&R Canal diversion)
###############################################################################


def _nj_delivery_reliability_weekly(data: dict) -> float:
    """Fraction of weeks NJ diversion meets >= 99% of its capped right. [0, 1]."""
    return _delivery_reliability_weekly(
        data["ibt_demands"]["demand_nj"],
        data["ibt_diversions"]["delivery_nj"],
        NJ_DELIVERY_CAP_MGD,
        reset="monthly",
    )


###############################################################################
# Metric Functions — Montague flow Decree (target = 1750 cfs = 1131.05 MGD)
###############################################################################
# NYC's downstream flow obligation. Reliability will not saturate at 1.0 because
# FFMP drought step-downs (L2-L5) intentionally drop releases below the target.


def _montague_flow_reliability_weekly(data: dict) -> float:
    """Fraction of weeks weekly-mean Montague flow >= 1131.05 MGD Decree target. [0, 1]."""
    return _flow_reliability_weekly(data["major_flow"]["delMontague"], MONTAGUE_DECREE_TARGET_MGD)


def _montague_flow_deficit_cvar90_pct(data: dict) -> float:
    """CVaR90 of weekly Montague flow deficit, % of Decree target. [0, 100]."""
    return _cvar_worst_mean(
        _weekly_flow_deficit_pct(
            _metric_window(data["major_flow"]["delMontague"]),
            MONTAGUE_DECREE_TARGET_MGD,
        ).values
    )


def _montague_flow_deficit_max_pct(data: dict) -> float:
    """DIAGNOSTIC: worst single-week Montague flow deficit, % of Decree target. [0, 100]."""
    s = _weekly_flow_deficit_pct(
        _metric_window(data["major_flow"]["delMontague"]), MONTAGUE_DECREE_TARGET_MGD,
    )
    return float(s.max()) if len(s) > 0 else 0.0


###############################################################################
# Metric Functions — Trenton flow Decree (target = 1938.95 MGD)
###############################################################################
# Lower-basin flow obligation; Trenton flow also proxies salt-front repulsion.


def _trenton_flow_reliability_weekly(data: dict) -> float:
    """Fraction of weeks weekly-mean Trenton flow >= 1938.95 MGD Decree target. [0, 1]."""
    return _flow_reliability_weekly(data["major_flow"]["delTrenton"], TRENTON_DECREE_TARGET_MGD)


def _trenton_flow_deficit_cvar90_pct(data: dict) -> float:
    """DIAGNOSTIC: CVaR90 of weekly Trenton flow deficit, % of Decree target. [0, 100]."""
    return _cvar_worst_mean(
        _weekly_flow_deficit_pct(
            _metric_window(data["major_flow"]["delTrenton"]),
            TRENTON_DECREE_TARGET_MGD,
        ).values
    )


###############################################################################
# Metric Functions — Downstream flood exposure (reservoir-tail gauges)
###############################################################################


def _downstream_flood_exceedance_minor(data: dict) -> float:
    """Mean annual ft·days above NWS minor flood stage at the worst gauge. [ft-days/yr]."""
    return _flood_exceedance_anygauge(data, "minor")


def _downstream_flood_days_minor(data: dict) -> float:
    """DIAGNOSTIC: mean annual days any tail gauge >= NWS minor flood stage. [days/yr]."""
    return _flood_days_anygauge(data, "minor")


def _downstream_flood_days_major(data: dict) -> float:
    """DIAGNOSTIC: mean annual days any tail gauge >= NWS major flood stage (severe). [days/yr]."""
    return _flood_days_anygauge(data, "major")


def _downstream_flood_days_action(data: dict) -> float:
    """DIAGNOSTIC: mean annual days any tail gauge >= FFMP L1 action stage. [days/yr]."""
    return _flood_days_anygauge(data, "action")


###############################################################################
# Metric Functions — NYC storage resilience
###############################################################################


def _nyc_storage_p5_pct(data: dict) -> float:
    """5th percentile of daily combined NYC storage, % of capacity. [0, 100]."""
    s = _nyc_combined_storage_pct(data)
    if len(s) == 0:
        return 0.0
    return float(np.percentile(s.values, 5))


def _nyc_storage_min_pct(data: dict) -> float:
    """DIAGNOSTIC: minimum daily combined NYC storage, % of capacity. [0, 100]."""
    s = _nyc_combined_storage_pct(data)
    return float(s.min()) if len(s) > 0 else 0.0


###############################################################################
# Metric Functions — Salt-front intrusion (LSTM) — DIAGNOSTIC ONLY
###############################################################################
# Diagnostic only; computed only when INCLUDE_SALINITY_MODEL=True.


def _salt_front_intrusion_max_rm(data: dict) -> float:
    """Maximum (most-upstream) salt-front position over the sim, in RM.

    Delaware River miles increase upstream from the bay mouth, so a HIGHER
    river-mile value means the salt front intruded farther upstream — worse for
    water supply at Trenton. NaN entries (e.g. the gate-skipped first sim day)
    are dropped before computing the max. Returns NaN if salinity is unavailable.
    """
    if "salinity" not in data:
        return float("nan")
    sf = data["salinity"].get("salt_front_location_mu")
    if sf is None:
        return float("nan")
    sf = _metric_window(sf).dropna()
    if sf.empty:
        return float("nan")
    return float(sf.max())


###############################################################################
# Metric Functions — Lordville thermal (LSTM) — DEFERRED
###############################################################################
# DEFERRED: inputs (multivariate meteorology) are unavailable for synthetic
# scenarios. Registered so the metric is one config flag from re-enable.


def _lordville_temp_exceedance_days(data: dict) -> float:
    """Days max water temp at Lordville exceeds the cold-water-fish threshold (°C).

    Reads data["temperature"]["temperature_after_thermal_release_mu"]; NaN
    entries (pre-LSTM-start) are dropped before counting.
    """
    if "temperature" not in data:
        return float("nan")
    temp = data["temperature"].get("temperature_after_thermal_release_mu")
    if temp is None:
        return float("nan")
    temp = _metric_window(temp).dropna()
    return float((temp > LORDVILLE_THERMAL_THRESHOLD_C).sum())


###############################################################################
# Objective Registry
###############################################################################
# The active subset is config.ACTIVE_OBJECTIVES; everything else is a
# registered diagnostic selectable by name.

OBJECTIVES: dict[str, Objective] = {}


# §1 epsilons apply to single-trace diagnostics only; the CAMPAIGN epsilons
# live on the annual-unit registry in src/objectives_ensemble.py.
def _register(name, direction, epsilon, description, func):
    OBJECTIVES[name] = Objective(
        name=name, direction=direction, epsilon=epsilon,
        description=description, func=func,
    )


# --- NYC water supply (Decree right = 800 MGD) ---
_register("nyc_delivery_reliability_weekly", "maximize", 0.07,
          f"Frac of weeks NYC delivery >= 99% of the running-avg entitlement "
          f"(min(demand, allowance); {NYC_DECREE_DIVERSION_CAP_MGD:.0f} MGD Decree right)",
          _nyc_delivery_reliability_weekly)
_register("nyc_delivery_deficit_cvar90_pct", "minimize", 1.5,
          f"CVaR90 of weekly NYC delivery deficit, % of "
          f"{NYC_DECREE_DIVERSION_CAP_MGD:.0f} MGD Decree cap [0-100]",
          _nyc_delivery_deficit_cvar90_pct)
_register("nyc_delivery_deficit_max_pct", "minimize", 3.0,
          "DIAGNOSTIC: worst-week NYC delivery deficit, % of Decree cap [0-100]",
          _nyc_delivery_deficit_max_pct)

# --- New Jersey water supply (D&R Canal diversion; active 8th objective) ---
_register("nj_delivery_reliability_weekly", "maximize", 0.007,
          f"Frac of weeks NJ diversion >= 99% of the running-avg entitlement "
          f"(min(demand, allowance); {NJ_DELIVERY_CAP_MGD:.0f} MGD baseline)",
          _nj_delivery_reliability_weekly)

# --- Montague flow Decree (NYC obligation; target = 1750 cfs = 1131.05 MGD) ---
_register("montague_flow_reliability_weekly", "maximize", 0.02,
          f"Frac of weeks Montague weekly-mean flow >= "
          f"{MONTAGUE_DECREE_TARGET_MGD:.0f} MGD Decree target",
          _montague_flow_reliability_weekly)
_register("montague_flow_deficit_cvar90_pct", "minimize", 1.5,
          f"CVaR90 of weekly Montague flow deficit, % of "
          f"{MONTAGUE_DECREE_TARGET_MGD:.0f} MGD Decree target [0-100]",
          _montague_flow_deficit_cvar90_pct)
_register("montague_flow_deficit_max_pct", "minimize", 3.0,
          "DIAGNOSTIC: worst-week Montague flow deficit, % of Decree target [0-100]",
          _montague_flow_deficit_max_pct)

# --- Trenton flow Decree (lower-basin / NJ obligation; target = 1938.95 MGD) ---
_register("trenton_flow_reliability_weekly", "maximize", 0.0003,
          f"Frac of weeks Trenton weekly-mean flow >= "
          f"{TRENTON_DECREE_TARGET_MGD:.0f} MGD Decree target",
          _trenton_flow_reliability_weekly)
_register("trenton_flow_deficit_cvar90_pct", "minimize", 0.03,
          "DIAGNOSTIC: CVaR90 of weekly Trenton flow deficit, % of Decree target [0-100]",
          _trenton_flow_deficit_cvar90_pct)

# --- Downstream flood exposure (any of Hale Eddy / Fishs Eddy / Bridgeville) ---
# ACTIVE metric = magnitude-weighted exceedance (flood_objective_diagnostics.md);
# the day counts stay registered as diagnostics.
_register("downstream_flood_exceedance_minor", "minimize", 0.01,
          "Mean annual ft-days above NWS minor flood stage at the "
          "worst-affected tail gauge (flood exceedance) [ft-days/yr]",
          _downstream_flood_exceedance_minor)
_register("downstream_flood_days_minor", "minimize", 0.02,
          "DIAGNOSTIC: mean annual days any tail gauge >= NWS minor flood "
          "stage [days/yr]",
          _downstream_flood_days_minor)
_register("downstream_flood_days_major", "minimize", 0.03,
          "DIAGNOSTIC: mean annual days any tail gauge >= NWS major flood "
          "stage (severe) [days/yr]",
          _downstream_flood_days_major)
_register("downstream_flood_days_action", "minimize", 0.03,
          "DIAGNOSTIC: mean annual days any tail gauge >= FFMP L1 action "
          "stage [days/yr]",
          _downstream_flood_days_action)

# --- NYC storage resilience ---
_register("nyc_storage_p5_pct", "maximize", 1.5,
          "5th-percentile combined NYC storage, % of total capacity [0-100]",
          _nyc_storage_p5_pct)
_register("nyc_storage_min_pct", "maximize", 1.0,
          "DIAGNOSTIC: minimum combined NYC storage, % of total capacity [0-100]",
          _nyc_storage_min_pct)

# --- Salt-front intrusion (LSTM) — DIAGNOSTIC only ---
_register("salt_front_intrusion_max_rm", "minimize", 0.5,
          "DIAGNOSTIC: max (most-upstream) salt-front river mile over sim "
          f"(DRBC reference RM {SALT_FRONT_REFERENCE_RM})",
          _salt_front_intrusion_max_rm)

# --- Lordville thermal (LSTM) — DEFERRED (inputs unavailable for synthetic scenarios) ---
_register("lordville_temp_exceedance_days", "minimize", 2.0,
          f"DEFERRED: days max water temp at Lordville > "
          f"{LORDVILLE_THERMAL_THRESHOLD_C} °C",
          _lordville_temp_exceedance_days)


###############################################################################
# Assembler
###############################################################################

def build_objective_set(items) -> ObjectiveSet:
    """Assemble an ObjectiveSet from a list of names and/or Objective instances.

    Items may be:
      - str:       look up in OBJECTIVES registry
      - Objective: use directly (for custom/ad-hoc metrics)

    Example:
        obj_set = build_objective_set([
            "nyc_delivery_reliability_weekly",
            "montague_flow_reliability_weekly",
            Objective("my_custom", "maximize", 0.01, "...", my_func),
        ])

    Args:
        items: Iterable of str | Objective.

    Returns:
        ObjectiveSet containing the resolved objectives in the given order.

    Raises:
        KeyError: If a string name is not in OBJECTIVES.
        TypeError: If an item is neither a string nor an Objective.
    """
    resolved = []
    for item in items:
        if isinstance(item, Objective):
            resolved.append(item)
        elif isinstance(item, str):
            if item not in OBJECTIVES:
                raise KeyError(
                    f"Unknown objective '{item}'. "
                    f"Available: {sorted(OBJECTIVES)}"
                )
            resolved.append(OBJECTIVES[item])
        else:
            raise TypeError(
                f"build_objective_set items must be str or Objective; "
                f"got {type(item).__name__}"
            )
    return ObjectiveSet(resolved)


def list_available_objectives() -> str:
    """Return a formatted table of all registered objectives."""
    lines = [f"Available objectives ({len(OBJECTIVES)}):"]
    for name, obj in OBJECTIVES.items():
        lines.append(f"  {name}  [{obj.direction}, eps={obj.epsilon}]  — {obj.description}")
    return "\n".join(lines)
