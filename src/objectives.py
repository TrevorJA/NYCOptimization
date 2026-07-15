"""
objectives.py - Objective function framework for NYC reservoir optimization.

Single source of truth for every objective METRIC. Provides an `Objective`
class, an `ObjectiveSet` container, a name-indexed `OBJECTIVES` registry of
pre-built metric instances, and a `build_objective_set()` assembler. Runs
select objectives by listing registry names (strings); swapping the objective
set is therefore a config edit (`config.ACTIVE_OBJECTIVES` /
`NYCOPT_OBJECTIVES`), never a code change.

Naming convention
-----------------
Every registered name is `{location}_{quantity}_{statistic}[_{unit}]` so the
name alone states *what is measured* and *how it is aggregated over time*:
    nyc_delivery_reliability_weekly   -> NYC delivery, weekly reliability frequency
    nyc_delivery_deficit_cvar90_pct   -> NYC delivery, CVaR90 of weekly deficit, in %
    montague_flow_deficit_max_pct     -> Montague flow, worst-week deficit, in %
    downstream_flood_days_minor       -> tail-gauge flooding, days >= NWS minor stage
    nyc_storage_p5_pct                -> NYC storage, 5th-percentile, in % of capacity

Temporal-aggregation design
----------------------------
- **Reliability** (Hashimoto reliability): fraction of weeks a Decree threshold
  is met. Stable, fast-converging satisficing frequency (Herman et al. 2015;
  Bonham et al. 2024). Used for NYC/NJ delivery and Montague/Trenton flow.
- **Deficit CVaR90** (recommended) vs **deficit max** (diagnostic): the worst-
  week maximum is a high-variance, low-information signal (Quinn et al. 2017);
  CVaR90 — the mean of the worst 10% of weekly deficits — keeps the tail-risk
  focus but is far more reproducible across realizations (Rockafellar & Uryasev
  2000; Fairbrother et al. 2022). The active set uses CVaR90; the max variants
  remain registered as diagnostics.
- **Flood days**: count of days any reservoir-tail gauge is at/above a named NWS
  stage. Count-over-threshold avoids the expectation-of-damage trap (Quinn et al.
  2017). Active objective uses the `minor` (NWS flood-onset) stage; `major` and
  `action` variants are registered for swapping.
- **Storage p5** (recommended) vs **storage min** (diagnostic): a low percentile
  is a stable vulnerability proxy; the single-day minimum is dominated by one
  drought event (Quinn et al. 2017).

Decree goalposts are the *static* 1954-Decree quantities (NYC 800 MGD; Montague
1131.05 MGD = 1750 cfs; Trenton 1938.95 MGD), never the time-varying live FFMP
`mrf_target` — scoring against the live target would let a policy "succeed" by
triggering drought step-downs that lower its own goalpost.

NYC (and NJ) delivery is a running-*average* right, not a daily ceiling: pywr-drb's
`FfmpNycRunningAvgParameter` / `FfmpNjRunningAvgParameter` let daily diversion
exceed the flat baseline by drawing down banked allowance, so long as the running
average stays within the right. The delivery metrics therefore do NOT clip daily
demand; they score against the reconstructed running-average entitlement
`min(demand, allowance)` (`_delivery_entitlement`), where the allowance bank is
accrued at the *static* Decree baseline — never the policy's drought-scaled
allowance — so demand spikes within the banked right are honored and a policy
cannot lower its own goalpost via drought step-downs.

Diagnostic-only metrics (registered, not in the default active set): the
worst-case variants above, NJ delivery reliability (optional 8th objective,
pending the redundancy screen), salt-front intrusion (replaced by the Trenton
flow objective — physically redundant, and the LSTM is unreliable in extreme
drought), and the deferred Lordville thermal metric.

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
    WARMUP_DAYS,
    NYC_DECREE_DIVERSION_CAP_MGD,
    NJ_DELIVERY_CAP_MGD,
    MONTAGUE_DECREE_TARGET_MGD,
    TRENTON_DECREE_TARGET_MGD,
    LORDVILLE_THERMAL_THRESHOLD_C,
    SALT_FRONT_REFERENCE_RM,
)


# Reservoir-tail USGS gauges, used by the downstream flood metrics:
#   01426500 Hale Eddy   (below Cannonsville)
#   01421000 Fishs Eddy  (below Pepacton)
#   01436690 Bridgeville (below Neversink)
# These respond to NYC release decisions, unlike Montague mainstem flow which is
# dominated by exogenous storms — so flooding here is operations-attributable.
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
# These factor the common reductions so the registered metric functions stay
# one-liners and so the CVaR vs. max (and p5 vs. min) variants are guaranteed to
# operate on identical underlying series.
#
# The `_weekly_*` / `_flood_over_*` / `_nyc_storage_pct_daily` cores operate on
# ALREADY-WINDOWED daily series (no warm-up handling inside): the §1 metrics
# below apply `_post_warmup` before calling them, and the annual-unit ensemble
# metrics in `src.objectives_ensemble` apply water-year unit slicing instead —
# guaranteeing the two paths share one weekly-accounting formula.


def _post_warmup(obj):
    """Drop the first WARMUP_DAYS daily steps (model spin-up).

    Args:
        obj: Daily-indexed pandas Series or DataFrame.

    Returns:
        The same type, with the first WARMUP_DAYS rows removed.
    """
    return obj.iloc[WARMUP_DAYS:]


def _cvar_worst_mean(values, frac: float = _CVAR_TAIL_FRAC) -> float:
    """Mean of the worst (largest) ``frac`` fraction of finite values.

    For a deficit/severity series where larger = worse this is the Conditional
    Value-at-Risk at level ``(1 - frac)`` — the mean of the worst
    ``ceil(frac * N)`` weekly values. Tail-averaging (rather than taking the
    single maximum) gives a lower-variance, more reproducible signal across
    realizations (Quinn et al. 2017; Rockafellar & Uryasev 2000). Non-finite
    entries are dropped; an empty series returns 0.0.
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

    Mirrors pywr-drb's running-average delivery parameters
    (``FfmpNycRunningAvgParameter`` / ``FfmpNjRunningAvgParameter``): the
    allowance starts at ``cap``, accrues ``cap - delivery`` each day, and resets
    to ``cap`` at the start of each budget period. It is accrued at the *static*
    Decree/baseline ``cap`` — never the model's drought-scaled allowance — so the
    deficit goalpost cannot be lowered by a policy's own drought step-downs (the
    same anti-gaming rule the flow-Decree goalposts use). The returned series is
    the maximum diversion the running-average right permits on each day, so a
    demand spike above the flat daily baseline is legitimate whenever prior
    under-use has banked the allowance for it.

    Implementation is the exact day-by-day recursion the model uses
    (``budget[t] = max(0, budget[t-1] + cap - delivery[t-1])``, or ``cap`` on a
    reset day). The zero-floor essentially never binds for valid NYC output —
    the static-``cap`` accrual is never slower than the model's own
    (drought-scaled) accrual from the shared May-31 reset, so delivery never
    exceeds the reconstructed bank — but it is applied exactly rather than
    assumed away (the NJ path does not mirror the model's drought-factor resets).
    The per-day loop is negligible beside the Pywr-DRB simulation that produced
    ``delivery``.

    Args:
        delivery: Daily delivery series (MGD) over the FULL realization window
            (the bank is path-dependent, so it must be built before any warm-up
            or water-year slicing).
        cap: Static running-average allowance (MGD); the daily accrual rate.
        reset: Budget-period reset — ``"annual"`` (NYC: the model resets on
            May 31, so the allowance is ``cap`` again on Jun 1) or ``"monthly"``
            (NJ: reset on the 1st of each month). The NJ drought-factor reset and
            its separate daily cap are intentionally not modeled — this is the
            static-right entitlement, and NJ is a diagnostic objective.

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

    Demand is NOT clipped at a flat daily ``cap``: the Decree limits the running
    *average* diversion, so a day's entitlement is the smaller of what was
    demanded and the running-average allowance banked to that day
    (:func:`_running_avg_budget`). Voluntary low-take days keep entitlement =
    demand (no penalty); demand spikes are honored up to the banked allowance;
    demand beyond the banked right is not counted as owed.

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

    ``target`` is the daily realizable entitlement (:func:`_delivery_entitlement`)
    — min(demand, running-average allowance) — so only shortfalls below the
    running-average Decree right count and demand spikes above the banked
    allowance do not. Normalized to the *static* ``cap`` so a fixed shortfall
    reads identically year-round.

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
    """Post-warmup weekly NYC delivery deficit, as % of the 800 MGD Decree cap."""
    delivery = data["ibt_diversions"]["delivery_nyc"]
    target = _delivery_entitlement(
        data["ibt_demands"]["demand_nyc"], delivery,
        NYC_DECREE_DIVERSION_CAP_MGD, reset="annual",
    )
    return _weekly_delivery_deficit_pct(
        _post_warmup(target), _post_warmup(delivery),
        NYC_DECREE_DIVERSION_CAP_MGD,
    )


def _weekly_flow_deficit_pct(flow: pd.Series, target: float) -> pd.Series:
    """Weekly flow deficit as % of a static Decree flow target (windowed series)."""
    weekly_flow = flow.resample("W").mean()
    deficit = (target - weekly_flow).clip(lower=0)
    return 100.0 * deficit / target


def _weekly_flow_ok(flow: pd.Series, target: float) -> pd.Series:
    """Weekly success indicators: weekly-mean flow >= a static Decree target.

    Operates on an already-windowed daily flow series. A week with a non-finite
    weekly mean compares False (a degenerate week is a failure week).
    """
    return flow.resample("W").mean() >= target


def _flow_reliability_weekly(flow: pd.Series, target: float) -> float:
    """Fraction of post-warmup weeks weekly-mean flow meets a Decree target."""
    ok = _weekly_flow_ok(_post_warmup(flow), target)
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
    """Fraction of post-warmup weeks weekly delivery >= 99% of the entitlement.

    The entitlement is the running-average Decree right
    (:func:`_delivery_entitlement`), reconstructed on the full series before
    warm-up is dropped so the allowance bank carries the correct initial state.
    """
    target = _delivery_entitlement(demand, delivery, cap, reset)
    ok = _weekly_delivery_ok(_post_warmup(target), _post_warmup(delivery))
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
    """Count of post-warmup days any tail gauge is at/above the named NWS stage."""
    stage = _post_warmup(data["flood_stage"][_DOWNSTREAM_GAUGES])
    return float(_flood_over_stage_daily(stage, level).sum())


def _nyc_storage_pct_daily(data: dict) -> pd.Series:
    """Daily combined NYC storage as % of total system capacity (full window)."""
    storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1)
    return 100.0 * storage / NYC_TOTAL_CAPACITY


def _nyc_combined_storage_pct(data: dict) -> pd.Series:
    """Post-warmup daily combined NYC storage as % of total system capacity."""
    return _post_warmup(_nyc_storage_pct_daily(data))


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
    """CVaR90 of weekly NYC delivery deficit, as % of the 800 MGD Decree cap. [0, 100].

    Mean of the worst 10% of weekly deficits — the stable tail-risk replacement
    for the single worst-week maximum.
    """
    return _cvar_worst_mean(_nyc_weekly_delivery_deficit_pct(data).values)


def _nyc_delivery_deficit_max_pct(data: dict) -> float:
    """DIAGNOSTIC: worst single-week NYC delivery deficit, % of Decree cap. [0, 100]."""
    s = _nyc_weekly_delivery_deficit_pct(data)
    return float(s.max()) if len(s) > 0 else 0.0


###############################################################################
# Metric Functions — New Jersey water supply (D&R Canal diversion)
###############################################################################


def _nj_delivery_reliability_weekly(data: dict) -> float:
    """Fraction of weeks NJ diversion meets >= 99% of its capped right. [0, 1].

    Optional second NJ stakeholder axis (the explicit NJ-supply leg). Included
    in the active set only if the redundancy screen shows it is not collinear
    with the Trenton flow objective.
    """
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
    """CVaR90 of weekly Montague flow deficit, % of Decree target. [0, 100].

    Montague flow is storm-dominated, so its single worst week is largely
    exogenous noise — CVaR90 is especially preferable to the maximum here.
    """
    return _cvar_worst_mean(
        _weekly_flow_deficit_pct(
            _post_warmup(data["major_flow"]["delMontague"]),
            MONTAGUE_DECREE_TARGET_MGD,
        ).values
    )


def _montague_flow_deficit_max_pct(data: dict) -> float:
    """DIAGNOSTIC: worst single-week Montague flow deficit, % of Decree target. [0, 100]."""
    s = _weekly_flow_deficit_pct(
        _post_warmup(data["major_flow"]["delMontague"]), MONTAGUE_DECREE_TARGET_MGD,
    )
    return float(s.max()) if len(s) > 0 else 0.0


###############################################################################
# Metric Functions — Trenton flow Decree (target = 1938.95 MGD)
###############################################################################
# Lower-basin / NJ flow obligation. Replaces the salt-front objective: the
# Trenton target exists largely to repel salt intrusion (so the two are
# physically redundant), and Trenton flow is a clean hydrologic signal whereas
# the salt-front LSTM is unreliable in extreme drought.


def _trenton_flow_reliability_weekly(data: dict) -> float:
    """Fraction of weeks weekly-mean Trenton flow >= 1938.95 MGD Decree target. [0, 1]."""
    return _flow_reliability_weekly(data["major_flow"]["delTrenton"], TRENTON_DECREE_TARGET_MGD)


def _trenton_flow_deficit_cvar90_pct(data: dict) -> float:
    """DIAGNOSTIC: CVaR90 of weekly Trenton flow deficit, % of Decree target. [0, 100]."""
    return _cvar_worst_mean(
        _weekly_flow_deficit_pct(
            _post_warmup(data["major_flow"]["delTrenton"]),
            TRENTON_DECREE_TARGET_MGD,
        ).values
    )


###############################################################################
# Metric Functions — Downstream flood exposure (reservoir-tail gauges)
###############################################################################


def _downstream_flood_days_minor(data: dict) -> float:
    """Days any tail gauge >= NWS minor flood stage (flood onset). [0, n_days]."""
    return _flood_days_anygauge(data, "minor")


def _downstream_flood_days_major(data: dict) -> float:
    """DIAGNOSTIC: days any tail gauge >= NWS major flood stage (severe). [0, n_days]."""
    return _flood_days_anygauge(data, "major")


def _downstream_flood_days_action(data: dict) -> float:
    """DIAGNOSTIC: days any tail gauge >= FFMP L1 action stage. [0, n_days]."""
    return _flood_days_anygauge(data, "action")


###############################################################################
# Metric Functions — NYC storage resilience
###############################################################################


def _nyc_storage_p5_pct(data: dict) -> float:
    """5th percentile of daily combined NYC storage, % of capacity. [0, 100].

    Stable low-percentile vulnerability proxy: "how depleted does the system
    routinely get", without the single-day minimum's dependence on one event.
    """
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
# Retained for re-evaluation diagnostics. Superseded as a search objective by
# the Trenton flow Decree metric (physically redundant; LSTM unreliable in
# extreme drought). Active only when INCLUDE_SALINITY_MODEL=True.


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
    sf = sf.iloc[WARMUP_DAYS:].dropna()
    if sf.empty:
        return float("nan")
    return float(sf.max())


###############################################################################
# Metric Functions — Lordville thermal (LSTM) — DEFERRED
###############################################################################
# Inputs require multivariate meteorology not available for stochastic re-eval
# scenarios. Kept registered so the metric is one config flag from re-enable.
# See local_notes/decisions/2026-04-29_temperature_lstm_deferred.md.


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
    temp = temp.iloc[WARMUP_DAYS:].dropna()
    return float((temp > LORDVILLE_THERMAL_THRESHOLD_C).sum())


###############################################################################
# Objective Registry
###############################################################################
# Single source of truth for all available objective metrics. The default
# active subset is config.ACTIVE_OBJECTIVES; everything else is a registered
# diagnostic available for swapping by name (no code change required).

OBJECTIVES: dict[str, Objective] = {}


# Epsilons calibrated to the signal scale (Reed et al. 2013): epsilon ~ IQR/10
# of each objective's spread across N=500 random DV policies on the historic
# reference trace (objective-sensitivity diagnostic, seed 42, 2026-06-17),
# rounded to clean steps. salt_front (no gradient), downstream_flood_days_major
# (binary on this trace), and lordville_temp (LSTM off -> all NaN) had no usable
# IQR and keep prior placeholders; revisit under the ensemble experiment.
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

# --- New Jersey water supply (D&R Canal diversion; optional 8th objective) ---
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
_register("downstream_flood_days_minor", "minimize", 1.0,
          "Days any tail gauge >= NWS minor flood stage (flood onset)",
          _downstream_flood_days_minor)
_register("downstream_flood_days_major", "minimize", 2.0,
          "DIAGNOSTIC: days any tail gauge >= NWS major flood stage (severe)",
          _downstream_flood_days_major)
_register("downstream_flood_days_action", "minimize", 2.0,
          "DIAGNOSTIC: days any tail gauge >= FFMP L1 action stage",
          _downstream_flood_days_action)

# --- NYC storage resilience ---
_register("nyc_storage_p5_pct", "maximize", 1.5,
          "5th-percentile combined NYC storage, % of total capacity [0-100]",
          _nyc_storage_p5_pct)
_register("nyc_storage_min_pct", "maximize", 1.0,
          "DIAGNOSTIC: minimum combined NYC storage, % of total capacity [0-100]",
          _nyc_storage_min_pct)

# --- Salt-front intrusion (LSTM) — DIAGNOSTIC; superseded by Trenton flow ---
_register("salt_front_intrusion_max_rm", "minimize", 0.5,
          "DIAGNOSTIC: max (most-upstream) salt-front river mile over sim "
          f"(DRBC reference RM {SALT_FRONT_REFERENCE_RM})",
          _salt_front_intrusion_max_rm)

# --- Lordville thermal (LSTM) — DEFERRED; see decision doc ---
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
