"""
objectives.py - Objective function framework for NYC reservoir optimization.

Provides an Objective class, ObjectiveSet container, a **name-indexed
registry** of pre-built metric instances, and a `build_objective_set()`
assembler. Users select objectives by listing names (strings) or passing
custom `Objective` instances directly.

Design principles:

- **Mirrored 1954-Decree pairs for NYC and Montague.** Both quantities
  (NYC 800 MGD diversion right; Montague 1750 cfs = 1131.05 MGD flow
  target) are scored against their *static* 1954 Decree values via a
  matching pair of metrics: a weekly reliability frequency and a max
  weekly deviation magnitude (% of Decree value). Static targets, not
  the time-varying live FFMP `mrf_target`, eliminate a gaming pathway
  where a policy could "succeed" by triggering drought step-downs that
  lower its own goalpost.

- **Salt-front intrusion** as the worst-case (max-upstream) river-mile
  position over the simulation. Worst-case framing rather than excursion-
  past-threshold preserves the Pareto gradient where most policies do
  not breach the DRBC standard.

- **Operations-attributable flood signal** as days at or above FFMP L1
  action stage on any of the three reservoir-tail gauges (Hale Eddy,
  Fishs Eddy, Bridgeville). Locates the metric on tunable downstream
  releases rather than on mainstem-storm flow at Montague.

- **Storage resilience** as the minimum combined NYC storage % of
  capacity over the post-warmup period.

Usage:
    from src.objectives import build_objective_set
    obj_set = build_objective_set([
        "nyc_reliability_weekly_decree",
        "nyc_max_deficit_weekly_decree",
        "montague_reliability_weekly_decree",
        "montague_max_deficit_weekly_decree",
        "salt_front_max_rm",
        "flood_days_downstream_action_anygauge",
        "storage_min_combined_pct",
    ])
    values = obj_set.compute(data)
    borg_values = obj_set.compute_for_borg(data)
"""

import numpy as np
import pandas as pd

from pywrdrb.flood_thresholds import flood_stage_thresholds

from config import (
    NYC_RESERVOIRS,
    NYC_TOTAL_CAPACITY,
    WARMUP_DAYS,
    NYC_DECREE_DIVERSION_CAP_MGD,
    MONTAGUE_DECREE_TARGET_MGD,
    LORDVILLE_THERMAL_THRESHOLD_C,
    SALT_FRONT_REFERENCE_RM,
)


# Reservoir-tail USGS gauges: Hale Eddy below Cannonsville, Fishs Eddy below
# Pepacton, Bridgeville below Neversink. Used by the action-stage flood metric.
_DOWNSTREAM_GAUGES = ["01426500", "01421000", "01436690"]


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
        ``EnsembleObjective`` from ``src.objectives_ensemble``. This is
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
# Metric Functions — NYC 1954 Decree pair
###############################################################################
# NYC's 1954 Decree right is up to 800 MGD. The reliability/max-deficit pair
# scores delivery against `min(demand, 800)` — the effective Decree right per
# day. Capping demand at 800 ensures voluntary winter low-takes (when demand
# < 800) are not penalized; only forced shortfalls below the Decree right
# count. Both metrics normalize to the *static* 800 MGD denominator so a
# 50-MGD shortfall reads identically in summer and winter.


def _nyc_reliability_weekly_decree(data: dict) -> float:
    """Fraction of weeks NYC delivery meets its 1954 Decree right.

    Demand is capped at the 800 MGD Decree cap before comparison. A week
    is "met" if weekly total delivery >= 99% of weekly total capped demand.
    Range [0, 1].
    """
    demand = (
        data["ibt_demands"]["demand_nyc"]
        .iloc[WARMUP_DAYS:]
        .clip(upper=NYC_DECREE_DIVERSION_CAP_MGD)
    )
    delivery = data["ibt_diversions"]["delivery_nyc"].iloc[WARMUP_DAYS:]
    weekly_demand = demand.resample("W").sum()
    weekly_delivery = delivery.resample("W").sum()
    met = (weekly_delivery >= 0.99 * weekly_demand).sum()
    total = len(weekly_demand)
    return float(met) / total if total > 0 else 0.0


def _nyc_max_deficit_weekly_decree(data: dict) -> float:
    """Worst single-week NYC delivery deficit as % of 1954 Decree cap.

    Per-week deficit is `max(0, weekly_mean_capped_demand - weekly_mean_delivery)`,
    normalized to the static 800 MGD Decree value. Returns the max over
    post-warmup weeks. Range [0, 100] pp.
    """
    demand = (
        data["ibt_demands"]["demand_nyc"]
        .iloc[WARMUP_DAYS:]
        .clip(upper=NYC_DECREE_DIVERSION_CAP_MGD)
    )
    delivery = data["ibt_diversions"]["delivery_nyc"].iloc[WARMUP_DAYS:]
    weekly_demand = demand.resample("W").mean()
    weekly_delivery = delivery.resample("W").mean()
    deficit = (weekly_demand - weekly_delivery).clip(lower=0)
    deficit_pct = 100.0 * deficit / NYC_DECREE_DIVERSION_CAP_MGD
    return float(deficit_pct.max()) if len(deficit_pct) > 0 else 0.0


###############################################################################
# Metric Functions — Montague 1954 Decree pair
###############################################################################
# Montague's 1954 Decree target is 1750 cfs = 1131.05 MGD. The reliability/
# max-deficit pair scores weekly-mean Montague flow against the static
# Decree value. Mirror of the NYC pair: same weekly aggregation, same
# % normalization, same units (pp) for max deficit.


def _montague_reliability_weekly_decree(data: dict) -> float:
    """Fraction of weeks weekly-mean Montague flow >= 1954 Decree target.

    Range [0, 1]. Will not saturate at 1.0 because under any FFMP drought
    step-down (L2-L5) releases drop below 1131.05 MGD by design.
    """
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    weekly_flow = flow.resample("W").mean()
    met = (weekly_flow >= MONTAGUE_DECREE_TARGET_MGD).sum()
    total = len(weekly_flow)
    return float(met) / total if total > 0 else 0.0


def _montague_max_deficit_weekly_decree(data: dict) -> float:
    """Worst single-week Montague flow deficit as % of 1954 Decree target.

    Per-week deficit is `max(0, decree_target - weekly_mean_flow)`,
    normalized to the static 1131.05 MGD Decree value. Returns the max
    over post-warmup weeks. Range [0, 100] pp.
    """
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    weekly_flow = flow.resample("W").mean()
    deficit = (MONTAGUE_DECREE_TARGET_MGD - weekly_flow).clip(lower=0)
    deficit_pct = 100.0 * deficit / MONTAGUE_DECREE_TARGET_MGD
    return float(deficit_pct.max()) if len(deficit_pct) > 0 else 0.0


###############################################################################
# Metric Functions — Salt-front intrusion (LSTM)
###############################################################################


def _salt_front_max_rm(data: dict) -> float:
    """Maximum (most-upstream) salt-front position over the sim, in RM.

    Delaware River miles increase upstream from the bay mouth, so a
    HIGHER river-mile value means the salt front intruded farther
    upstream — worse for water supply at Trenton. The objective is the
    single worst (largest-RM) day after the warmup window. NaN entries
    (e.g. the gate-skipped first sim day) are dropped before computing
    the max.
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
# Metric Functions — Flood risk below NYC reservoirs
###############################################################################


def _flood_days_downstream_action_anygauge(data: dict) -> float:
    """Days at or above FFMP L1 action stage on any reservoir-tail gauge.

    A day counts if **any** of Hale Eddy (below Cannonsville), Fishs Eddy
    (below Pepacton), or Bridgeville (below Neversink) is at or above its
    action stage threshold. Action stage is the FFMP L1 release cutoff,
    operationally meaningful and tunable by NYC release decisions
    (unlike Montague mainstem flow which is dominated by exogenous
    storms). Returns count of post-warmup days. Range [0, n_days].
    """
    stage = data["flood_stage"][_DOWNSTREAM_GAUGES].iloc[WARMUP_DAYS:]
    thresh = pd.Series(
        {g: flood_stage_thresholds[g]["action"] for g in _DOWNSTREAM_GAUGES}
    )
    over = stage.ge(thresh, axis=1)
    return float(over.any(axis=1).sum())


###############################################################################
# Metric Functions — Storage resilience
###############################################################################


def _min_storage_pct(data: dict) -> float:
    """Minimum combined NYC storage as percentage of total capacity. [0, 100]."""
    storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1).iloc[WARMUP_DAYS:]
    return 100.0 * float(storage.min()) / NYC_TOTAL_CAPACITY


###############################################################################
# Metric Functions — Lordville thermal (INACTIVE)
###############################################################################
# Inputs require multivariate meteorology not available for stochastic
# re-eval scenarios. Kept here so the metric is one config flag away from
# re-enable. See local_notes/decisions/2026-04-29_temperature_lstm_deferred.md.


def _lordville_thermal_exceedance_days(data: dict) -> float:
    """Number of days max water temp at Lordville exceeds threshold (°C).

    Reads from data["temperature"]["temperature_after_thermal_release_mu"].
    NaN entries (pre-LSTM-start) are dropped before counting.
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
# Single source of truth for all available objective metrics. Users select
# objectives by listing names from this registry (or by constructing their
# own `Objective` instances and passing them directly to build_objective_set).

OBJECTIVES: dict[str, Objective] = {}


def _register(name, direction, epsilon, description, func):
    OBJECTIVES[name] = Objective(
        name=name, direction=direction, epsilon=epsilon,
        description=description, func=func,
    )


# --- NYC 1954 Decree pair (right = 800 MGD) ---
_register("nyc_reliability_weekly_decree", "maximize", 0.01,
          "Fraction of weeks NYC delivery >= 99% of capped demand "
          f"({NYC_DECREE_DIVERSION_CAP_MGD:.0f} MGD Decree cap)",
          _nyc_reliability_weekly_decree)
_register("nyc_max_deficit_weekly_decree", "minimize", 1.0,
          "Worst-week NYC delivery deficit as pct of "
          f"{NYC_DECREE_DIVERSION_CAP_MGD:.0f} MGD Decree cap [0-100]",
          _nyc_max_deficit_weekly_decree)

# --- Montague 1954 Decree pair (target = 1750 cfs = 1131.05 MGD) ---
_register("montague_reliability_weekly_decree", "maximize", 0.01,
          "Fraction of weeks Montague weekly-mean flow >= "
          f"{MONTAGUE_DECREE_TARGET_MGD:.0f} MGD Decree target",
          _montague_reliability_weekly_decree)
_register("montague_max_deficit_weekly_decree", "minimize", 1.0,
          "Worst-week Montague flow deficit as pct of "
          f"{MONTAGUE_DECREE_TARGET_MGD:.0f} MGD Decree target [0-100]",
          _montague_max_deficit_weekly_decree)

# --- Salt-front intrusion (LSTM, active when INCLUDE_SALINITY_MODEL=True) ---
_register("salt_front_max_rm", "minimize", 0.5,
          "Max (most-upstream) salt-front river mile reached over sim "
          f"(DRBC reference: RM {SALT_FRONT_REFERENCE_RM})",
          _salt_front_max_rm)

# --- Flood risk below NYC reservoirs (action-stage at any tail gauge) ---
_register("flood_days_downstream_action_anygauge", "minimize", 3.0,
          "Days any of Hale Eddy/Fishs Eddy/Bridgeville >= FFMP L1 action stage",
          _flood_days_downstream_action_anygauge)

# --- Storage resilience ---
_register("storage_min_combined_pct", "maximize", 0.5,
          "Minimum combined NYC storage as pct of total capacity [0-100]",
          _min_storage_pct)

# --- Temperature (LSTM) — INACTIVE; see decision doc ---
_register("lordville_thermal_exceedance_days", "minimize", 2.0,
          f"Days max water temp at Lordville > "
          f"{LORDVILLE_THERMAL_THRESHOLD_C} °C",
          _lordville_thermal_exceedance_days)


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
            "nyc_reliability_weekly_decree",
            "montague_reliability_weekly_decree",
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
