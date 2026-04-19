"""
objectives.py - Objective function framework for NYC reservoir optimization.

Provides an Objective class, ObjectiveSet container, a **name-indexed
registry** of pre-built metric instances, and a `build_objective_set()`
assembler. Users select objectives by listing names (strings) or passing
custom `Objective` instances directly — no pre-defined "sets".

Metric design principles:
- Reliability metrics use WEEKLY aggregation (daily is insensitive;
  shortages are rare events that daily frequency fails to discriminate).
- Flow compliance at Montague/Trenton is available in two flavors:
    * `_fixed`    — against constant FFMP baseline targets
                    (Montague 1,131 MGD, Trenton 1,939 MGD)
    * `_dynamic`  — against the simulation's time-varying mrf_target
  Fixed-target metrics are preferred for cross-architecture comparison
  because all architectures see the same target; dynamic-target metrics
  reflect live FFMP step-down logic and are useful for baseline runs only.
- Vulnerability metrics capture the worst-case shortage magnitude.

Usage:
    from src.objectives import build_objective_set
    obj_set = build_objective_set([
        "nyc_reliability_weekly",
        "nyc_vulnerability",
        "montague_reliability_weekly_fixed",
        "trenton_reliability_weekly_fixed",
        ...
    ])
    values = obj_set.compute(data)
    borg_values = obj_set.compute_for_borg(data)
"""

import numpy as np
import pandas as pd

from config import (
    NYC_RESERVOIRS,
    NYC_TOTAL_CAPACITY,
    WARMUP_DAYS,
    MONTAGUE_FIXED_TARGET_MGD,
    TRENTON_FIXED_TARGET_MGD,
)


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

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [f"ObjectiveSet ({self.n_objs} objectives):"]
        for obj in self._objectives:
            lines.append(
                f"  {obj.name}: {obj.direction} (eps={obj.epsilon}) — {obj.description}"
            )
        return "\n".join(lines)


###############################################################################
# Metric Functions
###############################################################################

# --- NYC Supply ---

def _nyc_reliability_weekly(data: dict) -> float:
    """Fraction of weeks NYC delivery meets or exceeds demand.

    A week is "met" if weekly total delivery >= 99% of weekly total demand.
    Weekly aggregation provides better discrimination than daily for
    rare shortage events.
    """
    demand = data["ibt_demands"]["demand_nyc"].iloc[WARMUP_DAYS:]
    delivery = data["ibt_diversions"]["delivery_nyc"].iloc[WARMUP_DAYS:]

    weekly_demand = demand.resample("W").sum()
    weekly_delivery = delivery.resample("W").sum()

    met = (weekly_delivery >= 0.99 * weekly_demand).sum()
    total = len(weekly_demand)
    return float(met) / total if total > 0 else 0.0


def _nyc_vulnerability(data: dict) -> float:
    """Worst single-week NYC shortage as percent of that week's demand.

    Captures the worst-case shortage magnitude. Returns value in [0, 100].
    """
    demand = data["ibt_demands"]["demand_nyc"].iloc[WARMUP_DAYS:]
    delivery = data["ibt_diversions"]["delivery_nyc"].iloc[WARMUP_DAYS:]

    weekly_demand = demand.resample("W").sum()
    weekly_delivery = delivery.resample("W").sum()
    weekly_shortage = (weekly_demand - weekly_delivery).clip(lower=0)

    deficit_pct = np.where(
        weekly_demand > 0,
        100.0 * weekly_shortage / weekly_demand,
        0.0,
    )
    return float(np.max(deficit_pct)) if len(deficit_pct) > 0 else 0.0


# --- Flow Compliance ------------------------------------------------------
# Each downstream gauge has two flavors:
#   _dynamic : target is the simulation's live mrf_target (FFMP step-down)
#   _fixed   : target is a constant (FFMP baseline MGD)
# Fixed-target metrics are recommended for cross-architecture comparison
# because the target does not vary with the policy being evaluated.


def _weekly_reliability_vs_target(flow: pd.Series, target) -> float:
    weekly_flow = flow.resample("W").mean()
    if hasattr(target, "resample"):
        weekly_target = target.resample("W").mean()
    else:
        weekly_target = target  # scalar — broadcasts
    met = (weekly_flow >= weekly_target).sum()
    total = len(weekly_flow)
    return float(met) / total if total > 0 else 0.0


def _weekly_vulnerability_vs_target(flow: pd.Series, target) -> float:
    weekly_flow = flow.resample("W").mean()
    if hasattr(target, "resample"):
        weekly_target = target.resample("W").mean()
    else:
        weekly_target = pd.Series(target, index=weekly_flow.index)
    deficit = (weekly_target - weekly_flow).clip(lower=0)
    denom = np.where(weekly_target > 0, weekly_target, np.nan)
    deficit_pct = 100.0 * deficit / denom
    deficit_pct = np.nan_to_num(deficit_pct, nan=0.0)
    return float(np.max(deficit_pct)) if len(deficit_pct) > 0 else 0.0


def _montague_reliability_weekly_dynamic(data: dict) -> float:
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    target = data["mrf_target"]["delMontague"].iloc[WARMUP_DAYS:]
    return _weekly_reliability_vs_target(flow, target)


def _montague_vulnerability_dynamic(data: dict) -> float:
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    target = data["mrf_target"]["delMontague"].iloc[WARMUP_DAYS:]
    return _weekly_vulnerability_vs_target(flow, target)


def _montague_reliability_weekly_fixed(data: dict) -> float:
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    return _weekly_reliability_vs_target(flow, MONTAGUE_FIXED_TARGET_MGD)


def _montague_vulnerability_fixed(data: dict) -> float:
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    return _weekly_vulnerability_vs_target(flow, MONTAGUE_FIXED_TARGET_MGD)


def _trenton_reliability_weekly_dynamic(data: dict) -> float:
    flow = data["major_flow"]["delTrenton"].iloc[WARMUP_DAYS:]
    target = data["mrf_target"]["delTrenton"].iloc[WARMUP_DAYS:]
    return _weekly_reliability_vs_target(flow, target)


def _trenton_vulnerability_dynamic(data: dict) -> float:
    flow = data["major_flow"]["delTrenton"].iloc[WARMUP_DAYS:]
    target = data["mrf_target"]["delTrenton"].iloc[WARMUP_DAYS:]
    return _weekly_vulnerability_vs_target(flow, target)


def _trenton_reliability_weekly_fixed(data: dict) -> float:
    flow = data["major_flow"]["delTrenton"].iloc[WARMUP_DAYS:]
    return _weekly_reliability_vs_target(flow, TRENTON_FIXED_TARGET_MGD)


def _trenton_vulnerability_fixed(data: dict) -> float:
    flow = data["major_flow"]["delTrenton"].iloc[WARMUP_DAYS:]
    return _weekly_vulnerability_vs_target(flow, TRENTON_FIXED_TARGET_MGD)


# --- Flood Risk ---

def _flood_days(data: dict) -> float:
    """Number of days aggregate NYC storage exceeds spill risk threshold.

    Uses 95% of total NYC capacity as a proxy for spill risk.
    """
    storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1).iloc[WARMUP_DAYS:]
    flood_threshold = 0.95 * NYC_TOTAL_CAPACITY
    return float((storage >= flood_threshold).sum())


def _flood_days_downstream(data: dict) -> float:
    """Number of days downstream flow at Montague exceeds flood threshold.

    Uses a flow threshold of 25,000 CFS (~16,148 MGD) at Montague.
    """
    FLOOD_FLOW_THRESHOLD_MGD = 16148.0
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    return float((flow >= FLOOD_FLOW_THRESHOLD_MGD).sum())


# --- NJ Supply ---

def _nj_reliability_weekly(data: dict) -> float:
    """Fraction of weeks NJ delivery meets or exceeds demand.

    Mirrors NYC reliability metric. A week is "met" if weekly total
    delivery >= 99% of weekly total demand.
    """
    demand = data["ibt_demands"]["demand_nj"].iloc[WARMUP_DAYS:]
    delivery = data["ibt_diversions"]["delivery_nj"].iloc[WARMUP_DAYS:]

    weekly_demand = demand.resample("W").sum()
    weekly_delivery = delivery.resample("W").sum()

    met = (weekly_delivery >= 0.99 * weekly_demand).sum()
    total = len(weekly_demand)
    return float(met) / total if total > 0 else 0.0


def _nj_vulnerability(data: dict) -> float:
    """Worst single-week NJ shortage as percent of that week's demand.

    Mirrors NYC vulnerability metric. Returns value in [0, 100].
    """
    demand = data["ibt_demands"]["demand_nj"].iloc[WARMUP_DAYS:]
    delivery = data["ibt_diversions"]["delivery_nj"].iloc[WARMUP_DAYS:]

    weekly_demand = demand.resample("W").sum()
    weekly_delivery = delivery.resample("W").sum()
    weekly_shortage = (weekly_demand - weekly_delivery).clip(lower=0)

    deficit_pct = np.where(
        weekly_demand > 0,
        100.0 * weekly_shortage / weekly_demand,
        0.0,
    )
    return float(np.max(deficit_pct)) if len(deficit_pct) > 0 else 0.0


# --- Storage Resilience ---

def _min_storage_pct(data: dict) -> float:
    """Minimum combined NYC storage as percentage of total capacity. [0, 100]."""
    storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1).iloc[WARMUP_DAYS:]
    return 100.0 * float(storage.min()) / NYC_TOTAL_CAPACITY


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


# --- NYC Supply ---
_register("nyc_reliability_weekly", "maximize", 0.01,
          "Fraction of weeks NYC delivery >= 99% of demand",
          _nyc_reliability_weekly)
_register("nyc_vulnerability", "minimize", 1.0,
          "Worst-week NYC shortage as pct of demand [0-100]",
          _nyc_vulnerability)

# --- NJ Supply ---
_register("nj_reliability_weekly", "maximize", 0.01,
          "Fraction of weeks NJ delivery >= 99% of demand",
          _nj_reliability_weekly)
_register("nj_vulnerability", "minimize", 1.0,
          "Worst-week NJ shortage as pct of demand [0-100]",
          _nj_vulnerability)

# --- Flow Compliance (fixed targets — recommended default) ---
_register("montague_reliability_weekly_fixed", "maximize", 0.01,
          "Fraction of weeks Montague flow >= FFMP fixed target "
          f"({MONTAGUE_FIXED_TARGET_MGD:.0f} MGD)",
          _montague_reliability_weekly_fixed)
_register("montague_vulnerability_fixed", "minimize", 1.0,
          "Worst-week Montague deficit as pct of fixed target",
          _montague_vulnerability_fixed)
_register("trenton_reliability_weekly_fixed", "maximize", 0.005,
          "Fraction of weeks Trenton flow >= FFMP fixed target "
          f"({TRENTON_FIXED_TARGET_MGD:.0f} MGD)",
          _trenton_reliability_weekly_fixed)
_register("trenton_vulnerability_fixed", "minimize", 1.0,
          "Worst-week Trenton deficit as pct of fixed target",
          _trenton_vulnerability_fixed)

# --- Flow Compliance (dynamic targets — for baseline diagnostics only) ---
_register("montague_reliability_weekly_dynamic", "maximize", 0.01,
          "Fraction of weeks Montague flow >= time-dynamic MRF target",
          _montague_reliability_weekly_dynamic)
_register("montague_vulnerability_dynamic", "minimize", 1.0,
          "Worst-week Montague deficit as pct of dynamic target",
          _montague_vulnerability_dynamic)
_register("trenton_reliability_weekly_dynamic", "maximize", 0.005,
          "Fraction of weeks Trenton flow >= time-dynamic MRF target",
          _trenton_reliability_weekly_dynamic)
_register("trenton_vulnerability_dynamic", "minimize", 1.0,
          "Worst-week Trenton deficit as pct of dynamic target",
          _trenton_vulnerability_dynamic)

# --- Flood Risk ---
_register("flood_risk_storage_spill_days", "minimize", 10.0,
          "Days aggregate NYC storage > 95% capacity (spill risk proxy)",
          _flood_days)
_register("flood_risk_downstream_flow_days", "minimize", 5.0,
          "Days Montague flow exceeds 25,000 CFS action stage proxy",
          _flood_days_downstream)

# --- Storage Resilience ---
_register("storage_min_combined_pct", "maximize", 0.5,
          "Minimum combined NYC storage as pct of total capacity [0-100]",
          _min_storage_pct)


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
            "nyc_reliability_weekly",
            "montague_reliability_weekly_fixed",
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
