"""
objectives.py - Objective function framework for NYC reservoir optimization.

Provides an Objective class and ObjectiveSet container that support:
- Registering individual objective metrics as callable functions
- Configurable objective sets for testing different priorities
- Borg-compatible output (all minimized) with direction handling
- Epsilon values for Borg epsilon-dominance archiving

Metric design principles:
- Reliability metrics use WEEKLY aggregation (daily is insensitive;
  shortages are rare events that daily frequency fails to discriminate).
- Flow compliance at Montague/Trenton uses TIME-DYNAMIC targets from
  the simulation (mrf_target), not hardcoded constants.
- Vulnerability metrics capture the worst-case shortage magnitude.
- Trenton uses the full simulation period (no a priori seasonal assumption).

Usage:
    from src.objectives import DEFAULT_OBJECTIVES
    values = DEFAULT_OBJECTIVES.compute(data)
    borg_values = DEFAULT_OBJECTIVES.compute_for_borg(data)
"""

import numpy as np
import pandas as pd

from config import (
    NYC_RESERVOIRS,
    NYC_TOTAL_CAPACITY,
    WARMUP_DAYS,
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


# --- Flow Compliance (time-dynamic targets) ---

def _montague_reliability_weekly(data: dict) -> float:
    """Fraction of weeks flow at Montague meets the time-dynamic MRF target.

    Uses the actual FFMP-driven target from simulation output (varies by
    drought level and month), not a hardcoded constant.
    """
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    target = data["mrf_target"]["delMontague"].iloc[WARMUP_DAYS:]

    weekly_flow = flow.resample("W").mean()
    weekly_target = target.resample("W").mean()

    met = (weekly_flow >= weekly_target).sum()
    total = len(weekly_flow)
    return float(met) / total if total > 0 else 0.0


def _montague_vulnerability(data: dict) -> float:
    """Worst single-week Montague flow deficit as percent of target.

    Returns the maximum weekly shortfall below the time-dynamic MRF
    target, expressed as a percentage of that week's target. [0, 100+].
    """
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    target = data["mrf_target"]["delMontague"].iloc[WARMUP_DAYS:]

    weekly_flow = flow.resample("W").mean()
    weekly_target = target.resample("W").mean()

    deficit = (weekly_target - weekly_flow).clip(lower=0)
    deficit_pct = np.where(
        weekly_target > 0,
        100.0 * deficit / weekly_target,
        0.0,
    )
    return float(np.max(deficit_pct)) if len(deficit_pct) > 0 else 0.0


def _trenton_reliability_weekly(data: dict) -> float:
    """Fraction of weeks flow at Trenton meets the time-dynamic MRF target.

    Uses the full simulation period (no seasonal assumption).
    Target comes from FFMP simulation output.
    """
    flow = data["major_flow"]["delTrenton"].iloc[WARMUP_DAYS:]
    target = data["mrf_target"]["delTrenton"].iloc[WARMUP_DAYS:]

    weekly_flow = flow.resample("W").mean()
    weekly_target = target.resample("W").mean()

    met = (weekly_flow >= weekly_target).sum()
    total = len(weekly_flow)
    return float(met) / total if total > 0 else 0.0


def _trenton_vulnerability(data: dict) -> float:
    """Worst single-week Trenton flow deficit as percent of target.

    Full period, time-dynamic target. [0, 100+].
    """
    flow = data["major_flow"]["delTrenton"].iloc[WARMUP_DAYS:]
    target = data["mrf_target"]["delTrenton"].iloc[WARMUP_DAYS:]

    weekly_flow = flow.resample("W").mean()
    weekly_target = target.resample("W").mean()

    deficit = (weekly_target - weekly_flow).clip(lower=0)
    deficit_pct = np.where(
        weekly_target > 0,
        100.0 * deficit / weekly_target,
        0.0,
    )
    return float(np.max(deficit_pct)) if len(deficit_pct) > 0 else 0.0


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
# Pre-built Objective Instances
###############################################################################

# --- NYC Supply ---
obj_nyc_reliability_weekly = Objective(
    name="nyc_reliability_weekly",
    direction="maximize",
    epsilon=0.01,
    description="Fraction of weeks NYC delivery >= 99% of demand",
    func=_nyc_reliability_weekly,
)

obj_nyc_vulnerability = Objective(
    name="nyc_vulnerability",
    direction="minimize",
    epsilon=1.0,
    description="Worst-week NYC shortage as pct of demand [0-100]",
    func=_nyc_vulnerability,
)

# --- Flow Compliance ---
obj_montague_reliability_weekly = Objective(
    name="montague_reliability_weekly",
    direction="maximize",
    epsilon=0.01,
    description="Fraction of weeks Montague flow >= time-dynamic MRF target",
    func=_montague_reliability_weekly,
)

obj_montague_vulnerability = Objective(
    name="montague_vulnerability",
    direction="minimize",
    epsilon=1.0,
    description="Worst-week Montague deficit as pct of dynamic target",
    func=_montague_vulnerability,
)

obj_trenton_reliability_weekly = Objective(
    name="trenton_reliability_weekly",
    direction="maximize",
    epsilon=0.005,
    description="Fraction of weeks Trenton flow >= time-dynamic MRF target (full period)",
    func=_trenton_reliability_weekly,
)

obj_trenton_vulnerability = Objective(
    name="trenton_vulnerability",
    direction="minimize",
    epsilon=1.0,
    description="Worst-week Trenton deficit as pct of dynamic target",
    func=_trenton_vulnerability,
)

# --- NJ Supply ---
obj_nj_reliability_weekly = Objective(
    name="nj_reliability_weekly",
    direction="maximize",
    epsilon=0.01,
    description="Fraction of weeks NJ delivery >= 99% of demand",
    func=_nj_reliability_weekly,
)

obj_nj_vulnerability = Objective(
    name="nj_vulnerability",
    direction="minimize",
    epsilon=1.0,
    description="Worst-week NJ shortage as pct of demand [0-100]",
    func=_nj_vulnerability,
)

# --- Flood Risk ---
obj_flood_risk_storage_spill_days = Objective(
    name="flood_risk_storage_spill_days",
    direction="minimize",
    epsilon=10.0,
    description="Days aggregate NYC storage > 95% capacity (spill risk proxy)",
    func=_flood_days,
)

obj_flood_risk_downstream_flow_days = Objective(
    name="flood_risk_downstream_flow_days",
    direction="minimize",
    epsilon=5.0,
    description="Days Montague flow exceeds 25,000 CFS action stage proxy",
    func=_flood_days_downstream,
)

# --- Storage Resilience ---
obj_storage_min_combined_pct = Objective(
    name="storage_min_combined_pct",
    direction="maximize",
    epsilon=0.5,
    description="Minimum combined NYC storage as pct of total capacity [0-100]",
    func=_min_storage_pct,
)


###############################################################################
# Pre-built Objective Sets
###############################################################################

# Default 7-objective set: NYC + NJ supply, Montague/Trenton compliance,
# downstream flood risk, storage resilience.
# Uses downstream flow-based flood metric (Montague > 25,000 CFS action stage)
# instead of the 95% storage threshold (which is rarely triggered even under
# FFMP Normal operations).
DEFAULT_OBJECTIVES = ObjectiveSet([
    obj_nyc_reliability_weekly,
    obj_nyc_vulnerability,
    obj_nj_reliability_weekly,
    obj_montague_reliability_weekly,
    obj_trenton_reliability_weekly,
    obj_flood_risk_downstream_flow_days,
    obj_storage_min_combined_pct,
])

# Extended: adds vulnerability metrics for all stakeholders
EXTENDED_OBJECTIVES = ObjectiveSet([
    obj_nyc_reliability_weekly,
    obj_nyc_vulnerability,
    obj_nj_reliability_weekly,
    obj_nj_vulnerability,
    obj_montague_reliability_weekly,
    obj_montague_vulnerability,
    obj_trenton_reliability_weekly,
    obj_trenton_vulnerability,
    obj_flood_risk_downstream_flow_days,
    obj_storage_min_combined_pct,
])

# Compact 4-objective set for quick diagnostic / debugging runs
COMPACT_OBJECTIVES = ObjectiveSet([
    obj_nyc_reliability_weekly,
    obj_montague_reliability_weekly,
    obj_flood_risk_downstream_flow_days,
    obj_storage_min_combined_pct,
])

# Registry of named sets for CLI selection
OBJECTIVE_SETS = {
    "default": DEFAULT_OBJECTIVES,
    "extended": EXTENDED_OBJECTIVES,
    "compact": COMPACT_OBJECTIVES,
}
