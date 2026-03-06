"""
objectives.py - Objective function framework for NYC reservoir optimization.

Provides an Objective class and ObjectiveSet container that support:
- Registering individual objective metrics as callable functions
- Configurable objective sets for testing different priorities
- Borg-compatible output (all minimized) with direction handling
- Epsilon values for Borg epsilon-dominance archiving

Usage:
    # Use the default set
    from src.objectives import DEFAULT_OBJECTIVES
    values = DEFAULT_OBJECTIVES.compute(data)
    borg_values = DEFAULT_OBJECTIVES.compute_for_borg(data)

    # Create a custom set
    custom = ObjectiveSet([obj_nyc_supply_reliability_daily,
                           obj_flood_risk_storage_spill_days,
                           obj_storage_min_combined_pct])
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
        """Compute all raw objective values from simulation data.

        Returns:
            List of floats in the order defined by this ObjectiveSet.
        """
        return [obj.compute(data) for obj in self._objectives]

    def compute_for_borg(self, data: dict) -> list:
        """Compute all objectives in Borg-compatible format (all minimized).

        Returns:
            List of floats, negated where direction is 'maximize'.
        """
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

def _nyc_reliability(data: dict) -> float:
    """Fraction of days NYC delivery meets or exceeds demand.

    Uses ibt_demands["demand_nyc"] and ibt_diversions["delivery_nyc"].
    Returns value in [0, 1] where 1.0 = perfect reliability.
    """
    demand = data["ibt_demands"]["demand_nyc"].iloc[WARMUP_DAYS:]
    delivery = data["ibt_diversions"]["delivery_nyc"].iloc[WARMUP_DAYS:]

    # A day is "met" if delivery >= 99% of demand (tolerance for numerical noise)
    met = (delivery >= 0.99 * demand).sum()
    total = len(demand)
    return float(met) / total if total > 0 else 0.0


def _nyc_max_deficit_pct(data: dict) -> float:
    """Maximum monthly shortage as a percentage of monthly demand.

    Aggregates daily shortages to monthly totals, then finds the
    worst month. Returns value in [0, 100].
    """
    demand = data["ibt_demands"]["demand_nyc"].iloc[WARMUP_DAYS:]
    delivery = data["ibt_diversions"]["delivery_nyc"].iloc[WARMUP_DAYS:]

    daily_shortage = (demand - delivery).clip(lower=0)
    monthly_shortage = daily_shortage.resample("ME").sum()
    monthly_demand = demand.resample("ME").sum()

    # Avoid division by zero for months with no demand
    monthly_deficit_pct = np.where(
        monthly_demand > 0,
        100.0 * monthly_shortage / monthly_demand,
        0.0,
    )
    return float(np.max(monthly_deficit_pct))


def _nyc_max_consecutive_shortfall(data: dict) -> float:
    """Maximum consecutive days NYC delivery falls below demand.

    Returns integer count of longest consecutive shortfall streak.
    A shortfall day is defined as delivery < 99% of demand.
    """
    demand = data["ibt_demands"]["demand_nyc"].iloc[WARMUP_DAYS:]
    delivery = data["ibt_diversions"]["delivery_nyc"].iloc[WARMUP_DAYS:]

    shortfall = (delivery < 0.99 * demand).astype(int)
    if shortfall.sum() == 0:
        return 0.0

    # Find longest consecutive run of 1s
    groups = shortfall.diff().ne(0).cumsum()
    max_streak = shortfall.groupby(groups).sum().max()
    return float(max_streak)


def _montague_reliability(data: dict) -> float:
    """Fraction of days flow at Montague meets or exceeds the target.

    Uses fixed Decree-mandated minimum of 1750 CFS (~1131 MGD).
    """
    MONTAGUE_TARGET_MGD = 1131.05
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    met = (flow >= MONTAGUE_TARGET_MGD).sum()
    return float(met) / len(flow) if len(flow) > 0 else 0.0


def _trenton_reliability(data: dict) -> float:
    """Fraction of days flow at Trenton meets the target (Jun 15 - Mar 15).

    The Trenton target of ~3000 CFS (~1939 MGD) is active during the
    period June 15 through March 15. Outside this window, there is no
    binding Trenton target under normal operations.
    """
    TRENTON_TARGET_MGD = 1938.95
    flow = data["major_flow"]["delTrenton"].iloc[WARMUP_DAYS:]

    # Filter to active period: Jun 15 (DOY 166) through Mar 15 (DOY 74)
    doy = flow.index.dayofyear
    active = (doy >= 166) | (doy <= 74)
    active_flow = flow[active]

    if len(active_flow) == 0:
        return 0.0
    met = (active_flow >= TRENTON_TARGET_MGD).sum()
    return float(met) / len(active_flow)


def _flood_days(data: dict) -> float:
    """Number of days aggregate NYC storage exceeds spill risk threshold.

    Uses 95% of total NYC capacity as a proxy for spill risk.
    NOTE: This is a crude proxy. A better metric would use actual
    downstream flow exceedances at gauge locations.
    """
    storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1).iloc[WARMUP_DAYS:]
    flood_threshold = 0.95 * NYC_TOTAL_CAPACITY
    return float((storage >= flood_threshold).sum())


def _flood_days_downstream(data: dict) -> float:
    """Number of days downstream flow at Montague exceeds flood threshold.

    Uses a flow threshold of 25,000 CFS (~16,148 MGD) at Montague
    as an action stage proxy. This better captures actual downstream
    flood risk from reservoir releases.
    """
    FLOOD_FLOW_THRESHOLD_MGD = 16148.0  # ~25,000 CFS
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    return float((flow >= FLOOD_FLOW_THRESHOLD_MGD).sum())


def _min_storage_pct(data: dict) -> float:
    """Minimum combined NYC storage as percentage of total capacity.

    Captures worst-case drought vulnerability. Returns value in [0, 100].
    """
    storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1).iloc[WARMUP_DAYS:]
    return 100.0 * float(storage.min()) / NYC_TOTAL_CAPACITY


def _avg_storage_pct(data: dict) -> float:
    """Average combined NYC storage as percentage of total capacity.

    Captures overall storage utilization. Returns value in [0, 100].
    """
    storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1).iloc[WARMUP_DAYS:]
    return 100.0 * float(storage.mean()) / NYC_TOTAL_CAPACITY


def _montague_deficit_mgd(data: dict) -> float:
    """Average daily Montague shortfall below target (MGD).

    Only counts days with shortfall. Returns 0.0 if no shortfall.
    """
    MONTAGUE_TARGET_MGD = 1131.05
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    deficit = (MONTAGUE_TARGET_MGD - flow).clip(lower=0)
    shortfall_days = (deficit > 0).sum()
    if shortfall_days == 0:
        return 0.0
    return float(deficit.sum()) / shortfall_days


###############################################################################
# Pre-built Objective Instances
#
# Naming convention: obj_<stakeholder/system>_<metric>_<variant>
# When multiple metrics measure similar things, each variant gets a
# distinct verbose name so they can coexist without ambiguity.
###############################################################################

# --- NYC Supply ---

obj_nyc_supply_reliability_daily = Objective(
    name="nyc_supply_reliability_daily",
    direction="maximize",
    epsilon=0.005,
    description="Fraction of days NYC delivery >= 99% of demand",
    func=_nyc_reliability,
)

obj_nyc_drought_max_monthly_deficit_pct = Objective(
    name="nyc_drought_max_monthly_deficit_pct",
    direction="minimize",
    epsilon=1.0,
    description="Worst-month shortage as percent of that month's demand [0-100]",
    func=_nyc_max_deficit_pct,
)

obj_nyc_drought_max_consecutive_shortfall_days = Objective(
    name="nyc_drought_max_consecutive_shortfall_days",
    direction="minimize",
    epsilon=5.0,
    description="Longest streak of consecutive days delivery < 99% demand",
    func=_nyc_max_consecutive_shortfall,
)

# --- Flow Compliance ---

obj_montague_flow_reliability_daily = Objective(
    name="montague_flow_reliability_daily",
    direction="maximize",
    epsilon=0.005,
    description="Fraction of days Delaware at Montague flow >= 1131 MGD (Decree min)",
    func=_montague_reliability,
)

obj_trenton_flow_reliability_seasonal = Objective(
    name="trenton_flow_reliability_seasonal",
    direction="maximize",
    epsilon=0.005,
    description="Fraction of active-period days (Jun15-Mar15) Trenton flow >= 1939 MGD",
    func=_trenton_reliability,
)

obj_montague_flow_avg_deficit_mgd = Objective(
    name="montague_flow_avg_deficit_mgd",
    direction="minimize",
    epsilon=5.0,
    description="Mean daily Montague shortfall on violation days (MGD)",
    func=_montague_deficit_mgd,
)

# --- Flood Risk ---

obj_flood_risk_storage_spill_days = Objective(
    name="flood_risk_storage_spill_days",
    direction="minimize",
    epsilon=5.0,
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

obj_storage_avg_combined_pct = Objective(
    name="storage_avg_combined_pct",
    direction="maximize",
    epsilon=1.0,
    description="Average combined NYC storage as pct of total capacity [0-100]",
    func=_avg_storage_pct,
)


###############################################################################
# Pre-built Objective Sets
#
# Multiple sets allow testing different priority framings without
# committing to a single formulation prematurely.
###############################################################################

# Default 6-objective set: monthly deficit variant, storage-proxy flood
DEFAULT_OBJECTIVES = ObjectiveSet([
    obj_nyc_supply_reliability_daily,
    obj_nyc_drought_max_monthly_deficit_pct,
    obj_montague_flow_reliability_daily,
    obj_trenton_flow_reliability_seasonal,
    obj_flood_risk_storage_spill_days,
    obj_storage_min_combined_pct,
])

# Drought-duration variant: consecutive shortfall days instead of monthly deficit %
DROUGHT_DURATION_OBJECTIVES = ObjectiveSet([
    obj_nyc_supply_reliability_daily,
    obj_nyc_drought_max_consecutive_shortfall_days,
    obj_montague_flow_reliability_daily,
    obj_trenton_flow_reliability_seasonal,
    obj_flood_risk_storage_spill_days,
    obj_storage_min_combined_pct,
])

# Downstream flood variant: Montague flow exceedance instead of storage proxy
DOWNSTREAM_FLOOD_OBJECTIVES = ObjectiveSet([
    obj_nyc_supply_reliability_daily,
    obj_nyc_drought_max_monthly_deficit_pct,
    obj_montague_flow_reliability_daily,
    obj_trenton_flow_reliability_seasonal,
    obj_flood_risk_downstream_flow_days,
    obj_storage_min_combined_pct,
])

# Comprehensive 8-objective set: includes both flood metrics and both
# drought severity metrics. Good for initial exploration before
# deciding which variants to keep.
COMPREHENSIVE_OBJECTIVES = ObjectiveSet([
    obj_nyc_supply_reliability_daily,
    obj_nyc_drought_max_monthly_deficit_pct,
    obj_nyc_drought_max_consecutive_shortfall_days,
    obj_montague_flow_reliability_daily,
    obj_trenton_flow_reliability_seasonal,
    obj_flood_risk_storage_spill_days,
    obj_flood_risk_downstream_flow_days,
    obj_storage_min_combined_pct,
])

# Compact 4-objective set for quick diagnostic / debugging runs
COMPACT_OBJECTIVES = ObjectiveSet([
    obj_nyc_supply_reliability_daily,
    obj_montague_flow_reliability_daily,
    obj_flood_risk_storage_spill_days,
    obj_storage_min_combined_pct,
])

# Registry of named sets for CLI selection
OBJECTIVE_SETS = {
    "default": DEFAULT_OBJECTIVES,
    "drought_duration": DROUGHT_DURATION_OBJECTIVES,
    "downstream_flood": DOWNSTREAM_FLOOD_OBJECTIVES,
    "comprehensive": COMPREHENSIVE_OBJECTIVES,
    "compact": COMPACT_OBJECTIVES,
}


