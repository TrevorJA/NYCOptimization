"""
objectives.py - Objective function metric calculations.

Each function takes a dict of Pywr-DRB simulation output DataFrames
and returns a scalar metric value. The functions are registered in
METRIC_FUNCTIONS and called by the simulation wrapper to compute
the objective vector returned to Borg.
"""

import numpy as np
import pandas as pd

from config import (
    NYC_RESERVOIRS,
    NYC_TOTAL_CAPACITY,
    WARMUP_DAYS,
)


###############################################################################
# Individual Metric Functions
###############################################################################

def nyc_reliability(data: dict) -> float:
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


def nyc_max_deficit_pct(data: dict) -> float:
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


def montague_reliability(data: dict) -> float:
    """Fraction of days flow at Montague meets or exceeds the target.

    The Montague target varies by drought level and season in the FFMP,
    but as a first-order metric we use a fixed baseline of 1750 CFS
    (~1131 MGD). This is the Decree-mandated minimum.
    """
    MONTAGUE_TARGET_MGD = 1131.05
    flow = data["major_flow"]["delMontague"].iloc[WARMUP_DAYS:]
    met = (flow >= MONTAGUE_TARGET_MGD).sum()
    return float(met) / len(flow) if len(flow) > 0 else 0.0


def trenton_reliability(data: dict) -> float:
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


def flood_days(data: dict) -> float:
    """Number of days where downstream stage exceeds action threshold.

    Uses combined reservoir storage in flood zone (Level 1a/1b) as
    a proxy. A day is counted as a "flood day" if aggregate NYC storage
    exceeds 95% of total capacity, indicating spill risk.

    TODO: Replace with actual downstream stage calculation using
    flood monitoring nodes (Hale Eddy, Fishs Eddy, Bridgeville)
    once flood operations are integrated into the metric pipeline.
    """
    storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1).iloc[WARMUP_DAYS:]
    flood_threshold = 0.95 * NYC_TOTAL_CAPACITY
    return float((storage >= flood_threshold).sum())


def min_storage_pct(data: dict) -> float:
    """Minimum combined NYC storage as percentage of total capacity.

    Captures worst-case drought vulnerability. Returns value in [0, 100].
    """
    storage = data["res_storage"][NYC_RESERVOIRS].sum(axis=1).iloc[WARMUP_DAYS:]
    return 100.0 * float(storage.min()) / NYC_TOTAL_CAPACITY


###############################################################################
# Metric Registry
###############################################################################

# Maps objective names (from config.OBJECTIVES) to their computation functions.
# Order must match config.OBJECTIVES for correct Borg objective indexing.

METRIC_FUNCTIONS = {
    "nyc_reliability": nyc_reliability,
    "nyc_max_deficit_pct": nyc_max_deficit_pct,
    "montague_reliability": montague_reliability,
    "trenton_reliability": trenton_reliability,
    "flood_days": flood_days,
    "min_storage_pct": min_storage_pct,
}


def compute_all_objectives(data: dict) -> list:
    """Compute all objectives from simulation output data.

    Args:
        data: Dict of DataFrames from Pywr-DRB simulation
              (keyed by results set name).

    Returns:
        List of objective values in the order defined by config.OBJECTIVES.
    """
    from config import OBJECTIVES
    values = []
    for obj_name in OBJECTIVES:
        func = METRIC_FUNCTIONS[obj_name]
        values.append(func(data))
    return values


def objectives_for_borg(data: dict) -> list:
    """Compute objectives in Borg-compatible format (all minimized).

    Borg minimizes all objectives. For "maximize" objectives, we negate
    the value so that minimizing the negated value = maximizing the original.

    Returns:
        List of objective values, negated where direction is "maximize".
    """
    from config import get_obj_directions
    raw = compute_all_objectives(data)
    directions = get_obj_directions()
    # directions: 1 for maximize, -1 for minimize
    # For Borg (all minimize): negate maximize objectives
    return [-d * v for d, v in zip(directions, raw)]
