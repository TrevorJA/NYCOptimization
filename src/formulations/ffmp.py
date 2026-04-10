"""
ffmp.py - Parameterized FFMP (2017 Flexible Flow Management Program) formulation.

Defines decision variables, bounds, and baselines for the FFMP formulation
that re-optimizes existing FFMP parameters within plausible ranges.
"""

from collections import OrderedDict


# FFMP decision variable specification.
# Each entry: {"baseline": <default value>, "bounds": [lo, hi], "units": <str>}
FFMP_FORMULATION = {
    "description": "Parameterized 2017 FFMP rule structure",
    "decision_variables": OrderedDict({

        # --- MRF baselines (MGD) ---
        "mrf_cannonsville": {
            "baseline": 122.8,
            "bounds": [60.0, 250.0],
            "units": "MGD",
        },
        "mrf_pepacton": {
            "baseline": 64.63,
            "bounds": [30.0, 130.0],
            "units": "MGD",
        },
        "mrf_neversink": {
            "baseline": 48.47,
            "bounds": [20.0, 100.0],
            "units": "MGD",
        },
        "mrf_montague": {
            "baseline": 1131.05,
            "bounds": [800.0, 1500.0],
            "units": "MGD",
        },
        "mrf_trenton": {
            "baseline": 1938.95,
            "bounds": [1400.0, 2500.0],
            "units": "MGD",
        },

        # --- NYC delivery constraints ---
        "max_nyc_delivery": {
            "baseline": 800.0,
            "bounds": [500.0, 900.0],
            "units": "MGD",
        },

        # --- NYC drought factors (L3, L4, L5) ---
        # L1a-L2 factors are effectively unconstrained (set to large values)
        "nyc_drought_factor_L3": {
            "baseline": 0.85,
            "bounds": [0.60, 1.0],
            "units": "fraction",
        },
        "nyc_drought_factor_L4": {
            "baseline": 0.70,
            "bounds": [0.40, 0.95],
            "units": "fraction",
        },
        "nyc_drought_factor_L5": {
            "baseline": 0.65,
            "bounds": [0.30, 0.90],
            "units": "fraction",
        },

        # --- NJ drought factors (L4, L5) ---
        "nj_drought_factor_L4": {
            "baseline": 0.90,
            "bounds": [0.60, 1.0],
            "units": "fraction",
        },
        "nj_drought_factor_L5": {
            "baseline": 0.80,
            "bounds": [0.50, 1.0],
            "units": "fraction",
        },

        # --- Storage zone vertical shifts (fraction of capacity) ---
        # Applied as additive shifts to each drought level threshold curve
        "zone_shift_level1b": {
            "baseline": 0.0,
            "bounds": [-0.10, 0.10],
            "units": "fraction",
        },
        "zone_shift_level1c": {
            "baseline": 0.0,
            "bounds": [-0.10, 0.10],
            "units": "fraction",
        },
        "zone_shift_level2": {
            "baseline": 0.0,
            "bounds": [-0.10, 0.10],
            "units": "fraction",
        },
        "zone_shift_level3": {
            "baseline": 0.0,
            "bounds": [-0.10, 0.10],
            "units": "fraction",
        },
        "zone_shift_level4": {
            "baseline": 0.0,
            "bounds": [-0.10, 0.10],
            "units": "fraction",
        },
        "zone_shift_level5": {
            "baseline": 0.0,
            "bounds": [-0.10, 0.10],
            "units": "fraction",
        },

        # --- Flood release maximums (CFS) ---
        "flood_max_cannonsville": {
            "baseline": 4200.0,
            "bounds": [2000.0, 8000.0],
            "units": "CFS",
        },
        "flood_max_pepacton": {
            "baseline": 2400.0,
            "bounds": [1200.0, 5000.0],
            "units": "CFS",
        },
        "flood_max_neversink": {
            "baseline": 3400.0,
            "bounds": [1500.0, 7000.0],
            "units": "CFS",
        },

        # --- MRF seasonal profile scaling (4 seasons) ---
        "mrf_profile_scale_winter": {
            "baseline": 1.0,
            "bounds": [0.5, 2.0],
            "units": "multiplier",
        },
        "mrf_profile_scale_spring": {
            "baseline": 1.0,
            "bounds": [0.5, 2.0],
            "units": "multiplier",
        },
        "mrf_profile_scale_summer": {
            "baseline": 1.0,
            "bounds": [0.5, 2.0],
            "units": "multiplier",
        },
        "mrf_profile_scale_fall": {
            "baseline": 1.0,
            "bounds": [0.5, 2.0],
            "units": "multiplier",
        },
    }),
}


def generate_ffmp_formulation():
    """Return the FFMP formulation dictionary.

    Returns the standard parameterized FFMP formulation with all 24 decision
    variables covering MRF baselines, delivery constraints, drought factors,
    storage zone shifts, flood limits, and MRF seasonal scaling.

    Returns:
        Dict with "description" and "decision_variables" keys.
    """
    return FFMP_FORMULATION
