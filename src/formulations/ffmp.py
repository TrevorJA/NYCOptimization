"""
ffmp.py - Parameterized FFMP (2017 Flexible Flow Management Program) formulation.

Defines decision variables, bounds, and baselines for the FFMP formulation
that re-optimizes existing FFMP parameters within plausible ranges, and
supports N-zone variable-resolution variants.
"""

import numpy as np
from collections import OrderedDict


###############################################################################
# N-zone interpolation helper
###############################################################################

def _interpolate_factors(default_values, n_target):
    """Linearly interpolate a list of values to n_target points.

    Used to scale default 7-level FFMP drought factor arrays to an arbitrary
    number of drought levels in generate_ffmp_formulation(n_zones).

    Args:
        default_values: List/array of source values.
        n_target: Number of output points.

    Returns:
        List of length n_target.
    """
    x_default = np.linspace(0, 1, len(default_values))
    x_target = np.linspace(0, 1, n_target)
    return list(np.interp(x_target, x_default, default_values))


###############################################################################
# Standard FFMP formulation (24 DVs, 7 drought levels)
###############################################################################

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


###############################################################################
# Formulation factory
###############################################################################

def generate_ffmp_formulation(n_zones=None):
    """Generate an FFMP formulation, optionally with variable zone resolution.

    With n_zones=None (default), returns the standard 24-DV formulation
    matching the 2017 FFMP's 7 drought levels (level1a..level5).

    With n_zones=N, generates an N-zone variant where:
    - N storage zone boundary curves are optimized (zone_1..zone_N)
    - N+1 drought levels (zone_0=normal, zone_1..zone_N=drought)
    - Delivery factors only included for levels where interpolated
      baseline is < the unconstrained threshold (< 100 for NYC, < 1.0 for NJ)
    - N=6 is equivalent to the standard 7-level FFMP in zone count

    Args:
        n_zones: Number of storage zone boundary curves, or None for standard.

    Returns:
        Dict with "description" and "decision_variables" keys.
    """
    if n_zones is None:
        return FFMP_FORMULATION

    # Default 7-level baselines for interpolation
    default_nyc_factors = [1_000_000, 1_000_000, 1_000_000, 1_000_000,
                           0.85, 0.70, 0.65]
    default_nj_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 0.90, 0.80]

    interp_nyc = _interpolate_factors(default_nyc_factors, n_zones + 1)
    interp_nj = _interpolate_factors(default_nj_factors, n_zones + 1)

    drought_levels = ["zone_0"] + [f"zone_{i+1}" for i in range(n_zones)]
    storage_levels = [f"zone_{i+1}" for i in range(n_zones)]

    dvs = OrderedDict()

    # MRF baselines (same across all N-zone variants)
    dvs["mrf_cannonsville"] = {"baseline": 122.8, "bounds": [60.0, 250.0], "units": "MGD"}
    dvs["mrf_pepacton"] = {"baseline": 64.63, "bounds": [30.0, 130.0], "units": "MGD"}
    dvs["mrf_neversink"] = {"baseline": 48.47, "bounds": [20.0, 100.0], "units": "MGD"}
    dvs["mrf_montague"] = {"baseline": 1131.05, "bounds": [800.0, 1500.0], "units": "MGD"}
    dvs["mrf_trenton"] = {"baseline": 1938.95, "bounds": [1400.0, 2500.0], "units": "MGD"}

    # Max NYC delivery
    dvs["max_nyc_delivery"] = {"baseline": 800.0, "bounds": [500.0, 900.0], "units": "MGD"}

    # Zone shifts (N curves)
    for level in storage_levels:
        dvs[f"zone_shift_{level}"] = {
            "baseline": 0.0,
            "bounds": [-0.10, 0.10],
            "units": "fraction",
        }

    # NYC delivery factors: only for levels where baseline < unconstrained threshold
    for i, level in enumerate(drought_levels):
        if interp_nyc[i] < 100:
            dvs[f"nyc_drought_factor_{level}"] = {
                "baseline": float(np.clip(interp_nyc[i], 0.30, 1.0)),
                "bounds": [0.30, 1.0],
                "units": "fraction",
            }

    # NJ delivery factors: only for levels where baseline < 1.0
    for i, level in enumerate(drought_levels):
        if interp_nj[i] < 1.0:
            dvs[f"nj_drought_factor_{level}"] = {
                "baseline": float(np.clip(interp_nj[i], 0.50, 1.0)),
                "bounds": [0.50, 1.0],
                "units": "fraction",
            }

    # Flood limits (same across all N-zone variants)
    dvs["flood_max_cannonsville"] = {"baseline": 4200.0, "bounds": [2000.0, 8000.0], "units": "CFS"}
    dvs["flood_max_pepacton"] = {"baseline": 2400.0, "bounds": [1200.0, 5000.0], "units": "CFS"}
    dvs["flood_max_neversink"] = {"baseline": 3400.0, "bounds": [1500.0, 7000.0], "units": "CFS"}

    # MRF seasonal profile scaling
    for season in ["winter", "spring", "summer", "fall"]:
        dvs[f"mrf_profile_scale_{season}"] = {
            "baseline": 1.0,
            "bounds": [0.5, 2.0],
            "units": "multiplier",
        }

    return {
        "description": f"Parameterized FFMP with {n_zones}-zone storage curves",
        "decision_variables": dvs,
    }
