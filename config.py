"""
config.py - Central configuration for NYCOptimization study.

Defines problem formulations, decision variable specifications,
objective functions, simulation settings, and Borg MOEA parameters.
"""

import numpy as np
from pathlib import Path
from collections import OrderedDict

###############################################################################
# Paths
###############################################################################

PROJECT_DIR = Path(__file__).parent
SRC_DIR = PROJECT_DIR / "src"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
SCRIPTS_DIR = PROJECT_DIR / "scripts"

# Borg shared libraries (user must compile and place here)
BORG_DIR = PROJECT_DIR / "borg"

# Pywr-DRB pre-simulated releases for trimmed model
PRESIM_DIR = OUTPUTS_DIR / "presim"
PRESIM_FILE = PRESIM_DIR / "presimulated_releases_mgd.csv"


###############################################################################
# Simulation Settings
###############################################################################

START_DATE = "1945-01-01"
END_DATE = "2022-12-31"
INFLOW_TYPE = "pub_nhmv10_BC_withObsScaled"
USE_TRIMMED_MODEL = True
INITIAL_VOLUME_FRAC = 0.80

# Results sets to export from Pywr-DRB simulations
RESULTS_SETS = [
    "major_flow",
    "res_storage",
    "res_level",
    "ibt_diversions",
    "ibt_demands",
]

# Warmup period (days) to exclude from metric calculations
WARMUP_DAYS = 365


###############################################################################
# NYC System Constants
###############################################################################

NYC_RESERVOIRS = ["cannonsville", "pepacton", "neversink"]

# Capacities in MG
NYC_RESERVOIR_CAPACITIES = {
    "cannonsville": 95706.0,
    "pepacton": 140190.0,
    "neversink": 34941.0,
}
NYC_TOTAL_CAPACITY = sum(NYC_RESERVOIR_CAPACITIES.values())  # 270,837 MG


###############################################################################
# Problem Formulations
###############################################################################

# Each formulation defines a set of decision variables with names, bounds,
# and a mapping function that converts a flat decision variable vector
# into a NYCOperationsConfig object.

FORMULATIONS = {}

# ---- Formulation A: Parameterized FFMP ------------------------------------
# Re-optimize existing FFMP parameters within their plausible ranges.
# This mirrors the parameter groups from NYCOperationExploration but
# enables all groups and uses optimization-appropriate bounds.

FORMULATIONS["ffmp"] = {
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
# Objectives
###############################################################################

# Objective definitions live in src/objectives.py using the Objective class.
# The active objective set is selected here by name.
# Available sets: "default", "drought_duration", "downstream_flood", "comprehensive", "compact"
ACTIVE_OBJECTIVE_SET = "default"


###############################################################################
# Borg MOEA Settings
###############################################################################

BORG_SETTINGS = {
    "max_evaluations": 1_000_000,
    "runtime_frequency": 1000,       # Print archive every N NFE
    "n_seeds": 10,
}

# Multi-Master Borg parallel configuration
MMBORG_SETTINGS = {
    "n_islands": None,               # Set per HPC allocation
    "n_workers_per_island": None,     # Set per HPC allocation
    "max_time_hours": 24,
}


###############################################################################
# MOEA Diagnostics (MOEAFramework v5.0)
###############################################################################

DIAGNOSTICS_SETTINGS = {
    "moea_framework_jar": "MOEAFramework-5.0/cli",
    "hypervolume_delta": 0.01,       # HV improvement threshold
}


###############################################################################
# Re-evaluation Settings
###############################################################################

REEVALUATION_SETTINGS = {
    "n_realizations": None,          # TBD
    "realization_length_years": 70,
    "generator": "kirsch_nowak",
    "climate_scenarios": ["stationary"],
    "robustness_metrics": ["satisficing", "regret"],
}


###############################################################################
# Helper Functions
###############################################################################

def get_formulation(name="ffmp"):
    """Get a formulation definition by name."""
    if name not in FORMULATIONS:
        raise ValueError(
            f"Unknown formulation '{name}'. "
            f"Available: {list(FORMULATIONS.keys())}"
        )
    return FORMULATIONS[name]


def get_n_vars(formulation_name="ffmp"):
    """Get number of decision variables for a formulation."""
    f = get_formulation(formulation_name)
    return len(f["decision_variables"])


def get_objective_set(name=None):
    """Get the active ObjectiveSet instance.

    Args:
        name: Objective set name. If None, uses ACTIVE_OBJECTIVE_SET from config.
    """
    from src.objectives import OBJECTIVE_SETS
    if name is None:
        name = ACTIVE_OBJECTIVE_SET
    if name not in OBJECTIVE_SETS:
        raise ValueError(
            f"Unknown objective set '{name}'. "
            f"Available: {list(OBJECTIVE_SETS.keys())}"
        )
    return OBJECTIVE_SETS[name]


def get_n_objs():
    """Get number of objectives in the active set."""
    return get_objective_set().n_objs


def get_bounds(formulation_name="ffmp"):
    """Get decision variable bounds as (lower, upper) arrays."""
    f = get_formulation(formulation_name)
    lower = []
    upper = []
    for var_spec in f["decision_variables"].values():
        lower.append(var_spec["bounds"][0])
        upper.append(var_spec["bounds"][1])
    return np.array(lower), np.array(upper)


def get_epsilons():
    """Get epsilon values for Borg epsilon-dominance, ordered by objective."""
    return get_objective_set().epsilons


def get_var_names(formulation_name="ffmp"):
    """Get ordered list of decision variable names."""
    f = get_formulation(formulation_name)
    return list(f["decision_variables"].keys())


def get_obj_names():
    """Get ordered list of objective names."""
    return get_objective_set().names


def get_obj_directions():
    """Get list of objective directions (1 for maximize, -1 for minimize).

    Borg minimizes all objectives. For maximization objectives, we negate
    the value before returning to Borg and negate again when reading results.
    """
    return get_objective_set().directions


def get_baseline_values(formulation_name="ffmp"):
    """Get baseline (default FFMP) decision variable values."""
    f = get_formulation(formulation_name)
    return np.array([v["baseline"] for v in f["decision_variables"].values()])


def print_config_summary(formulation_name="ffmp"):
    """Print a summary of the current configuration."""
    f = get_formulation(formulation_name)
    obj_set = get_objective_set()
    print(f"Formulation: {formulation_name}")
    print(f"Description: {f['description']}")
    print(f"Decision variables: {get_n_vars(formulation_name)}")
    print(f"Objective set: {ACTIVE_OBJECTIVE_SET} ({obj_set.n_objs} objectives)")
    print(f"\nDecision Variables:")
    for name, spec in f["decision_variables"].items():
        print(f"  {name}: [{spec['bounds'][0]}, {spec['bounds'][1]}] "
              f"({spec['units']}) baseline={spec['baseline']}")
    print(f"\n{obj_set.summary()}")
    print(f"\nSimulation: {INFLOW_TYPE}, {START_DATE} to {END_DATE}")
    print(f"Trimmed model: {USE_TRIMMED_MODEL}")
    print(f"Borg NFE: {BORG_SETTINGS['max_evaluations']:,}")
    print(f"Seeds: {BORG_SETTINGS['n_seeds']}")
