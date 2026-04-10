"""
config.py - Central configuration for NYCOptimization study.

Contains paths, simulation settings, NYC system constants, and Borg MOEA
parameters.  Problem formulation logic lives in src/formulations/.
"""

import numpy as np
from pathlib import Path

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

START_DATE = "1945-10-01"   # Water-year start matching presimulated release data
END_DATE = "2022-09-30"     # Water-year end matching presimulated release data
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
# Active Objective Set
###############################################################################

# Available sets: "default", "drought_duration", "downstream_flood",
#                 "comprehensive", "compact"
ACTIVE_OBJECTIVE_SET = "default"


###############################################################################
# Borg MOEA Settings
###############################################################################

BORG_SETTINGS = {
    "max_evaluations": 1_000_000,    # Per island (total NFE = islands * max_evaluations)
    "runtime_frequency": 500,        # Archive snapshot every N NFE
    "n_seeds": 10,
}

# Multi-Master Borg parallel configuration
MMBORG_SETTINGS = {
    "n_islands": None,               # Set per HPC allocation
    "n_workers_per_island": None,    # Set per HPC allocation
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
# Backward-compatible re-exports from src.formulations
###############################################################################
# Callers that do `from config import get_bounds` etc. continue to work.

from src.formulations import (           # noqa: E402
    FORMULATIONS,
    get_formulation,
    get_bounds,
    get_var_names,
    get_n_vars,
    get_baseline_values,
    get_n_objs,
    get_obj_names,
    get_obj_directions,
    get_objective_set,
    make_objective_function,
    is_external_policy,
    get_architecture,
    generate_ffmp_formulation,
)


###############################################################################
# Thin helpers (kept here for API compatibility)
###############################################################################

def get_epsilons():
    """Epsilon values for Borg epsilon-dominance, ordered by objective."""
    return get_objective_set().epsilons


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
