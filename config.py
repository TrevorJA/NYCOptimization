"""
config.py - Central configuration for NYCOptimization study.

Single source of truth for paths, simulation settings, NYC system constants,
Borg MOEA parameters, T/S coupling, re-evaluation sizing, and the slug
naming convention. Problem formulation logic lives in src/formulations/.

Every methodologic knob has a default constant here and a `NYCOPT_*` env
override read at import time. SLURM scripts source per-experiment env files
under slurm/envs/ to set these without relying on remembered CLI flags;
see local_notes/configuration/knob_reference.md for the full table.

Environment overrides (selected — full list in knob_reference.md):
    NYCOPT_OBJECTIVES           -> ACTIVE_OBJECTIVES (comma-separated names)
    NYCOPT_STATE_FEATURES       -> STATE_FEATURES    (comma-separated names)
    NYCOPT_FORMULATIONS         -> PRODUCTION_FORMULATIONS (comma-separated)
    NYCOPT_FFMP_VR_N            -> FFMP_VR_N_SWEEP (comma-separated ints)
    NYCOPT_TEMPERATURE_ON       -> INCLUDE_TEMPERATURE_MODEL (bool, default 0)
    NYCOPT_SALINITY_ON          -> INCLUDE_SALINITY_MODEL    (bool, default 0)
    NYCOPT_TS_ON                -> shortcut: sets both above (legacy convenience)
    NYCOPT_THERMAL_THRESHOLD_C  -> LORDVILLE_THERMAL_THRESHOLD_C (float)
    NYCOPT_SALT_FRONT_RM        -> SALT_FRONT_REFERENCE_RM (float)
    NYCOPT_SALINITY_ASYNC       -> SALINITY_ASYNC_UPDATE (bool)
    NYCOPT_LSTM_START_DATE      -> LSTM_START_DATE (LSTM training data start)
    NYCOPT_REEVAL_N             -> REEVAL_REALIZATIONS (int)
    NYCOPT_REEVAL_NODES         -> REEVAL_NODES (int)
    NYCOPT_REEVAL_RANKS         -> REEVAL_RANKS_PER_NODE (int)
    NYCOPT_REEVAL_MODE          -> REEVAL_MODE ("mpi" | "single")
    NYCOPT_CLUSTER              -> CLUSTER ("anvil" | "hopper")
    NYCOPT_TEMPERATURE_LSTM_DIR -> TEMPERATURE_LSTM_DIR (path)
    NYCOPT_SALINITY_LSTM_DIR    -> SALINITY_LSTM_DIR (path)
    RUN_SLUG_TAG                -> appended as a free-form suffix to derive_slug()
"""

import os
import numpy as np
from pathlib import Path


###############################################################################
# Env-parsing helpers
###############################################################################

def _parse_list_env(name: str, default: list[str]) -> list[str]:
    """Parse a comma-separated environment variable into a list of names."""
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_int_list_env(name: str, default: list[int]) -> list[int]:
    """Parse a comma-separated env var into a list of ints."""
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def _parse_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var. Truthy: 1, true, yes, on (case-insensitive)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None and raw.strip() else default


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw.strip() else default


def _parse_str_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw is not None and raw.strip() else default


def _parse_path_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw is not None and raw.strip() else default


###############################################################################
# Paths
###############################################################################

PROJECT_DIR = Path(__file__).parent
SRC_DIR = PROJECT_DIR / "src"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = PROJECT_DIR / "figures"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
NOTES_DIR = PROJECT_DIR / "local_notes"

# Borg shared libraries (user must compile and place here)
BORG_DIR = PROJECT_DIR / "lib" / "borg"

# Pywr-DRB pre-simulated releases for trimmed model
PRESIM_DIR = OUTPUTS_DIR / "presim"
PRESIM_FILE = PRESIM_DIR / "presimulated_releases_mgd.csv"

# PywrDRB-ML plugin (sibling repo) — temperature + salinity LSTM weights
PYWRDRB_ML_DIR = _parse_path_env(
    "NYCOPT_PYWRDRB_ML_DIR",
    (PROJECT_DIR / ".." / "PywrDRB-ML").resolve(),
)
TEMPERATURE_LSTM_DIR = _parse_path_env(
    "NYCOPT_TEMPERATURE_LSTM_DIR",
    PYWRDRB_ML_DIR / "models" / "TempLSTM",
)
SALINITY_LSTM_DIR = _parse_path_env(
    "NYCOPT_SALINITY_LSTM_DIR",
    PYWRDRB_ML_DIR / "models" / "SalinityLSTM",
)

# Specific artifact paths the ModelBuilder options dict consumes.
# These are YAML/JSON file paths (NOT Python objects) — the parameter
# classes do their own loading from these paths.
TEMPERATURE_LSTM_MODEL1 = _parse_path_env(
    "NYCOPT_TEMPERATURE_LSTM_MODEL1",
    TEMPERATURE_LSTM_DIR / "TempLSTM1.yml",
)
TEMPERATURE_LSTM_MODEL2 = _parse_path_env(
    "NYCOPT_TEMPERATURE_LSTM_MODEL2",
    TEMPERATURE_LSTM_DIR / "TempLSTM2.yml",
)
TEMPERATURE_LSTM_TAVG2TMAX = _parse_path_env(
    "NYCOPT_TEMPERATURE_LSTM_TAVG2TMAX",
    TEMPERATURE_LSTM_DIR / "Tavg2Tmax_coefs.json",
)
SALINITY_LSTM_MODEL = _parse_path_env(
    "NYCOPT_SALINITY_LSTM_MODEL",
    SALINITY_LSTM_DIR / "SalinityLSTM.yml",
)


###############################################################################
# Output category subdirectories (slug-aware, hierarchical)
###############################################################################
# Categories below are diagnostic, not publication. Plotting + reeval scripts
# write to {category}/{slug}/. New categories may be added freely as analyses
# emerge; the convention is "category by analysis type, slug as inner partition".
# `_exploratory/` is the workbench area for in-flight analyses; promote a topic
# to a top-level category once the analysis stabilizes.

OUTPUT_BASELINE_DIR = OUTPUTS_DIR / "baseline"
OUTPUT_OPTIMIZATION_DIR = OUTPUTS_DIR / "optimization"
OUTPUT_REEVALUATION_DIR = OUTPUTS_DIR / "reevaluation"
OUTPUT_DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"
OUTPUT_REFERENCE_SETS_DIR = OUTPUTS_DIR / "reference_sets"
OUTPUT_RUN_MANIFESTS_DIR = OUTPUTS_DIR / "run_manifests"

FIG_CONVERGENCE_DIR = FIGURES_DIR / "convergence"
FIG_PARETO_DIR = FIGURES_DIR / "pareto"
FIG_PARALLEL_COORDS_DIR = FIGURES_DIR / "parallel_coords"
FIG_POLICY_INSPECTION_DIR = FIGURES_DIR / "policy_inspection"
FIG_ROBUSTNESS_DIR = FIGURES_DIR / "robustness"
FIG_EXPLORATORY_DIR = FIGURES_DIR / "_exploratory"


def output_dir_for(category: str, slug: str) -> Path:
    """Return a slug-partitioned output subdir, creating it if needed.

    Args:
        category: One of "baseline", "optimization", "reevaluation",
            "diagnostics", or any free-form name (auto-created).
        slug: The methodologic slug from `derive_slug()`.
    """
    p = OUTPUTS_DIR / category / slug
    p.mkdir(parents=True, exist_ok=True)
    return p


def figure_dir_for(category: str, slug: str) -> Path:
    """Return a slug-partitioned figure subdir, creating it if needed.

    Args:
        category: e.g. "convergence", "pareto", "parallel_coords",
            "policy_inspection", "robustness". Free-form names land
            under `_exploratory/{slug}/{category}/`.
        slug: The methodologic slug from `derive_slug()`.
    """
    stable = {"convergence", "pareto", "parallel_coords",
              "policy_inspection", "robustness"}
    if category in stable:
        p = FIGURES_DIR / category / slug
    else:
        p = FIG_EXPLORATORY_DIR / slug / category
    p.mkdir(parents=True, exist_ok=True)
    return p


###############################################################################
# Simulation Settings
###############################################################################

START_DATE = "1945-10-01"   # Water-year start matching presimulated release data
END_DATE = "2022-09-30"     # Water-year end matching presimulated release data
INFLOW_TYPE = "pub_nhmv10_BC_withObsScaled"
USE_TRIMMED_MODEL = True
INITIAL_VOLUME_FRAC = 0.80

# State features observed by external policies (RBF/Tree/ANN).
#
# Each entry is the name of a feature registered in
# src.external_policy.STATE_FEATURE_REGISTRY. The default below mirrors
# the information used by FFMP's decision logic:
#   - combined NYC storage (drought zone classification)
#   - Montague non-NYC flow at lag 2d (Montague MRF look-ahead)
#   - Trenton non-NYC flow at lag 4d  (Trenton MRF look-ahead)
#   - NJ demand at lag 4d             (Trenton equivalent-flow forecast)
# sin(DOY) and cos(DOY) are appended automatically by the state extractor
# (seasonality is always cheap and justifiable).
#
# Add features by listing additional registry names or, for ad-hoc use,
# passing dict entries directly to build_state_config().
_DEFAULT_STATE_FEATURES = [
    "combined_nyc_storage_frac",
    "montague_flow_lag2",
    "trenton_flow_lag4",
    "nj_demand_lag4",
]

STATE_FEATURES = _parse_list_env("NYCOPT_STATE_FEATURES", _DEFAULT_STATE_FEATURES)

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

# Fixed MRF targets (MGD) used by fixed-target objective variants.
# These are the FFMP baseline release floors at Montague and Trenton.
# Fixed targets are preferred for cross-architecture comparison because
# they do not depend on the live FFMP step-down logic.
MONTAGUE_FIXED_TARGET_MGD = 1131.05
TRENTON_FIXED_TARGET_MGD = 1938.95


###############################################################################
# Active Objectives
###############################################################################
# User-facing list of objective names. See src.objectives.OBJECTIVES for
# the full registry; call src.objectives.list_available_objectives() to print.
#
# Guideline: for cross-architecture publication runs, prefer the _fixed
# flow-compliance variants so that every architecture is scored against
# the same Montague/Trenton targets.

_DEFAULT_OBJECTIVES = [
    "nyc_reliability_weekly",
    "nyc_vulnerability",
    "nj_reliability_weekly",
    "montague_reliability_weekly_fixed",
    "trenton_reliability_weekly_fixed",
    "flood_risk_downstream_flow_days",
    "storage_min_combined_pct",
]

ACTIVE_OBJECTIVES = _parse_list_env("NYCOPT_OBJECTIVES", _DEFAULT_OBJECTIVES)


###############################################################################
# Variable-Resolution FFMP Sweep
###############################################################################

# Values of N (storage zone boundary curves) to sweep for the complexity
# frontier experiment. Each value maps to formulation "ffmp_{N}".
#
# NOTE on baseline equivalence: generate_ffmp_formulation(n_zones=6) produces
# the 7-level zone count of the standard FFMP; values ≥ 6 are higher-resolution
# variants. The user requirement is N ≥ baseline, so this sweep starts at 6.
FFMP_VR_N_SWEEP = _parse_int_list_env("NYCOPT_FFMP_VR_N", [6, 8, 10, 12])


###############################################################################
# Manuscript Production Defaults
###############################################################################
# Default formulation set for the manuscript experiment: structure-preserving
# FFMP -> resolution-extended FFMP_VR -> structure-free ANN. This is the
# *default*, not a lock-in: RBF, Tree, Spline remain fully functional and
# can be added back via NYCOPT_FORMULATIONS or by editing the env file
# under slurm/envs/. See local_notes/decisions/2026-04-29_manuscript_scope.md.

_DEFAULT_PRODUCTION_FORMULATIONS = (
    ["ffmp"]
    + [f"ffmp_{n}" for n in FFMP_VR_N_SWEEP]
    + ["ann"]
)
PRODUCTION_FORMULATIONS = _parse_list_env(
    "NYCOPT_FORMULATIONS", _DEFAULT_PRODUCTION_FORMULATIONS,
)


###############################################################################
# Temperature & Salinity LSTM Coupling
###############################################################################
# When enabled, the LSTMs (from PywrDRB-ML) run as pywrdrb Parameters during
# simulation. Both default off; salinity is the manuscript-active path
# (temperature is deferred — see decisions/2026-04-29_temperature_lstm_deferred.md).
#
# `NYCOPT_TS_ON` is a legacy convenience that turns on whichever LSTM is
# considered active for the manuscript (currently: salinity only). New
# scripts should prefer `NYCOPT_SALINITY_ON` / `NYCOPT_TEMPERATURE_ON` for
# clarity.
#
# See:
#   local_notes/methodology/temperature_salinity.md
#   local_notes/decisions/2026-04-29_ts_observe_only.md
#   local_notes/decisions/2026-04-29_temperature_lstm_deferred.md

_TS_ON_LEGACY = _parse_bool_env("NYCOPT_TS_ON", False)
INCLUDE_TEMPERATURE_MODEL = _parse_bool_env("NYCOPT_TEMPERATURE_ON", False)
INCLUDE_SALINITY_MODEL = _parse_bool_env("NYCOPT_SALINITY_ON", _TS_ON_LEGACY)

# LSTM training data window starts in 1979; pre-1979 simulation days are
# returned as NaN by the LSTM parameters and must be dropped at objective
# computation time. Set this to the earliest date the LSTMs trust.
LSTM_START_DATE = _parse_str_env("NYCOPT_LSTM_START_DATE", "1979-01-01")

# Threshold above which Lordville thermal exceedance days are counted.
# 23.89 °C (75 °F) is the DRBC cold-water-fish thermal stress threshold.
# (Inactive while INCLUDE_TEMPERATURE_MODEL=False.)
LORDVILLE_THERMAL_THRESHOLD_C = _parse_float_env(
    "NYCOPT_THERMAL_THRESHOLD_C", 23.89,
)

# Reference river mile for salt-front excursion. RM 92.47 is the DRBC
# Trenton salinity standard reference; lower RM means salt has moved
# farther upstream (worse for water supply at Trenton).
SALT_FRONT_REFERENCE_RM = _parse_float_env(
    "NYCOPT_SALT_FRONT_RM", 92.47,
)

# Salinity coupling mode. When True (default), the salinity LSTM runs in
# asynchronous mode and does NOT rewrite mrf_target_{Montague,Trenton} —
# strictly observational. Set False only to *intentionally* let the salinity
# LSTM feed back into MRF targets. Doing so changes the meaning of the
# Montague/Trenton dynamic-target objectives. Note: in either mode the LSTM
# still updates per-timestep and publishes salt_front_location_mu — only the
# MRF-rewrite side effect is gated.
SALINITY_ASYNC_UPDATE = _parse_bool_env("NYCOPT_SALINITY_ASYNC", True)

# Extend RESULTS_SETS so pywrdrb.Data().load_output() pulls the LSTM
# parameter outputs out of the HDF5 (or in-memory recorder). 'salinity'
# yields columns including 'salt_front_location_mu'; 'temperature' yields
# 'temperature_after_thermal_release_mu'. Defined as side effect so the
# constant doesn't change shape based on env at module import; instead the
# code that uses RESULTS_SETS reads from this single source of truth.
if INCLUDE_SALINITY_MODEL and "salinity" not in RESULTS_SETS:
    RESULTS_SETS = list(RESULTS_SETS) + ["salinity"]
if INCLUDE_TEMPERATURE_MODEL and "temperature" not in RESULTS_SETS:
    RESULTS_SETS = list(RESULTS_SETS) + ["temperature"]


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
# Centralized knobs for the post-optimization re-simulation step. The MPI
# variant (Phase 1+) reads cluster sizing from these values, so a single
# env-file edit reshapes the SLURM submission.

# Number of stochastic realizations per solution. 1 = deterministic single-trace
# re-eval (Phase 1 default). Phase 3 raises this for ensemble robustness.
REEVAL_REALIZATIONS = _parse_int_env("NYCOPT_REEVAL_N", 1)

# Re-eval execution mode: "mpi" uses src/reevaluate_mpi.py (multi-node);
# "single" falls back to src/reevaluate.py (multiprocessing.Pool).
REEVAL_MODE = _parse_str_env("NYCOPT_REEVAL_MODE", "single")

# Cluster sizing for MPI re-eval. Sourced into SLURM SBATCH directives.
REEVAL_NODES = _parse_int_env("NYCOPT_REEVAL_NODES", 4)
REEVAL_RANKS_PER_NODE = _parse_int_env("NYCOPT_REEVAL_RANKS", 16)

REEVALUATION_SETTINGS = {
    "n_realizations": REEVAL_REALIZATIONS,
    "realization_length_years": 70,
    "generator": "kirsch_nowak",
    "climate_scenarios": ["stationary"],
    "robustness_metrics": ["satisficing", "regret"],
}


###############################################################################
# Cluster Target
###############################################################################
# Selects the SLURM template family (Anvil for production, Hopper for smoke
# tests). _common.sh and workflow/05_reevaluate.sh consult this value when
# they need cluster-specific defaults (MCA flags, node sizing).

CLUSTER = _parse_str_env("NYCOPT_CLUSTER", "hopper")


###############################################################################
# Slug Naming Convention
###############################################################################
# Slugs identify a methodologic configuration so outputs from different
# configs never collide. Format:
#   {formulation}_obj{N_OBJ}{ts_suffix}{state_suffix}{custom_suffix}
#
# Examples:
#   ffmp_obj7                    — current production baseline (no T/S)
#   ffmp_obj9_ts                 — production after T/S lands
#   ffmp_6_obj9_ts               — variable-resolution N=6 with T/S
#   ann_obj9_ts_state4           — ANN with reduced 4-feature state
#   ffmp_obj9_ts_pilot42         — ad-hoc tagged run (RUN_SLUG_TAG=pilot42)
#
# `RUN_SLUG_TAG` env appends a free-form suffix; useful for one-off variants
# without polluting the canonical slug grammar.
# A non-empty `RUN_SLUG` env wins outright (escape hatch for legacy paths).

def derive_slug(formulation: str, *, custom_tag: str | None = None) -> str:
    """Derive a slug from active config + a formulation name.

    Suffix grammar (LSTM portion):
      - both temperature + salinity on  -> "_ts"
      - salinity only                    -> "_sal"
      - temperature only                 -> "_temp"
      - neither                          -> (omitted)

    Args:
        formulation: e.g. "ffmp", "ffmp_6", "ann".
        custom_tag: appended after auto-derived components if non-empty.
            Falls back to the `RUN_SLUG_TAG` env var.

    Returns:
        Slug string used as the inner partition under outputs/{category}/
        and figures/{category}/.
    """
    explicit = os.environ.get("RUN_SLUG", "").strip()
    if explicit:
        return explicit

    parts = [formulation, f"obj{len(ACTIVE_OBJECTIVES)}"]
    if INCLUDE_TEMPERATURE_MODEL and INCLUDE_SALINITY_MODEL:
        parts.append("ts")
    elif INCLUDE_SALINITY_MODEL:
        parts.append("sal")
    elif INCLUDE_TEMPERATURE_MODEL:
        parts.append("temp")
    if len(STATE_FEATURES) != len(_DEFAULT_STATE_FEATURES):
        parts.append(f"state{len(STATE_FEATURES)}")

    tag = custom_tag if custom_tag else os.environ.get("RUN_SLUG_TAG", "").strip()
    if tag:
        parts.append(tag)

    return "_".join(parts)


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
    print(f"Active objectives ({obj_set.n_objs}): {ACTIVE_OBJECTIVES}")
    print(f"\nDecision Variables:")
    for name, spec in f["decision_variables"].items():
        print(f"  {name}: [{spec['bounds'][0]}, {spec['bounds'][1]}] "
              f"({spec['units']}) baseline={spec['baseline']}")
    print(f"\n{obj_set.summary()}")
    print(f"\nSimulation: {INFLOW_TYPE}, {START_DATE} to {END_DATE}")
    print(f"Trimmed model: {USE_TRIMMED_MODEL}")
    print(f"Borg NFE: {BORG_SETTINGS['max_evaluations']:,}")
    print(f"Seeds: {BORG_SETTINGS['n_seeds']}")
