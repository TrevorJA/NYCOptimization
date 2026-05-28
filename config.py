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
    NYCOPT_INFLOW_TYPE          -> INFLOW_TYPE (pywrdrb inflow-dataset key)
    NYCOPT_FORMULATIONS         -> PRODUCTION_FORMULATIONS (comma-separated)
    NYCOPT_FFMP_VR_N            -> FFMP_VR_N_SWEEP (comma-separated ints)
    NYCOPT_TEMPERATURE_ON       -> INCLUDE_TEMPERATURE_MODEL (bool, default 0)
    NYCOPT_SALINITY_ON          -> INCLUDE_SALINITY_MODEL    (bool, default 0)
    NYCOPT_TS_ON                -> shortcut: sets both above (legacy convenience)
    NYCOPT_THERMAL_THRESHOLD_C  -> LORDVILLE_THERMAL_THRESHOLD_C (float)
    NYCOPT_SALT_FRONT_RM        -> SALT_FRONT_REFERENCE_RM (float)
    NYCOPT_SALINITY_ASYNC       -> SALINITY_ASYNC_UPDATE (bool)
    NYCOPT_LSTM_START_DATE      -> LSTM_START_DATE (earliest sim day the salinity LSTM updates; defaults to START_DATE since 2026-04-30)
    NYCOPT_REEVAL_N             -> REEVAL_REALIZATIONS (int)
    NYCOPT_REEVAL_NODES         -> REEVAL_NODES (int)
    NYCOPT_REEVAL_RANKS         -> REEVAL_RANKS_PER_NODE (int)
    NYCOPT_REEVAL_MODE          -> REEVAL_MODE ("mpi" | "single")
    NYCOPT_CLUSTER              -> CLUSTER ("anvil" | "hopper")
    NYCOPT_TEMPERATURE_LSTM_DIR -> TEMPERATURE_LSTM_DIR (path)
    NYCOPT_SALINITY_LSTM_DIR    -> SALINITY_LSTM_DIR (path)
    NYCOPT_ENSEMBLE_PRESET      -> SEARCH_ENSEMBLE_SPEC (name from src.ensembles.PRESETS)
    NYCOPT_REEVAL_ENSEMBLE_PRESET -> REEVAL_ENSEMBLE_SPEC (name from src.ensembles.PRESETS)
    NYCOPT_ENSEMBLE_INDICES     -> overrides realization_indices on SEARCH_ENSEMBLE_SPEC
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

# Staged synthetic-ensemble inputs. Each ensemble preset (e.g. wcu_kirsch_n5)
# stages two HDF5 files under STAGED_ENSEMBLE_DIR/{inflow_type}/:
#   catchment_inflow_mgd.hdf5   - per-realization inflows (FlowEnsemble)
#   predicted_inflows_mgd.hdf5  - per-realization Montague/Trenton lag forecasts
#                                 (PredictionEnsemble; required when running with
#                                 inflow_ensemble_indices)
# Generated by `scripts/build_ensemble.py`; gitignored at the per-file level.
# Registered with pywrdrb's path navigator at simulation start (see
# src/ensembles.py::register_ensemble_path).
STAGED_ENSEMBLE_DIR = OUTPUTS_DIR / "synthetic_ensembles"

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

# Default inflow source = Amestoy et al. (2026) Bayesian-bias-corrected
# reconstructed DRB streamflow ensemble (1945-2023; Environmental Modelling
# & Software, 195, 106756). The key "pub_nhmv10_BC_withObsScaled" is the
# median realization of that ensemble and ships pre-packaged with pywrdrb's
# Data() loader. See local_notes/decisions/2026-04-30_inflow_and_du_search.md.
# Override per-experiment via NYCOPT_INFLOW_TYPE in slurm/envs/*.env.
INFLOW_TYPE = os.environ.get("NYCOPT_INFLOW_TYPE", "pub_nhmv10_BC_withObsScaled")
USE_TRIMMED_MODEL = True
INITIAL_VOLUME_FRAC = 0.80

# NYC and NJ interbasin diversion demand mode passed to pywrdrb.ModelBuilder.
# Default = constant_max so every candidate policy is stressed with the same
# decree-maximum demand profile (NYC=800 MGD, NJ=100 MGD monthly avg, defined
# in pywrdrb constants.csv). Set to "historical" to use the extrapolated
# historical time series instead — useful for sensitivity studies.
# "custom" is also accepted by pywrdrb but not wired into NYCOpt experiments.
_NYC_NJ_DEMAND_SOURCE_MODES = ("constant_max", "historical", "custom")
NYC_NJ_DEMAND_SOURCE = _parse_str_env(
    "NYCOPT_NYC_NJ_DEMAND_SOURCE", "constant_max"
).lower()
if NYC_NJ_DEMAND_SOURCE not in _NYC_NJ_DEMAND_SOURCE_MODES:
    raise ValueError(
        f"Invalid NYC_NJ_DEMAND_SOURCE='{NYC_NJ_DEMAND_SOURCE}'; "
        f"expected one of {_NYC_NJ_DEMAND_SOURCE_MODES}"
    )

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
    "flood_stage",
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

# 1954 Supreme Court Decree quantities used by NYC and Montague objectives.
# These static values are the goalposts the optimizer is scored against.
# They DO NOT depend on the live FFMP step-down logic — scoring against the
# time-varying FFMP target would create a perverse incentive where a policy
# could "succeed" by triggering drought step-downs that lower its own target.
NYC_DECREE_DIVERSION_CAP_MGD = 800.0     # NYC right under 1954 Decree
MONTAGUE_DECREE_TARGET_MGD = 1131.05     # = 1750 cfs, 1954 Supreme Court Decree


###############################################################################
# Active Objectives
###############################################################################
# User-facing list of objective names. See src.objectives.OBJECTIVES for
# the full registry; call src.objectives.list_available_objectives() to print.
#
# The default 7-objective set pairs NYC and Montague compliance against the
# 1954 Supreme Court Decree quantities (mirrored reliability + max-deficit
# pair for each), plus three single-axis metrics: salt-front intrusion,
# downstream flood days, and storage resilience.

_DEFAULT_OBJECTIVES = [
    "nyc_reliability_weekly_decree",
    "nyc_max_deficit_weekly_decree",
    "montague_reliability_weekly_decree",
    "montague_max_deficit_weekly_decree",
    "salt_front_max_rm",
    "flood_days_downstream_action_anygauge",
    "storage_min_combined_pct",
]

ACTIVE_OBJECTIVES = _parse_list_env("NYCOPT_OBJECTIVES", _DEFAULT_OBJECTIVES)


###############################################################################
# Variable-Resolution FFMP Sweep
###############################################################################

# Values of N (storage zone boundary curves) to sweep for the complexity
# frontier experiment. Each value maps to formulation "ffmp_{N}".
#
# NOTE on baseline equivalence: generate_ffmp_formulation(n_zones=6) reproduces
# the standard FFMP's 7-level zone count and the same 24-DV layout, so
# `ffmp_6` is operationally identical to `ffmp` and is omitted from the sweep
# to avoid a redundant 10-seed slot. The sweep starts at N=8 (one resolution
# step above baseline). Re-include 6 via NYCOPT_FFMP_VR_N if intentionally
# studying baseline equivalence.
FFMP_VR_N_SWEEP = _parse_int_list_env("NYCOPT_FFMP_VR_N", [8, 10, 12])


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

# Earliest date at which the salinity LSTM is allowed to begin updating.
# After the SalinityLSTM database/NPZ were extended to 1945-01-01 (matching
# the Amestoy et al. (2026) reconstructed inflow record), this defaults to
# the simulation START_DATE so the LSTM runs over the full optimization
# window. The override knob remains for experiments that want to clip the
# salinity sub-period (e.g. evaluate post-1979-only behavior).
LSTM_START_DATE = _parse_str_env("NYCOPT_LSTM_START_DATE", START_DATE)

# Threshold above which Lordville thermal exceedance days are counted.
# 23.89 °C (75 °F) is the DRBC cold-water-fish thermal stress threshold.
# (Inactive while INCLUDE_TEMPERATURE_MODEL=False.)
LORDVILLE_THERMAL_THRESHOLD_C = _parse_float_env(
    "NYCOPT_THERMAL_THRESHOLD_C", 23.89,
)

# DRBC Trenton salinity-standard reference river mile. Used as a plotting
# reference and for narrative framing; the salt-front objective itself is
# the absolute max-upstream RM and does not subtract this constant.
# Convention: river miles increase upstream from the bay mouth, so HIGHER
# RM = salt front intruded farther upstream = worse for water supply.
SALT_FRONT_REFERENCE_RM = _parse_float_env(
    "NYCOPT_SALT_FRONT_RM", 92.47,
)

# Salinity coupling mode. **Default False (sync).**
# Why sync, not async: in async mode the LSTM does NOT advance its internal
# time index `ml_model.t` during pywrdrb's run loop, so every sim-day's flow
# overwrites `X[0, :]` and the LSTM forward pass over the full window is
# dominated by historical training data. In sync mode the LSTM advances 1
# step per sim day and produces a per-day sf_mu series that is genuinely
# responsive to NYC operational decisions — verified by the random-sample
# diagnostics 2026-04-29.
#
# Side effect of sync mode: when the system enters NYC drought emergency
# (drought_level_agg_nyc_idx == n_drought_levels - 1), the salinity LSTM
# rewrites mrf_target_{delMontague,delTrenton} via FlowTargetSaltFrontAdj-
# ustmentRatio. This is internal model state; the default Montague flow
# objectives score against the *static* 1954 Decree value
# (MONTAGUE_DECREE_TARGET_MGD), not against the live mrf_target, so they
# are unaffected by this rewrite. Outside drought emergency, the
# adjustment ratio is 1.0 (no-op).
#
# Set True only to *intentionally* disable the LSTM's responsiveness to
# simulated flows (e.g., to study LSTM sensitivity in isolation).
SALINITY_ASYNC_UPDATE = _parse_bool_env("NYCOPT_SALINITY_ASYNC", False)

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
# Salt-front MRF adjustment parameterization (FFMP-family DVs)
###############################################################################
# The salinity LSTM, when enabled in sync mode, drives a salt-front-based
# adjustment of `mrf_target_{delMontague,delTrenton}` during NYC drought
# emergency. The adjustment is a lookup table indexed by (RM band, season).
# By default the table is fixed at the FFMP-Appendix-A values. Setting
# `NYCOPT_SALT_FRONT_PARAM_MODE` to one of the modes below exposes parts
# of that table as decision variables for FFMP-family formulations only
# (RBF/Tree/ANN/Spline are unaffected — they don't use FFMP drought levels
# so the adjustment never fires for them).
#
# Modes (configurable subset of the full operational table):
#   "fixed"               -> 0 new DVs (default; behavior identical to today)
#   "multipliers"         -> 15 multiplier cells (5 reference cells pinned 1.0)
#   "multipliers_with_gate" -> +1 activation drought-level DV (16 total)
#   "full"                -> +3 RM-band threshold DVs (19 total)
#
# See:
#   local_notes/decisions/2026-04-29_salt_front_parameterization.md
#   local_notes/methodology/salt_front_adjustment_dvs.md

_SALT_FRONT_PARAM_MODES = ("fixed", "multipliers", "multipliers_with_gate", "full")
SALT_FRONT_PARAM_MODE = _parse_str_env("NYCOPT_SALT_FRONT_PARAM_MODE", "fixed").lower()
if SALT_FRONT_PARAM_MODE not in _SALT_FRONT_PARAM_MODES:
    raise ValueError(
        f"Invalid SALT_FRONT_PARAM_MODE='{SALT_FRONT_PARAM_MODE}'; "
        f"expected one of {_SALT_FRONT_PARAM_MODES}"
    )

# DV bounds for multiplier cells. The FFMP-default values currently span
# ~0.69–1.19, so a [0.5, 1.5] window gives meaningful exploration room.
_SALT_FRONT_MULT_BOUNDS_RAW = _parse_str_env(
    "NYCOPT_SALT_FRONT_MULTIPLIER_BOUNDS", "0.5,1.5"
)
SALT_FRONT_MULTIPLIER_BOUNDS = tuple(
    float(x) for x in _SALT_FRONT_MULT_BOUNDS_RAW.split(",")
)
if len(SALT_FRONT_MULTIPLIER_BOUNDS) != 2 or SALT_FRONT_MULTIPLIER_BOUNDS[0] >= SALT_FRONT_MULTIPLIER_BOUNDS[1]:
    raise ValueError(
        f"Invalid SALT_FRONT_MULTIPLIER_BOUNDS={SALT_FRONT_MULTIPLIER_BOUNDS}; "
        "expected 'lo,hi' with lo<hi"
    )

# RM-band thresholds (lo, mid, hi). Defaults are the DRBC §2.5.3 operational
# triggers (82.9, 87.0, 92.5). Per-threshold bounds prevent the optimizer
# from violating physical ordering. Encoded as 3 (lo, hi) tuples in lo->hi
# order. Stored as flat string for env override; parsed below.
_SALT_FRONT_RM_BOUNDS_RAW = _parse_str_env(
    "NYCOPT_SALT_FRONT_RM_BAND_BOUNDS",
    # lo (82.9 default): allow [76, 86]
    # mid (87.0 default): allow [84, 90]
    # hi (92.5 default): allow [89, 95]
    "76,86;84,90;89,95",
)
def _parse_rm_band_bounds(s: str) -> list[tuple[float, float]]:
    out = []
    for chunk in s.split(";"):
        parts = chunk.strip().split(",")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid RM band bound '{chunk}'; expected 'lo,hi'"
            )
        lo, hi = float(parts[0]), float(parts[1])
        if lo >= hi:
            raise ValueError(f"RM band bound has lo>=hi: {chunk}")
        out.append((lo, hi))
    return out

SALT_FRONT_RM_BAND_BOUNDS = _parse_rm_band_bounds(_SALT_FRONT_RM_BOUNDS_RAW)
if len(SALT_FRONT_RM_BAND_BOUNDS) != 3:
    raise ValueError(
        f"SALT_FRONT_RM_BAND_BOUNDS must have 3 entries (lo, mid, hi); got {SALT_FRONT_RM_BAND_BOUNDS}"
    )

# Allowed activation drought levels when activation is parameterized.
# In stock FFMP (7-level config), L3=index 4, L4=index 5, L5=index 6.
# We expose the high-end levels because earlier activation would dramatically
# change behavior; downstream applications can override via env.
SALT_FRONT_ACTIVATION_LEVEL_OPTIONS = _parse_int_list_env(
    "NYCOPT_SALT_FRONT_ACTIVATION_LEVELS", [4, 5, 6]
)

# When activation is NOT a DV, this fixed level fires the rule. Default 6
# (= L5 / Drought Emergency) matches FFMP. For N-zone configs this should
# normally be n_drought_levels - 1; the simulation layer resolves that.
SALT_FRONT_FIXED_ACTIVATION_LEVEL = _parse_int_env(
    "NYCOPT_SALT_FRONT_FIXED_ACTIVATION_LEVEL", 6
)


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
# Ensemble Evaluation
###############################################################################
# The MOEA evaluator can run a candidate policy on a single historic streamflow
# trace (legacy default) or on a multi-realization ensemble. Both modes are
# represented as ``EnsembleSpec`` instances from src.ensembles; the legacy
# default is the ``historic_single`` preset, which preserves byte-identical
# behavior of all pre-existing runs.
#
# Two specs are resolved at import time:
#   SEARCH_ENSEMBLE_SPEC  - used inside Borg's evaluate() during optimization.
#   REEVAL_ENSEMBLE_SPEC  - used by src/reevaluate.py + reevaluate_mpi.py.
#
# When SEARCH_ENSEMBLE_SPEC.is_ensemble is True we emit a slug fragment so
# outputs land under a distinct directory; the historic_single preset emits
# no fragment to keep legacy paths unchanged. Re-eval output is partitioned
# further by the re-eval preset name (see workflow/05_reevaluate.sh).

from src.ensembles import (             # noqa: E402
    get_ensemble_spec,
    with_indices_override,
)

NYCOPT_ENSEMBLE_PRESET = _parse_str_env("NYCOPT_ENSEMBLE_PRESET", "historic_single")
NYCOPT_REEVAL_ENSEMBLE_PRESET = _parse_str_env(
    "NYCOPT_REEVAL_ENSEMBLE_PRESET", "historic_single",
)

SEARCH_ENSEMBLE_SPEC = get_ensemble_spec(NYCOPT_ENSEMBLE_PRESET)
REEVAL_ENSEMBLE_SPEC = get_ensemble_spec(NYCOPT_REEVAL_ENSEMBLE_PRESET)

# Optional realization-index override on the search preset. Useful for smoke
# testing a subset of an N=300 ensemble without authoring a new preset.
_ensemble_indices_override = _parse_int_list_env("NYCOPT_ENSEMBLE_INDICES", [])
if _ensemble_indices_override:
    SEARCH_ENSEMBLE_SPEC = with_indices_override(
        SEARCH_ENSEMBLE_SPEC, _ensemble_indices_override,
    )

# Selection-bias guard (Bonham 2024): re-eval ensemble should be independent
# of the search ensemble. Warn loudly if a user accidentally points both at
# the same preset and the spec is actually an ensemble (single-trace re-eval
# with single-trace search is the legacy case and need not warn).
if (
    SEARCH_ENSEMBLE_SPEC.is_ensemble
    and NYCOPT_REEVAL_ENSEMBLE_PRESET == NYCOPT_ENSEMBLE_PRESET
):
    print(
        f"  [config] WARN: search and re-eval ensemble preset are identical "
        f"('{NYCOPT_ENSEMBLE_PRESET}'). This is a selection-bias risk per "
        f"Bonham (2024) — prefer an independent re-eval preset (different "
        f"seed and/or realization indices)."
    )


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

    Ensemble portion (after sfdv, before state-feature suffix):
      - SEARCH_ENSEMBLE_SPEC.slug_fragment when non-empty (e.g. "wcu5").
        ``historic_single`` ships an empty fragment to preserve legacy slugs.

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
    # Salt-front DV mode (only meaningful when salinity is on AND formulation
    # is FFMP-family; non-FFMP runs ignore the mode but the suffix still
    # captures the campaign intent).
    _sfdv_suffix = {
        "multipliers":           "sfdv_mult",
        "multipliers_with_gate": "sfdv_multgate",
        "full":                  "sfdv_full",
    }.get(SALT_FRONT_PARAM_MODE, "")
    if _sfdv_suffix:
        parts.append(_sfdv_suffix)
    if SEARCH_ENSEMBLE_SPEC.slug_fragment:
        parts.append(SEARCH_ENSEMBLE_SPEC.slug_fragment)
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
    print(
        f"\nSearch ensemble: preset='{SEARCH_ENSEMBLE_SPEC.preset_name}', "
        f"is_ensemble={SEARCH_ENSEMBLE_SPEC.is_ensemble}, "
        f"N={SEARCH_ENSEMBLE_SPEC.n_realizations}, "
        f"inflow_type='{SEARCH_ENSEMBLE_SPEC.inflow_type}'"
    )
    print(
        f"Re-eval ensemble: preset='{REEVAL_ENSEMBLE_SPEC.preset_name}', "
        f"is_ensemble={REEVAL_ENSEMBLE_SPEC.is_ensemble}, "
        f"N={REEVAL_ENSEMBLE_SPEC.n_realizations}, "
        f"inflow_type='{REEVAL_ENSEMBLE_SPEC.inflow_type}'"
    )
