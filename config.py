"""
config.py - Central configuration for NYCOptimization study.

Single source of truth for paths, simulation settings, NYC system constants,
Borg MOEA parameters, T/S coupling, re-evaluation sizing, ensemble selection,
and the slug naming convention. Problem formulation logic lives in
src/formulations/ (FFMP + variable-resolution FFMP).

Every methodologic knob has a default constant here and a `NYCOPT_*` env
override read at import time. SLURM scripts source per-experiment env files
under workflow/envs/ to set these without relying on remembered CLI flags.

Environment overrides (selected):
    NYCOPT_OBJECTIVES           -> ACTIVE_OBJECTIVES (comma-separated names)
    NYCOPT_INFLOW_TYPE          -> INFLOW_TYPE (pywrdrb inflow-dataset key)
    NYCOPT_FORMULATIONS         -> PRODUCTION_FORMULATIONS (comma-separated)
    NYCOPT_FFMP_VR_N            -> FFMP_VR_N_SWEEP (comma-separated ints)
    NYCOPT_TEMPERATURE_ON       -> INCLUDE_TEMPERATURE_MODEL (bool, default 0)
    NYCOPT_SALINITY_ON          -> INCLUDE_SALINITY_MODEL    (bool, default 0)
    NYCOPT_TS_ON                -> shortcut: sets both above (legacy convenience)
    NYCOPT_THERMAL_THRESHOLD_C  -> LORDVILLE_THERMAL_THRESHOLD_C (float)
    NYCOPT_SALT_FRONT_RM        -> SALT_FRONT_REFERENCE_RM (float)
    NYCOPT_SALINITY_ASYNC       -> SALINITY_ASYNC_UPDATE (bool)
    NYCOPT_LSTM_START_DATE      -> LSTM_START_DATE
    NYCOPT_REEVAL_N             -> REEVAL_REALIZATIONS (int)
    NYCOPT_REEVAL_NODES         -> REEVAL_NODES (int)
    NYCOPT_REEVAL_RANKS         -> REEVAL_RANKS_PER_NODE (int)
    NYCOPT_REEVAL_MODE          -> REEVAL_MODE ("mpi" | "single")
    NYCOPT_CLUSTER              -> CLUSTER ("anvil" | "hopper")
    NYCOPT_TEMPERATURE_LSTM_DIR -> TEMPERATURE_LSTM_DIR (path)
    NYCOPT_SALINITY_LSTM_DIR    -> SALINITY_LSTM_DIR (path)
    NYCOPT_SCENARIO_DESIGN      -> ACTIVE_SCENARIO_DESIGN (src.scenario_designs) -> SEARCH_ENSEMBLE_SPEC
    NYCOPT_MOEA_CONFIG          -> ACTIVE_MOEA_CONFIG (src.moea_config) -> BORG/MMBORG settings
    NYCOPT_REEVAL_ENSEMBLE_PRESET -> REEVAL_ENSEMBLE_SPEC (name from src.ensembles.PRESETS)
    NYCOPT_ENSEMBLE_INDICES     -> overrides realization_indices on SEARCH_ENSEMBLE_SPEC
    NYCOPT_ENSEMBLE_KN_YEARS    -> ENSEMBLE_KN_YEARS (Step 1 Kirsch-Nowak generator)
    NYCOPT_ENSEMBLE_KN_REALIZATIONS -> ENSEMBLE_KN_REALIZATIONS (Step 1 generator)
    NYCOPT_ENSEMBLE_KN_SEED     -> ENSEMBLE_KN_SEED (Step 1 generator)
    NYCOPT_ENSEMBLE_KN_FORCE    -> ENSEMBLE_KN_FORCE (overwrite existing staged ensemble)
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
# Generated by the workflow/02-04 ensemble pipeline (scripts/main/
# {generate_stochastic_ensemble,subsample_hazard_filling,prep_pywrdrb_inputs}.py);
# gitignored at the per-file level.
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
# Output tree (two-axis, hierarchical)
###############################################################################
# Run outputs are partitioned as outputs/{scenario}/{moea_slug}/{artifact}/,
# where {scenario} is ACTIVE_SCENARIO_DESIGN.name, {moea_slug} is derive_slug(),
# and {artifact} is the output type (sets, runtime, metrics, reeval,
# diagnostics, ...). Cross-design comparison reads across {scenario} dirs.
#
# A few non-run outputs keep a flat top-level home (manifests, presim).

OUTPUT_REFERENCE_SETS_DIR = OUTPUTS_DIR / "reference_sets"
OUTPUT_RUN_MANIFESTS_DIR = OUTPUTS_DIR / "run_manifests"
OUTPUT_BASELINE_DIR = OUTPUTS_DIR / "baseline"
# Ad-hoc diagnostics not tied to a single run (benchmarks, samplers). Per-run
# diagnostics use run_output_dir(scenario, slug, "diagnostics") instead.
OUTPUT_DIAGNOSTICS_DIR = OUTPUTS_DIR / "diagnostics"

FIG_EXPLORATORY_DIR = FIGURES_DIR / "_exploratory"


def run_output_dir(scenario: str, moea_slug: str, artifact: str) -> Path:
    """Return a run's artifact subdir, creating it if needed.

    Args:
        scenario: Scenario-design name (top-level partition); typically
            ``active_scenario_name()``.
        moea_slug: The moea slug from ``derive_slug()``.
        artifact: Output type, e.g. "sets", "runtime", "metrics", "reeval",
            "diagnostics", "checkpoints".

    Returns:
        ``outputs/{scenario}/{moea_slug}/{artifact}/`` (created).
    """
    p = OUTPUTS_DIR / scenario / moea_slug / artifact
    p.mkdir(parents=True, exist_ok=True)
    return p


def figure_dir_for(scenario: str, moea_slug: str, kind: str) -> Path:
    """Return a two-axis-partitioned figure subdir, creating it if needed.

    Args:
        scenario: Scenario-design name (top-level partition).
        moea_slug: The moea slug from ``derive_slug()``.
        kind: e.g. "convergence", "pareto", "parallel_coords",
            "policy_inspection", "robustness". Free-form names land under an
            ``_exploratory/`` subdir.

    Returns:
        ``figures/{scenario}/{moea_slug}/{kind}/`` (created), or the
        ``_exploratory`` variant for non-stable kinds.
    """
    stable = {"convergence", "pareto", "parallel_coords",
              "policy_inspection", "robustness"}
    if kind in stable:
        p = FIGURES_DIR / scenario / moea_slug / kind
    else:
        p = FIG_EXPLORATORY_DIR / scenario / moea_slug / kind
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
# Override per-experiment via NYCOPT_INFLOW_TYPE in workflow/envs/*.env.
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
# Stochastic Ensemble Generation (Step 1: Kirsch-Nowak)
###############################################################################
# Settings consumed by scripts/main/generate_stochastic_ensemble.py. Override
# per-ensemble via workflow/envs/ensemble_kn_*.env so a single shell submission
# fully specifies which ensemble is being built.
#
# These knobs do NOT affect optimization runs — they only describe Step 1
# generation. To use a *built* ensemble during optimization, point a scenario
# design (src/scenario_designs.py) at it via its ensemble_preset (e.g. a
# kn_{Y}yr_n{N} slug, resolved on the fly by src.ensembles.get_ensemble_spec);
# for re-eval set NYCOPT_REEVAL_ENSEMBLE_PRESET.

ENSEMBLE_KN_YEARS        = _parse_int_env("NYCOPT_ENSEMBLE_KN_YEARS", 50)
ENSEMBLE_KN_REALIZATIONS = _parse_int_env("NYCOPT_ENSEMBLE_KN_REALIZATIONS", 1000)
ENSEMBLE_KN_SEED         = _parse_int_env("NYCOPT_ENSEMBLE_KN_SEED", 42)
ENSEMBLE_KN_FORCE        = _parse_bool_env("NYCOPT_ENSEMBLE_KN_FORCE", False)


###############################################################################
# Forcing-based master ensemble (Step 1, forcing designs: methods §3.1-3.2)
###############################################################################
# Settings for the CMIP6-forced master ensemble that backs the hazard_filling
# and input_stratified designs (scengen.forcing_space + master_ensemble). The
# forcing designs read their sizes (N_forcing x realizations_per_profile, L)
# from src/scenario_designs.py; these knobs supply the shared forcing-space
# configuration and the streaming-storage mode. No CLI value flags — override
# via workflow/envs/*.env. The CMIP6 tables live in the sibling repo.

_CMIP6_STATS_DIR = PROJECT_DIR.parent / "CMIP6_multimodel_streamflow" / "stats"
ENSEMBLE_FORCING_MEAN_FRAC_CSV = _parse_path_env(
    "NYCOPT_ENSEMBLE_FORCING_MEAN_FRAC_CSV",
    _CMIP6_STATS_DIR / "diff_relative_to_dataset_baseline"
    / "nyc_inflow_monthly_mean_frac_by_dataset_ssp_and_period.csv",
)
ENSEMBLE_FORCING_MEAN_ABS_CSV = _parse_path_env(
    "NYCOPT_ENSEMBLE_FORCING_MEAN_ABS_CSV",
    _CMIP6_STATS_DIR / "datasets_nyc_inflow_monthly_means.csv",
)
ENSEMBLE_FORCING_STD_CSV = _parse_path_env(
    "NYCOPT_ENSEMBLE_FORCING_STD_CSV",
    _CMIP6_STATS_DIR / "datasets_nyc_inflow_monthly_stds.csv",
)
ENSEMBLE_FORCING_VARIANCE_AXIS = _parse_bool_env("NYCOPT_ENSEMBLE_FORCING_VARIANCE_AXIS", False)
ENSEMBLE_FORCING_BOUND_PCT = (
    _parse_float_env("NYCOPT_ENSEMBLE_FORCING_BOUND_LO", 5.0),
    _parse_float_env("NYCOPT_ENSEMBLE_FORCING_BOUND_HI", 95.0),
)
ENSEMBLE_FORCING_MARGIN = _parse_float_env("NYCOPT_ENSEMBLE_FORCING_MARGIN", 0.0)
ENSEMBLE_MASTER_SEED = _parse_int_env("NYCOPT_ENSEMBLE_MASTER_SEED", 0)
# stream_only discards daily traces after hazard computation (the ~1e6 production
# master); default False keeps the daily HDF5s so workflow/03 consumes M unchanged.
ENSEMBLE_MASTER_STREAM_ONLY = _parse_bool_env("NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY", False)
ENSEMBLE_MASTER_HAZARD_BLOCK = _parse_int_env("NYCOPT_ENSEMBLE_MASTER_HAZARD_BLOCK", 256)
# Chunk the stored daily master into contiguous chunks of this many realizations (must be a
# multiple of realizations_per_profile). 0 = single directory. Bounds generation write-memory and
# avoids a monolithic multi-hundred-GB HDF5 for a large master (methods §3.2).
ENSEMBLE_MASTER_CHUNK_SIZE = _parse_int_env("NYCOPT_ENSEMBLE_MASTER_CHUNK_SIZE", 0)


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
TRENTON_DECREE_TARGET_MGD = 1938.95      # Trenton-equiv. flow objective (mrf_baseline_delTrenton)
NJ_DELIVERY_CAP_MGD = 100.0              # NJ diversion baseline (monthly-avg D&R Canal right)


###############################################################################
# Active Objectives
###############################################################################
# User-facing list of objective names. See src.objectives.OBJECTIVES for
# the full registry; call src.objectives.list_available_objectives() to print.
#
# The default 7-objective set spans every stakeholder axis Pywr-DRB simulates:
#   - NYC supply: weekly delivery reliability + tail (CVaR90) delivery deficit
#   - Montague flow Decree (NYC's downstream obligation): reliability + CVaR90 deficit
#   - Trenton flow Decree (lower-basin / NJ obligation; also repels salinity): reliability
#   - downstream flood exposure: days any reservoir-tail gauge >= NWS minor flood stage
#   - storage resilience: 5th-percentile combined NYC storage
# Worst-case extremes (max-deficit, min-storage, salt-front-max) were replaced
# with stable tail/percentile/count forms (see docs/notes/methods/objective_definitions.md
# §1-2; Quinn et al. 2017; Bonham et al. 2024). The full registry in
# src.objectives.OBJECTIVES also exposes diagnostic variants (max-deficit,
# min-storage, flood-major/action, salt-front, NJ delivery) for easy swapping.

_DEFAULT_OBJECTIVES = [
    "nyc_delivery_reliability_weekly",
    "nyc_delivery_deficit_cvar90_pct",
    "montague_flow_reliability_weekly",
    "montague_flow_deficit_cvar90_pct",
    "trenton_flow_reliability_weekly",
    "downstream_flood_days_minor",
    "nyc_storage_p5_pct",
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
# Production Formulation Set
###############################################################################
# Default formulation set: base FFMP + variable-resolution FFMP at each N
# in FFMP_VR_N_SWEEP. Override via NYCOPT_FORMULATIONS or per-experiment
# env file under workflow/envs/.

_DEFAULT_PRODUCTION_FORMULATIONS = (
    ["ffmp"] + [f"ffmp_{n}" for n in FFMP_VR_N_SWEEP]
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
# of that table as decision variables.
#
# Modes (configurable subset of the full operational table):
#   "fixed"               -> 0 new DVs (default; behavior identical to today)
#   "multipliers"         -> 15 multiplier cells (5 reference cells pinned 1.0)
#   "multipliers_with_gate" -> +1 activation drought-level DV (16 total)
#   "full"                -> +3 RM-band threshold DVs (19 total)

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
# Borg MOEA Settings (algorithm axis)
###############################################################################
# Algorithm settings now live in a named, versioned registry
# (src/moea_config.py). The active config is selected by NYCOPT_MOEA_CONFIG and
# is the second axis (alongside the scenario design) that specifies a run. The
# legacy BORG_SETTINGS / MMBORG_SETTINGS dicts are kept as the public read
# surface for existing callers, but are now *projected* from ACTIVE_MOEA_CONFIG
# rather than hand-edited. Change algorithm settings by editing/selecting a
# MOEAConfig, not these dicts.

from src.moea_config import get_moea_config   # noqa: E402

NYCOPT_MOEA_CONFIG = _parse_str_env("NYCOPT_MOEA_CONFIG", "smoke")
ACTIVE_MOEA_CONFIG = get_moea_config(NYCOPT_MOEA_CONFIG)

# Slug grammar: the MOEA-config name is appended to the moea slug for every
# config except the production default, keeping production output paths clean
# while disambiguating dev/experimental algorithm variants.
_DEFAULT_MOEA_SLUG_CONFIG = "production"

BORG_SETTINGS = {
    "max_evaluations": ACTIVE_MOEA_CONFIG.max_evaluations,  # Per island
    "runtime_frequency": ACTIVE_MOEA_CONFIG.runtime_frequency,
    "n_seeds": ACTIVE_MOEA_CONFIG.n_seeds,
}

# Multi-Master Borg parallel configuration
MMBORG_SETTINGS = {
    "n_islands": ACTIVE_MOEA_CONFIG.n_islands,
    "n_workers_per_island": ACTIVE_MOEA_CONFIG.n_workers_per_island,
    "max_time_hours": ACTIVE_MOEA_CONFIG.max_time_hours,
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
    # Metric identifiers scored offline by src.robustness from the persisted raw
    # per-realization matrix. This is the only key consumed in code (the default
    # metric set for the manuscript; src.robustness --metrics overrides). The
    # realization count is REEVAL_REALIZATIONS above; the ensemble length /
    # generator / climate scenarios come from the resolved REEVAL_ENSEMBLE_SPEC,
    # not from static labels here.
    "robustness_metrics": [
        "satisficing_univariate",
        "satisficing_multivariate",
        "regret_from_best",
        "regret_from_baseline",
    ],
}


###############################################################################
# Cluster Target
###############################################################################
# Selects the SLURM template family (Anvil for production, Hopper for smoke
# tests). workflow/_common.sh and workflow/08_reevaluate.sh consult this value when
# they need cluster-specific defaults (MCA flags, node sizing).

CLUSTER = _parse_str_env("NYCOPT_CLUSTER", "hopper")


###############################################################################
# Scenario design + ensemble evaluation (scenario axis)
###############################################################################
# The scenario design is the first of the two run axes (the other is the MOEA
# config above). It names the construction recipe for the streamflow ensemble
# used during search — the methodological contribution of the study. Designs
# are registered in src/scenario_designs.py; the active one is selected by
# NYCOPT_SCENARIO_DESIGN and becomes the TOP level of the output tree:
#   outputs/{scenario}/{moea_slug}/...
#
# The MOEA evaluator runs a candidate policy on whatever ensemble the active
# scenario design resolves to (an ``EnsembleSpec`` from src.ensembles). The
# legacy single-trace default is the ``historic`` design (-> ``historic_single``
# preset), which preserves byte-identical behavior of all pre-existing runs.
#
# Two specs are resolved at import time:
#   SEARCH_ENSEMBLE_SPEC  - from ACTIVE_SCENARIO_DESIGN; used inside Borg's
#                           evaluate() during optimization.
#   REEVAL_ENSEMBLE_SPEC  - the held-out test ensemble; used by
#                           src/reevaluate.py + reevaluate_mpi.py. Still selected
#                           directly by preset name for now (its design is an
#                           open decision — see experimental_design.md #3).

from src.ensembles import (             # noqa: E402
    get_ensemble_spec,
    with_indices_override,
)
from src.scenario_designs import get_scenario_design   # noqa: E402

NYCOPT_SCENARIO_DESIGN = _parse_str_env("NYCOPT_SCENARIO_DESIGN", "historic")
ACTIVE_SCENARIO_DESIGN = get_scenario_design(NYCOPT_SCENARIO_DESIGN)

NYCOPT_REEVAL_ENSEMBLE_PRESET = _parse_str_env(
    "NYCOPT_REEVAL_ENSEMBLE_PRESET", "historic_single",
)

# Resolve the search ensemble. Designs whose construction is not yet wired
# (input_stratified / hazard_filling) leave SEARCH_ENSEMBLE_SPEC None
# so config stays importable — diagnostics/reeval/plotting on such a design's
# outputs only need active_scenario_name(). Optimization fails fast with a clear
# message (see src/mmborg.py) when the spec is None.
try:
    SEARCH_ENSEMBLE_SPEC = ACTIVE_SCENARIO_DESIGN.resolve_search_spec()
except NotImplementedError as _e:
    SEARCH_ENSEMBLE_SPEC = None
    print(
        f"  [config] NOTE: scenario design '{ACTIVE_SCENARIO_DESIGN.name}' has "
        f"no search ensemble wired yet; optimization is unavailable for it "
        f"(diagnostics/reeval/plotting on its outputs still work). {_e}"
    )
REEVAL_ENSEMBLE_SPEC = get_ensemble_spec(NYCOPT_REEVAL_ENSEMBLE_PRESET)

# Realizations simulated per Pywr model build inside Borg's evaluate() ensemble
# path (src/simulation.py::run_simulation_ensemble_batched). 0 (default) keeps
# the legacy single-model behavior — all N realizations as one scenario block,
# byte-identical to prior runs. A positive value bounds peak memory per
# evaluation by simulating the ensemble in sequential batches of this size and
# reducing each realization to its per-objective base metric before the next
# batch (the same memory-batching the ensemble objective-sensitivity diagnostic
# uses, so search and diagnostic handle realizations identically). Recommended
# for large search ensembles (e.g. N>=128) to avoid OOM on a Borg worker.
SEARCH_REALIZATION_BATCH = _parse_int_env("NYCOPT_SEARCH_REALIZATION_BATCH", 0)

# Optional realization-index override on the search ensemble. Useful for smoke
# testing a subset of a large ensemble without authoring a new preset.
_ensemble_indices_override = _parse_int_list_env("NYCOPT_ENSEMBLE_INDICES", [])
if _ensemble_indices_override and SEARCH_ENSEMBLE_SPEC is not None:
    SEARCH_ENSEMBLE_SPEC = with_indices_override(
        SEARCH_ENSEMBLE_SPEC, _ensemble_indices_override,
    )

# Selection-bias guard (Bonham 2024): re-eval ensemble should be independent
# of the search ensemble. Warn loudly if both resolve to the same preset and
# the search spec is actually an ensemble (single-trace re-eval with
# single-trace search is the legacy case and need not warn).
if (
    SEARCH_ENSEMBLE_SPEC is not None
    and SEARCH_ENSEMBLE_SPEC.is_ensemble
    and NYCOPT_REEVAL_ENSEMBLE_PRESET == SEARCH_ENSEMBLE_SPEC.preset_name
):
    print(
        f"  [config] WARN: search and re-eval ensemble resolve to the same "
        f"preset ('{SEARCH_ENSEMBLE_SPEC.preset_name}'). This is a "
        f"selection-bias risk per Bonham (2024) — prefer an independent "
        f"re-eval preset (different seed and/or realization indices)."
    )


###############################################################################
# Slug Naming Convention
###############################################################################
# A run is partitioned on two axes: the scenario design (top-level output dir,
# from ACTIVE_SCENARIO_DESIGN.name) and the moea slug below. derive_slug()
# builds the moea slug — the problem-definition identity plus the non-default
# algorithm-config name. The ensemble is NOT in the slug; it is the parent
# {scenario} directory. Format:
#   {formulation}_obj{N_OBJ}{ts_suffix}{sfdv_suffix}{moea_cfg_suffix}{custom_suffix}
#
# Full output path: outputs/{scenario}/{moea_slug}/{artifact}/
#
# Examples (moea slug only):
#   ffmp_obj7                    — FFMP, 7 objectives, production algo config
#   ffmp_obj7_sal                — salinity LSTM on
#   ffmp_8_obj7_sal              — variable-resolution N=8 with salinity
#   ffmp_obj7_sal_smoke          — dev smoke algorithm config
#   ffmp_obj7_sal_pilot42        — ad-hoc tagged run (RUN_SLUG_TAG=pilot42)
#
# `RUN_SLUG_TAG` env appends a free-form suffix; useful for one-off variants
# without polluting the canonical slug grammar.
# A non-empty `RUN_SLUG` env wins outright (escape hatch for legacy paths).

def active_scenario_name() -> str:
    """Return the active scenario-design name (top-level output partition)."""
    return ACTIVE_SCENARIO_DESIGN.name


def derive_slug(formulation: str, *, custom_tag: str | None = None) -> str:
    """Derive the moea slug from active config + a formulation name.

    Suffix grammar (LSTM portion):
      - both temperature + salinity on  -> "_ts"
      - salinity only                    -> "_sal"
      - temperature only                 -> "_temp"
      - neither                          -> (omitted)

    Algorithm portion:
      - ACTIVE_MOEA_CONFIG.name appended unless it is the production default,
        keeping production paths clean while disambiguating dev variants.

    Args:
        formulation: e.g. "ffmp", "ffmp_8".
        custom_tag: appended after auto-derived components if non-empty.
            Falls back to the `RUN_SLUG_TAG` env var.

    Returns:
        The moea slug, used as the inner partition under
        outputs/{scenario}/ and figures/{scenario}/.
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
    _sfdv_suffix = {
        "multipliers":           "sfdv_mult",
        "multipliers_with_gate": "sfdv_multgate",
        "full":                  "sfdv_full",
    }.get(SALT_FRONT_PARAM_MODE, "")
    if _sfdv_suffix:
        parts.append(_sfdv_suffix)
    if ACTIVE_MOEA_CONFIG.name != _DEFAULT_MOEA_SLUG_CONFIG:
        parts.append(ACTIVE_MOEA_CONFIG.name)

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
    print(f"Scenario design: {ACTIVE_SCENARIO_DESIGN.name} "
          f"({ACTIVE_SCENARIO_DESIGN.family})")
    _nfe = BORG_SETTINGS['max_evaluations']
    print(f"MOEA config: {ACTIVE_MOEA_CONFIG.name} "
          f"(NFE/island={_nfe:,} seeds={BORG_SETTINGS['n_seeds']})"
          if _nfe is not None else
          f"MOEA config: {ACTIVE_MOEA_CONFIG.name} (NFE/island=TBD, seeds=TBD)")
    if SEARCH_ENSEMBLE_SPEC is None:
        print("\nSearch ensemble: <not wired for this scenario design>")
    else:
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
