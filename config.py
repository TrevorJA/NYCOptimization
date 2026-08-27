"""
config.py - Central configuration for NYCOptimization.

Paths, simulation settings, NYC system constants, the active objective set,
LSTM coupling toggles, the two run axes (scenario design x MOEA config),
re-evaluation knobs, and the slug grammar. Problem formulation logic lives in
src/formulations/ (FFMP + variable-resolution FFMP).

Every knob has a default here and a NYCOPT_* env override read at import time,
documented in the section that defines it; SLURM scripts source per-run env
files under workflow/envs/ to set them. Run-identity knobs:
    NYCOPT_SCENARIO_DESIGN        -> ACTIVE_SCENARIO_DESIGN -> SEARCH_ENSEMBLE_SPEC
    NYCOPT_ENSEMBLE_DRAW          -> SCENARIO_ENSEMBLE_DRAW
    NYCOPT_MOEA_CONFIG            -> ACTIVE_MOEA_CONFIG -> BORG/MMBORG settings
    NYCOPT_REEVAL_ENSEMBLE_PRESET -> REEVAL_ENSEMBLE_SPEC (the test ensemble)
    NYCOPT_OBJECTIVES             -> ACTIVE_OBJECTIVES (comma-separated names)
    NYCOPT_FORMULATIONS           -> PRODUCTION_FORMULATIONS (comma-separated)
    NYCOPT_RESULTS_SLUG           -> pins the slug post-processing reads
    RUN_SLUG / RUN_SLUG_TAG       -> override / suffix the derived moea slug
"""

import os
import sys
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
OUTPUTS_DIR = PROJECT_DIR / "outputs"
# All generated figures live under the outputs tree (regenerable, gitignored).
FIGURES_DIR = OUTPUTS_DIR / "figures"
# Manuscript-candidate and SI figure trees: the only figure locations at the
# repo root, git-tracked, rendered at manuscript style (PNG + PDF). Everything
# else lives under the gitignored outputs tree.
MANUSCRIPT_FIG_DIR = PROJECT_DIR / "figures" / "manuscript"
SI_FIG_DIR = PROJECT_DIR / "figures" / "si"

# Borg shared libraries (user must compile and place here)
BORG_DIR = PROJECT_DIR / "lib" / "borg"

# Pywr-DRB pre-simulated releases for trimmed model
PRESIM_DIR = OUTPUTS_DIR / "presim"
PRESIM_FILE = PRESIM_DIR / "presimulated_releases_mgd.csv"

# Staged synthetic-ensemble inputs, one directory per slug under
# STAGED_ENSEMBLE_DIR/{inflow_type}/ (required files: src.ensembles.
# STAGED_ENSEMBLE_FILES, written by workflow steps 02-04; registered with
# pywrdrb's path navigator at simulation start). Gitignored.
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
OUTPUT_BASELINE_DIR = OUTPUTS_DIR / "baseline"
# Ad-hoc diagnostics not tied to a single run (benchmarks, samplers). Per-run
# diagnostics use run_output_dir(scenario, slug, "diagnostics") instead.

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


def baseline_objectives_csv(formulation: str = "ffmp",
                            scenario: str = None) -> Path:
    """Path of the baseline FFMP objective vector comparable to a scenario.

    A baseline vector is only comparable to a front evaluated on the SAME
    substrate, so the file is scenario-partitioned: the historic baseline
    (single trace, full model, step 05 default path) keeps its legacy flat
    location, while every other scenario design points at the same default
    policy scored on that design's search ensemble (step 05
    ``--search-ensemble``). Consumers get "no file" — not a historic-record
    vector — when a scenario has not been scored yet.

    Args:
        formulation: Problem formulation name.
        scenario: Scenario-design name; defaults to the active design.

    Returns:
        ``outputs/baseline/{formulation}_baseline_objectives.csv`` for the
        historic design, else
        ``outputs/baseline/{scenario}/{formulation}_baseline_objectives.csv``
        (parent NOT created — this is a locator, not a writer).
    """
    if scenario is None:
        scenario = active_scenario_name()
    name = f"{formulation}_baseline_objectives.csv"
    if scenario == "historic":
        return OUTPUT_BASELINE_DIR / name
    return OUTPUT_BASELINE_DIR / scenario / name


#: Per-run figure kinds routed to the stable results tree.
FIGURE_KINDS_STABLE = frozenset({
    "convergence", "pareto", "parallel_coords", "policy_inspection",
    "robustness", "satisficing", "criteria", "robustness_cdf", "factor_maps",
})

#: Figure kinds routed EXPLICITLY to the exploratory tree (internal
#: understanding, never manuscript candidates).
FIGURE_KINDS_EXPLORATORY = frozenset({"scenario_discovery", "explore"})


def figure_dir_for(scenario: str, moea_slug: str, kind: str) -> Path:
    """Return a two-axis-partitioned figure subdir, creating it if needed.

    Args:
        scenario: Scenario-design name (top-level partition).
        moea_slug: The moea slug from ``derive_slug()``.
        kind: A registered figure kind: one of
            :data:`FIGURE_KINDS_STABLE` (results tree) or
            :data:`FIGURE_KINDS_EXPLORATORY` (exploratory tree).

    Returns:
        ``outputs/figures/{scenario}/{moea_slug}/{kind}/`` (created), or the
        ``_exploratory`` variant for exploratory kinds.

    Raises:
        ValueError: For an unregistered kind.
    """
    if kind in FIGURE_KINDS_STABLE:
        p = FIGURES_DIR / scenario / moea_slug / kind
    elif kind in FIGURE_KINDS_EXPLORATORY:
        p = FIG_EXPLORATORY_DIR / scenario / moea_slug / kind
    else:
        raise ValueError(
            f"unregistered figure kind {kind!r}; add it to "
            f"config.FIGURE_KINDS_STABLE or FIGURE_KINDS_EXPLORATORY."
        )
    p.mkdir(parents=True, exist_ok=True)
    return p


###############################################################################
# Simulation Settings
###############################################################################

# HISTORIC-design simulation window ONLY (step-01 presim slice, step-05
# baseline): December-anchored bounds of the reconstructed record (which spans
# 1945-01-01..2023-12-31), sharing the synthetic epoch's December anchor so the
# 6-month metric exclusion ends on June 1 (the FFMP operating-year boundary).
# Synthetic-ensemble windows derive from each staged ensemble's own _meta.json
# start_date (src/simulation.py::_ensemble_window), never from these.
START_DATE = "1945-12-01"
END_DATE = "2023-11-30"

# Epoch of every synthetic realization (ENSEMBLE_START_DATE, a December 1) is
# defined in src/ensembles.py and re-exported below with the ensemble registry.

# Default inflow source = Amestoy et al. (2026) Bayesian-bias-corrected
# reconstructed DRB streamflow ensemble (1945-2023; Environmental Modelling
# & Software, 195, 106756). The key "pub_nhmv10_BC_withObsScaled" is the
# median realization of that ensemble and ships pre-packaged with pywrdrb's
# Data() loader.
# Override per-experiment via NYCOPT_INFLOW_TYPE in workflow/envs/*.env.
INFLOW_TYPE = os.environ.get("NYCOPT_INFLOW_TYPE", "pub_nhmv10_BC_withObsScaled")
# Trimmed model (presimulated lower-basin releases) is BOTH the search and the
# re-evaluation path: the non-NYC STARFIT releases are policy-independent, so
# step 04 presimulates them once per staged realization and every policy
# evaluation reuses them. The full model (all reservoirs simulated live) is
# used only for the presim pass (step 01/04) and the single-trace historic
# baseline (step 05); the env knob exists so benchmarks and those jobs can
# select it without editing this file.
USE_TRIMMED_MODEL = _parse_bool_env("NYCOPT_USE_TRIMMED_MODEL", True)
INITIAL_VOLUME_FRAC = 0.80

# Montague/Trenton flow-forecast mode passed to pywrdrb.ModelBuilder
# (Options.flow_prediction_mode): perfect_foresight for every Pywr-DRB
# simulation, pinned explicitly at every model build; never rely on pywrdrb's
# default.
PYWRDRB_FLOW_PREDICTION_MODE = "perfect_foresight"

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

# Metrics (objectives AND hazard-selection metrics) exclude the first 6 months
# of each scenario window — the SSI-6 accumulation spin-up, before which the
# drought index has no defined value. On the December-start windows the
# exclusion ends exactly on June 1, the FFMP operating-year boundary, and both
# layers then score the IDENTICAL window [Jun 1 year 1, May 31 year L]: the
# objectives keep complete Jun-May FFMP years (src.objectives_ensemble.
# ffmp_year_unit_slices) and the hazard image trims the same trailing partial
# (src.ensemble_generation._hazard_block). Simulations still start from fixed
# initial storage (INITIAL_VOLUME_FRAC); the exclusion is applied BY DATE from
# each window's DatetimeIndex, never as a fixed day count.
METRIC_EXCLUSION_MONTHS = 6


###############################################################################
# Forcing-ensemble generation (workflow step 02)
###############################################################################
# Shared forcing-space configuration and storage mode for the generators
# (scengen.forcing_space; src.ensemble_generation). Designs read their sizes
# (N, R, L, P) from src/scenario_designs.py. The CMIP6 tables live in the
# sibling repo; override via workflow/envs/*.env.

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
# OFF for the campaign: the forcing space is the 3-D mean box [m, r1, r2] with
# the CV-preserving c = a (the CV axis adds DU dimensions without widening
# hazard-space tail stress). Opt-in sensitivity via the env var.
ENSEMBLE_FORCING_VARIANCE_AXIS = _parse_bool_env("NYCOPT_ENSEMBLE_FORCING_VARIANCE_AXIS", False)
ENSEMBLE_FORCING_BOUND_PCT = (
    _parse_float_env("NYCOPT_ENSEMBLE_FORCING_BOUND_LO", 5.0),
    _parse_float_env("NYCOPT_ENSEMBLE_FORCING_BOUND_HI", 95.0),
)
ENSEMBLE_FORCING_MARGIN = _parse_float_env("NYCOPT_ENSEMBLE_FORCING_MARGIN", 0.0)
# stream_only discards daily traces after hazard computation (the ~1e6
# production candidate pool); default False keeps the daily HDF5s so
# workflow/03 consumes them unchanged.
ENSEMBLE_MASTER_STREAM_ONLY = _parse_bool_env("NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY", False)
ENSEMBLE_MASTER_HAZARD_BLOCK = _parse_int_env("NYCOPT_ENSEMBLE_MASTER_HAZARD_BLOCK", 256)
# Chunk the stored daily candidate pool into contiguous chunks of this many
# realizations (must be a multiple of realizations_per_profile). 0 = single
# directory. Bounds generation write-memory and avoids a monolithic
# multi-hundred-GB HDF5 for a large pool (methods §3.2).
ENSEMBLE_MASTER_CHUNK_SIZE = _parse_int_env("NYCOPT_ENSEMBLE_MASTER_CHUNK_SIZE", 0)

# Campaign hazard selection axes (m = 6), chosen via the nested-P saturation
# diagnostic (docs/notes/methods/hazard_selector_diagnostics.md).
# drought_duration and flood_rise_rate stay in every hazard image but never
# enter the snap distance.
HAZARD_SELECTION_AXES = _parse_list_env("NYCOPT_HAZARD_SELECTION_AXES", [
    "drought_magnitude",
    "drought_severity",
    "drought_onset_rate",
    "drought_recovery_rate",
    "flood_peak_discharge",
    "flood_pulse_duration",
])


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
# Objective names in base (single-trace registry) spelling; the annual-unit
# registry (src.objectives_ensemble) renames them whenever a scenario design
# is wired. The default eight objectives:
#   - NYC supply: weekly delivery reliability + tail (CVaR90) delivery deficit
#   - Montague flow Decree: reliability + CVaR90 deficit
#   - Trenton flow Decree: reliability
#   - downstream flood exposure: ft-days above NWS minor flood stage
#     (docs/notes/methods/flood_objective_diagnostics.md)
#   - storage resilience: 5th-percentile combined NYC storage
#   - NJ supply: weekly delivery reliability
# Definitions: docs/notes/methods/objective_definitions.md.

_DEFAULT_OBJECTIVES = [
    "nyc_delivery_reliability_weekly",
    "nyc_delivery_deficit_cvar90_pct",
    "montague_flow_reliability_weekly",
    "montague_flow_deficit_cvar90_pct",
    "trenton_flow_reliability_weekly",
    "downstream_flood_exceedance_minor",
    "nyc_storage_p5_pct",
    "nj_delivery_reliability_weekly",
]

ACTIVE_OBJECTIVES = _parse_list_env("NYCOPT_OBJECTIVES", _DEFAULT_OBJECTIVES)

# Stakeholder floor on NYC delivery reliability, enforced during search as the
# formal post-simulation Borg constraint `nyc_reliability_floor` (violation =
# max(0, floor - reliability) on the natural 0-1 scale;
# src.formulations.make_post_sim_constraint_function). The floor reads the
# active set's reliability objective (the annual non-failure frequency in
# every ensemble search context). src/pareto_filter.py applies the same floor
# as a post-hoc screen.
NYC_RELIABILITY_FLOOR = _parse_float_env("NYCOPT_NYC_RELIABILITY_FLOOR", 0.5)


###############################################################################
# Variable-Resolution FFMP Sweep
###############################################################################

# Values of N (storage zone boundary curves) for the variable-resolution FFMP
# extension; each maps to formulation "ffmp_{N}". ffmp_6 is structurally
# identical to ffmp and is omitted; re-include it via NYCOPT_FFMP_VR_N.
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
# simulation. Both default off and neither is used in the manuscript (the
# salinity LSTM does not perform well under extreme droughts; temperature is
# deferred). The coupling machinery is retained, dormant, behind
# NYCOPT_SALINITY_ON / NYCOPT_TEMPERATURE_ON.

INCLUDE_TEMPERATURE_MODEL = _parse_bool_env("NYCOPT_TEMPERATURE_ON", False)
INCLUDE_SALINITY_MODEL = _parse_bool_env("NYCOPT_SALINITY_ON", False)

# Earliest date the salinity LSTM may update; defaults to START_DATE. Dormant.
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

# Salinity coupling mode. Default False (sync): the LSTM advances one step per
# sim day and responds to simulated flows; True disables that responsiveness.
# In sync mode the LSTM rewrites mrf_target_{delMontague,delTrenton} during
# NYC drought emergency, which the Decree-scored objectives never read. Dormant.
SALINITY_ASYNC_UPDATE = _parse_bool_env("NYCOPT_SALINITY_ASYNC", False)

# Extend RESULTS_SETS so pywrdrb.Data().load_output() pulls the LSTM outputs
# ('salinity' -> 'salt_front_location_mu'; 'temperature' ->
# 'temperature_after_thermal_release_mu').
if INCLUDE_SALINITY_MODEL and "salinity" not in RESULTS_SETS:
    RESULTS_SETS = list(RESULTS_SETS) + ["salinity"]
if INCLUDE_TEMPERATURE_MODEL and "temperature" not in RESULTS_SETS:
    RESULTS_SETS = list(RESULTS_SETS) + ["temperature"]


###############################################################################
# Salt-front MRF adjustment parameterization (FFMP-family DVs)
###############################################################################
# Dormant (default "fixed"; requires the salinity LSTM). The salt-front MRF
# adjustment table, indexed by (RM band, season), can be exposed as decision
# variables via NYCOPT_SALT_FRONT_PARAM_MODE. DV counts per mode are defined by
# src.formulations.salt_front_dvs.salt_front_dv_specs:
#   "fixed"                 -> 0 DVs
#   "multipliers"           -> 11 free multiplier cells (reference 1.0 cells implicit)
#   "multipliers_with_gate" -> + 1 activation drought-level DV (12)
#   "full"                  -> + 3 RM-band threshold DVs (15)

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
# Algorithm settings live in a named, versioned registry (src/moea_config.py).
# The active config is selected by NYCOPT_MOEA_CONFIG and is the second axis
# (alongside the scenario design) that specifies a run. The BORG_SETTINGS /
# MMBORG_SETTINGS dicts are the public read surface, projected from
# ACTIVE_MOEA_CONFIG. Change algorithm settings by editing/selecting a
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
# Re-evaluation knobs; the MPI driver reads node/rank sizing from these.

REEVALUATION_SETTINGS = {
    # Metric identifiers scored offline by src.robustness from the persisted
    # per-SOW annual-unit objective matrix (one metric currency across search,
    # robustness and regret); `src.robustness --metrics` overrides. The SOW
    # count comes from REEVAL_ENSEMBLE_SPEC. Set-relative and search-vs-test
    # gap metrics are deliberately absent (src/robustness.py).
    "robustness_metrics": [
        "satisficing_multivariate_sow",  # PRIMARY: Starr domain criterion over SOWs
        "satisficing_univariate_sow",    # the PRIMARY's per-objective decomposition
        "laplace_mean",                  # McPhail T3 = mean  (risk-neutral anchor)
        "maximin",                       # McPhail T3 = worst (risk-averse anchor)
        "regret_magnitudes",             # incumbent-relative regret, natural units
        "regret_frequencies",            # its unit-free harm frequencies
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
# single-trace default is the ``historic`` design (-> ``historic_single``
# preset).
#
# Two specs are resolved at import time:
#   SEARCH_ENSEMBLE_SPEC  - from ACTIVE_SCENARIO_DESIGN; used inside Borg's
#                           evaluate() during optimization.
#   REEVAL_ENSEMBLE_SPEC  - the held-out test ensemble E_test; used by
#                           src/reevaluate.py + reevaluate_mpi.py. It is an
#                           EnsembleSpec, never a ScenarioDesign: it never enters
#                           search, and no search ensemble is drawn from it.
#
# Resolution is a PURE LOOKUP. Every design's ensemble is constructed by workflow
# step 02 (and step 03 for hazard-filling), so importing config performs no RNG
# draws and no bulk I/O.

from src.ensembles import (             # noqa: E402
    ENSEMBLE_START_DATE,  # epoch of every synthetic realization (a December 1)
    get_ensemble_spec,
    staged_ensemble_dir,
    with_indices_override,
)
from src.scenario_designs import (   # noqa: E402
    SCENARIO_YEARS,  # single source of truth for the scenario length L
    SEARCH_ENSEMBLE_N,  # common ensemble size N across the matched designs
    SEED_ROOT,
    assert_seed_domains_disjoint,
    get_scenario_design,
)

NYCOPT_SCENARIO_DESIGN = _parse_str_env("NYCOPT_SCENARIO_DESIGN", "historic")
ACTIVE_SCENARIO_DESIGN = get_scenario_design(NYCOPT_SCENARIO_DESIGN)

# Independent ensemble-draw replication index. A draw is the design's
# construction RE-RUN FROM SCRATCH with a fresh seed — an independent
# generation, not a re-indexing of shared data. Designs with no fixed ensemble
# to redraw (historic) accept only 0 and raise otherwise (fail fast at import).
# Nonzero draws are appended to the moea slug as "_d{k}" so replicate runs
# partition to distinct output directories.
SCENARIO_ENSEMBLE_DRAW = _parse_int_env("NYCOPT_ENSEMBLE_DRAW", 0)

NYCOPT_REEVAL_ENSEMBLE_PRESET = _parse_str_env(
    "NYCOPT_REEVAL_ENSEMBLE_PRESET", "historic_single",
)

# Seed domains must not collide: every design GENERATES its own ensemble, so
# two designs sharing a seed would produce correlated realizations. Fail fast
# at import.
assert_seed_domains_disjoint()

# Resolve the search ensemble. Designs whose ensemble is not staged yet leave
# SEARCH_ENSEMBLE_SPEC None so config stays importable — diagnostics/reeval/
# plotting on such a design's outputs only need active_scenario_name().
# Optimization fails fast with a clear message (see src/mmborg.py).
try:
    SEARCH_ENSEMBLE_SPEC = ACTIVE_SCENARIO_DESIGN.resolve_search_spec(
        draw=SCENARIO_ENSEMBLE_DRAW
    )
except NotImplementedError as _e:
    SEARCH_ENSEMBLE_SPEC = None
    print(
        f"  [config] NOTE: scenario design '{ACTIVE_SCENARIO_DESIGN.name}' has "
        f"no search ensemble wired yet; optimization is unavailable for it "
        f"(diagnostics/reeval/plotting on its outputs still work). {_e}"
    )
REEVAL_ENSEMBLE_SPEC = get_ensemble_spec(NYCOPT_REEVAL_ENSEMBLE_PRESET)

# Realizations per Pywr model run inside evaluate()
# (src/simulation.py::run_simulation_ensemble_batched); 0 = one block. Results
# are identical to the unbatched path (tests/test_ensemble_simulation.py); the
# production env files set 150 for N=300 (see search_node_rss_gb).
SEARCH_REALIZATION_BATCH = _parse_int_env("NYCOPT_SEARCH_REALIZATION_BATCH", 0)

# --- Per-node memory model for the MM-Borg pre-flight (workflow/_common.sh
# nycopt_check_memory). Per-rank RSS = 600 MB + 0.49 MB per scenario-year held
# in one Pywr model (min(N, batch) x L), fitted to the measured N=100
# production steady state and conservative above N=200; the safety line is
# 85 % of the 256 GB node.
RANK_RSS_INTERCEPT_MB = 600.0
RANK_RSS_MB_PER_SCENARIO_YEAR = 0.49
NODE_MEMORY_GB = _parse_float_env("NYCOPT_NODE_MEMORY_GB", 256.0)
NODE_MEMORY_SAFETY_FRACTION = 0.85


def search_rank_rss_mb(n_realizations: int, realization_years: int,
                       realization_batch: int = 0) -> float:
    """Estimated resident set of one evaluator rank, in MB.

    Args:
        n_realizations: Search-ensemble size N (1 for a single trace).
        realization_years: Realization length L in years.
        realization_batch: Realizations per Pywr model build; ``<= 0`` means
            the whole ensemble is one scenario block.

    Returns:
        ``RANK_RSS_INTERCEPT_MB + RANK_RSS_MB_PER_SCENARIO_YEAR * min(N, batch) * L``.
    """
    held = n_realizations if realization_batch <= 0 else min(n_realizations, realization_batch)
    return RANK_RSS_INTERCEPT_MB + RANK_RSS_MB_PER_SCENARIO_YEAR * held * realization_years


def search_node_rss_gb(ranks_per_node: int, n_realizations: int,
                       realization_years: int, realization_batch: int = 0) -> float:
    """Estimated resident set of one node, in GB, at ``ranks_per_node`` evaluators."""
    return ranks_per_node * search_rank_rss_mb(
        n_realizations, realization_years, realization_batch) / 1024.0

# Print a one-line build/run/extract wall-time split per ensemble model run
# (src/simulation.py::run_simulation_ensemble_inmemory). Logging only.
SIM_PHASE_TIMING = _parse_int_env("NYCOPT_SIM_PHASE_TIMING", 0)

# --- Chunked re-evaluation (step 09, src/chunk_reeval.py) execution knobs ---
# None of these change results: unit rows are keyed by (solution id, GLOBAL
# realization id, objective), so layout, scheduling and merge placement cannot
# alter the merged cube (tests/test_chunk_reeval.py). Single documented copy.
#
# CHUNK_INCREMENTAL  1: flush each (solution, chunk) unit atomically and skip
#                    done units on restart (resubmitting IS the resume); 0:
#                    accumulate in memory, one write per rank at the end.
# CHUNK_SCHEDULE     claim: ranks pull units via O_CREAT|O_EXCL claim files;
#                    interleave: static strided; contiguous: static blocks.
# CHUNK_MERGE        job: rank 0 merges after all ranks finish; off: ranks only
#                    write units (merge via workflow/09b_merge_test_chunks.sh).
# CHUNK_MERGE_ALLOW_PARTIAL  1: merge with missing units as NaN rows.
# CHUNK_DONE_DEADLINE_S      await_all_done deadline for the in-job merge.
# CHUNK_RETRY_FAILED 1: a resume re-attempts units whose previous run raised.
# CHUNK_STOP_EPOCH / CHUNK_UNIT_SECONDS  wall guard: no unit starts within
#                    1.25 x CHUNK_UNIT_SECONDS of CHUNK_STOP_EPOCH (unix time).
CHUNK_INCREMENTAL = _parse_int_env("NYCOPT_CHUNK_INCREMENTAL", 1)
CHUNK_SCHEDULE = os.environ.get("NYCOPT_CHUNK_SCHEDULE", "claim")
CHUNK_MERGE = os.environ.get("NYCOPT_CHUNK_MERGE", "job")
CHUNK_MERGE_ALLOW_PARTIAL = _parse_int_env("NYCOPT_CHUNK_MERGE_ALLOW_PARTIAL", 0)
CHUNK_DONE_DEADLINE_S = _parse_float_env("NYCOPT_CHUNK_DONE_DEADLINE_S", 10800.0)
CHUNK_RETRY_FAILED = _parse_int_env("NYCOPT_CHUNK_RETRY_FAILED", 0)
CHUNK_STOP_EPOCH = _parse_float_env("NYCOPT_CHUNK_STOP_EPOCH", 0.0)
CHUNK_UNIT_SECONDS = _parse_float_env("NYCOPT_CHUNK_UNIT_SECONDS", 0.0)

# FIFO bound on the per-process base model_dict cache
# (src/simulation.py::_get_cached_model_dict). Speed/memory only — a cache miss
# rebuilds the dict (~1 s) with identical content. The bound matters because
# the chunked re-eval visits 50 chunk presets (x batch offsets) per rank.
MODEL_DICT_CACHE_MAX = _parse_int_env("NYCOPT_MODEL_DICT_CACHE_MAX", 8)

# Optional realization-index override on the search ensemble. Useful for smoke
# testing a subset of a large ensemble without authoring a new preset.
_ensemble_indices_override = _parse_int_list_env("NYCOPT_ENSEMBLE_INDICES", [])
if _ensemble_indices_override and SEARCH_ENSEMBLE_SPEC is not None:
    SEARCH_ENSEMBLE_SPEC = with_indices_override(
        SEARCH_ENSEMBLE_SPEC, _ensemble_indices_override,
    )

def _staged_seed_domain(spec) -> str | None:
    """Read the ``seed_domain`` recorded in a staged ensemble's ``_meta.json``.

    Written by ``src.ensemble_generation.generate_forcing_ensemble`` (from
    ``ForcingEnsembleConfig.seed_domain``), which every generator path (steps
    02, 03 and 12) supplies. Returns ``None`` for specs with no staged metadata
    (static presets, the historic trace), which cannot collide by construction.
    """
    import json

    if spec is None or not spec.is_ensemble:
        return None
    meta_path = staged_ensemble_dir(spec.inflow_type) / "_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text()).get("seed_domain")


def assert_search_test_seed_domains_disjoint(search_spec, reeval_spec) -> None:
    """Selection-bias guard (Bonham et al. 2024): E_test must be seed-independent of search.

    A HARD ERROR, not a warning — a warning nobody reads is not a guard, and a search
    ensemble drawn from the same seed stream as the test ensemble would make the
    held-out re-evaluation not held out at all. Compared on the RECORDED seed domain
    rather than the preset name, because two ensembles can share a generator stream
    under different slugs. E_test's reserved domains are ``etest:*``
    (``scengen.seeds.SEED_DOMAINS``), which are disjoint from every search-side domain.

    Args:
        search_spec: The active search ``EnsembleSpec`` (or ``None``).
        reeval_spec: The re-eval (test) ``EnsembleSpec``.

    Raises:
        RuntimeError: If both are staged and record the same seed domain.
    """
    search_domain = _staged_seed_domain(search_spec)
    reeval_domain = _staged_seed_domain(reeval_spec)
    if search_domain is not None and search_domain == reeval_domain:
        raise RuntimeError(
            f"Search and re-evaluation ensembles share seed domain "
            f"'{search_domain}' (search='{search_spec.inflow_type}', "
            f"re-eval='{reeval_spec.inflow_type}'). E_test must be generated "
            f"from an independent seed stream ('etest:*'), or the held-out "
            f"re-evaluation is not held out (selection bias, Bonham et al. 2024)."
        )


assert_search_test_seed_domains_disjoint(SEARCH_ENSEMBLE_SPEC, REEVAL_ENSEMBLE_SPEC)


###############################################################################
# Slug Naming Convention
###############################################################################
# A run is partitioned on two axes: the scenario design (top-level output dir,
# from ACTIVE_SCENARIO_DESIGN.name) and the moea slug below. derive_slug()
# builds the moea slug — the problem-definition identity plus the non-default
# algorithm-config name. The ensemble is NOT in the slug; it is the parent
# {scenario} directory. Format:
#   {formulation}_obj{N_OBJ}{ts_suffix}{sfdv_suffix}{draw_suffix}{moea_cfg_suffix}{custom_suffix}
# where {draw_suffix} = "_d{k}" only for a nonzero NYCOPT_ENSEMBLE_DRAW replicate.
#
# Full output path: outputs/{scenario}/{moea_slug}/{artifact}/
#
# Examples (moea slug only):
#   ffmp_obj8                    — FFMP, 8 objectives, production algo config
#   ffmp_8_obj8                  — variable-resolution N=8, 8 objectives
#   ffmp_obj8_smoke              — dev smoke algorithm config
#   ffmp_obj8_pilot42            — ad-hoc tagged run (RUN_SLUG_TAG=pilot42)
#   ffmp_obj8_sal                — salinity LSTM on (dormant; only if
#                                  NYCOPT_SALINITY_ON=1 — not used in manuscript)
#
# `RUN_SLUG_TAG` env appends a free-form suffix; useful for one-off variants
# without polluting the canonical slug grammar.
# A non-empty `RUN_SLUG` env wins outright (escape hatch for nonstandard paths).

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
    if SCENARIO_ENSEMBLE_DRAW:
        # Independent ensemble-draw replicate: partition its outputs away from
        # draw 0 (the ensemble itself is staged per draw, e.g. fixprob_*_d{k}).
        parts.append(f"d{SCENARIO_ENSEMBLE_DRAW}")
    if ACTIVE_MOEA_CONFIG.name != _DEFAULT_MOEA_SLUG_CONFIG:
        parts.append(ACTIVE_MOEA_CONFIG.name)

    tag = custom_tag if custom_tag else os.environ.get("RUN_SLUG_TAG", "").strip()
    if tag:
        parts.append(tag)

    return "_".join(parts)


###############################################################################
# Reading a campaign's outputs back (post-processing slug resolution)
###############################################################################
# derive_slug() above answers "where does the run I am ABOUT TO EXECUTE write?"
# — it is built from the ACTIVE run identity, so it needs NYCOPT_ENV_FILE (or
# the equivalent NYCOPT_* knobs) to be set. Post-processing asks the opposite
# question — "where did the campaign I am READING already write?" — and there
# the same call is a trap: with no env file, ACTIVE_MOEA_CONFIG falls back to
# the dev-smoke config and derive_slug() returns `ffmp_obj8_smoke`, which
# either fails with a confusing missing-file error or, worse, silently scores
# leftover smoke-scale results. results_slug() below is the resolver every
# read-side entry point uses instead; it never guesses quietly.

#: Env var pinning the moea slug that post-processing reads campaign outputs
#: from. Explicit always wins over the inference in :func:`results_slug`.
RESULTS_SLUG_ENV = "NYCOPT_RESULTS_SLUG"

#: Nominal campaign slug, used only where a resolution failure must not be
#: fatal (e.g. the figure driver on a machine holding no campaign outputs, so
#: each figure can SKIP with its own message instead of the pass dying).
CAMPAIGN_RESULTS_SLUG = "ffmp_obj8"


def slugs_carrying_reeval(reeval_tag: str) -> list[str]:
    """Moea slugs holding a re-eval on ``reeval_tag`` for EVERY campaign design.

    Args:
        reeval_tag: The held-out ensemble tag (a re-eval leaf directory name).

    Returns:
        Sorted slugs present under every campaign design, so a slug that only
        one design happens to carry is never a resolution candidate.
    """
    from src.scenario_designs import campaign_designs

    common: set | None = None
    for design in campaign_designs():
        root = OUTPUTS_DIR / design
        here = ({s.name for s in root.iterdir()
                 if s.is_dir() and (s / "reeval" / reeval_tag).is_dir()}
                if root.is_dir() else set())
        common = here if common is None else (common & here)
    return sorted(common or ())


def results_slug(reeval_tag: str, formulation: str | None = None) -> str:
    """The moea slug carrying the campaign re-eval outputs for ``reeval_tag``.

    Resolution order, first match wins:

    1. ``NYCOPT_RESULTS_SLUG`` when set — explicit beats every inference.
    2. ``derive_slug(formulation)`` when that slug actually carries the tag
       for every campaign design (i.e. the ambient run identity is the
       campaign's). Skipped when ``formulation`` is None.
    3. The unique on-disk slug that does, reported on stderr so a discovered
       resolution is never invisible in a job log.

    Args:
        reeval_tag: The held-out ensemble tag being post-processed.
        formulation: Formulation name enabling step 2; None to skip it.

    Returns:
        The resolved moea slug.

    Raises:
        FileNotFoundError: no slug carries the tag for every campaign design.
        ValueError: several do — ambiguous, so the caller must pin
            ``NYCOPT_RESULTS_SLUG``.
    """
    from src.scenario_designs import campaign_designs

    explicit = os.environ.get(RESULTS_SLUG_ENV, "").strip()
    if explicit:
        return explicit

    candidates = slugs_carrying_reeval(reeval_tag)
    derived = derive_slug(formulation) if formulation else None
    if derived is not None and derived in candidates:
        return derived

    if len(candidates) == 1:
        slug = candidates[0]
        why = (f"derive_slug gave '{derived}', which carries no re-eval on "
               f"this tag" if derived is not None
               else "no formulation was supplied")
        print(f"[config] results slug resolved by discovery: '{slug}' "
              f"(tag '{reeval_tag}'; {why}). Set {RESULTS_SLUG_ENV} to pin it.",
              file=sys.stderr)
        return slug

    if not candidates:
        raise FileNotFoundError(
            f"no moea slug under {OUTPUTS_DIR} carries a re-eval on tag "
            f"'{reeval_tag}' for every campaign design "
            f"({', '.join(campaign_designs())}). "
            f"Run the re-evaluation (workflow steps 08/09) first, or set "
            f"{RESULTS_SLUG_ENV} if the outputs live under a nonstandard slug."
        )
    raise ValueError(
        f"tag '{reeval_tag}' is carried by several moea slugs "
        f"({', '.join(candidates)}); set {RESULTS_SLUG_ENV} to the one to "
        f"post-process."
    )


###############################################################################
# Re-exports from src.formulations (the public formulation API)
###############################################################################

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
# Thin helpers
###############################################################################

def get_epsilons():
    """Epsilon values for Borg epsilon-dominance, ordered by objective."""
    return get_objective_set().epsilons


