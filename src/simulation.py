"""
simulation.py - Pywr-DRB simulation wrapper for optimization and re-evaluation.

Core evaluation path: flat decision-variable vector -> NYCOperationsConfig ->
Pywr-DRB simulation -> objective vector. Search (``evaluate``) and
re-evaluation (``evaluate_annual_units``) both run in memory: the model dict
is cached and patched per evaluation, a temporary JSON model file is written
to a per-rank temp dir (pywr.Model.load requires a file), and an
InMemoryRecorder captures results to numpy arrays. HDF5 output is written
only by the step-05 baseline (``run_simulation_to_disk``).
"""

import os
import sys
import copy
import json
import time
import tempfile
import numpy as np
import pandas as pd
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    START_DATE,
    END_DATE,
    INFLOW_TYPE,
    USE_TRIMMED_MODEL,
    INITIAL_VOLUME_FRAC,
    NYC_NJ_DEMAND_SOURCE,
    PYWRDRB_FLOW_PREDICTION_MODE,
    RESULTS_SETS,
    PRESIM_DIR,
    PRESIM_FILE,
    NYC_RESERVOIRS,
    INCLUDE_SALINITY_MODEL,
    INCLUDE_TEMPERATURE_MODEL,
    NYC_DECREE_DIVERSION_CAP_MGD,
    SIM_PHASE_TIMING,
)
from src.formulations import get_formulation, get_var_names
from src.formulations.salt_front_dvs import apply_salt_front_dvs
from src.ts_options import build_lstm_options_block

# Allow debug override of the SINGLE-TRACE (historic) simulation date range via
# environment variables. Set PYWRDRB_SIM_START_DATE / PYWRDRB_SIM_END_DATE
# (YYYY-MM-DD) before launching mpirun to run a shorter period for fast
# debugging. Ensemble windows are derived from each staged ensemble's own
# _meta.json stamp (see _ensemble_window) and ignore these overrides.
# Example (5-year debug period, ~13s/eval vs ~150s for full 78-year run):
#   export PYWRDRB_SIM_START_DATE=2018-01-01
#   export PYWRDRB_SIM_END_DATE=2022-12-31
_env_start = os.environ.get("PYWRDRB_SIM_START_DATE")
_env_end = os.environ.get("PYWRDRB_SIM_END_DATE")
if _env_start:
    START_DATE = _env_start
if _env_end:
    END_DATE = _env_end


###############################################################################
# Cached model components
###############################################################################

_CACHED_PRESIM_FILE = None
_PRESIM_SEARCHED = False
_CACHED_DEFAULTS_CONFIG = None   # Cached NYCOperationsConfig.from_defaults()
_CACHED_MODEL_DICTS = {}         # Keyed by tuple(drought_levels) for N-zone support
_CACHED_NZONE_CONFIGS = {}       # Keyed by n_zones

# CFS→MGD conversion (from pywrdrb.utils.constants)
_CFS_TO_MGD = 0.645932368556

# Parameter keys affected by decision variables
_DROUGHT_LEVELS = ["level1a", "level1b", "level1c", "level2", "level3", "level4", "level5"]
_ZONE_LEVELS = ["level1b", "level1c", "level2", "level3", "level4", "level5"]
_NYC_RESERVOIRS_OPT = ["cannonsville", "pepacton", "neversink"]
_DOWNSTREAM_LOCS = ["delMontague", "delTrenton"]

# Day-of-year windows for the seasonal MRF profile-scaling DVs (the
# monthly flow-target tables use the matching month lists in
# _apply_flow_target_scaling). These are the FFMP's OWN season definitions
# from the Tables 4a-4g headers — winter Dec 1-Mar 31, spring Apr 1-May 31,
# summer Jun 1-Aug 31, fall Sep 1-Nov 30 — indexed as calendar dates on the
# 366-column (leap-year) daily profiles. The FFMP's finer date bins nest
# inside these seasons, so seasonally scaled schedules only ever step on
# the FFMP's own bin edges.
_SEASON_DOY_RANGES = {
    "winter": list(range(336, 367)) + list(range(1, 92)),
    "spring": list(range(92, 153)),
    "summer": list(range(153, 245)),
    "fall": list(range(245, 336)),
}

# Below this baseline peak-to-trough span a storage curve is treated as flat
# (no plateaus to remap); the two vertical shifts collapse to one offset.
_ZONE_FLAT_EPS = 1e-9

# Evaluation counter for progress reporting
_EVAL_COUNT = 0
_EVAL_START_TIME = None
_EVAL_LOG_INTERVAL = 1  # Print status every N evaluations (set to 1 for debugging)


def _find_presim_file() -> Optional[Path]:
    """Find the pre-simulated releases CSV for the trimmed model (cached).

    Returns None if the canonical PRESIM_FILE does not exist. The caller
    is responsible for raising a useful error if trimmed mode is requested
    but the file is missing.
    """
    global _CACHED_PRESIM_FILE, _PRESIM_SEARCHED
    if _PRESIM_SEARCHED:
        return _CACHED_PRESIM_FILE

    _PRESIM_SEARCHED = True
    if PRESIM_FILE.exists():
        _CACHED_PRESIM_FILE = PRESIM_FILE
    return _CACHED_PRESIM_FILE


def _require_presim_file() -> Path:
    """Return the presimulated releases file, raising a clear error if missing."""
    f = _find_presim_file()
    if f is None:
        raise FileNotFoundError(
            f"Presimulated releases file not found: {PRESIM_FILE}\n"
            "Run the setup step first:\n"
            "  bash workflow/01_generate_presim.sh\n"
            "  (or: python scripts/main/generate_presim.py)"
        )
    return f


###############################################################################
# Cached NYCOperationsConfig and Model Dict
###############################################################################

def _get_cached_defaults():
    """Return cached NYCOperationsConfig.from_defaults() (avoids re-reading CSVs).

    Applies a compatibility shim for pywrdrb versions where STORAGE_LEVELS and
    DROUGHT_LEVELS are dynamic properties that return all rows in storage_zones_df
    (including MRF factor rows) rather than just the storage zone threshold rows.
    When detected, we reset storage_zones_df to only the 6 storage zone rows so
    that n_drought_levels == 7 and update_delivery_constraints() works correctly.
    """
    global _CACHED_DEFAULTS_CONFIG
    if _CACHED_DEFAULTS_CONFIG is None:
        from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig
        cfg = NYCOperationsConfig.from_defaults()
        # Shim: if STORAGE_LEVELS returns more than 6 entries (buggy property),
        # filter storage_zones_df down to just the known 6 zone rows.
        if len(cfg.STORAGE_LEVELS) != 6:
            zones_df = cfg.storage_zones_df.loc[_ZONE_LEVELS]
            object.__setattr__(cfg, 'storage_zones_df', zones_df)
        _CACHED_DEFAULTS_CONFIG = cfg
    return _CACHED_DEFAULTS_CONFIG


def _get_cached_nzone_defaults(n_zones):
    """Return cached NYCOperationsConfig for N storage zones."""
    if n_zones not in _CACHED_NZONE_CONFIGS:
        _CACHED_NZONE_CONFIGS[n_zones] = build_nzone_config(n_zones)
    return _CACHED_NZONE_CONFIGS[n_zones]


def _get_cached_model_dict(use_trimmed: bool = None, nyc_config=None,
                           ensemble_spec=None):
    """Build and cache the base model_dict (first call per level structure only).

    Subsequent evaluations deep-copy this dict and patch only the DV-affected
    parameters, avoiding the ~1s cost of make_model() on every eval.

    Cache key includes drought-level structure, T/S toggles, the resolved
    trimmed/full model variant, and ensemble preset name + DU factor signature
    so that switching ensemble presets (or enabling salinity, or switching to
    the full model) does not silently reuse a cache built for a different
    model. ``use_trimmed`` is resolved against ``USE_TRIMMED_MODEL`` *before*
    keying: every production caller passes ``None``, so keying on the raw
    argument would collapse both variants onto one entry.
    """
    global _CACHED_MODEL_DICTS
    if nyc_config is None:
        nyc_config = _get_cached_defaults()
    if ensemble_spec is None:
        from src.ensembles import get_ensemble_spec
        ensemble_spec = get_ensemble_spec("historic_single")
    use_trimmed = USE_TRIMMED_MODEL if use_trimmed is None else bool(use_trimmed)
    drought_levels, _ = _config_levels(nyc_config)
    key = (
        tuple(drought_levels),
        bool(INCLUDE_TEMPERATURE_MODEL),
        bool(INCLUDE_SALINITY_MODEL),
        use_trimmed,
        ensemble_spec.preset_name,
        ensemble_spec.du_factor_signature,
    )
    if key not in _CACHED_MODEL_DICTS:
        mb = _build_model_builder(
            nyc_config,
            use_trimmed=use_trimmed,
            ensemble_spec=ensemble_spec,
        )
        _CACHED_MODEL_DICTS[key] = mb.model_dict
        # FIFO bound: the chunked re-eval walks 50 chunk presets (x batch
        # offsets) per rank; unbounded growth is GBs/rank. Eviction is
        # speed-only — a re-miss rebuilds an identical dict (~1 s).
        from config import MODEL_DICT_CACHE_MAX

        while len(_CACHED_MODEL_DICTS) > max(1, MODEL_DICT_CACHE_MAX):
            _CACHED_MODEL_DICTS.pop(next(iter(_CACHED_MODEL_DICTS)))
    return _CACHED_MODEL_DICTS[key]


def _config_levels(nyc_config):
    """Return (drought_levels, storage_levels) from config.

    pywrdrb exposes DROUGHT_LEVELS and STORAGE_LEVELS as properties.
    The default config's storage_zones_df includes all profile rows (storage zones
    AND MRF factor rows), so the STORAGE_LEVELS property returns too many rows.
    We detect this by checking if the first entry is 'zone_' prefixed (N-zone)
    or if the result contains known non-zone rows, and fall back to module
    constants when needed.
    """
    raw_storage = nyc_config.STORAGE_LEVELS
    # N-zone configs have zone_1..zone_N naming (all correct, df is filtered)
    if raw_storage and raw_storage[0].startswith('zone_'):
        return nyc_config.DROUGHT_LEVELS, raw_storage
    # Default config: fall back to the known-correct module-level constants
    return _DROUGHT_LEVELS, _ZONE_LEVELS


def _patch_model_dict(model_dict: dict, nyc_config):
    """Update DV-affected parameters in a model_dict from NYCOperationsConfig.

    This replaces the full make_model() rebuild by directly setting the
    parameter values that correspond to decision variables.

    Works for both the default 7-level config and N-zone configs — drought
    and storage level names are read dynamically from nyc_config.
    """
    params = model_dict["parameters"]
    drought_levels, storage_levels = _config_levels(nyc_config)

    # --- Constants ---
    # MRF baselines
    for key in ["mrf_baseline_cannonsville", "mrf_baseline_pepacton",
                "mrf_baseline_neversink", "mrf_baseline_delMontague",
                "mrf_baseline_delTrenton"]:
        params[key]["value"] = nyc_config.get_constant(key)

    # Max NYC delivery
    params["max_flow_baseline_delivery_nyc"]["value"] = nyc_config.get_constant(
        "max_flow_baseline_delivery_nyc"
    )

    # Drought delivery factors (NYC and NJ, all levels from config)
    for level in drought_levels:
        params[f"{level}_factor_delivery_nyc"]["value"] = float(
            nyc_config.get_constant(f"{level}_factor_delivery_nyc")
        )
        params[f"{level}_factor_delivery_nj"]["value"] = float(
            nyc_config.get_constant(f"{level}_factor_delivery_nj")
        )

    # Flood release limits (CFS→MGD conversion)
    for res in _NYC_RESERVOIRS_OPT:
        cfs_val = nyc_config.get_constant(f"flood_max_release_{res}_cfs")
        params[f"flood_max_release_{res}"]["value"] = cfs_val * _CFS_TO_MGD

    # --- Daily profiles (366 values) ---
    # Storage zone thresholds
    for level in storage_levels:
        params[level]["values"] = nyc_config.get_storage_zone_profile(level).tolist()

    # MRF daily factor profiles (per reservoir × per level)
    for level in drought_levels:
        for res in _NYC_RESERVOIRS_OPT:
            key = f"{level}_factor_mrf_{res}"
            if key in params:
                params[key]["values"] = nyc_config.get_mrf_factor_profile(
                    key, daily=True
                ).tolist()

    # --- Monthly profiles (12 values) ---
    # MRF monthly factor profiles (Montague & Trenton × per level)
    for level in drought_levels:
        for loc in _DOWNSTREAM_LOCS:
            key = f"{level}_factor_mrf_{loc}"
            if key in params:
                params[key]["values"] = nyc_config.get_mrf_factor_profile(
                    key, daily=False
                ).tolist()

    # --- Salt-front parameter substitution (FFMP-family + salinity on) ---
    _patch_salt_front_parameter(model_dict, nyc_config)


def _patch_salt_front_parameter(model_dict: dict, nyc_config) -> None:
    """Substitute the upstream salt-front parameter for our parameterized
    subclass and inject DV-derived values.

    Reads salt-front options stashed on `nyc_config` by `_stash_salt_front_options`.
    Idempotent: safe to call on either a fresh or an already-patched model
    dict. No-op when no salt-front options are stashed (mode=fixed or
    no-salinity path).
    """
    sf_options = getattr(nyc_config, "_salt_front_options", None)
    if sf_options is None:
        return
    params = model_dict.get("parameters", {})
    for loc in ("delMontague", "delTrenton"):
        key = f"flow_target_salt_front_adjustment_ratio_{loc}"
        if key not in params:
            continue
        params[key]["type"] = "NYCOptParameterizedSaltFrontAdjustmentRatio"
        params[key]["multipliers"] = sf_options["multipliers"]
        params[key]["rm_band_thresholds"] = sf_options["rm_band_thresholds"]
        params[key]["nyc_drought_emergency_level"] = sf_options["activation_level"]


###############################################################################
# N-zone Config Builder
###############################################################################

def build_nzone_config(n_zones):
    """Build NYCOperationsConfig with N storage zones via pywrdrb's native interpolation.

    Delegates to pywrdrb's NYCOperationsConfig.from_n_zones(N), which linearly
    interpolates the 6-curve FFMP defaults to N boundary curves (producing N+1
    drought levels zone_0..zone_N).

    Args:
        n_zones: Number of storage zone boundary curves (>= 2).

    Returns:
        NYCOperationsConfig instance.
    """
    from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig
    return NYCOperationsConfig.from_n_zones(n_zones)


###############################################################################
# Decision Variable -> NYCOperationsConfig Conversion
###############################################################################

def dvs_to_config(dv_vector, formulation_name="ffmp"):
    """Convert a flat decision variable vector to a NYCOperationsConfig.

    Uses cached defaults to avoid re-reading CSVs on every evaluation.

    Args:
        dv_vector: Array-like of decision variable values.
        formulation_name: Name of the formulation to use.

    Returns:
        NYCOperationsConfig instance.
    """
    import copy as _copy

    var_names = get_var_names(formulation_name)
    params = dict(zip(var_names, dv_vector))

    if formulation_name == "ffmp":
        base = _get_cached_defaults()
        config = _copy.deepcopy(base)
        _apply_ffmp_params(config, params)
    elif formulation_name.startswith("ffmp_"):
        n_zones = int(formulation_name.split("_")[1])
        base = _get_cached_nzone_defaults(n_zones)
        config = _copy.deepcopy(base)
        _apply_nzone_ffmp_params(config, params)
    else:
        raise NotImplementedError(
            f"Formulation '{formulation_name}' not yet implemented."
        )
    return config


def _stash_salt_front_options(config, params: dict) -> None:
    """Compute salt-front DV-derived options and stash on the config object.

    Reads salt-front-DV-named entries from `params` (any subset present),
    composes the multiplier table / RM thresholds / activation level via
    `apply_salt_front_dvs`, and attaches the resulting dict to `config`
    under the `_salt_front_options` attribute. Picked up later by
    `_patch_model_dict` to substitute the parameter type and inject values.

    No-op when `SALT_FRONT_PARAM_MODE == "fixed"` (no salt-front DVs were
    appended to the formulation, so `params` won't contain them).

    Activation-level resolution is N-zone aware: when the activation level is
    NOT a DV, we use `config.n_drought_levels - 1` (which is 6 for stock FFMP
    and N+0 for FFMP_VR(N=N+0)... e.g. 8 for N=8, 12 for N=12). This matches
    the upstream `model_builder.py:2704` default and ensures the rule fires
    at the actual drought-emergency band regardless of N.
    `SALT_FRONT_FIXED_ACTIVATION_LEVEL` is honored only when `config` does
    not expose `n_drought_levels` (defensive fallback).
    """
    from config import (
        SALT_FRONT_PARAM_MODE,
        SALT_FRONT_FIXED_ACTIVATION_LEVEL,
    )
    if SALT_FRONT_PARAM_MODE == "fixed":
        return
    sf_params = {k: v for k, v in params.items() if k.startswith("sf_")}
    n_drought = getattr(config, "n_drought_levels", None)
    if n_drought is not None:
        fixed_level = int(n_drought) - 1
    else:
        fixed_level = SALT_FRONT_FIXED_ACTIVATION_LEVEL
    sf_options = apply_salt_front_dvs(
        sf_params,
        fixed_activation_level=fixed_level,
    )
    object.__setattr__(config, "_salt_front_options", sf_options)


def _apply_ffmp_params(config, params: dict):
    """Apply Formulation A (parameterized FFMP) parameters to config.

    Maps flat DV parameters to NYCOperationsConfig.update_*() methods.
    Method signatures verified against pywrdrb source (nyc_operations_config.py).
    """
    # MRF baselines are fixed at the FFMP defaults (config.constants values
    # from the pywrdrb constants CSV) — not DVs. Montague/Trenton baseline
    # targets are 1954-Decree quantities, likewise fixed; only the
    # drought-zone adjustment factors are optimized (below).

    # Delivery constraints: factor arrays have 7 elements for levels 1a, 1b,
    # 1c, 2, 3, 4, 5. The DVs are stage-wise allocation reductions (decode rule
    # in src/formulations/ffmp.py): effective factor = 1 minus the running sum
    # (NYC L3-L5, NJ L4-L5). NYC L1a-L2 defaults are 1,000,000 (unconstrained),
    # NJ L1a-L3 defaults are 1.0; both must come from config.constants.
    defaults = config.constants
    _NYC_EXPECTED_KEYS = [
        "level1a_factor_delivery_nyc", "level1b_factor_delivery_nyc",
        "level1c_factor_delivery_nyc", "level2_factor_delivery_nyc",
    ]
    _NJ_EXPECTED_KEYS = [
        "level1a_factor_delivery_nj", "level1b_factor_delivery_nj",
        "level1c_factor_delivery_nj", "level2_factor_delivery_nj",
        "level3_factor_delivery_nj",
    ]
    for key in _NYC_EXPECTED_KEYS + _NJ_EXPECTED_KEYS:
        if key not in defaults:
            raise KeyError(
                f"Expected key '{key}' not found in config.constants. "
                f"NYCOperationsConfig.from_defaults() may not have loaded "
                f"the constants CSV correctly. Available keys: "
                f"{sorted(k for k in defaults if 'factor_delivery' in k)}"
            )

    nyc_l3 = 1.0 - params["nyc_allocation_reduction_L3"]
    nyc_l4 = nyc_l3 - params["nyc_allocation_reduction_L4"]
    nyc_l5 = nyc_l4 - params["nyc_allocation_reduction_L5"]
    nyc_factors = np.array([
        float(defaults["level1a_factor_delivery_nyc"]),  # 1,000,000 (unconstrained)
        float(defaults["level1b_factor_delivery_nyc"]),  # 1,000,000
        float(defaults["level1c_factor_delivery_nyc"]),  # 1,000,000
        float(defaults["level2_factor_delivery_nyc"]),   # 1,000,000
        nyc_l3,
        nyc_l4,
        nyc_l5,
    ])
    nj_l4 = 1.0 - params["nj_allocation_reduction_L4"]
    nj_l5 = nj_l4 - params["nj_allocation_reduction_L5"]
    nj_factors = np.array([
        float(defaults["level1a_factor_delivery_nj"]),   # 1.0
        float(defaults["level1b_factor_delivery_nj"]),   # 1.0
        float(defaults["level1c_factor_delivery_nj"]),   # 1.0
        float(defaults["level2_factor_delivery_nj"]),    # 1.0
        float(defaults["level3_factor_delivery_nj"]),    # 1.0
        nj_l4,
        nj_l5,
    ])
    # The NYC diversion cap is Decree-fixed (800 MGD), not a DV.
    config.update_delivery_constraints(
        max_nyc_delivery=NYC_DECREE_DIVERSION_CAP_MGD,
        drought_factors_nyc=nyc_factors,
        drought_factors_nj=nj_factors,
    )

    # Storage zone shifts (low-plateau vertical shift and one temporal shift
    # per curve; high-plateau shifts exist only for curves whose refill
    # plateau sits below capacity — level3/4/5 — the rest default to 0.0)
    zone_levels = ["level1b", "level1c", "level2", "level3", "level4", "level5"]
    vshifts_lower = {level: params.get(f"zone_vshift_{level}_lower", 0.0)
                     for level in zone_levels}
    vshifts_upper = {level: params.get(f"zone_vshift_{level}_upper", 0.0)
                     for level in zone_levels}
    tshifts = {level: params.get(f"zone_tshift_{level}", 0.0)
               for level in zone_levels}
    _apply_zone_shifts(config, vshifts_lower, vshifts_upper, tshifts)

    # MRF seasonal profile scaling (conservation zones only), then
    # flood-zone (L1a/L1b) release scaling on the still-pristine flood rows.
    # The flood_max_release_{res}_cfs constants (FFMP Table 5) are fixed —
    # no update_flood_limits call.
    _apply_mrf_profile_scaling(config, params)
    _apply_flood_release_scaling(config, params)

    # Downstream flow-target factor scaling (Montague/Trenton, per level)
    _apply_flow_target_scaling(config, params)

    # Stash salt-front DV-derived options for downstream model-dict patching.
    _stash_salt_front_options(config, params)


def _apply_nzone_ffmp_params(config, params: dict):
    """Apply N-zone FFMP params to an NYCOperationsConfig with zone_0..zone_N naming.

    Mirrors _apply_ffmp_params but works with any N-zone config built by
    build_nzone_config(). DV names use zone_{i} naming; missing DV keys fall
    back to the interpolated defaults already stored in config.constants.
    """
    # MRF baselines fixed at defaults; Montague/Trenton targets Decree-fixed
    # (as in the base path).

    # Delivery constraints — decode stage-wise allocation reductions into
    # factor arrays (see _apply_ffmp_params). A level is reduction-decoded
    # iff its interpolated default factor is below the party's unconstrained
    # gate — the same rule generate_ffmp_formulation uses to emit DVs — and
    # a missing DV key falls back to the level's default increment.
    drought_levels, storage_levels_nz = _config_levels(config)

    def _reduced_factors(party: str, gate: float) -> np.ndarray:
        factors = []
        prev_default = 1.0
        prev_factor = 1.0
        for level in drought_levels:
            default = float(config.constants[f"{level}_factor_delivery_{party}"])
            if default >= gate:
                factors.append(default)
                continue
            default = min(default, 1.0)
            reduction = float(params.get(
                f"{party}_allocation_reduction_{level}",
                prev_default - default,
            ))
            prev_factor = prev_factor - reduction
            factors.append(prev_factor)
            prev_default = default
        return np.array(factors)

    nyc_factors = _reduced_factors("nyc", 100.0)
    nj_factors = _reduced_factors("nj", 1.0)
    config.update_delivery_constraints(
        max_nyc_delivery=NYC_DECREE_DIVERSION_CAP_MGD,
        drought_factors_nyc=nyc_factors,
        drought_factors_nj=nj_factors,
    )

    # Zone shifts (low-plateau vertical shift and one temporal shift per
    # curve; high-plateau shift DVs exist only for curves whose refill plateau
    # sits below capacity — the rest default to 0.0)
    vshifts_lower = {level: params.get(f"zone_vshift_{level}_lower", 0.0)
                     for level in storage_levels_nz}
    vshifts_upper = {level: params.get(f"zone_vshift_{level}_upper", 0.0)
                     for level in storage_levels_nz}
    tshifts = {level: params.get(f"zone_tshift_{level}", 0.0)
               for level in storage_levels_nz}
    _apply_zone_shifts(config, vshifts_lower, vshifts_upper, tshifts)

    # MRF seasonal scaling (conservation zones only), then flood-zone
    # release scaling. Table 5 flood caps stay at their constants.
    _apply_mrf_profile_scaling(config, params)
    _apply_flood_release_scaling(config, params)

    # Downstream flow-target factor scaling (Montague/Trenton, per level)
    _apply_flow_target_scaling(config, params)

    # Stash salt-front DV-derived options for downstream model-dict patching.
    _stash_salt_front_options(config, params)


def _apply_zone_shifts(config, vshifts_lower: dict,
                       vshifts_upper: dict = None, tshifts: dict = None):
    """Apply two vertical (per-plateau) and one temporal shift per curve.

    Each storage curve is a trapezoid with a low plateau (its baseline minimum,
    the fall/winter void) and a high plateau (its baseline maximum, the
    spring/summer refill target). The two plateau levels are moved independently
    — the low plateau by ``vshifts_lower`` and the high plateau by
    ``vshifts_upper`` — and the daily values are affinely remapped between the
    new plateau levels, so the two ramps re-interpolate to connect and the
    trapezoidal shape is preserved (no new kinks). This decouples void depth
    from refill target. A within-curve clamp keeps the low plateau at or below
    the high plateau (void <= refill); if a shift would invert them the curve
    flattens to the high-plateau level. Each curve is then slid along the
    day-of-year axis by its temporal-shift DV (a circular roll, rounded to whole
    days), clipped to [0, 1], and cross-curve monotonicity-clamped (each more
    severe level's curve is capped at the less severe one's).

    Args:
        config: NYCOperationsConfig to mutate.
        vshifts_lower: Mapping level -> additive low-plateau shift
            (zone_vshift_{level}_lower DV, fraction of capacity).
        vshifts_upper: Mapping level -> additive high-plateau shift
            (zone_vshift_{level}_upper DV, fraction of capacity).
        tshifts: Mapping level -> temporal shift in days
            (zone_tshift_{level} DV). Positive shifts move the curve to
            later days of the year.

    Operates on config.storage_zones_df (rows=levels, 366 daily columns).
    Works for both default level1b..level5 and N-zone zone_1..zone_N naming.
    """
    vshifts_upper = vshifts_upper or {}
    tshifts = tshifts or {}
    _, zone_order = _config_levels(config)
    zones = config.storage_zones_df.copy()
    date_cols = [c for c in zones.columns if c != "doy"]

    for level in zone_order:
        if level not in zones.index:
            continue
        baseline_row = zones.loc[level, date_cols].values.astype(float)
        lo, hi = baseline_row.min(), baseline_row.max()
        vlow = float(vshifts_lower.get(level, 0.0))
        vup = float(vshifts_upper.get(level, 0.0))
        tshift = int(round(float(tshifts.get(level, 0.0))))
        hi_new = hi + vup
        # Within-curve clamp: the void level may not exceed the refill target.
        lo_new = min(lo + vlow, hi_new)
        span = hi - lo
        if span > _ZONE_FLAT_EPS:
            remapped = lo_new + (baseline_row - lo) / span * (hi_new - lo_new)
        else:
            remapped = np.full_like(baseline_row, lo_new)
        zones.loc[level, date_cols] = np.roll(remapped, tshift)

    zones[date_cols] = zones[date_cols].clip(lower=0.0, upper=1.0)

    # Enforce monotonicity: more severe levels must be <= less severe
    for i in range(1, len(zone_order)):
        more_severe = zone_order[i]
        less_severe = zone_order[i - 1]
        if more_severe in zones.index and less_severe in zones.index:
            zones.loc[more_severe, date_cols] = np.minimum(
                zones.loc[more_severe, date_cols].values.astype(float),
                zones.loc[less_severe, date_cols].values.astype(float),
            )
    config.storage_zones_df = zones


def _apply_mrf_profile_scaling(config, params: dict):
    """Apply seasonal scaling to MRF daily factor profiles.

    Operates on config.mrf_factors_daily_df. The DataFrame is loaded from
    ffmp_reservoir_operation_daily_profiles.csv (index_col='profile') which
    contains both storage zone rows and MRF factor rows. This function
    scales all rows EXCEPT the flood-zone (L1a/L1b) reservoir release rows,
    which are controlled exclusively by the flood_release_scale_* DVs
    (_apply_flood_release_scaling). Storage zone rows are also scaled but
    unaffected in practice because config.storage_zones_df is a separate
    copy that was already modified by _apply_zone_shifts.

    NOTE: In pywrdrb from_defaults(), mrf_factors_daily_df is initialized
    as storage_zones_df.copy(). Both reference the same CSV. This is the
    pywrdrb design — the daily profiles CSV contains all profile types
    (zone thresholds and MRF factors) indexed by the 'profile' column.
    """
    season_ranges = _SEASON_DOY_RANGES
    mrf_factors = config.mrf_factors_daily_df.copy()

    # Flood zones = drought level indices below flood_conservation_boundary
    # (= 2): level1a/level1b for standard FFMP, zone_0/zone_1 for N-zone.
    drought_levels, _ = _config_levels(config)
    flood_rows = {f"{lev}_factor_mrf_{res}"
                  for lev in drought_levels[:2] for res in _NYC_RESERVOIRS_OPT}
    scalable = ~mrf_factors.index.isin(flood_rows)

    # The DataFrame columns are date strings (e.g. "1-Jan") or DOY integers,
    # NOT including a "doy" column (index_col='profile' was used at load).
    # Filter defensively in case future versions add metadata columns.
    numeric_or_date_cols = [c for c in mrf_factors.columns
                           if c not in ("doy", "profile", "type")]

    if len(numeric_or_date_cols) < 365:
        import warnings
        warnings.warn(
            f"mrf_factors_daily_df has only {len(numeric_or_date_cols)} data columns "
            f"(expected 365-366). Columns: {mrf_factors.columns.tolist()[:5]}... "
            f"MRF seasonal scaling may not work correctly. Verify the CSV structure "
            f"of ffmp_reservoir_operation_daily_profiles.csv."
        )

    for season, doy_range in season_ranges.items():
        scale = params[f"mrf_profile_scale_{season}"]
        # Map day-of-year to column indices (DOY is 1-indexed, list is 0-indexed)
        cols_to_scale = [numeric_or_date_cols[d - 1] for d in doy_range
                         if d - 1 < len(numeric_or_date_cols)]
        mrf_factors.loc[scalable, cols_to_scale] *= scale

    config.mrf_factors_daily_df = mrf_factors


def _apply_flood_release_scaling(config, params: dict):
    """Scale the flood-zone (L1a/L1b) reservoir release factor rows.

    DVs named ``flood_release_scale_{l1a|l1b}_{res}`` multiply the DEFAULT
    flood-zone release schedule (FFMP Tables 4a-4g spill-mitigation rows),
    season-invariant — matching the FFMP, which holds these rows constant
    across its tables and seasons. The within-year shape (L1a-absent window
    Apr 16-Jun 15, Neversink's L1b step) is preserved by the multiplier
    form; seasonal flood policy is carried by the zone-boundary shift DVs.
    The mrf_baseline_{res} constants are fixed (not DVs), so scaling the
    factor rows scales the effective release rates directly.

    Guardrails, applied in effective-MGD space: cap at the Table 5
    combined-discharge constant (flood_max_release_{res}_cfs), then clamp
    effective L1b <= L1a elementwise. Flood zones are drought_levels[0]
    and [1] (indices below flood_conservation_boundary = 2): level1a and
    level1b for the standard FFMP, zone_0 and zone_1 for N-zone variants.
    Missing DV keys default to 1.0 (no-op for formulations without them).

    Must run after _apply_mrf_profile_scaling, which leaves the flood-zone
    rows pristine (default values) for this function to read.
    """
    drought_levels, _ = _config_levels(config)
    flood_levels = {"l1a": drought_levels[0], "l1b": drought_levels[1]}
    factors = config.mrf_factors_daily_df.copy()
    date_cols = [c for c in factors.columns
                 if c not in ("doy", "profile", "type")]

    for res in _NYC_RESERVOIRS_OPT:
        baseline = float(config.constants[f"mrf_baseline_{res}"])
        cap_mgd = (float(config.constants[f"flood_max_release_{res}_cfs"])
                   * _CFS_TO_MGD)
        eff = {}
        for key, level in flood_levels.items():
            row = factors.loc[
                f"{level}_factor_mrf_{res}", date_cols
            ].values.astype(float)
            mult = float(params.get(f"flood_release_scale_{key}_{res}", 1.0))
            eff[key] = np.minimum(row * baseline * mult, cap_mgd)
        eff["l1b"] = np.minimum(eff["l1b"], eff["l1a"])
        for key, level in flood_levels.items():
            factors.loc[f"{level}_factor_mrf_{res}", date_cols] = (
                eff[key] / baseline
            )

    config.mrf_factors_daily_df = factors


def _apply_flow_target_scaling(config, params: dict):
    """Scale the Montague/Trenton monthly flow-target factor tables per DV.

    DVs named ``mrf_target_scale_{montague|trenton}_{level}`` multiply the
    corresponding drought level's default monthly factors across all months
    (non-seasonal). The effective factor is capped at 1.0 so an adjusted
    target never exceeds the Decree-fixed baseline target. Levels without a
    matching DV (normal-operation levels, factors == 1.0) are left untouched.
    Missing DV keys default to 1.0 (no change), which keeps this a no-op for
    formulations without these DVs.

    Operates on config.mrf_factors_monthly_df, which _patch_model_dict
    reads back via get_mrf_factor_profile(..., daily=False).
    """
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    loc_keys = {"montague": "delMontague", "trenton": "delTrenton"}
    drought_levels, _ = _config_levels(config)

    monthly = config.mrf_factors_monthly_df.copy()
    for loc, loc_key in loc_keys.items():
        for level in drought_levels:
            row = f"{level}_factor_mrf_{loc_key}"
            if row not in monthly.index:
                continue
            scale = params.get(f"mrf_target_scale_{loc}_{level}", 1.0)
            monthly.loc[row, months] = np.minimum(
                monthly.loc[row, months].values.astype(float) * scale,
                1.0,
            )
    config.mrf_factors_monthly_df = monthly


###############################################################################
# Formal Borg constraints (pure DV arithmetic — no simulation)
###############################################################################

# Violations at or below this magnitude are returned as exact 0.0. Borg
# treats ANY nonzero constraint value as infeasible, so arithmetic rounding
# noise must not leak into the feasibility signal.
_CONSTRAINT_VIOLATION_TOL = 1e-9


def _constraint_defaults(formulation_name: str):
    """Return the cached pristine defaults config for a formulation."""
    if formulation_name == "ffmp":
        return _get_cached_defaults()
    if formulation_name.startswith("ffmp_"):
        return _get_cached_nzone_defaults(int(formulation_name.split("_")[1]))
    raise NotImplementedError(
        f"Formulation '{formulation_name}' not yet implemented."
    )


def _flood_zone_ordering_violation(cfg, params: dict) -> float:
    """Worst-day exceedance of effective L1b over L1a, summed over reservoirs.

    Mirrors ``_apply_flood_release_scaling`` elementwise in effective-MGD
    space (default factor rows x baseline x scale DV, Table 5 cap applied),
    normalized by the reservoir MRF baseline so the value is dimensionless
    and order-1. Zero exactly when the L1b <= L1a clamp is a no-op.
    """
    drought_levels, _ = _config_levels(cfg)
    flood_levels = {"l1a": drought_levels[0], "l1b": drought_levels[1]}
    factors = cfg.mrf_factors_daily_df
    date_cols = [c for c in factors.columns
                 if c not in ("doy", "profile", "type")]

    total = 0.0
    for res in _NYC_RESERVOIRS_OPT:
        baseline = float(cfg.constants[f"mrf_baseline_{res}"])
        cap_mgd = (float(cfg.constants[f"flood_max_release_{res}_cfs"])
                   * _CFS_TO_MGD)
        eff = {}
        for key, level in flood_levels.items():
            row = factors.loc[
                f"{level}_factor_mrf_{res}", date_cols
            ].values.astype(float)
            mult = float(params.get(f"flood_release_scale_{key}_{res}", 1.0))
            eff[key] = np.minimum(row * baseline * mult, cap_mgd)
        total += max(0.0, float((eff["l1b"] - eff["l1a"]).max())) / baseline
    return total


def compute_constraint_violations(dv_vector,
                                  formulation_name: str = "ffmp") -> list:
    """Compute the DV-SPACE formal Borg constraint violations for a DV vector.

    Pure DV arithmetic on the cached defaults config (no model build or
    simulation). Each value is a violation magnitude: 0.0 = feasible,
    positive values scale with the degree of violation. Covers only
    ``src.formulations.DV_CONSTRAINT_NAMES``; the post-simulation constraint
    (``nyc_reliability_floor``) lives in
    ``src.formulations.make_post_sim_constraint_function``.

    Args:
        dv_vector: Array-like of decision variable values.
        formulation_name: "ffmp" or "ffmp_N".

    Returns:
        Single-element list [flood_zone_ordering]. Violations at or below
        ``_CONSTRAINT_VIOLATION_TOL`` are returned as exact 0.0.
    """
    cfg = _constraint_defaults(formulation_name)
    params = dict(zip(get_var_names(formulation_name), dv_vector))
    violations = [
        _flood_zone_ordering_violation(cfg, params),
    ]
    return [0.0 if v <= _CONSTRAINT_VIOLATION_TOL else float(v)
            for v in violations]


###############################################################################
# Model Building Helpers
###############################################################################

def _build_model_builder(nyc_config, use_trimmed: bool = None,
                         ensemble_spec=None):
    """Create and configure a ModelBuilder. Shared by both simulation paths.

    Args:
        nyc_config: NYCOperationsConfig instance.
        use_trimmed: Whether to use the trimmed model. If None, falls back to
            USE_TRIMMED_MODEL from config. Trimmed mode requires that
            00_generate_presim.sh has been run first.
        ensemble_spec: Optional EnsembleSpec. When ``is_ensemble=True``, the
            ModelBuilder is configured with the ensemble's ``inflow_type``
            (which routes pywrdrb's path navigator to the staged HDF5 dir)
            and ``inflow_ensemble_indices`` so pywr instantiates one scenario
            per requested realization. When None or ``is_ensemble=False``,
            the single-trace ``INFLOW_TYPE`` from config is used.
    """
    import pywrdrb

    if use_trimmed is None:
        use_trimmed = USE_TRIMMED_MODEL

    options = {
        "nyc_nj_demand_source": NYC_NJ_DEMAND_SOURCE,
        "use_trimmed_model": use_trimmed,
        "initial_volume_frac": INITIAL_VOLUME_FRAC,
        # Pinned explicitly for EVERY simulation; never rely on pywrdrb's default.
        "flow_prediction_mode": PYWRDRB_FLOW_PREDICTION_MODE,
        # Enable downstream stage recorders at Hale Eddy / Fishs Eddy /
        # Bridgeville. Required by the action-stage flood objective.
        "enable_nyc_flood_operations": True,
    }

    if use_trimmed:
        # Single-trace path: pin the project-local presim CSV.
        # Ensemble path: leave presimulated_releases_file unset so pywrdrb's
        # ModelBuilder auto-routes to {flows/inflow_type}/presimulated_releases_mgd.hdf5
        # (written by STARFITReleaseEnsemblePreprocessor); pywrdrb then wires
        # PresimulatedReleaseEnsemble parameters to that artifact.
        if ensemble_spec is None or not ensemble_spec.is_ensemble:
            presim_file = _require_presim_file()
            options["presimulated_releases_file"] = str(presim_file)

    # T/S LSTM options. Empty dict if both toggles are off, so this merge
    # is a no-op for the standard objective set.
    options.update(build_lstm_options_block())

    # Ensemble routing: register staged HDF5 directory with pywrdrb's path
    # navigator and pass realization indices through ModelBuilder's options
    # dict (pywrdrb stores it on self.options.inflow_ensemble_indices and
    # uses it to size the scenarios block + instantiate FlowEnsemble /
    # PredictionEnsemble parameters; see Pywr-DRB/.../model_builder.py:547).
    inflow_type_to_use = INFLOW_TYPE
    if ensemble_spec is not None and ensemble_spec.is_ensemble:
        from src.ensembles import register_ensemble_path
        register_ensemble_path(ensemble_spec.inflow_type)
        inflow_type_to_use = ensemble_spec.inflow_type
        options["inflow_ensemble_indices"] = list(
            ensemble_spec.realization_indices
        )

    # Ensemble window: derived from the spec's own staged stamp (meta
    # start_date + realization_years), never from the historic START_DATE,
    # so the pywr timestepper aligns with the staged HDF5 date axis.
    sim_start = START_DATE
    sim_end = END_DATE
    if ensemble_spec is not None and ensemble_spec.is_ensemble:
        sim_start, sim_end = _ensemble_window(ensemble_spec)

    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type_to_use,
        start_date=sim_start,
        end_date=sim_end,
        options=options,
        nyc_operations_config=nyc_config,
    )
    mb.make_model()
    return mb


def _ensemble_window(ensemble_spec) -> tuple[str, str]:
    """Return (start_date, end_date) of an ensemble simulation window.

    Derived entirely from the spec's own staged provenance: the window starts
    at the spec's ``start_date`` (the stamp of day 0, read from the staged
    ``_meta.json``) and ends ``realization_years`` years later minus one day,
    matching the staged HDF5 date axis exactly. The historic
    ``START_DATE``/``END_DATE`` and the ``PYWRDRB_SIM_*`` env overrides apply
    to single-trace (historic) simulations only, never to ensembles.

    Raises:
        ValueError: If the spec lacks ``start_date`` or ``realization_years`` —
            an ensemble window cannot be derived without them.
    """
    if ensemble_spec.start_date is None or ensemble_spec.realization_years is None:
        raise ValueError(
            f"ensemble spec '{ensemble_spec.preset_name}' lacks "
            f"start_date/realization_years "
            f"(start_date={ensemble_spec.start_date!r}, "
            f"realization_years={ensemble_spec.realization_years!r}); the "
            f"simulation window is derived from the staged stamp and length."
        )
    start_ts = pd.Timestamp(ensemble_spec.start_date)
    end_ts = start_ts + pd.DateOffset(years=int(ensemble_spec.realization_years)) - pd.Timedelta(days=1)
    return str(start_ts.date()), str(end_ts.date())


def _write_and_load_model(mb, model_json_path: str):
    """Write model dict to JSON and load as pywr.Model.

    pywr.Model requires loading from a JSON file. There is no dict constructor.
    """
    import pywrdrb
    mb.write_model(model_json_path)
    model = pywrdrb.Model.load(model_json_path)
    return model


###############################################################################
# Per-Rank Temp Directory
###############################################################################

# Cached temp dir per process
_TEMP_DIR = None
_MPI_RANK = None


def _get_mpi_rank() -> int:
    """Return MPI rank (0 if not in an MPI context)."""
    global _MPI_RANK
    if _MPI_RANK is None:
        try:
            from mpi4py import MPI
            _MPI_RANK = MPI.COMM_WORLD.Get_rank()
        except Exception:
            _MPI_RANK = 0
    return _MPI_RANK


def _get_temp_dir() -> str:
    """Get or create a persistent per-process temp directory.

    Uses /dev/shm (RAM-backed tmpfs) when available to avoid NFS I/O
    contention with many MPI workers writing JSON simultaneously.
    Falls back to /tmp if /dev/shm is not writable.
    """
    global _TEMP_DIR
    if _TEMP_DIR is None:
        rank = _get_mpi_rank()
        # Prefer /dev/shm (RAM filesystem) to avoid NFS contention
        shm_dir = "/dev/shm"
        if os.path.isdir(shm_dir) and os.access(shm_dir, os.W_OK):
            _TEMP_DIR = tempfile.mkdtemp(prefix=f"pywrdrb_opt_r{rank}_", dir=shm_dir)
        else:
            _TEMP_DIR = tempfile.mkdtemp(prefix=f"pywrdrb_opt_r{rank}_")
    return _TEMP_DIR


###############################################################################
# In-Memory Recorder (avoids HDF5 disk I/O and its threading side-effects)
###############################################################################

# Minimal-recorder selection: record only the result groups the objectives
# read (see _extract_results_from_recorder); identical objectives, far less
# memory. NYCOPT_MINIMAL_RECORDER=0 records everything.
# NYC/NJ demand+delivery are recorded by exact name; depending on the model
# build these can be either nodes or parameters, so match them in BOTH lists.
_OBJECTIVE_EXACT_NAMES = frozenset({
    "demand_nyc", "demand_nj", "delivery_nyc", "delivery_nj",
})
_OBJECTIVE_PARAM_NAMES = frozenset({
    # salinity / temperature LSTM outputs (present only when those models are on)
    "salt_front_location_mu", "salt_front_location_sd",
    "temperature_after_thermal_release_mu", "temperature_after_thermal_release_sd",
    "thermal_release_requirement",
    "forecasted_temperature_before_thermal_release_mu",
})
_OBJECTIVE_PARAM_PREFIXES = ("mrf_target_", "stage_")


def _use_minimal_recorder() -> bool:
    return os.environ.get("NYCOPT_MINIMAL_RECORDER", "1").lower() not in (
        "0", "false", "no", "off",
    )


def _minimal_recorder_selection(model):
    """Return (nodes, parameters) object lists covering exactly the keys the
    objective extractors read (see _extract_results_from_recorder)."""
    from pywrdrb.utils.lists import reservoir_list, majorflow_list
    res = set(reservoir_list)
    mf = set(majorflow_list)

    def want_node(nm: str) -> bool:
        if nm in _OBJECTIVE_EXACT_NAMES:
            return True
        parts = nm.split("_", 1)
        if len(parts) < 2:
            return False
        head, tail = parts[0], parts[1]
        # res_storage: reservoir_{X in reservoir_list}; major_flow: link_{Y in majorflow_list}
        if head == "reservoir" and nm.split("_")[1] in res:
            return True
        if head == "link" and tail in mf:
            return True
        return False

    def want_param(nm: str) -> bool:
        return (nm in _OBJECTIVE_EXACT_NAMES
                or nm in _OBJECTIVE_PARAM_NAMES
                or nm.startswith(_OBJECTIVE_PARAM_PREFIXES))

    nodes = [n for n in model.nodes.values() if n.name and want_node(n.name)]
    params = [p for p in model.parameters if p.name and want_param(p.name)]
    return nodes, params


class InMemoryRecorder:
    """OutputRecorder wrapper with no-op lifecycle methods and no HDF5 output.

    pywr already calls setup/reset/after/finish on each registered recorder,
    so a second finish() from the wrapper double-frees. No HDF5 is written:
    its background threads corrupt the GIL inside Borg's ctypes callback.
    ``recorder.data`` stays accessible after ``model.run()``.
    """

    def __init__(self, model, minimal=None):
        from pywrdrb.recorder import OutputRecorder

        if minimal is None:
            minimal = _use_minimal_recorder()

        # OutputRecorder registers the individual NumpyArray*Recorders with the
        # pywr model. minimal=True records only the objective-relevant objects.
        if minimal:
            nodes, parameters = _minimal_recorder_selection(model)
            self._inner = OutputRecorder(
                model, output_filename="/dev/null",
                nodes=nodes, parameters=parameters,
            )
        else:
            self._inner = OutputRecorder(model, output_filename="/dev/null")

        # pywr calls setup/reset/after/finish on each registered recorder; the
        # wrapper must not double-call them.
        _noop = lambda: None
        self._inner.setup = _noop
        self._inner.reset = _noop
        self._inner.after = _noop
        self._inner.finish = _noop

    @property
    def recorder_dict(self):
        return self._inner.recorder_dict


###############################################################################
# Extract Results from Recorder (In-Memory)
###############################################################################

def _extract_results_from_recorder(recorder_dict, datetime_index, scenario=0) -> dict:
    """Extract simulation results from recorder dict into DataFrames.

    Args:
        recorder_dict: Dict mapping raw pywr names to NumpyArray*Recorder objects.
        datetime_index: Model timestepper datetime index.
        scenario: Scenario index to extract (default 0 for single-scenario runs).

    Returns:
        Dict of DataFrames keyed by results_set name.
    """
    from pywrdrb.utils.lists import reservoir_list, majorflow_list

    all_keys = list(recorder_dict.keys())
    dt_index = pd.DatetimeIndex(datetime_index)

    def _build_df(key_filter_fn, name_extract_fn, name_filter=None):
        data = {}
        for k in all_keys:
            if key_filter_fn(k):
                name = name_extract_fn(k)
                if name_filter is None or name in name_filter:
                    rec = recorder_dict[k]
                    data[name] = rec.data[:, scenario]
        if not data:
            return pd.DataFrame(index=dt_index)
        return pd.DataFrame(data, index=dt_index)

    results = {}

    results["res_storage"] = _build_df(
        key_filter_fn=lambda k: k.split("_")[0] == "reservoir",
        name_extract_fn=lambda k: k.split("_", 1)[1],
        name_filter=set(reservoir_list),
    )

    results["major_flow"] = _build_df(
        key_filter_fn=lambda k: k.split("_")[0] == "link",
        name_extract_fn=lambda k: k.split("_", 1)[1],
        name_filter=set(majorflow_list),
    )

    demand_data = {}
    for k in ["demand_nyc", "demand_nj"]:
        if k in recorder_dict:
            demand_data[k] = recorder_dict[k].data[:, scenario]
    results["ibt_demands"] = pd.DataFrame(demand_data, index=dt_index)

    delivery_data = {}
    for k in ["delivery_nyc", "delivery_nj"]:
        if k in recorder_dict:
            delivery_data[k] = recorder_dict[k].data[:, scenario]
    results["ibt_diversions"] = pd.DataFrame(delivery_data, index=dt_index)

    # MRF targets (time-dynamic, vary by drought level and month)
    mrf_data = {}
    for k in all_keys:
        if k.startswith("mrf_target_"):
            col = k.split("mrf_target_")[1]
            mrf_data[col] = recorder_dict[k].data[:, scenario]
    results["mrf_target"] = pd.DataFrame(mrf_data, index=dt_index)

    # Downstream stage at reservoir-tail gauges (only when
    # enable_nyc_flood_operations=True). Columns are gauge IDs:
    # 01426500 (Hale Eddy, below Cannonsville), 01421000 (Fishs Eddy,
    # below Pepacton), 01436690 (Bridgeville, below Neversink).
    stage_data = {}
    for k in all_keys:
        if k.startswith("stage_"):
            col = k.split("stage_", 1)[1]
            stage_data[col] = recorder_dict[k].data[:, scenario]
    if stage_data:
        results["flood_stage"] = pd.DataFrame(stage_data, index=dt_index)

    # Salinity LSTM outputs (only present when INCLUDE_SALINITY_MODEL is on).
    # The published parameter name is `salt_front_location_mu` (river mile,
    # 7-day average). Pre-LSTM-start dates produce NaN; downstream metrics
    # must dropna() rather than treating NaN as a real reading.
    salinity_keys = [
        "salt_front_location_mu", "salt_front_location_sd",
    ]
    sal_data = {k: recorder_dict[k].data[:, scenario]
                for k in salinity_keys if k in recorder_dict}
    if sal_data:
        results["salinity"] = pd.DataFrame(sal_data, index=dt_index)

    # Temperature LSTM outputs (only when INCLUDE_TEMPERATURE_MODEL is on;
    # currently inactive — the thermal metric is deferred).
    temperature_keys = [
        "temperature_after_thermal_release_mu",
        "temperature_after_thermal_release_sd",
        "thermal_release_requirement",
        "forecasted_temperature_before_thermal_release_mu",
    ]
    temp_data = {k: recorder_dict[k].data[:, scenario]
                 for k in temperature_keys if k in recorder_dict}
    if temp_data:
        results["temperature"] = pd.DataFrame(temp_data, index=dt_index)

    return results


def _extract_results_per_scenario(recorder_dict, datetime_index,
                                  n_scenarios: int) -> list:
    """Extract simulation results for every scenario in a multi-realization run.

    Calls ``_extract_results_from_recorder`` once per scenario index in
    ``[0, n_scenarios)`` and returns a list of N data dicts (one per
    realization). Each dict is shape-identical to the single-trace
    return so existing metric functions work unchanged.

    Salinity LSTM extraction is handled by the ensemble runner (a per-
    scenario ``_extract_salinity_records`` loop) rather than here, since the
    LSTM records live on the model — not on the recorder dict.
    """
    return [
        _extract_results_from_recorder(recorder_dict, datetime_index,
                                       scenario=s)
        for s in range(n_scenarios)
    ]


def _extract_salinity_records(model, datetime_index, results: dict,
                              scenario: int = 0) -> None:
    """Extract per-sim-day sf_mu/sf_sd from the salinity LSTM after model.run().

    In sync mode with ``debug=True`` the LSTM records each sim day's
    prediction in ``ml_model.records`` at ``t = sim_day_index``.
    ``records["sf_mu"]`` is ``(n_sim, n_scenarios)``; slice
    ``[:n_sim, scenario]`` and pair with the datetime index, replacing
    ``results["salinity"]``. Async mode is unsupported (``ml_model.t`` never
    advances during the run loop).

    Mutates `results` in place when salinity is enabled.
    """
    if not INCLUDE_SALINITY_MODEL:
        return
    try:
        salinity_param = model.parameters["salinity_model"]
    except (KeyError, AttributeError):
        return
    ml_model = getattr(salinity_param, "ml_model", None)
    if ml_model is None or not getattr(ml_model, "debug", False):
        return

    n_sim = len(datetime_index)
    sim_index = pd.DatetimeIndex(datetime_index)

    # Sync mode advances `ml_model.t` once per sim day after the first (day 1
    # is gate-skipped), so the expected count is `n_sim - 1`.
    if ml_model.t < n_sim - 1:
        print(f"  [salinity extract] WARN: ml_model.t={ml_model.t} < n_sim-1={n_sim-1}; "
              f"records likely incomplete (async mode?). Leaving recorder-based "
              f"data['salinity'] in place.")
        return

    sf_mu_full = np.asarray(ml_model.records.get("sf_mu", []), dtype=float)
    sf_sd_full = np.asarray(ml_model.records.get("sf_sd", []), dtype=float)
    if sf_mu_full.ndim != 2 or sf_mu_full.shape[0] < n_sim:
        print(f"  [salinity extract] WARN: records['sf_mu'] shape "
              f"{sf_mu_full.shape}; expected (>={n_sim}, n_scenarios). Skipping.")
        return
    if scenario >= sf_mu_full.shape[1]:
        print(f"  [salinity extract] WARN: scenario={scenario} out of range "
              f"for records shape {sf_mu_full.shape}. Skipping.")
        return

    sf_mu = sf_mu_full[:n_sim, scenario]
    sf_sd = sf_sd_full[:n_sim, scenario]

    results["salinity"] = pd.DataFrame(
        {
            "salt_front_location_mu": sf_mu,
            "salt_front_location_sd": sf_sd,
        },
        index=sim_index,
    )


###############################################################################
# In-Memory Simulation (for optimization)
###############################################################################

def run_simulation_inmemory(nyc_config, use_trimmed: bool = None) -> dict:
    """Run Pywr-DRB simulation with no HDF5 disk I/O.

    Uses cached model dict + parameter patching to avoid rebuilding the
    model from scratch on every evaluation. The base model_dict is built
    once (first call), then deep-copied and patched with DV-specific
    parameter values for each subsequent evaluation.

    Uses /dev/shm for temp JSON to minimize I/O contention under MPI.

    Args:
        nyc_config: NYCOperationsConfig instance.
        use_trimmed: Use trimmed model. Defaults to USE_TRIMMED_MODEL from config.

    Returns:
        Dict of DataFrames keyed by results set name.
    """
    import pywrdrb

    rank = _get_mpi_rank()
    tmp_dir = _get_temp_dir()
    model_json = str(Path(tmp_dir) / f"opt_model_r{rank}.json")

    # Deep-copy cached base model dict and patch with this eval's parameters.
    # Pass nyc_config so that N-zone configs get their own correctly-named dict.
    base_dict = _get_cached_model_dict(use_trimmed=use_trimmed, nyc_config=nyc_config)
    model_dict = copy.deepcopy(base_dict)
    _patch_model_dict(model_dict, nyc_config)

    # Write patched dict to JSON and load (pywr requires JSON file)
    with open(model_json, "w") as f:
        json.dump(model_dict, f)
    model = pywrdrb.Model.load(model_json)

    mem_recorder = InMemoryRecorder(model)
    model.run()

    # Access recorder data BEFORE deleting model (datetime_index lives on model)
    datetime_index = model.timestepper.datetime_index.to_timestamp()
    data = _extract_results_from_recorder(mem_recorder.recorder_dict, datetime_index)
    _extract_salinity_records(model, datetime_index, data)

    del model, mem_recorder
    return data


def run_simulation_ensemble_inmemory(nyc_config, ensemble_spec) -> list:
    """Run Pywr-DRB simulation across an inflow ensemble; no HDF5 disk I/O.

    Mirrors :func:`run_simulation_inmemory` with three differences:

      1. The cached base model_dict is keyed on the ensemble preset name +
         DU factor signature, so different presets cannot cross-contaminate.
      2. ``ModelBuilder`` is constructed with
         ``inflow_type=ensemble_spec.inflow_type`` and
         ``inflow_ensemble_indices=list(ensemble_spec.realization_indices)``,
         which routes pywrdrb's ``FlowEnsemble`` and ``PredictionEnsemble``
         parameters to the staged HDF5s under ``STAGED_ENSEMBLE_DIR``.
      3. Returns a list of N data dicts (one per scenario) instead of a
         single dict. Each dict has the same shape as
         :func:`run_simulation_inmemory`'s output, so existing metric
         functions in ``src/objectives.py`` work unchanged when wrapped by
         an ``AnnualUnitObjective`` from ``src/objectives_ensemble.py``.
         Salinity records are extracted per realization via
         :func:`_extract_salinity_records`.

    Args:
        nyc_config: NYCOperationsConfig instance (DV-applied).
        ensemble_spec: ``EnsembleSpec`` with ``is_ensemble=True``.

    Returns:
        list[dict] of length ``ensemble_spec.n_realizations``. Each dict
        has the same keys as :func:`run_simulation_inmemory`'s return.
    """
    import pywrdrb

    if not ensemble_spec.is_ensemble:
        raise ValueError(
            f"run_simulation_ensemble_inmemory called with is_ensemble=False "
            f"preset '{ensemble_spec.preset_name}'. Use run_simulation_inmemory "
            f"for the single-trace path."
        )

    rank = _get_mpi_rank()
    tmp_dir = _get_temp_dir()
    model_json = str(Path(tmp_dir) / f"opt_model_ensemble_r{rank}.json")

    # Trimmed-mode ensemble: pywrdrb's ModelBuilder auto-routes the trimmed-
    # model release parameters to PresimulatedReleaseEnsemble (reading from
    # presimulated_releases_mgd.hdf5 staged by STARFITReleaseEnsemble-
    # Preprocessor) when both use_trimmed_model=True AND
    # inflow_ensemble_indices are set. use_trimmed=None lets the cache pick up
    # USE_TRIMMED_MODEL from config, matching the single-trace behavior.
    t0 = time.perf_counter()
    base_dict = _get_cached_model_dict(
        use_trimmed=None,
        nyc_config=nyc_config,
        ensemble_spec=ensemble_spec,
    )
    model_dict = copy.deepcopy(base_dict)
    _patch_model_dict(model_dict, nyc_config)

    with open(model_json, "w") as f:
        json.dump(model_dict, f)
    model = pywrdrb.Model.load(model_json)
    t_build = time.perf_counter()

    mem_recorder = InMemoryRecorder(model)
    model.run()
    t_run = time.perf_counter()

    datetime_index = model.timestepper.datetime_index.to_timestamp()
    data_per_real = _extract_results_per_scenario(
        mem_recorder.recorder_dict,
        datetime_index,
        n_scenarios=ensemble_spec.n_realizations,
    )

    for s in range(ensemble_spec.n_realizations):
        _extract_salinity_records(
            model, datetime_index, data_per_real[s], scenario=s,
        )

    if SIM_PHASE_TIMING:
        t_extract = time.perf_counter()
        print(
            f"[phase] preset={ensemble_spec.preset_name} "
            f"n_scen={ensemble_spec.n_realizations} "
            f"build_s={t_build - t0:.2f} run_s={t_run - t_build:.2f} "
            f"extract_s={t_extract - t_run:.2f}",
            flush=True,
        )

    del model, mem_recorder
    return data_per_real


def run_simulation_ensemble_batched(
    nyc_config,
    ensemble_spec,
    batch_size: int,
    per_realization_fn: Callable,
    *,
    skip_failed_batches: bool = False,
    failed_value=None,
) -> list:
    """Simulate an inflow ensemble in sequential realization batches.

    The shared realization-handling path for Borg's ``evaluate()`` ensemble
    branch, re-evaluation, and the supplemental policy-sweep diagnostics, so
    all compute identical per-realization results. The ensemble is split into
    contiguous chunks of ``batch_size`` realizations; each chunk is simulated with one
    :func:`run_simulation_ensemble_inmemory` call (one Pywr model, ``batch_size``
    scenarios), each realization is reduced to a scalar/array via
    ``per_realization_fn``, and the chunk's timeseries are freed before the next
    chunk. Only the reduced per-realization values are retained, so peak memory
    is bounded by ``batch_size`` rather than the full ensemble.

    Realizations are independent (no cross-scenario coupling in Pywr), so a
    realization's reduced value is identical regardless of which batch it lands
    in; only peak memory changes with ``batch_size``. Each batch gets a distinct
    ``preset_name`` (``__b{offset}``) so the model-dict cache does not reuse a
    different batch's model.

    Args:
        nyc_config: NYCOperationsConfig (DV-applied).
        ensemble_spec: ``EnsembleSpec`` with ``is_ensemble=True``.
        batch_size: Realizations per simulation batch. ``<= 0`` (or ``None``)
            collapses to one batch of all realizations (single-model
            behavior, just with a ``__b0`` cache key).
        per_realization_fn: Callable ``data_dict -> value`` applied to each
            realization's result dict. Exceptions raised inside it are NOT
            caught here; the caller decides per-realization error tolerance.
        skip_failed_batches: If True, a batch whose *simulation* raises leaves
            its realizations set to ``failed_value`` and the sweep continues;
            if False (default) the exception propagates.
        failed_value: Value stored for each realization of a skipped batch.

    Returns:
        list of length ``ensemble_spec.n_realizations`` in realization order,
        holding ``per_realization_fn`` outputs (or ``failed_value``).
    """
    if not ensemble_spec.is_ensemble:
        raise ValueError(
            "run_simulation_ensemble_batched requires is_ensemble=True "
            f"(preset '{ensemble_spec.preset_name}'). Use run_simulation_inmemory "
            "for the single-trace path."
        )

    indices = list(ensemble_spec.realization_indices)
    n_real = len(indices)
    bs = batch_size if (batch_size and batch_size > 0) else n_real
    results: list = [failed_value] * n_real

    for b0 in range(0, n_real, bs):
        batch = indices[b0:b0 + bs]
        batch_spec = replace(
            ensemble_spec,
            preset_name=f"{ensemble_spec.preset_name}__b{b0}",
            realization_indices=tuple(batch),
        )
        try:
            data_per_real = run_simulation_ensemble_inmemory(nyc_config, batch_spec)
        except Exception as e:
            if not skip_failed_batches:
                raise
            # Leave this batch's rows as failed_value; other batches proceed.
            # Surface why (was silent before) so failures are diagnosable.
            msg = str(e).strip().splitlines()[-1] if str(e).strip() else ""
            print(f"  [ensemble_batched] batch b{b0} (n={len(batch)}) failed; "
                  f"rows -> failed_value: {type(e).__name__}: {msg}"[:200],
                  file=sys.stderr, flush=True)
            continue
        for j, data in enumerate(data_per_real):
            results[b0 + j] = per_realization_fn(data)
        del data_per_real

    return results


def check_dv_feasibility(nyc_config, ensemble_spec, *, probe_index=None):
    """Cheap structural-feasibility probe for a decision-variable policy.

    Simulates a SINGLE probe realization (full window, one scenario) and reports
    whether the model solves. Catches policies that produce an infeasible LP for
    every realization (structural / DV-level infeasibility) before the expensive
    full-ensemble run, so a known-bad DV can be skipped (recorded NaN) instead of
    crashing batch after batch downstream.

    Limitation: realization-specific infeasibility (a policy that solves for most
    inflow traces but fails on a few) is NOT caught — that would require running
    those realizations. Those remain handled by ``skip_failed_batches`` in
    :func:`run_simulation_ensemble_batched`.

    Args:
        nyc_config: NYCOperationsConfig (DV-applied).
        ensemble_spec: ``EnsembleSpec`` with ``is_ensemble=True``.
        probe_index: Realization index to probe. Defaults to the spec's first
            realization index.

    Returns:
        ``(feasible: bool, error: str | None)`` — ``error`` is a short
        ``"ExcType: message"`` string when infeasible, else ``None``.
    """
    if not ensemble_spec.is_ensemble:
        raise ValueError(
            "check_dv_feasibility requires is_ensemble=True "
            f"(preset '{ensemble_spec.preset_name}')."
        )
    probe = (probe_index if probe_index is not None
             else ensemble_spec.realization_indices[0])
    probe_spec = replace(
        ensemble_spec,
        preset_name=f"{ensemble_spec.preset_name}__probe",
        realization_indices=(int(probe),),
    )
    try:
        run_simulation_ensemble_inmemory(nyc_config, probe_spec)
        return True, None
    except Exception as e:  # noqa: BLE001 - any solver/build failure = infeasible
        msg = str(e).strip().splitlines()[-1] if str(e).strip() else ""
        return False, f"{type(e).__name__}: {msg}"[:200]


###############################################################################
# Disk-Based Simulation (step-05 baseline)
###############################################################################

def run_simulation_to_disk(nyc_config, output_file: Path,
                           use_trimmed: bool = None) -> dict:
    """Run Pywr-DRB simulation and save results to HDF5.

    Used for the baseline evaluation (and ad-hoc timeseries figures) where
    the full simulation output is kept for later analysis.

    Args:
        nyc_config: NYCOperationsConfig instance.
        output_file: Path to save HDF5 output.
        use_trimmed: Use trimmed model. Defaults to USE_TRIMMED_MODEL from
            config. For the historic baseline pass use_trimmed=False since
            it is a single run and the full model is more accurate.

    Returns:
        Dict of DataFrames keyed by results set name.
    """
    import pywrdrb

    mb = _build_model_builder(nyc_config, use_trimmed=use_trimmed)

    # Substitute the salt-front parameter type + values when DVs are active.
    # The cached path (run_simulation_inmemory) does this in _patch_model_dict;
    # here we do it inline before the model JSON is written.
    _patch_salt_front_parameter(mb.model_dict, nyc_config)

    model_json = output_file.with_suffix(".json")
    model = _write_and_load_model(mb, str(model_json))

    # Attach OutputRecorder (will write HDF5 on finish)
    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=str(output_file),
    )
    model.run()

    # Load results via pywrdrb.Data() for proper name mapping
    data = _load_results_from_hdf5(output_file)
    # Async-mode salinity LSTM populates only after model.run() finishes.
    # Compute here and overwrite data["salinity"] with the real time series.
    datetime_index = model.timestepper.datetime_index.to_timestamp()
    _extract_salinity_records(model, datetime_index, data)

    # Cleanup model JSON (keep HDF5 for analysis)
    model_json.unlink(missing_ok=True)
    return data


def _load_results_from_hdf5(output_file: Path) -> dict:
    """Load simulation results from HDF5 using pywrdrb.Data().

    Returns dict of DataFrames in the same format as the in-memory path,
    for API compatibility with objectives.py.
    """
    import pywrdrb

    data_loader = pywrdrb.Data()
    data_loader.load_output(
        output_filenames=[str(output_file)],
        results_sets=RESULTS_SETS,
    )

    # pywrdrb.Data stores results as data.results_set[label][scenario_id]
    label = output_file.stem
    results = {}
    for rs in RESULTS_SETS:
        if hasattr(data_loader, rs):
            rs_data = getattr(data_loader, rs)
            if label in rs_data and 0 in rs_data[label]:
                results[rs] = rs_data[label][0]
            else:
                # Try first available label/scenario
                for lbl in rs_data:
                    for scen in rs_data[lbl]:
                        results[rs] = rs_data[lbl][scen]
                        break
                    break

    return results


###############################################################################
# Borg Evaluation Function
###############################################################################

def _evaluate_ensemble_batched(nyc_config, ensemble_spec, objective_set,
                               batch_size: int) -> list:
    """Borg-format ensemble objectives via the memory-batched simulation path.

    Implements the two-layer annual-unit scheme (objective_definitions.md §2)
    batch by batch: each batch's realizations are reduced to their stage-(i)
    per-unit-year annual-metric vectors (NOT per-realization scalars) and the
    batch's timeseries are freed; after all batches, each objective's pooled
    unit-years (all realizations' units concatenated, in realization order)
    are collapsed with its stage-(ii) unit operator. Identical result to the
    unbatched ``ObjectiveSet.compute_for_borg_ensemble`` (same pooled units,
    same operator), but never holds all N data dicts at once.

    Requires an ObjectiveSet of ``AnnualUnitObjective`` instances (exposing
    ``annual_units`` and ``compute_for_borg_from_units``), as returned by
    ``formulations.get_objective_set()`` when the search ensemble is active.
    """
    ens_objs = list(objective_set)
    if not ens_objs or not all(
        hasattr(o, "annual_units") and hasattr(o, "compute_for_borg_from_units")
        for o in ens_objs
    ):
        raise NotImplementedError(
            "batched ensemble evaluation requires AnnualUnitObjective "
            "instances (with .annual_units and .compute_for_borg_from_units). "
            "Build the set via src.objectives_ensemble."
            "build_ensemble_objective_set or pass the active set returned by "
            "formulations.get_objective_set()."
        )

    def per_real(data):
        # Stage (i): one annual-metric vector per objective for this
        # realization (length = its metric-bearing FFMP-year unit count).
        return [o.annual_units(data) for o in ens_objs]

    unit_rows = run_simulation_ensemble_batched(
        nyc_config, ensemble_spec, batch_size, per_real,
    )
    return [
        o.compute_for_borg_from_units(
            np.concatenate([row[k] for row in unit_rows])
            if unit_rows else np.array([], dtype=float)
        )
        for k, o in enumerate(ens_objs)
    ]


def _units_tensor_from_rows(unit_rows: list, n_obj: int) -> np.ndarray:
    """Stack per-realization stage-(i) rows into a ``(R, M, U)`` float tensor.

    ``unit_rows[r]`` is either a list of ``n_obj`` equal-length annual-metric
    vectors, or ``None`` for a realization whose simulation batch failed. The
    unit-year count ``U`` is taken from the first successful row; failed rows
    become all-NaN slabs so the offline pooling can distinguish a FAILED
    realization (excluded from its SOW's unit pool) from a ran-but-degenerate
    one (whose non-finite unit-years the unit operators count as failures).
    """
    first = next((row for row in unit_rows if row is not None), None)
    if first is None:
        raise RuntimeError("every simulation batch failed; no unit-years produced")
    n_units = len(np.asarray(first[0], dtype=float).ravel())
    out = np.full((len(unit_rows), n_obj, n_units), np.nan, dtype=float)
    for r, row in enumerate(unit_rows):
        if row is None:
            continue
        for k in range(n_obj):
            vec = np.asarray(row[k], dtype=float).ravel()
            if vec.shape[0] != n_units:
                raise ValueError(
                    f"realization {r} yields {vec.shape[0]} unit-years, "
                    f"expected {n_units}; the ensemble's realizations must "
                    f"share one window length"
                )
            out[r, k, :] = vec
    return out


def _ensemble_units_tensor(nyc_config, ensemble_spec, objective_set,
                           batch_size: int, *, skip_failed_batches: bool = False):
    """Per-realization stage-(i) annual-metric tensor ``(R, M, U)``.

    The tensor-building half of the batched RE-EVAL path
    (:func:`evaluate_annual_units`): slab ``[r, k, :]`` holds objective ``k``'s
    per-unit-year annual metrics for realization ``r`` — the SAME stage-(i)
    reduction the search path pools (:func:`_evaluate_ensemble_batched`), so
    search and re-evaluation share one metric formula per quantity. The
    re-eval layer pools these per SOW with the §2 unit operators
    (``src.reeval_core.sow_objective_matrix``).

    Args:
        skip_failed_batches: When True (re-eval path), a batch whose simulation
            raises leaves its realizations as all-NaN slabs and the sweep
            continues, so one infeasible trace NaNs only its own batch rather
            than failing the whole solution. Search keeps the strict default
            (False) so its behavior is unchanged.

    Returns:
        ``(units, obj_names)`` — ``units`` float array ``(R, M, U)``;
        ``obj_names[k]`` is the annual objective name of slab ``k``.
    """
    ens_objs = list(objective_set)
    if not ens_objs or not all(hasattr(o, "annual_units") for o in ens_objs):
        raise NotImplementedError(
            "batched ensemble unit evaluation requires AnnualUnitObjective "
            "instances (with .annual_units). Build the set via "
            "src.objectives_ensemble.build_ensemble_objective_set or pass the "
            "active set returned by formulations.get_objective_set()."
        )

    def per_real(data):
        # Stage (i): one annual-metric vector per objective — identical to the
        # search path's per-realization reduction.
        return [o.annual_units(data) for o in ens_objs]

    unit_rows = run_simulation_ensemble_batched(
        nyc_config, ensemble_spec, batch_size, per_real,
        skip_failed_batches=skip_failed_batches,
        failed_value=None,  # post-processed to an all-NaN slab
    )
    units = _units_tensor_from_rows(unit_rows, len(ens_objs))
    return units, [o.name for o in ens_objs]


def evaluate_annual_units(dv_vector, formulation_name="ffmp", objective_set=None,
                          ensemble_spec=None, realization_batch=None):
    """Per-realization stage-(i) annual-metric tensor in NATURAL units.

    Re-eval-facing companion to :func:`evaluate`: runs the SAME simulation path
    and the SAME stage-(i) annual-metric reduction as search, returning the raw
    ``(n_realizations, n_objs, n_unit_years)`` tensor instead of search's
    pooled scalar objectives. The re-eval layer pools each SOW's unit-years
    with the §2 unit operators (``src.reeval_core.sow_objective_matrix``).
    Mirrors ``evaluate()``'s dispatch (resample / single-trace / batched /
    unbatched).

    Args:
        dv_vector: Decision-variable vector.
        formulation_name: Formulation name string.
        objective_set: ObjectiveSet of ``AnnualUnitObjective`` instances.
            Defaults to the active set from ``formulations.get_objective_set()``.
        ensemble_spec: EnsembleSpec; defaults to ``config.SEARCH_ENSEMBLE_SPEC``.
            Re-eval callers pass ``config.REEVAL_ENSEMBLE_SPEC``.
        realization_batch: Realizations per simulation batch; defaults to
            ``config.SEARCH_REALIZATION_BATCH``.

    Returns:
        ``(units, obj_names)`` — ``units`` float array ``(R, M, U)`` of
        stage-(i) annual metrics (all-NaN slab = failed realization);
        ``obj_names[k]`` is slab ``k``'s annual objective name. Single-trace
        (``is_ensemble=False``) returns ``R == 1``.
    """
    if objective_set is None:
        from src.formulations import get_objective_set
        objective_set = get_objective_set()

    ens_objs = list(objective_set)
    if not ens_objs or not all(hasattr(o, "annual_units") for o in ens_objs):
        raise NotImplementedError(
            "evaluate_annual_units requires AnnualUnitObjective instances "
            "(with .annual_units). Build the set via "
            "src.objectives_ensemble.build_ensemble_objective_set."
        )

    if ensemble_spec is None:
        from config import SEARCH_ENSEMBLE_SPEC
        ensemble_spec = SEARCH_ENSEMBLE_SPEC

    # Parity with evaluate(): redraw a resample-per-eval master pool. Re-eval
    # specs are not resample pools, so this is a no-op there.
    if ensemble_spec is not None and ensemble_spec.resample_per_eval:
        ensemble_spec = _resampled_eval_spec(ensemble_spec, _EVAL_COUNT)

    if realization_batch is None:
        from config import SEARCH_REALIZATION_BATCH
        realization_batch = SEARCH_REALIZATION_BATCH

    nyc_config = dvs_to_config(dv_vector, formulation_name)

    if not ensemble_spec.is_ensemble:
        # Single-trace re-eval: one realization's stage-(i) annual metrics.
        data = run_simulation_inmemory(nyc_config)
        rows = [[o.annual_units(data) for o in ens_objs]]
        return (_units_tensor_from_rows(rows, len(ens_objs)),
                [o.name for o in ens_objs])

    if realization_batch and realization_batch > 0:
        # Re-eval tolerates a failed batch (all-NaN slabs) instead of failing
        # the whole solution; the offline pooling excludes failed realizations
        # from their SOW's unit pool. Search (via _evaluate_ensemble_batched)
        # keeps the strict default so its behavior is unchanged.
        return _ensemble_units_tensor(
            nyc_config, ensemble_spec, objective_set, realization_batch,
            skip_failed_batches=True,
        )

    # Single-model ensemble path (all realizations as one scenario block).
    data_per_real = run_simulation_ensemble_inmemory(nyc_config, ensemble_spec)
    rows = [[o.annual_units(d) for o in ens_objs] for d in data_per_real]
    return (_units_tensor_from_rows(rows, len(ens_objs)),
            [o.name for o in ens_objs])


_RESAMPLE_BASE_SEED = 1_000_003  # salt for the resampled-probabilistic per-eval RNG


def _resampled_eval_spec(pool_spec, eval_count):
    """Draw a fresh per-evaluation subset from a resample-per-eval pool.

    Returns a copy of ``pool_spec`` whose ``realization_indices`` is a random
    size-``resample_size`` subset (without replacement) of the pool. The draw
    is keyed by (base salt, MPI rank, eval_count) so it differs every
    evaluation and is reproducible given the same rank/eval ordering
    (Trindade et al. 2017 per-evaluation reshuffling).

    Args:
        pool_spec: An ``EnsembleSpec`` with ``resample_per_eval=True`` whose
            ``realization_indices`` is the full master pool and ``resample_size``
            is the per-evaluation draw size.
        eval_count: The current evaluation counter (``_EVAL_COUNT``).

    Returns:
        An ``EnsembleSpec`` copy with the freshly drawn ``realization_indices``.
    """
    from src.ensembles import with_indices_override
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        rank = 0
    pool = pool_spec.realization_indices
    size = pool_spec.resample_size
    rng = np.random.default_rng([_RESAMPLE_BASE_SEED, rank, int(eval_count)])
    chosen = rng.choice(len(pool), size=size, replace=False)
    drawn = sorted(int(pool[i]) for i in chosen)
    return with_indices_override(pool_spec, drawn)


def evaluate(dv_vector, formulation_name="ffmp", objective_set=None,
             ensemble_spec=None, realization_batch=None):
    """Full evaluation pipeline: DVs -> simulation -> objectives.

    Called by Borg MOEA for each candidate solution. Uses in-memory
    simulation to minimize I/O overhead.

    Dispatches to either the single-trace path or the ensemble path
    based on ``ensemble_spec.is_ensemble``.

    Args:
        dv_vector: Array of decision variable values.
        formulation_name: Formulation name string.
        objective_set: ObjectiveSet instance. If None, uses the active set
            from config.ACTIVE_OBJECTIVE_SET.
        ensemble_spec: Optional EnsembleSpec override. If None, uses
            ``config.SEARCH_ENSEMBLE_SPEC``. The default ``historic_single``
            preset routes through the single-trace path.
        realization_batch: Realizations per simulation batch for the ensemble
            path. If None, uses ``config.SEARCH_REALIZATION_BATCH``. ``<= 0``
            keeps the single-model behavior (all realizations as one
            scenario block); a positive value bounds peak memory by simulating
            the ensemble in sequential batches via
            :func:`run_simulation_ensemble_batched` — the same shared path the
            objective-sensitivity diagnostic uses.

    Returns:
        List of objective values (Borg-compatible, all minimized).
    """
    global _EVAL_COUNT, _EVAL_START_TIME

    if _EVAL_START_TIME is None:
        _EVAL_START_TIME = time.time()

    _EVAL_COUNT += 1
    t0 = time.time()

    if objective_set is None:
        from src.formulations import get_objective_set
        objective_set = get_objective_set()

    if ensemble_spec is None:
        from config import SEARCH_ENSEMBLE_SPEC
        ensemble_spec = SEARCH_ENSEMBLE_SPEC

    # Resampled-probabilistic design: redraw the search ensemble from the master
    # pool for this evaluation (Trindade et al. 2017). The master-pool spec is
    # marked resample_per_eval=True by ScenarioDesign.resolve_search_spec.
    if ensemble_spec is not None and ensemble_spec.resample_per_eval:
        ensemble_spec = _resampled_eval_spec(ensemble_spec, _EVAL_COUNT)

    if realization_batch is None:
        from config import SEARCH_REALIZATION_BATCH
        realization_batch = SEARCH_REALIZATION_BATCH

    nyc_config = dvs_to_config(dv_vector, formulation_name)
    if not ensemble_spec.is_ensemble:
        # Single-trace (historic) design: the SAME annual-unit (§2) objective
        # as the ensembles, with the one trace as a single realization (N=1 ->
        # its consecutive FFMP-year units). Requires AnnualUnitObjective
        # instances (formulations.get_objective_set()).
        data = run_simulation_inmemory(nyc_config)
        objs = objective_set.compute_for_borg_ensemble([data])
    elif realization_batch and realization_batch > 0:
        # Memory-batched ensemble path: sequential batches keep only each
        # realization's per-unit-year annual metrics (stage i); the pooled
        # unit-years are collapsed with each objective's unit operator
        # (stage ii) without holding all N data dicts at once.
        objs = _evaluate_ensemble_batched(
            nyc_config, ensemble_spec, objective_set, realization_batch,
        )
    else:
        # Single-model ensemble path:
        # one Pywr model with all realizations as scenarios, then aggregate.
        data_per_real = run_simulation_ensemble_inmemory(
            nyc_config, ensemble_spec,
        )
        # Requires an ObjectiveSet built via
        # src.objectives_ensemble.build_ensemble_objective_set; fail loudly on
        # a hand-built single-trace set.
        if not hasattr(objective_set, "compute_for_borg_ensemble"):
            raise NotImplementedError(
                "ensemble evaluation requested but ObjectiveSet has no "
                "compute_for_borg_ensemble. Build the set via "
                "src.objectives_ensemble.build_ensemble_objective_set or "
                "pass the active set returned by formulations.get_objective_set()."
            )
        objs = objective_set.compute_for_borg_ensemble(data_per_real)

    elapsed = time.time() - t0
    if _EVAL_COUNT % _EVAL_LOG_INTERVAL == 0 or _EVAL_COUNT == 1:
        total_elapsed = time.time() - _EVAL_START_TIME
        avg_time = total_elapsed / _EVAL_COUNT
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            rank = 0
        obj_str = ", ".join(f"{o:.4f}" for o in objs)
        sys.stdout.write(
            f"[Rank {rank}] Eval #{_EVAL_COUNT}: {elapsed:.1f}s this eval, "
            f"{avg_time:.1f}s avg, {total_elapsed:.0f}s total | objs=[{obj_str}]\n"
        )
        sys.stdout.flush()

    return objs
