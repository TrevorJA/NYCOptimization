"""
simulation.py - Pywr-DRB simulation wrapper for optimization.

Provides the core evaluation function that:
1. Takes a flat decision variable vector
2. Converts it to a NYCOperationsConfig
3. Builds and runs a Pywr-DRB simulation
4. Computes and returns the objective vector

Memory/I/O design:
    During optimization, Borg calls evaluate() ~1M times. Each call
    must be fast. We still need to write a temporary JSON model file
    (required by pywr.Model.load), but we avoid HDF5 output by using
    an InMemoryRecorder that captures data to numpy arrays without
    writing to disk.

    For baseline and re-evaluation runs, the standard OutputRecorder
    writes full HDF5 results for post-hoc analysis.

This module is imported by the Borg optimization driver.
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
    RESULTS_SETS,
    PRESIM_DIR,
    PRESIM_FILE,
    NYC_RESERVOIRS,
    INCLUDE_SALINITY_MODEL,
    INCLUDE_TEMPERATURE_MODEL,
)
from src.formulations import get_formulation, get_var_names
from src.formulations.salt_front_dvs import apply_salt_front_dvs
from src.ts_options import build_lstm_options_block

# Allow debug override of simulation date range via environment variables.
# Set PYWRDRB_SIM_START_DATE / PYWRDRB_SIM_END_DATE (YYYY-MM-DD) before
# launching mpirun to run a shorter period for fast debugging.
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
_CACHED_MODEL_DICT = None        # Cached base model dict (avoids make_model per eval)
_CACHED_MODEL_DICTS = {}         # Keyed by tuple(drought_levels) for N-zone support
_CACHED_NZONE_CONFIGS = {}       # Keyed by n_zones

# CFS→MGD conversion (from pywrdrb.utils.constants)
_CFS_TO_MGD = 0.645932368556

# Parameter keys affected by decision variables
_DROUGHT_LEVELS = ["level1a", "level1b", "level1c", "level2", "level3", "level4", "level5"]
_ZONE_LEVELS = ["level1b", "level1c", "level2", "level3", "level4", "level5"]
_NYC_RESERVOIRS_OPT = ["cannonsville", "pepacton", "neversink"]
_DOWNSTREAM_LOCS = ["delMontague", "delTrenton"]

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
            "  bash workflow/00_generate_presim.sh\n"
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

    Cache key includes drought-level structure, T/S toggles, and ensemble
    preset name + DU factor signature so that switching ensemble presets
    (or enabling salinity) does not silently reuse a cache built for a
    different inflow source.
    """
    global _CACHED_MODEL_DICT, _CACHED_MODEL_DICTS
    if nyc_config is None:
        nyc_config = _get_cached_defaults()
    if ensemble_spec is None:
        from src.ensembles import get_ensemble_spec
        ensemble_spec = get_ensemble_spec("historic_single")
    drought_levels, _ = _config_levels(nyc_config)
    key = (
        tuple(drought_levels),
        bool(INCLUDE_TEMPERATURE_MODEL),
        bool(INCLUDE_SALINITY_MODEL),
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
        # Keep legacy single reference in sync for backward compatibility
        # (only when T/S is off, no ensemble, so legacy callers still see
        # a sane default).
        legacy_levels = tuple(
            ["level1a", "level1b", "level1c", "level2", "level3", "level4", "level5"]
        )
        if key == (legacy_levels, False, False, "historic_single", ""):
            _CACHED_MODEL_DICT = _CACHED_MODEL_DICTS[key]
    return _CACHED_MODEL_DICTS[key]


def _config_levels(nyc_config):
    """Return (drought_levels, storage_levels) from config, handling both old and new API.

    The Phase 1 pywrdrb adds DROUGHT_LEVELS and STORAGE_LEVELS as properties.
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
    drought levels zone_0..zone_N). Byte-level equivalence with the previous
    local implementation was verified at N ∈ {6, 8, 10, 12}.

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
    # MRF baselines
    # update_mrf_baselines(cannonsville, pepacton, neversink, montague, trenton)
    config.update_mrf_baselines(
        cannonsville=params["mrf_cannonsville"],
        pepacton=params["mrf_pepacton"],
        neversink=params["mrf_neversink"],
        montague=params["mrf_montague"],
        trenton=params["mrf_trenton"],
    )

    # Delivery constraints
    # update_delivery_constraints(max_nyc_delivery, drought_factors_nyc, drought_factors_nj, ...)
    # drought_factors arrays have 7 elements for levels: 1a, 1b, 1c, 2, 3, 4, 5
    # We only optimize L3, L4, L5; levels 1a, 1b, 1c, 2 use defaults from config.
    #
    # IMPORTANT: NYC L1a-L2 defaults are 1,000,000 (effectively unconstrained),
    # NOT 1.0. NJ L1a-L3 defaults are 1.0 (no reduction). These must come from
    # the loaded config.constants to preserve correct FFMP behavior.
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

    nyc_factors = np.array([
        float(defaults["level1a_factor_delivery_nyc"]),  # 1,000,000 (unconstrained)
        float(defaults["level1b_factor_delivery_nyc"]),  # 1,000,000
        float(defaults["level1c_factor_delivery_nyc"]),  # 1,000,000
        float(defaults["level2_factor_delivery_nyc"]),   # 1,000,000
        params["nyc_drought_factor_L3"],
        params["nyc_drought_factor_L4"],
        params["nyc_drought_factor_L5"],
    ])
    nj_factors = np.array([
        float(defaults["level1a_factor_delivery_nj"]),   # 1.0
        float(defaults["level1b_factor_delivery_nj"]),   # 1.0
        float(defaults["level1c_factor_delivery_nj"]),   # 1.0
        float(defaults["level2_factor_delivery_nj"]),    # 1.0
        float(defaults["level3_factor_delivery_nj"]),    # 1.0
        params["nj_drought_factor_L4"],
        params["nj_drought_factor_L5"],
    ])
    config.update_delivery_constraints(
        max_nyc_delivery=params["max_nyc_delivery"],
        drought_factors_nyc=nyc_factors,
        drought_factors_nj=nj_factors,
    )

    # Storage zone vertical shifts
    zone_levels = ["level1b", "level1c", "level2", "level3", "level4", "level5"]
    shifts = {level: params[f"zone_shift_{level}"] for level in zone_levels}
    _apply_zone_shifts(config, shifts)

    # Flood release maximums
    # update_flood_limits(max_release_cannonsville, max_release_pepacton, max_release_neversink)
    config.update_flood_limits(
        max_release_cannonsville=params["flood_max_cannonsville"],
        max_release_pepacton=params["flood_max_pepacton"],
        max_release_neversink=params["flood_max_neversink"],
    )

    # MRF seasonal profile scaling
    _apply_mrf_profile_scaling(config, params)

    # Stash salt-front DV-derived options for downstream model-dict patching.
    _stash_salt_front_options(config, params)


def _apply_nzone_ffmp_params(config, params: dict):
    """Apply N-zone FFMP params to an NYCOperationsConfig with zone_0..zone_N naming.

    Mirrors _apply_ffmp_params but works with any N-zone config built by
    build_nzone_config(). DV names use zone_{i} naming; missing DV keys fall
    back to the interpolated defaults already stored in config.constants.
    """
    # MRF baselines
    config.update_mrf_baselines(
        cannonsville=params["mrf_cannonsville"],
        pepacton=params["mrf_pepacton"],
        neversink=params["mrf_neversink"],
        montague=params["mrf_montague"],
        trenton=params["mrf_trenton"],
    )

    # Delivery constraints — build factor arrays from defaults + DV overrides
    drought_levels, storage_levels_nz = _config_levels(config)
    nyc_factors = np.array([
        params.get(f"nyc_drought_factor_{level}",
                   float(config.constants[f"{level}_factor_delivery_nyc"]))
        for level in drought_levels
    ])
    nj_factors = np.array([
        params.get(f"nj_drought_factor_{level}",
                   float(config.constants[f"{level}_factor_delivery_nj"]))
        for level in drought_levels
    ])
    config.update_delivery_constraints(
        max_nyc_delivery=params["max_nyc_delivery"],
        drought_factors_nyc=nyc_factors,
        drought_factors_nj=nj_factors,
    )

    # Zone shifts
    shifts = {level: params.get(f"zone_shift_{level}", 0.0) for level in storage_levels_nz}
    _apply_zone_shifts(config, shifts)

    # Flood limits
    config.update_flood_limits(
        max_release_cannonsville=params["flood_max_cannonsville"],
        max_release_pepacton=params["flood_max_pepacton"],
        max_release_neversink=params["flood_max_neversink"],
    )

    # MRF seasonal scaling
    _apply_mrf_profile_scaling(config, params)

    # Stash salt-front DV-derived options for downstream model-dict patching.
    _stash_salt_front_options(config, params)


def _apply_zone_shifts(config, shifts: dict):
    """Apply vertical shifts to storage zone thresholds with monotonic constraint.

    Operates on config.storage_zones_df (rows=levels, 366 daily columns).
    Works for both default level1b..level5 and N-zone zone_1..zone_N naming.
    """
    _, zone_order = _config_levels(config)
    zones = config.storage_zones_df.copy()
    date_cols = [c for c in zones.columns if c != "doy"]

    for level in zone_order:
        if level in zones.index:
            zones.loc[level, date_cols] = zones.loc[level, date_cols] + shifts[level]

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
    scales ALL rows by the seasonal multiplier — storage zone rows are
    unaffected because config.storage_zones_df is a separate copy that
    was already modified by _apply_zone_shifts.

    NOTE: In pywrdrb from_defaults(), mrf_factors_daily_df is initialized
    as storage_zones_df.copy(). Both reference the same CSV. This is the
    pywrdrb design — the daily profiles CSV contains all profile types
    (zone thresholds and MRF factors) indexed by the 'profile' column.
    """
    season_ranges = {
        "winter": list(range(335, 367)) + list(range(1, 60)),
        "spring": list(range(60, 152)),
        "summer": list(range(152, 244)),
        "fall": list(range(244, 335)),
    }
    mrf_factors = config.mrf_factors_daily_df.copy()

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
        mrf_factors.loc[:, cols_to_scale] *= scale

    config.mrf_factors_daily_df = mrf_factors


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
            the legacy single-trace ``INFLOW_TYPE`` from config is used.
    """
    import pywrdrb

    if use_trimmed is None:
        use_trimmed = USE_TRIMMED_MODEL

    options = {
        "nyc_nj_demand_source": NYC_NJ_DEMAND_SOURCE,
        "use_trimmed_model": use_trimmed,
        "initial_volume_frac": INITIAL_VOLUME_FRAC,
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

    # Ensemble window override: when the spec carries a realization_years
    # value, the staged HDF5s span a clipped window starting at START_DATE.
    # Use that window so the pywr timestepper aligns with the staged dates.
    sim_start = START_DATE
    sim_end = END_DATE
    if (
        ensemble_spec is not None
        and ensemble_spec.is_ensemble
        and ensemble_spec.realization_years is not None
    ):
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
    """Return (start_date, end_date) for an ensemble with realization_years set.

    Starts at the configured START_DATE and ends ``realization_years`` years
    later (minus one day) so the window matches the staged HDF5's date axis
    produced by ``KirschNowakGenerator._generate``. Honors the
    ``PYWRDRB_SIM_START_DATE`` / ``PYWRDRB_SIM_END_DATE`` env overrides
    indirectly via the module-level ``START_DATE`` (already populated at
    import time).
    """
    if ensemble_spec.realization_years is None:
        return START_DATE, END_DATE
    start_ts = pd.Timestamp(START_DATE)
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

class InMemoryRecorder:
    """Wrapper around pywrdrb.OutputRecorder that skips HDF5 output.

    Root cause of prior crash (double-free in pywr C extension):
      OutputRecorder.finish() called rec.finish() on each individual
      NumpyArray*Recorder. But pywr's C code ALREADY calls finish() on
      every registered recorder automatically at the end of model.run().
      The double-call freed the internal buffer twice → double-free / abort.

    Fix: replace ALL lifecycle methods on the OutputRecorder wrapper with
    no-ops. Each NumpyArray*Recorder is registered with the pywr model and
    receives exactly one lifecycle call (setup/reset/after/finish) from pywr.
    After model.run() completes, recorder.data is still accessible.

    We also avoid writing HDF5 (which would spawn HDF5 background threads and
    corrupt the Python GIL state inside Borg's ctypes C→Python callback).
    """

    def __init__(self, model):
        from pywrdrb.recorder import OutputRecorder

        # Create OutputRecorder (this also registers the individual
        # NumpyArray*Recorders with the pywr model).
        self._inner = OutputRecorder(model, output_filename="/dev/null")

        # Make ALL wrapper lifecycle methods no-ops.
        # pywr's C engine calls setup/reset/after/finish on each individual
        # registered recorder — the wrapper must not double-call them.
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
    # currently inactive — see decisions/2026-04-29_temperature_lstm_deferred.md).
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
    realization). Each dict is shape-identical to the legacy single-trace
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

    In sync mode (`asycronized_update=False`, our default), the LSTM advances
    `ml_model.t` once per simulation day, writing each sim day's flow into
    `ml_model.X[t, :]` and computing that day's forward pass. With `debug=True`
    on the salinity options, the per-day `sf_mu`/`sf_sd` is recorded in
    `ml_model.records` at index `t = sim_day_index` — i.e., `records[0]` is
    the first sim day's salt-front prediction regardless of the LSTM's own
    `start_date`.

    `ml_model.records["sf_mu"]` is shape `(n_sim, n_scenarios)` after the
    PywrDRB-ML/Pywr-DRB scenario-aware refactor (2026-05-06), even when
    `n_scenarios=1`. We slice `[:n_sim, scenario]` to pull the per-realization
    series and pair it with the simulation's datetime_index. Replaces
    `results["salinity"]` with this DataFrame; this is the canonical source of
    truth for the salinity objective.

    Single-trace callers (`run_simulation_inmemory`, `run_simulation_to_disk`)
    leave `scenario` at its default 0 and get the same series they got pre-
    refactor. Ensemble callers loop over scenario indices.

    Async mode (`asycronized_update=True`) is intentionally unsupported: in
    async mode `ml_model.t` never advances during pywrdrb's run loop, so all
    sim days overwrite `X[0, :]` and the LSTM's forward pass over the full
    window is dominated by historical training data — not what we want for
    NYC-policy-driven optimization. The upstream SalinityLSTMModel.update
    additionally raises NotImplementedError if asycronized_update=True is
    combined with n_scenarios > 1.

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

    # Under sync mode `ml_model.t` advances once per sim day after the
    # first (the gate at salt_front_location.py skips day 1 because
    # `previous_date < ml_model.current_date`), so the expected count is
    # `n_sim - 1`. A genuine async-mode misconfiguration would leave
    # `ml_model.t` near zero.
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
         an ``EnsembleObjective`` aggregator.

      4. Salinity LSTM is now scenario-aware (PywrDRB-ML + Pywr-DRB
         salt_front_location refactor, 2026-05-06): ``ml_model.records``
         is shape ``(n_sim, n_scenarios)``, and we call
         :func:`_extract_salinity_records` once per realization to populate
         ``data_per_real[s]["salinity"]``.

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
    # inflow_ensemble_indices are set. We pass use_trimmed=None so the
    # cache picks up USE_TRIMMED_MODEL from config, matching the legacy
    # single-trace behavior.
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

    mem_recorder = InMemoryRecorder(model)
    model.run()

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

    The shared realization-handling path for both Borg's ``evaluate()`` ensemble
    branch and the ensemble objective-sensitivity diagnostic, so the two compute
    identical per-realization results. The ensemble is split into contiguous
    chunks of ``batch_size`` realizations; each chunk is simulated with one
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
            collapses to one batch of all realizations (legacy single-model
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
        except Exception:
            if not skip_failed_batches:
                raise
            # Leave this batch's rows as failed_value; other batches proceed.
            continue
        for j, data in enumerate(data_per_real):
            results[b0 + j] = per_realization_fn(data)
        del data_per_real

    return results


###############################################################################
# Disk-Based Simulation (for baseline runs and re-evaluation)
###############################################################################

def run_simulation_to_disk(nyc_config, output_file: Path,
                           use_trimmed: bool = None) -> dict:
    """Run Pywr-DRB simulation and save results to HDF5.

    Use this for baseline evaluation and re-evaluation where we want
    to keep the full simulation output for later analysis.

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

    Builds the per-realization base-metric matrix ``(n_real, n_obj)`` by
    simulating the ensemble in sequential batches (freeing each batch's
    timeseries), then collapses each objective's column with its satisficing
    aggregator. Identical result to the legacy
    ``ObjectiveSet.compute_for_borg_ensemble`` (same per-realization base
    metrics, same aggregation), but never holds all N data dicts at once.

    Requires an ObjectiveSet of ``EnsembleObjective`` instances (exposing
    ``.base`` and ``.compute_for_borg_from_values``), as returned by
    ``formulations.get_objective_set()`` when the search ensemble is active.
    """
    ens_objs = list(objective_set)
    if not ens_objs or not all(
        hasattr(o, "base") and hasattr(o, "compute_for_borg_from_values")
        for o in ens_objs
    ):
        raise NotImplementedError(
            "batched ensemble evaluation requires EnsembleObjective instances "
            "(with .base and .compute_for_borg_from_values). Build the set via "
            "src.objectives_ensemble.build_ensemble_objective_set or pass the "
            "active set returned by formulations.get_objective_set()."
        )

    def per_real(data):
        # Per-realization base-metric vector (one column per objective). Strict
        # (no per-metric try/except) so behavior matches the legacy path.
        return [o.base.compute(data) for o in ens_objs]

    base_rows = run_simulation_ensemble_batched(
        nyc_config, ensemble_spec, batch_size, per_real,
    )
    base_matrix = np.asarray(base_rows, dtype=float)  # (n_real, n_obj)
    return [o.compute_for_borg_from_values(base_matrix[:, k])
            for k, o in enumerate(ens_objs)]


_RESAMPLE_BASE_SEED = 1_000_003  # salt for the resampled-probabilistic per-eval RNG


def _resampled_eval_spec(pool_spec, eval_count):
    """Draw a fresh per-evaluation subset from a resample-per-eval master pool.

    Returns a copy of ``pool_spec`` whose ``realization_indices`` is a random
    size-``resample_size`` subset (without replacement) of the master pool. The
    draw is keyed by (base salt, MPI rank, eval_count) so it differs every
    evaluation and is reproducible given the same rank/eval ordering. This is
    the Trindade et al. (2017) per-evaluation reshuffling: each candidate sees a
    fresh random scenario set, so in-search fitness is a noisy estimate and
    cross-design comparison rests on the held-out re-evaluation.

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

    Dispatches to either the legacy single-trace path or the ensemble path
    based on ``ensemble_spec.is_ensemble``. The legacy path (default) is
    byte-identical to the manuscript baseline.

    Args:
        dv_vector: Array of decision variable values.
        formulation_name: Formulation name string.
        objective_set: ObjectiveSet instance. If None, uses the active set
            from config.ACTIVE_OBJECTIVE_SET.
        ensemble_spec: Optional EnsembleSpec override. If None, uses
            ``config.SEARCH_ENSEMBLE_SPEC``. The default ``historic_single``
            preset routes through the legacy single-trace path.
        realization_batch: Realizations per simulation batch for the ensemble
            path. If None, uses ``config.SEARCH_REALIZATION_BATCH``. ``<= 0``
            keeps the legacy single-model behavior (all realizations as one
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
        data = run_simulation_inmemory(nyc_config)
        objs = objective_set.compute_for_borg(data)
    elif realization_batch and realization_batch > 0:
        # Memory-batched ensemble path: simulate in sequential batches and
        # collapse the per-realization base-metric matrix to satisficing
        # objectives, never holding all N data dicts at once. Shares
        # run_simulation_ensemble_batched with the objective-sensitivity
        # diagnostic, so search and diagnostic handle realizations identically.
        objs = _evaluate_ensemble_batched(
            nyc_config, ensemble_spec, objective_set, realization_batch,
        )
    else:
        # Legacy single-model ensemble path (byte-identical to prior behavior):
        # one Pywr model with all realizations as scenarios, then aggregate.
        data_per_real = run_simulation_ensemble_inmemory(
            nyc_config, ensemble_spec,
        )
        # The ensemble dispatch expects an ObjectiveSet built via
        # `src.objectives_ensemble.build_ensemble_objective_set`, which is
        # what `formulations.get_objective_set()` returns when
        # `SEARCH_ENSEMBLE_SPEC.is_ensemble` is True. A legacy single-trace
        # set leaks through only if a caller hand-built one and passed it
        # explicitly — fail loudly there rather than silently invoking the
        # wrong compute path.
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
