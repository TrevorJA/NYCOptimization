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
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    START_DATE,
    END_DATE,
    INFLOW_TYPE,
    USE_TRIMMED_MODEL,
    INITIAL_VOLUME_FRAC,
    RESULTS_SETS,
    PRESIM_DIR,
    PRESIM_FILE,
    NYC_RESERVOIRS,
)
from src.formulations import get_formulation, get_var_names

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
            "  bash 00_generate_presim.sh\n"
            "  (or: python scripts/generate_presim.py)"
        )
    return f


###############################################################################
# Cached NYCOperationsConfig and Model Dict
###############################################################################

def _get_cached_defaults():
    """Return cached NYCOperationsConfig.from_defaults() (avoids re-reading CSVs)."""
    global _CACHED_DEFAULTS_CONFIG
    if _CACHED_DEFAULTS_CONFIG is None:
        from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig
        _CACHED_DEFAULTS_CONFIG = NYCOperationsConfig.from_defaults()
    return _CACHED_DEFAULTS_CONFIG


def _get_cached_model_dict(use_trimmed: bool = None):
    """Build and cache the base model_dict using default config (first call only).

    Subsequent evaluations deep-copy this dict and patch only the DV-affected
    parameters, avoiding the ~1s cost of make_model() on every eval.
    """
    global _CACHED_MODEL_DICT
    if _CACHED_MODEL_DICT is None:
        defaults = _get_cached_defaults()
        mb = _build_model_builder(defaults, use_trimmed=use_trimmed)
        _CACHED_MODEL_DICT = mb.model_dict
    return _CACHED_MODEL_DICT


def _patch_model_dict(model_dict: dict, nyc_config):
    """Update DV-affected parameters in a model_dict from NYCOperationsConfig.

    This replaces the full make_model() rebuild by directly setting the
    parameter values that correspond to decision variables.
    """
    params = model_dict["parameters"]

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

    # Drought delivery factors (NYC and NJ, all 7 levels)
    for level in _DROUGHT_LEVELS:
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
    for level in _ZONE_LEVELS:
        params[level]["values"] = nyc_config.get_storage_zone_profile(level).tolist()

    # MRF daily factor profiles (per reservoir × per level)
    for level in _DROUGHT_LEVELS:
        for res in _NYC_RESERVOIRS_OPT:
            key = f"{level}_factor_mrf_{res}"
            if key in params:
                params[key]["values"] = nyc_config.get_mrf_factor_profile(
                    key, daily=True
                ).tolist()

    # --- Monthly profiles (12 values) ---
    # MRF monthly factor profiles (Montague & Trenton × per level)
    for level in _DROUGHT_LEVELS:
        for loc in _DOWNSTREAM_LOCS:
            key = f"{level}_factor_mrf_{loc}"
            if key in params:
                params[key]["values"] = nyc_config.get_mrf_factor_profile(
                    key, daily=False
                ).tolist()


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
    base = _get_cached_defaults()
    config = _copy.deepcopy(base)

    var_names = get_var_names(formulation_name)
    params = dict(zip(var_names, dv_vector))

    if formulation_name == "ffmp":
        _apply_ffmp_params(config, params)
    else:
        raise NotImplementedError(
            f"Formulation '{formulation_name}' not yet implemented."
        )
    return config


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


def _apply_zone_shifts(config, shifts: dict):
    """Apply vertical shifts to storage zone thresholds with monotonic constraint.

    Operates on config.storage_zones_df (rows=levels, 366 daily columns).
    """
    zone_order = ["level1b", "level1c", "level2", "level3", "level4", "level5"]
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

def _build_model_builder(nyc_config, use_trimmed: bool = None):
    """Create and configure a ModelBuilder. Shared by both simulation paths.

    Args:
        nyc_config: NYCOperationsConfig instance.
        use_trimmed: Whether to use the trimmed model. If None, falls back to
            USE_TRIMMED_MODEL from config. Trimmed mode requires that
            00_generate_presim.sh has been run first.
    """
    import pywrdrb

    if use_trimmed is None:
        use_trimmed = USE_TRIMMED_MODEL

    options = {
        "nyc_nj_demand_source": "historical",
        "use_trimmed_model": use_trimmed,
        "initial_volume_frac": INITIAL_VOLUME_FRAC,
    }

    if use_trimmed:
        presim_file = _require_presim_file()
        options["presimulated_releases_file"] = str(presim_file)

    mb = pywrdrb.ModelBuilder(
        inflow_type=INFLOW_TYPE,
        start_date=START_DATE,
        end_date=END_DATE,
        options=options,
        nyc_operations_config=nyc_config,
    )
    mb.make_model()
    return mb


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

    return results


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

    # Deep-copy cached base model dict and patch with this eval's parameters
    base_dict = _get_cached_model_dict(use_trimmed=use_trimmed)
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

    del model, mem_recorder
    return data


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

def evaluate(dv_vector, formulation_name="ffmp", objective_set=None):
    """Full evaluation pipeline: DVs -> simulation -> objectives.

    Called by Borg MOEA for each candidate solution. Uses in-memory
    simulation to minimize I/O overhead.

    Args:
        dv_vector: Array of decision variable values.
        formulation_name: Formulation name string.
        objective_set: ObjectiveSet instance. If None, uses the active set
            from config.ACTIVE_OBJECTIVE_SET.

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

    config = dvs_to_config(dv_vector, formulation_name)
    data = run_simulation_inmemory(config)
    objs = objective_set.compute_for_borg(data)

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
