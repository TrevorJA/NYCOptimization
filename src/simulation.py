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

import sys
import gc
import json
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
    get_formulation,
    get_var_names,
)


###############################################################################
# Cached model components
###############################################################################

_CACHED_PRESIM_FILE = None
_PRESIM_SEARCHED = False


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
# Decision Variable -> NYCOperationsConfig Conversion
###############################################################################

def dvs_to_config(dv_vector, formulation_name="ffmp"):
    """Convert a flat decision variable vector to a NYCOperationsConfig.

    Args:
        dv_vector: Array-like of decision variable values.
        formulation_name: Name of the formulation to use.

    Returns:
        NYCOperationsConfig instance.
    """
    from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig

    var_names = get_var_names(formulation_name)
    params = dict(zip(var_names, dv_vector))
    config = NYCOperationsConfig.from_defaults()

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
# In-Memory Recorder (skips HDF5 write)
###############################################################################

class InMemoryRecorder:
    """Lightweight recorder that captures simulation data to numpy arrays.

    Wraps pywrdrb.OutputRecorder but overrides finish() to skip the
    HDF5 write. Data is still captured during simulation via the
    internal NumpyArray*Recorder objects.

    After model.run(), access data via self.recorder_dict[key].data.
    """

    def __init__(self, model):
        from pywrdrb.recorder import OutputRecorder
        # Use /dev/null as dummy output path (finish is overridden so never written)
        self._inner = OutputRecorder.__new__(OutputRecorder)
        self._inner.model = model
        self._inner.output_filename = "/dev/null"
        self._inner.recorder_dict = {}

        # Replicate OutputRecorder.__init__ logic for recorder creation
        from pywr.recorders import (
            NumpyArrayNodeRecorder,
            NumpyArrayParameterRecorder,
            NumpyArrayStorageRecorder,
        )
        from pywr.recorders import Recorder
        from pywrdrb.utils.lists import reservoir_list

        # Register as a Pywr recorder so setup/after/finish are called
        Recorder.__init__(self._inner, model)

        nodes = [n for n in model.nodes.values() if n.name]
        parameters = [p for p in model.parameters if p.name]

        for p in parameters:
            self._inner.recorder_dict[p.name] = NumpyArrayParameterRecorder(model, p)
        for n in nodes:
            if n.name.split("_")[0] == "reservoir" and n.name.split("_")[1] in reservoir_list:
                self._inner.recorder_dict[n.name] = NumpyArrayStorageRecorder(
                    model, n, proportional=False
                )
            else:
                self._inner.recorder_dict[n.name] = NumpyArrayNodeRecorder(model, n)

        # Monkey-patch finish to skip HDF5 write
        original_finish = self._inner.finish
        def finish_no_hdf5():
            # Call finish on child recorders (frees Cython buffers) but skip to_hdf5()
            for rec in self._inner.recorder_dict.values():
                rec.finish()
        self._inner.finish = finish_no_hdf5

    @property
    def recorder_dict(self):
        return self._inner.recorder_dict


###############################################################################
# Extract Results from Recorder (In-Memory)
###############################################################################

def _extract_results_from_recorder(recorder_dict, datetime_index, scenario=0) -> dict:
    """Extract simulation results from recorder dict into DataFrames.

    Applies the same name mapping as pywrdrb's Output.get_keys_and_column_names_for_results_set()
    to produce DataFrames with standard pywrdrb node names.

    Args:
        recorder_dict: Dict mapping raw pywr names to NumpyArray*Recorder objects.
        datetime_index: Model timestepper datetime index.
        scenario: Scenario index to extract (default 0 for single-scenario runs).

    Returns:
        Dict of DataFrames keyed by results_set name, matching the format
        expected by objectives.py.
    """
    from pywrdrb.utils.lists import reservoir_list, majorflow_list

    all_keys = list(recorder_dict.keys())
    dt_index = pd.DatetimeIndex(datetime_index)

    def _build_df(key_filter_fn, name_extract_fn, name_filter=None):
        """Build a DataFrame from recorders matching a filter."""
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

    # res_storage: "reservoir_X" -> "X" where X in reservoir_list
    results["res_storage"] = _build_df(
        key_filter_fn=lambda k: k.split("_")[0] == "reservoir",
        name_extract_fn=lambda k: k.split("_", 1)[1],
        name_filter=set(reservoir_list),
    )

    # major_flow: "link_X" -> "X" where X in majorflow_list
    results["major_flow"] = _build_df(
        key_filter_fn=lambda k: k.split("_")[0] == "link",
        name_extract_fn=lambda k: k.split("_", 1)[1],
        name_filter=set(majorflow_list),
    )

    # ibt_demands: exact keys "demand_nyc", "demand_nj"
    demand_data = {}
    for k in ["demand_nyc", "demand_nj"]:
        if k in recorder_dict:
            demand_data[k] = recorder_dict[k].data[:, scenario]
    results["ibt_demands"] = pd.DataFrame(demand_data, index=dt_index)

    # ibt_diversions: exact keys "delivery_nyc", "delivery_nj"
    delivery_data = {}
    for k in ["delivery_nyc", "delivery_nj"]:
        if k in recorder_dict:
            delivery_data[k] = recorder_dict[k].data[:, scenario]
    results["ibt_diversions"] = pd.DataFrame(delivery_data, index=dt_index)

    return results


###############################################################################
# In-Memory Simulation (for optimization, minimal disk I/O)
###############################################################################

# Cache the temp directory for model JSON to avoid repeated creation
_TEMP_DIR = None


def _get_temp_dir():
    """Get or create a persistent temp directory for model JSON files."""
    global _TEMP_DIR
    if _TEMP_DIR is None:
        _TEMP_DIR = tempfile.mkdtemp(prefix="pywrdrb_opt_")
    return _TEMP_DIR


def run_simulation_inmemory(nyc_config, use_trimmed: bool = None) -> dict:
    """Run Pywr-DRB simulation with minimal disk I/O.

    Writes a temporary model JSON (required by pywr.Model.load) but
    avoids HDF5 output. Results are extracted directly from in-memory
    recorders after the simulation completes.

    Args:
        nyc_config: NYCOperationsConfig instance.
        use_trimmed: Use trimmed model (fast, requires presim data). Defaults
            to USE_TRIMMED_MODEL from config.

    Returns:
        Dict of DataFrames keyed by results set name.
    """
    import pywrdrb

    mb = _build_model_builder(nyc_config, use_trimmed=use_trimmed)

    # pywr requires a JSON file to load the model
    tmp_dir = _get_temp_dir()
    model_json = str(Path(tmp_dir) / "opt_model.json")
    model = _write_and_load_model(mb, model_json)

    # Attach in-memory recorder (captures data without writing HDF5)
    mem_recorder = InMemoryRecorder(model)

    # Run simulation
    model.run()

    # Extract results from in-memory recorder data
    datetime_index = model.timestepper.datetime_index.to_timestamp()
    data = _extract_results_from_recorder(
        mem_recorder.recorder_dict, datetime_index
    )

    # Cleanup
    del model, mb, mem_recorder
    gc.collect()

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
    if objective_set is None:
        from config import get_objective_set
        objective_set = get_objective_set()

    config = dvs_to_config(dv_vector, formulation_name)
    data = run_simulation_inmemory(config)
    objs = objective_set.compute_for_borg(data)
    return objs
