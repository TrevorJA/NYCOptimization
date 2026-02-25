"""
simulation.py - Pywr-DRB simulation wrapper for optimization.

Provides the core function that:
1. Takes a flat decision variable vector
2. Converts it to a NYCOperationsConfig
3. Builds and runs a Pywr-DRB simulation
4. Computes and returns the objective vector

This module is imported by the Borg optimization script.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    START_DATE,
    END_DATE,
    INFLOW_TYPE,
    USE_TRIMMED_MODEL,
    INITIAL_VOLUME_FRAC,
    RESULTS_SETS,
    PRESIM_DIR,
    NYC_RESERVOIRS,
    get_formulation,
    get_var_names,
)


###############################################################################
# Decision Variable -> NYCOperationsConfig Conversion
###############################################################################

def dvs_to_config(dv_vector, formulation_name="ffmp"):
    """Convert a flat decision variable vector to a NYCOperationsConfig.

    Args:
        dv_vector: Array-like of decision variable values, ordered
                   according to the formulation's decision_variables dict.
        formulation_name: Name of the formulation to use.

    Returns:
        pywrdrb.parameters.NYCOperationsConfig instance.
    """
    from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig

    var_names = get_var_names(formulation_name)
    params = dict(zip(var_names, dv_vector))

    # Start from default FFMP config
    config = NYCOperationsConfig.from_defaults()

    if formulation_name == "ffmp":
        _apply_ffmp_params(config, params)
    else:
        raise NotImplementedError(
            f"Formulation '{formulation_name}' conversion not yet implemented."
        )

    return config


def _apply_ffmp_params(config, params: dict):
    """Apply Formulation A (parameterized FFMP) parameters to config.

    Modifies config in-place.
    """
    # --- MRF baselines ---
    config.update_mrf_baselines(
        cannonsville=params["mrf_cannonsville"],
        pepacton=params["mrf_pepacton"],
        neversink=params["mrf_neversink"],
    )

    # Downstream MRF targets
    config.update_mrf_baselines(
        delMontague=params["mrf_montague"],
        delTrenton=params["mrf_trenton"],
    )

    # --- Delivery constraints ---
    # NYC drought factors: L1a through L2 stay at high values (unconstrained),
    # L3-L5 use optimized factors
    config.update_delivery_constraints(
        max_nyc_delivery=params["max_nyc_delivery"],
        nyc_drought_factor_L3=params["nyc_drought_factor_L3"],
        nyc_drought_factor_L4=params["nyc_drought_factor_L4"],
        nyc_drought_factor_L5=params["nyc_drought_factor_L5"],
        nj_drought_factor_L4=params["nj_drought_factor_L4"],
        nj_drought_factor_L5=params["nj_drought_factor_L5"],
    )

    # --- Storage zone vertical shifts ---
    zone_levels = ["level1b", "level1c", "level2", "level3", "level4", "level5"]
    shifts = {level: params[f"zone_shift_{level}"] for level in zone_levels}
    _apply_zone_shifts(config, shifts)

    # --- Flood release maximums ---
    config.update_flood_limits(
        cannonsville=params["flood_max_cannonsville"],
        pepacton=params["flood_max_pepacton"],
        neversink=params["flood_max_neversink"],
    )

    # --- MRF seasonal profile scaling ---
    _apply_mrf_profile_scaling(config, params)


def _apply_zone_shifts(config, shifts: dict):
    """Apply vertical shifts to storage zone thresholds.

    Shifts are additive (fraction of capacity). After shifting,
    enforces monotonic ordering: level1b >= level1c >= ... >= level5.
    """
    zone_order = ["level1b", "level1c", "level2", "level3", "level4", "level5"]
    zones = config.storage_zones.copy()

    # Apply shifts
    for level in zone_order:
        if level in zones.index:
            zones.loc[level] = zones.loc[level] + shifts[level]

    # Clip to [0, 1]
    zones = zones.clip(lower=0.0, upper=1.0)

    # Enforce monotonic ordering (each level must be <= the one above it)
    for i in range(1, len(zone_order)):
        more_severe = zone_order[i]
        less_severe = zone_order[i - 1]
        if more_severe in zones.index and less_severe in zones.index:
            zones.loc[more_severe] = np.minimum(
                zones.loc[more_severe].values,
                zones.loc[less_severe].values,
            )

    config.storage_zones = zones


def _apply_mrf_profile_scaling(config, params: dict):
    """Apply seasonal scaling to MRF daily factor profiles.

    Divides the year into four seasons and scales all MRF daily
    factor profiles by the corresponding season multiplier.

    Season boundaries (approximate DOY):
        Winter: Dec 1 - Feb 28  (DOY 335-365, 1-59)
        Spring: Mar 1 - May 31  (DOY 60-151)
        Summer: Jun 1 - Aug 31  (DOY 152-243)
        Fall:   Sep 1 - Nov 30  (DOY 244-334)
    """
    season_ranges = {
        "winter": list(range(335, 367)) + list(range(1, 60)),
        "spring": list(range(60, 152)),
        "summer": list(range(152, 244)),
        "fall": list(range(244, 335)),
    }

    mrf_factors = config.mrf_daily_factors.copy()

    for season, doy_range in season_ranges.items():
        scale = params[f"mrf_profile_scale_{season}"]
        # Convert DOY to 0-indexed column positions
        cols = [d - 1 for d in doy_range if d - 1 < mrf_factors.shape[1]]
        mrf_factors.iloc[:, cols] *= scale

    config.mrf_daily_factors = mrf_factors


###############################################################################
# Simulation Execution
###############################################################################

def run_simulation(nyc_config, output_file: Optional[Path] = None) -> dict:
    """Run a single Pywr-DRB simulation with given NYC operations config.

    Args:
        nyc_config: NYCOperationsConfig instance.
        output_file: Optional path to save HDF5 output. If None, uses
                     a temporary file.

    Returns:
        Dict of DataFrames keyed by results set name.
    """
    import pywrdrb
    import tempfile

    if output_file is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False)
        output_file = Path(tmp.name)
        tmp.close()

    # Find pre-simulated releases file (for trimmed model)
    presim_file = _find_presim_file()

    # Build model
    options = {
        "nyc_nj_demand_source": "historical",
        "use_trimmed_model": USE_TRIMMED_MODEL,
        "initial_volume_frac": INITIAL_VOLUME_FRAC,
    }
    if presim_file is not None:
        options["presimulated_releases_file"] = str(presim_file)

    mb = pywrdrb.ModelBuilder(
        inflow_type=INFLOW_TYPE,
        start_date=START_DATE,
        end_date=END_DATE,
        options=options,
        nyc_operations_config=nyc_config,
    )
    mb.make_model()

    # Write model JSON to temp location
    model_json = output_file.with_suffix(".json")
    mb.write_model(str(model_json))

    # Load and run
    model = pywrdrb.Model.load(str(model_json))
    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=str(output_file),
    )
    model.run()

    # Load results
    data = _load_results(output_file)

    # Cleanup temp model JSON
    model_json.unlink(missing_ok=True)

    return data


def _find_presim_file() -> Optional[Path]:
    """Find the pre-simulated releases file for the trimmed model."""
    if not USE_TRIMMED_MODEL:
        return None

    candidates = list(PRESIM_DIR.glob("*.hdf5")) + list(PRESIM_DIR.glob("*.csv"))
    if candidates:
        return candidates[0]
    return None


def _load_results(output_file: Path) -> dict:
    """Load simulation results from HDF5 into a dict of DataFrames."""
    import pywrdrb

    data_loader = pywrdrb.Data()
    data_loader.load_output(
        [str(output_file)],
        results_sets=RESULTS_SETS,
    )

    results = {}
    for rs in RESULTS_SETS:
        if hasattr(data_loader, rs):
            df = getattr(data_loader, rs)
            # Data loader returns list of DataFrames (one per output file)
            if isinstance(df, list) and len(df) > 0:
                results[rs] = df[0]
            elif isinstance(df, pd.DataFrame):
                results[rs] = df
    return results


###############################################################################
# Combined: DV Vector -> Objective Vector (for Borg)
###############################################################################

def evaluate(dv_vector, formulation_name="ffmp"):
    """Full evaluation pipeline: DVs -> simulation -> objectives.

    This is the function called by the Borg MOEA for each candidate solution.

    Args:
        dv_vector: Array of decision variable values.
        formulation_name: Formulation name string.

    Returns:
        List of objective values (Borg-compatible, all minimized).
    """
    from src.objectives import objectives_for_borg

    # 1. Convert DVs to config
    config = dvs_to_config(dv_vector, formulation_name)

    # 2. Run simulation
    data = run_simulation(config)

    # 3. Compute objectives
    objs = objectives_for_borg(data)

    return objs
