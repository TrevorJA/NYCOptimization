"""
results.py - Load Pywr-DRB simulation output from HDF5.

Uses pywrdrb.Data() to load output files with proper name mapping.
The pywrdrb.Data() class stores results as:
    data.<results_set>[output_label][scenario_id] = pd.DataFrame

where output_label is derived from the HDF5 filename and scenario_id
is an integer (0 for single-scenario runs).
"""

import pandas as pd
from pathlib import Path


def load_simulation_results(
    output_file: Path,
    results_sets: list = None,
    scenario: int = 0,
) -> dict:
    """Load simulation results from HDF5 into a dict of DataFrames.

    Args:
        output_file: Path to HDF5 output file.
        results_sets: List of result set names to load. If None, loads
            defaults from config.RESULTS_SETS.
        scenario: Scenario index to extract (default 0).

    Returns:
        Dict of DataFrames keyed by results set name, with standard
        pywrdrb node names as columns.
    """
    import pywrdrb

    if results_sets is None:
        from config import RESULTS_SETS
        results_sets = RESULTS_SETS

    data_loader = pywrdrb.Data()
    data_loader.load_output(
        output_filenames=[str(output_file)],
        results_sets=results_sets,
    )

    # The output label is the filename stem (without extension)
    label = Path(output_file).stem

    results = {}
    for rs in results_sets:
        if hasattr(data_loader, rs):
            rs_data = getattr(data_loader, rs)
            # rs_data is a dict: {label: {scenario_id: DataFrame}}
            if isinstance(rs_data, dict):
                if label in rs_data and scenario in rs_data[label]:
                    results[rs] = rs_data[label][scenario]
                else:
                    # Fallback: try first available label and scenario
                    for lbl in rs_data:
                        for scen in rs_data[lbl]:
                            results[rs] = rs_data[lbl][scen]
                            break
                        break

    return results
