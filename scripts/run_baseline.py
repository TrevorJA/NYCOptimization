"""
run_baseline.py - Run baseline (default FFMP) simulation and compute metrics.

Establishes the reference performance of the current 2017 FFMP rules
against all objectives. This baseline is used for:
    1. Comparison against optimized solutions
    2. Setting satisficing thresholds for robustness analysis
    3. Verifying the simulation pipeline works end-to-end

Usage:
    python run_baseline.py [--formulation ffmp]
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    get_baseline_values,
    get_var_names,
    get_obj_names,
    get_obj_directions,
    OUTPUTS_DIR,
)
from src.simulation import dvs_to_config, run_simulation
from src.objectives import compute_all_objectives


def main(formulation_name="ffmp"):
    print(f"Running baseline evaluation for formulation: {formulation_name}")

    # Get default FFMP parameter values
    baseline_dvs = get_baseline_values(formulation_name)
    var_names = get_var_names(formulation_name)
    obj_names = get_obj_names()
    directions = get_obj_directions()

    print(f"\nBaseline decision variables ({len(baseline_dvs)}):")
    for name, val in zip(var_names, baseline_dvs):
        print(f"  {name}: {val}")

    # Convert to config and run simulation
    print("\nBuilding and running Pywr-DRB simulation...")
    config = dvs_to_config(baseline_dvs, formulation_name)

    output_dir = OUTPUTS_DIR / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "baseline_output.hdf5"

    data = run_simulation(config, output_file=output_file)

    # Compute objectives
    print("\nComputing objectives...")
    obj_values = compute_all_objectives(data)

    print(f"\nBaseline Objective Values:")
    print("-" * 50)
    for name, val, d in zip(obj_names, obj_values, directions):
        direction_str = "maximize" if d == 1 else "minimize"
        print(f"  {name}: {val:.4f} ({direction_str})")

    # Save results
    results = {
        "formulation": formulation_name,
        "decision_variables": dict(zip(var_names, baseline_dvs.tolist())),
        "objectives": dict(zip(obj_names, obj_values)),
    }

    results_file = output_dir / "baseline_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"HDF5 output saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--formulation", type=str, default="ffmp")
    args = parser.parse_args()
    main(args.formulation)
