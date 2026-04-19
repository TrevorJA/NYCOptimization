"""
run_baseline.py - Evaluate the default FFMP policy (no optimization).

Runs a single Pywr-DRB simulation with baseline decision variable values
and saves full HDF5 output for analysis. This provides the "status quo"
reference point against which optimized solutions are compared.

Model mode:
    The baseline uses the FULL model (use_trimmed=False) by default. The
    historic baseline is a single run so efficiency is not a concern, and
    the full model is more accurate (all STARFIT reservoirs simulate freely).

    The trimmed model path (use_trimmed=True) is available for quick tests
    after running 00_generate_presim.sh, but is not recommended for the
    final baseline result.

Usage:
    python scripts/run_baseline.py [--formulation ffmp] [--use-trimmed] [--test-inmemory]

Outputs:
    outputs/baseline/{formulation}_baseline.hdf5
    outputs/baseline/{formulation}_baseline_objectives.csv
"""

import sys
import time
import argparse
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import OUTPUTS_DIR
from src.formulations import get_baseline_values, get_var_names
from src.simulation import dvs_to_config, run_simulation_to_disk, run_simulation_inmemory
from src.formulations import get_objective_set


def run_baseline(formulation: str = "ffmp", use_trimmed: bool = False):
    """Run baseline simulation with full model and compute objectives.

    Args:
        formulation: Problem formulation name.
        use_trimmed: If True, use trimmed model (requires presim data from
            00_generate_presim.sh). Default False for accurate baseline.

    Returns:
        Tuple of (data dict, objectives list).
    """
    _ACTIVE_OBJS = get_objective_set()
    baseline_dir = OUTPUTS_DIR / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    dv_values = get_baseline_values(formulation)
    var_names = get_var_names(formulation)
    model_mode = "trimmed" if use_trimmed else "full"

    print(f"\n--- Baseline config ({formulation}, {model_mode} model) ---")
    for name, val in zip(var_names, dv_values):
        print(f"  {name} = {val}")

    config = dvs_to_config(dv_values, formulation)

    output_file = baseline_dir / f"{formulation}_baseline.hdf5"
    print(f"\n--- Running simulation ({model_mode} model) ---")
    t0 = time.perf_counter()
    data = run_simulation_to_disk(config, output_file, use_trimmed=use_trimmed)
    elapsed = time.perf_counter() - t0
    print(f"  Output : {output_file}")
    print(f"  Elapsed: {elapsed:.1f}s")

    print(f"\n--- Objectives ---")
    obj_values = _ACTIVE_OBJS.compute(data)
    obj_names = _ACTIVE_OBJS.names
    for name, val in zip(obj_names, obj_values):
        print(f"  {name} = {val:.6f}")

    obj_df = pd.DataFrame([obj_values], columns=obj_names)
    obj_csv = baseline_dir / f"{formulation}_baseline_objectives.csv"
    obj_df.to_csv(obj_csv, index=False)
    print(f"\n  Objectives saved: {obj_csv}")

    return data, obj_values


def run_inmemory_test(formulation: str = "ffmp", use_trimmed: bool = False):
    """Test the in-memory simulation path against the disk-based result.

    Useful for verifying that the in-memory extraction in simulation.py
    produces results consistent with the full HDF5/Data() path.

    Args:
        formulation: Problem formulation name.
        use_trimmed: Use trimmed model. Requires presim data if True.
    """
    _ACTIVE_OBJS = get_objective_set()
    model_mode = "trimmed" if use_trimmed else "full"
    print(f"\n--- In-memory test ({formulation}, {model_mode} model) ---")

    dv_values = get_baseline_values(formulation)
    config = dvs_to_config(dv_values, formulation)

    t0 = time.perf_counter()
    data = run_simulation_inmemory(config, use_trimmed=use_trimmed)
    elapsed = time.perf_counter() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    for key in ["major_flow", "res_storage", "ibt_demands", "ibt_diversions"]:
        df = data.get(key)
        if df is not None and not df.empty:
            print(f"  {key}: shape={df.shape}, cols={list(df.columns)[:4]}")
        else:
            print(f"  {key}: MISSING or EMPTY")

    obj_values = _ACTIVE_OBJS.compute(data)
    obj_names = _ACTIVE_OBJS.names
    print(f"\n  Objectives (in-memory):")
    for name, val in zip(obj_names, obj_values):
        print(f"    {name} = {val:.6f}")

    return data, obj_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline FFMP evaluation")
    parser.add_argument("--formulation", type=str, default="ffmp")
    parser.add_argument(
        "--use-trimmed", action="store_true",
        help="Use trimmed model (requires presim data). Default: full model."
    )
    parser.add_argument(
        "--test-inmemory", action="store_true",
        help="Also test the in-memory path and compare with disk-based results."
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Baseline Evaluation")
    print(f"  Formulation : {args.formulation}")
    print(f"  Model mode  : {'trimmed' if args.use_trimmed else 'full'}")
    print("=" * 50)

    data_disk, objs_disk = run_baseline(
        args.formulation, use_trimmed=args.use_trimmed
    )

    if args.test_inmemory:
        data_mem, objs_mem = run_inmemory_test(
            args.formulation, use_trimmed=args.use_trimmed
        )
        print(f"\n--- Comparison: disk vs in-memory ---")
        obj_names = get_objective_set().names
        for name, vd, vm in zip(obj_names, objs_disk, objs_mem):
            diff = abs(vd - vm)
            flag = " <-- MISMATCH" if diff > 1e-6 else ""
            print(f"  {name}: disk={vd:.6f}  mem={vm:.6f}  diff={diff:.2e}{flag}")

    print("\n--- Baseline complete ---")
