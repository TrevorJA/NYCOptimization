"""
run_baseline.py - Evaluate the default FFMP policy (no optimization).

Runs a single Pywr-DRB simulation with baseline decision variable values
and saves full HDF5 output for analysis. This provides the "status quo"
reference point against which optimized solutions are compared.

Objectives:
    Scored with the active objective set from ``config.get_objective_set()`` —
    the same annual-unit objective function that search and re-evaluation use,
    with the single baseline trace passed as one realization (``[data]``, N=1).
    This is what makes the baseline vector directly comparable to Pareto
    objective vectors and to the re-evaluation matrix.

Model mode:
    The baseline uses the FULL model (use_trimmed=False) by default. The
    historic baseline is a single run so efficiency is not a concern, and
    the full model is more accurate (all STARFIT reservoirs simulate freely).

    The trimmed model path (use_trimmed=True) is available for quick tests
    after running workflow/01_generate_presim.sh, but is not recommended
    for the final baseline result.

Usage:
    python scripts/main/run_baseline.py [--formulation ffmp] [--use-trimmed] [--test-inmemory]

Outputs:
    outputs/baseline/{formulation}_baseline.hdf5
    outputs/baseline/{formulation}_baseline_objectives.csv
    outputs/baseline/{scenario}/{formulation}_baseline_objectives.csv
        (--search-ensemble: the same policy scored on an ensemble scenario's
        search ensemble — the vector comparable to that scenario's fronts)
"""

import sys
import time
import argparse
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from config import OUTPUTS_DIR, get_objective_set
from src.formulations import get_baseline_values, get_var_names
from src.simulation import dvs_to_config, run_simulation_to_disk, run_simulation_inmemory


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
    obj_values = _ACTIVE_OBJS.compute([data])
    obj_names = _ACTIVE_OBJS.names
    for name, val in zip(obj_names, obj_values):
        print(f"  {name} = {val:.6f}")

    obj_df = pd.DataFrame([obj_values], columns=obj_names)
    obj_csv = baseline_dir / f"{formulation}_baseline_objectives.csv"
    obj_df.to_csv(obj_csv, index=False)
    print(f"\n  Objectives saved: {obj_csv}")

    return data, obj_values


def run_baseline_reeval(formulation: str = "ffmp", seed=None):
    """Run the default policy through the re-eval ensemble; persist its raw matrix.

    Provides the incumbent's per-SOW objective matrix for the incumbent-relative
    regret family (``src.robustness --baseline-dir``). Uses the SAME common
    re-eval ensemble and per-SOW annual-unit objective computation as the policy
    re-eval, so the two are on equal footing. Writes ``reeval_raw.parquet`` +
    ``reeval_raw_meta.json`` under ``.../reeval/{tag}[/seed_NN]/baseline``.

    Args:
        formulation: Problem formulation name.
        seed: Optional seed (for the per-seed re-eval subdir).

    Returns:
        Path to the baseline ``reeval_raw`` file.
    """
    from config import REEVAL_ENSEMBLE_SPEC, active_scenario_name, derive_slug
    from src.reeval_core import (
        evaluate_solution_raw, persist_reeval_raw, reeval_output_dir, reeval_tag,
    )

    scenario = active_scenario_name()
    slug = derive_slug(formulation)
    base_dir = (reeval_output_dir(scenario, slug, REEVAL_ENSEMBLE_SPEC, seed)
                / "baseline")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Baseline re-eval ({formulation}) on common ensemble "
          f"'{reeval_tag(REEVAL_ENSEMBLE_SPEC)}' ---")
    dv_values = get_baseline_values(formulation)
    _sid, mat, names, err = evaluate_solution_raw(0, dv_values, formulation)
    if err:
        raise RuntimeError(f"baseline re-eval failed: {err}")
    _summary, raw_path, meta_path = persist_reeval_raw(
        base_dir, [(0, mat, names, None)], formulation, 1, seed,
    )
    print(f"  baseline raw  -> {raw_path}")
    print(f"  baseline meta -> {meta_path}")
    return raw_path


def run_baseline_search_ensemble(formulation: str = "ffmp"):
    """Score the default FFMP policy on the active scenario's search ensemble.

    Runs the baseline DV vector through ``src.simulation.evaluate`` — the
    exact evaluation path (search ensemble, model mode, realization batching,
    active annual-unit objective set) used for every candidate during that
    scenario's search — so the resulting vector is directly comparable to the
    scenario's Pareto-front objectives, which the historic single-trace
    baseline is not. Writes the scenario-partitioned CSV located by
    ``config.baseline_objectives_csv`` (natural orientation, same column
    convention as the historic baseline CSV) plus a provenance sidecar.

    Returns:
        Tuple of (natural objective values, csv path).
    """
    import json

    from config import (SEARCH_ENSEMBLE_SPEC, active_scenario_name,
                        baseline_objectives_csv)
    from src.formulations import get_obj_directions
    from src.simulation import evaluate

    scenario = active_scenario_name()
    if not SEARCH_ENSEMBLE_SPEC.is_ensemble:
        raise SystemExit(
            f"--search-ensemble: active scenario '{scenario}' evaluates on a "
            f"single trace; use the default historic baseline path instead."
        )

    _ACTIVE_OBJS = get_objective_set()
    dv_values = get_baseline_values(formulation)

    print(f"\n--- Baseline on search ensemble ({formulation}, "
          f"scenario={scenario}) ---")
    t0 = time.perf_counter()
    objs_min = evaluate(dv_values, formulation_name=formulation)
    elapsed = time.perf_counter() - t0
    if any(v >= 1e6 for v in objs_min):
        raise RuntimeError(
            "baseline search-ensemble evaluation returned the failure "
            f"penalty vector: {objs_min}"
        )

    directions = get_obj_directions()
    natural = [-v if d == 1 else v for v, d in zip(objs_min, directions)]
    obj_names = _ACTIVE_OBJS.names
    print(f"  Elapsed: {elapsed:.1f}s")
    for name, val in zip(obj_names, natural):
        print(f"  {name} = {val:.6f}")

    obj_csv = baseline_objectives_csv(formulation, scenario)
    obj_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([natural], columns=obj_names).to_csv(obj_csv, index=False)
    meta = {
        "scenario": scenario,
        "formulation": formulation,
        "ensemble_spec": repr(SEARCH_ENSEMBLE_SPEC),
        "objective_names": list(obj_names),
        "eval_seconds": round(elapsed, 1),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = obj_csv.with_name(obj_csv.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\n  Objectives saved: {obj_csv}")
    print(f"  Meta saved      : {meta_path}")
    return natural, obj_csv


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

    obj_values = _ACTIVE_OBJS.compute([data])
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
    parser.add_argument(
        "--reeval", action="store_true",
        help="Run the baseline through the common re-eval ensemble and persist "
             "its per-SOW matrix (for the incumbent-relative regret family)."
    )
    parser.add_argument(
        "--search-ensemble", action="store_true",
        help="Score the baseline policy on the active scenario's SEARCH "
             "ensemble via the same evaluate() path the search used, and "
             "write the scenario-partitioned objectives CSV. Requires an "
             "ensemble scenario design (env file)."
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed subdir for --reeval baseline output.")
    args = parser.parse_args()

    if args.reeval:
        run_baseline_reeval(args.formulation, seed=args.seed)
        print("\n--- Baseline re-eval complete ---")
        sys.exit(0)

    if args.search_ensemble:
        run_baseline_search_ensemble(args.formulation)
        print("\n--- Baseline search-ensemble scoring complete ---")
        sys.exit(0)

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
