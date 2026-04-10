"""
plot_results.py - Generate diagnostic plots from pre-computed metrics.

Reads .metrics files (produced by run_diagnostics.py) and .set files
to generate HV convergence and parallel coordinates plots. Does NOT
rerun MOEAFramework — just reads existing outputs.

Usage:
    python scripts/plot_results.py [--formulation ffmp] [--seed 1] [--baseline]

    --baseline    Run a single simulation to compute FFMP baseline objectives
                  for overlay on the parallel coordinates plot (~2-3 min).

Produces:
    outputs/figures/hv_convergence_{formulation}_seed{seed}.png
    outputs/figures/parallel_coords_{formulation}_seed{seed}.png
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OUTPUTS_DIR, BORG_SETTINGS
from src.formulations import get_n_vars, get_obj_names, get_obj_directions
from src.plotting.parallel_coordinates import plot_parallel_coordinates
from src.plotting.pareto_evolution import (
    plot_pareto_evolution_scatter,
    plot_pareto_evolution_overlay,
)


def load_metrics(metrics_file: Path) -> pd.DataFrame:
    """Load a MOEAFramework .metrics file into a DataFrame."""
    df = pd.read_csv(metrics_file, sep=r"\s+", comment=None, header=0)
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    return df


def find_metrics_files(formulation: str, seed: int) -> list:
    """Find .metrics files for a given formulation and seed."""
    metrics_dir = OUTPUTS_DIR / "optimization" / formulation / "metrics"
    pattern = f"seed_{seed:02d}_{formulation}_*.metrics"
    files = sorted(metrics_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No metrics files found matching {pattern} in {metrics_dir}.\n"
            f"Run diagnostics first: python scripts/run_diagnostics.py "
            f"--formulation {formulation} --seed {seed}"
        )
    return files


def find_set_file(formulation: str, seed: int) -> Path:
    """Find the .set file for a given formulation and seed."""
    sets_dir = OUTPUTS_DIR / "optimization" / formulation / "sets"
    # Prefer the Borg-written set file
    borg_set = sets_dir / f"seed_{seed:02d}_{formulation}.set"
    if borg_set.exists():
        return borg_set
    # Fall back to MOEAFramework-merged set
    merged_set = sets_dir / f"{formulation}_seed{seed:02d}_merged.set"
    if merged_set.exists():
        return merged_set
    raise FileNotFoundError(f"No .set file found in {sets_dir}")


def plot_hv_convergence(metrics_files, formulation, seed, runtime_freq, output_file):
    """Plot hypervolume convergence from MOEAFramework metrics files."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for mf in metrics_files:
        island = mf.stem.split("_")[-1]
        df = load_metrics(mf)

        if "Hypervolume" not in df.columns:
            print(f"  Warning: no Hypervolume column in {mf.name}")
            continue

        nfe = np.arange(1, len(df) + 1) * runtime_freq
        ax.plot(nfe, df["Hypervolume"], "o-", markersize=3, linewidth=1.5,
                label=f"Island {island}")

    ax.set_xlabel("Number of Function Evaluations (NFE)")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"Hypervolume Convergence ({formulation}, seed {seed})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_file}")


def compute_baseline_objectives(formulation):
    """Run a single in-memory simulation with default FFMP DVs."""
    try:
        from src.formulations import get_baseline_values, get_objective_set
        from src.simulation import dvs_to_config, run_simulation_inmemory

        baseline_dvs = get_baseline_values(formulation)
        obj_set = get_objective_set()

        print(f"  Running baseline simulation ({len(baseline_dvs)} DVs)...")
        config = dvs_to_config(baseline_dvs, formulation)
        data = run_simulation_inmemory(config)
        raw_objs = obj_set.compute(data)
        print(f"  Baseline objectives: {[f'{v:.4f}' for v in raw_objs]}")
        return np.array(raw_objs)
    except Exception as e:
        print(f"  Could not compute baseline objectives: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Plot optimization results")
    parser.add_argument("--formulation", type=str, default="ffmp")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baseline", action="store_true",
                        help="Compute baseline objectives via simulation (~2 min)")
    args = parser.parse_args()

    formulation = args.formulation
    seed = args.seed
    runtime_freq = BORG_SETTINGS.get("runtime_frequency", 500)

    fig_dir = OUTPUTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- HV convergence ---
    print(f"\n=== Plotting results: {formulation}, seed {seed} ===\n")
    metrics_files = find_metrics_files(formulation, seed)
    print(f"Found {len(metrics_files)} metrics file(s)")

    hv_file = fig_dir / f"hv_convergence_{formulation}_seed{seed:02d}.png"
    plot_hv_convergence(metrics_files, formulation, seed, runtime_freq, hv_file)

    # --- Parallel coordinates ---
    set_file = find_set_file(formulation, seed)
    print(f"\nParallel coordinates from {set_file.name}")

    baseline_objs = None
    if args.baseline:
        print("  Computing baseline objectives...")
        baseline_objs = compute_baseline_objectives(formulation)

    pc_file = fig_dir / f"parallel_coords_{formulation}_seed{seed:02d}.png"
    plot_parallel_coordinates(
        set_file=set_file,
        formulation=formulation,
        output_file=pc_file,
        baseline_objs=baseline_objs,
    )

    # --- Pareto front evolution ---
    runtime_dir = OUTPUTS_DIR / "optimization" / formulation / "runtime"
    runtime_files = sorted(runtime_dir.glob(
        f"seed_{seed:02d}_{formulation}_*.runtime"
    ))
    if runtime_files:
        # Use island 0 runtime file for evolution plots
        rt_file = runtime_files[0]
        print(f"\nPareto evolution from {rt_file.name}")

        # Pairwise scatter: NYC reliability vs flood risk
        scatter_file = fig_dir / f"pareto_evolution_scatter_{formulation}_seed{seed:02d}.png"
        plot_pareto_evolution_scatter(
            runtime_file=rt_file,
            formulation=formulation,
            obj_pair=(0, 4),  # nyc_reliability vs flood_risk
            output_file=scatter_file,
            baseline_objs=baseline_objs,
        )

        # Overlay plot
        overlay_file = fig_dir / f"pareto_evolution_overlay_{formulation}_seed{seed:02d}.png"
        plot_pareto_evolution_overlay(
            runtime_file=rt_file,
            formulation=formulation,
            obj_pair=(0, 4),
            output_file=overlay_file,
            baseline_objs=baseline_objs,
        )

    # --- Summary ---
    print(f"\n=== Done ===")
    for mf in metrics_files:
        df = load_metrics(mf)
        if "Hypervolume" in df.columns:
            print(f"  {mf.stem}: final HV = {df['Hypervolume'].iloc[-1]:.6f}")


if __name__ == "__main__":
    main()
