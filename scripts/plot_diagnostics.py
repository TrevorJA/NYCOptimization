"""
plot_diagnostics.py - Visualize MOEA runtime diagnostics.

Reads metrics files produced by MOEAFramework and generates:
    1. Hypervolume convergence across NFE (per seed and envelope)
    2. Seed reliability assessment
    3. Parallel coordinates of the reference set

Usage:
    python plot_diagnostics.py --formulation ffmp
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    get_obj_names,
    get_obj_directions,
    get_n_vars,
    OUTPUTS_DIR,
    BORG_SETTINGS,
)


def load_metrics(diag_dir: Path, formulation: str) -> dict:
    """Load MOEAFramework metrics files for all seeds.

    Returns:
        Dict mapping seed number to DataFrame of metrics over NFE.
    """
    metrics = {}
    for mf in sorted(diag_dir.glob(f"seed_*_{formulation}.metrics")):
        seed_str = mf.stem.split("_")[1]
        seed = int(seed_str)
        # MOEAFramework metrics files are whitespace-delimited
        # Columns: NFE, Hypervolume, GenerationalDistance, ...
        try:
            df = pd.read_csv(mf, sep=r"\s+", comment="#")
            metrics[seed] = df
        except Exception as e:
            print(f"Warning: Could not load {mf}: {e}")
    return metrics


def load_reference_set(ref_file: Path, formulation: str) -> np.ndarray:
    """Load reference set and return objective values."""
    n_vars = get_n_vars(formulation)
    data = []
    with open(ref_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            # Objectives start after decision variables
            data.append(values[n_vars:])
    return np.array(data)


def plot_hypervolume_convergence(metrics: dict, fig_dir: Path, formulation: str):
    """Plot hypervolume vs NFE for all seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for seed, df in sorted(metrics.items()):
        if "Hypervolume" in df.columns and "NFE" in df.columns:
            ax.plot(df["NFE"], df["Hypervolume"],
                    alpha=0.5, linewidth=1, label=f"Seed {seed}")

    ax.set_xlabel("Number of Function Evaluations (NFE)")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"Hypervolume Convergence: {formulation}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / f"hypervolume_convergence_{formulation}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: hypervolume_convergence_{formulation}.png")


def plot_seed_reliability(metrics: dict, fig_dir: Path, formulation: str):
    """Plot final hypervolume distribution across seeds (boxplot)."""
    final_hvs = []
    for seed, df in sorted(metrics.items()):
        if "Hypervolume" in df.columns:
            final_hvs.append(df["Hypervolume"].iloc[-1])

    if not final_hvs:
        print("  No hypervolume data available for reliability plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(final_hvs, vert=True)
    ax.scatter(np.ones(len(final_hvs)), final_hvs, alpha=0.6, zorder=3)
    ax.set_ylabel("Final Hypervolume")
    ax.set_title(f"Seed Reliability: {formulation}")
    ax.set_xticks([1])
    ax.set_xticklabels([formulation])

    fig.tight_layout()
    fig.savefig(fig_dir / f"seed_reliability_{formulation}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: seed_reliability_{formulation}.png")


def plot_parallel_coordinates(ref_set: np.ndarray, fig_dir: Path, formulation: str):
    """Plot parallel coordinates of the reference set objectives."""
    obj_names = get_obj_names()
    directions = get_obj_directions()
    n_objs = len(obj_names)

    if ref_set.shape[0] == 0:
        print("  Empty reference set, skipping parallel coordinates.")
        return

    # Un-negate maximization objectives (Borg stores minimized values)
    display = ref_set.copy()
    for i in range(n_objs):
        if directions[i] == 1:  # was negated for Borg
            display[:, i] = -display[:, i]

    # Normalize each objective to [0, 1] for visualization
    normed = np.zeros_like(display)
    for i in range(n_objs):
        col = display[:, i]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            normed[:, i] = (col - cmin) / (cmax - cmin)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_objs)

    for row in normed:
        ax.plot(x, row, alpha=0.15, color="steelblue", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(obj_names, rotation=30, ha="right")
    ax.set_ylabel("Normalized Objective Value")
    ax.set_title(f"Parallel Coordinates: {formulation} Reference Set "
                 f"({ref_set.shape[0]} solutions)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(fig_dir / f"parallel_coords_{formulation}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: parallel_coords_{formulation}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot MOEA diagnostics."
    )
    parser.add_argument("--formulation", type=str, default="ffmp")
    args = parser.parse_args()

    formulation = args.formulation
    diag_dir = OUTPUTS_DIR / "diagnostics" / formulation
    ref_file = OUTPUTS_DIR / "reference_sets" / f"{formulation}.ref"
    fig_dir = OUTPUTS_DIR / "figures" / formulation
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting diagnostics for: {formulation}")

    # Metrics
    if diag_dir.exists():
        metrics = load_metrics(diag_dir, formulation)
        if metrics:
            plot_hypervolume_convergence(metrics, fig_dir, formulation)
            plot_seed_reliability(metrics, fig_dir, formulation)
        else:
            print("  No metrics files found.")
    else:
        print(f"  Diagnostics directory not found: {diag_dir}")

    # Reference set parallel coordinates
    if ref_file.exists():
        ref_set = load_reference_set(ref_file, formulation)
        plot_parallel_coordinates(ref_set, fig_dir, formulation)
    else:
        print(f"  Reference set not found: {ref_file}")


if __name__ == "__main__":
    main()
