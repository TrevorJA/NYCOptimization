"""
pareto_evolution.py - Visualize Pareto front evolution across NFE snapshots.

Reads Borg runtime files and plots how the Pareto-approximate set improves
at selected NFE checkpoints. Two views:
  1. Pairwise scatter plots at selected NFE snapshots
  2. Parallel coordinates at selected NFE snapshots (small multiples)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

from config import get_obj_names, get_obj_directions, get_n_vars


def parse_runtime_snapshots(runtime_file: Path, n_vars: int):
    """Parse a Borg runtime file into per-NFE snapshots.

    Returns:
        List of (nfe, obj_array) tuples, where obj_array has shape
        (n_solutions, n_objs) with raw (un-negated) objective values.
    """
    directions = get_obj_directions()
    n_objs = len(directions)
    snapshots = []
    current_nfe = None
    current_rows = []

    with open(runtime_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("//NFE="):
                if current_nfe is not None and current_rows:
                    obj_array = _rows_to_objs(current_rows, n_vars, n_objs, directions)
                    snapshots.append((current_nfe, obj_array))
                current_nfe = int(line.split("=")[1])
                current_rows = []
            elif line.startswith("//"):
                continue
            elif line and current_nfe is not None:
                try:
                    values = [float(x) for x in line.split()]
                    current_rows.append(values)
                except ValueError:
                    continue

    # Last snapshot
    if current_nfe is not None and current_rows:
        obj_array = _rows_to_objs(current_rows, n_vars, n_objs, directions)
        snapshots.append((current_nfe, obj_array))

    return snapshots


def _rows_to_objs(rows, n_vars, n_objs, directions):
    """Extract objectives from raw rows, un-negating maximize objectives."""
    data = np.array(rows)
    objs = data[:, n_vars:n_vars + n_objs]
    for i in range(n_objs):
        if directions[i] == 1:
            objs[:, i] = -objs[:, i]
    return objs


def plot_pareto_evolution_scatter(
    runtime_file: Path,
    formulation: str,
    obj_pair: tuple = (0, 4),
    n_snapshots: int = 6,
    output_file: Path = None,
    baseline_objs: np.ndarray = None,
    figsize: tuple = (14, 8),
):
    """Plot pairwise scatter of Pareto front at selected NFE snapshots.

    Args:
        runtime_file: Path to a Borg .runtime file.
        formulation: Formulation name.
        obj_pair: Tuple of (obj_x_index, obj_y_index) for scatter axes.
        n_snapshots: Number of NFE snapshots to show.
        output_file: Path to save figure.
        baseline_objs: Optional baseline objective values (raw).
        figsize: Figure size.
    """
    n_vars = get_n_vars(formulation)
    obj_names = get_obj_names()
    snapshots = parse_runtime_snapshots(runtime_file, n_vars)

    if not snapshots:
        print("No snapshots found in runtime file.")
        return

    # Select evenly spaced snapshots
    indices = np.linspace(0, len(snapshots) - 1, n_snapshots, dtype=int)
    selected = [snapshots[i] for i in indices]

    ix, iy = obj_pair
    ncols = 3
    nrows = (n_snapshots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, (ax, (nfe, objs)) in enumerate(zip(axes.flat, selected)):
        ax.scatter(objs[:, ix], objs[:, iy], s=8, alpha=0.5, color="steelblue")

        if baseline_objs is not None:
            ax.scatter(
                baseline_objs[ix], baseline_objs[iy],
                s=80, color="firebrick", marker="*", zorder=10,
                edgecolors="black", linewidths=0.5,
            )

        ax.set_title(f"NFE = {nfe:,}  ({objs.shape[0]} sol.)", fontsize=10)
        ax.set_xlabel(obj_names[ix], fontsize=8)
        ax.set_ylabel(obj_names[iy], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Pareto Front Evolution: {obj_names[ix]} vs {obj_names[iy]}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close(fig)
    else:
        plt.show()


def plot_pareto_evolution_overlay(
    runtime_file: Path,
    formulation: str,
    obj_pair: tuple = (0, 4),
    n_snapshots: int = 5,
    output_file: Path = None,
    baseline_objs: np.ndarray = None,
    figsize: tuple = (8, 6),
):
    """Overlay Pareto fronts at different NFE on a single scatter plot.

    Color encodes NFE progression (light = early, dark = late).

    Args:
        runtime_file: Path to a Borg .runtime file.
        formulation: Formulation name.
        obj_pair: Tuple of (obj_x_index, obj_y_index).
        n_snapshots: Number of NFE snapshots to overlay.
        output_file: Path to save figure.
        baseline_objs: Optional baseline objective values (raw).
        figsize: Figure size.
    """
    n_vars = get_n_vars(formulation)
    obj_names = get_obj_names()
    snapshots = parse_runtime_snapshots(runtime_file, n_vars)

    if not snapshots:
        print("No snapshots found in runtime file.")
        return

    indices = np.linspace(0, len(snapshots) - 1, n_snapshots, dtype=int)
    selected = [snapshots[i] for i in indices]

    ix, iy = obj_pair
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=selected[0][0], vmax=selected[-1][0])

    for nfe, objs in selected:
        color = cmap(norm(nfe))
        ax.scatter(
            objs[:, ix], objs[:, iy],
            s=12, alpha=0.6, color=color,
            label=f"NFE={nfe:,} ({objs.shape[0]})",
        )

    if baseline_objs is not None:
        ax.scatter(
            baseline_objs[ix], baseline_objs[iy],
            s=120, color="firebrick", marker="*", zorder=10,
            edgecolors="black", linewidths=0.5, label="FFMP Baseline",
        )

    ax.set_xlabel(obj_names[ix], fontsize=11)
    ax.set_ylabel(obj_names[iy], fontsize=11)
    ax.set_title(f"Pareto Front Evolution ({formulation})", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close(fig)
    else:
        plt.show()
