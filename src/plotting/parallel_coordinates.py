"""
parallel_coordinates.py - Parallel coordinate plot of Pareto reference set.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import get_obj_names, get_obj_directions, get_n_vars
from src.load.reference_set import load_reference_set


def plot_parallel_coordinates(
    ref_file: Path,
    formulation: str,
    output_file: Path = None,
    figsize: tuple = (10, 5),
    highlight_baseline: bool = True,
):
    """Plot parallel coordinates of the reference set objectives.

    Each axis represents a normalized objective. Solutions are drawn
    as polylines. All objectives are oriented so that "up" is better.

    Args:
        ref_file: Path to .ref file.
        formulation: Formulation name.
        output_file: Path to save figure.
        figsize: Figure size.
        highlight_baseline: If True, overlay baseline performance.
    """
    n_vars = get_n_vars(formulation)
    _, obj_data = load_reference_set(ref_file, n_vars)
    obj_names = get_obj_names()
    directions = get_obj_directions()
    n_objs = len(obj_names)

    if obj_data.shape[0] == 0:
        print("Empty reference set.")
        return

    # Un-negate maximization objectives (Borg stores all-minimized)
    display = obj_data.copy()
    for i in range(n_objs):
        if directions[i] == 1:
            display[:, i] = -display[:, i]

    # Normalize to [0, 1]
    normed = np.zeros_like(display)
    for i in range(n_objs):
        col = display[:, i]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            normed[:, i] = (col - cmin) / (cmax - cmin)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_objs)

    for row in normed:
        ax.plot(x, row, alpha=0.12, color="steelblue", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(obj_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Normalized Value (higher = better)")
    ax.set_title(
        f"Reference Set ({formulation}, {obj_data.shape[0]} solutions)"
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
