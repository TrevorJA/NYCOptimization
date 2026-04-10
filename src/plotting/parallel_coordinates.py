"""
parallel_coordinates.py - Parallel coordinate plot of Pareto approximate set.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.formulations import get_obj_names, get_obj_directions, get_n_vars
from src.load.reference_set import load_reference_set


def plot_parallel_coordinates(
    set_file: Path,
    formulation: str,
    output_file: Path = None,
    baseline_objs: np.ndarray = None,
    figsize: tuple = (12, 5),
):
    """Plot parallel coordinates of objectives from a .set or .ref file.

    All axes are oriented so that "up" is the preferred direction.
    Min/max raw values are annotated at the bottom/top of each axis.

    Args:
        set_file: Path to .set or .ref file (vars + objs, whitespace-delimited).
        formulation: Formulation name (for n_vars and title).
        output_file: Path to save figure. If None, displays interactively.
        baseline_objs: Optional array of baseline objective values (raw, not
            Borg-negated). If provided, drawn as a bold highlighted line.
        figsize: Figure size.
    """
    n_vars = get_n_vars(formulation)
    _, obj_data = load_reference_set(set_file, n_vars)
    obj_names = get_obj_names()
    directions = get_obj_directions()
    n_objs = len(obj_names)

    if obj_data.shape[0] == 0:
        print("Empty solution set — nothing to plot.")
        return

    # Un-negate maximization objectives (Borg stores all-minimized)
    raw = obj_data.copy()
    for i in range(n_objs):
        if directions[i] == 1:
            raw[:, i] = -raw[:, i]

    # Prepare baseline in raw space
    baseline_raw = None
    if baseline_objs is not None:
        baseline_raw = np.array(baseline_objs, dtype=float)

    # Compute normalization range (include baseline if present)
    all_data = raw.copy()
    if baseline_raw is not None:
        all_data = np.vstack([all_data, baseline_raw.reshape(1, -1)])

    col_min = all_data.min(axis=0)
    col_max = all_data.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0

    # Normalize to [0, 1], then flip minimization objectives so "up" = preferred
    normed = (raw - col_min) / col_range
    if baseline_raw is not None:
        baseline_normed = (baseline_raw - col_min) / col_range

    for i in range(n_objs):
        if directions[i] == -1:  # minimize: flip so low raw value -> top
            normed[:, i] = 1.0 - normed[:, i]
            if baseline_raw is not None:
                baseline_normed[i] = 1.0 - baseline_normed[i]

    # Determine axis labels: top = best, bottom = worst
    top_labels = []  # best raw value (displayed at top of axis)
    bot_labels = []  # worst raw value (displayed at bottom of axis)
    for i in range(n_objs):
        if directions[i] == 1:  # maximize: best = max, worst = min
            top_labels.append(col_max[i])
            bot_labels.append(col_min[i])
        else:  # minimize: best = min, worst = max
            top_labels.append(col_min[i])
            bot_labels.append(col_max[i])

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_objs)

    # Draw Pareto solutions
    for row in normed:
        ax.plot(x, row, alpha=0.15, color="steelblue", linewidth=0.8)

    # Draw baseline
    if baseline_raw is not None:
        ax.plot(
            x, baseline_normed, color="firebrick", linewidth=2.5,
            marker="o", markersize=5, label="FFMP Baseline", zorder=10,
        )
        ax.legend(loc="lower right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(obj_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Preference Direction  (↑ better)")
    ax.set_title(
        f"Pareto Approximate Set ({formulation}, {obj_data.shape[0]} solutions)"
    )
    ax.set_ylim(-0.12, 1.12)
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate min/max raw values at bottom/top of each axis
    for i in range(n_objs):
        top_val = top_labels[i]
        bot_val = bot_labels[i]
        fmt = _format_value(top_val)
        ax.text(i, 1.04, fmt, ha="center", va="bottom", fontsize=7, color="0.3")
        fmt = _format_value(bot_val)
        ax.text(i, -0.04, fmt, ha="center", va="top", fontsize=7, color="0.3")

    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close(fig)
    else:
        plt.show()


def _format_value(v):
    """Format a raw objective value for axis annotation."""
    if abs(v) >= 100:
        return f"{v:.0f}"
    elif abs(v) >= 1:
        return f"{v:.2f}"
    else:
        return f"{v:.4f}"
