"""
src/plotting/tree_policy_plot.py - Operational visualization for
`SoftTreePolicy`.

Left side: PDP grid showing output vs. each input with others at baseline.
Right side: compact node-link diagram of the soft oblique tree. Internal
nodes are labeled with the dominant split feature and the x-location where
the gate probability crosses 0.5; edge labels are the soft-gate probabilities
p_L / p_R rather than hard thresholds. Leaves are colored by their output
value in physical units.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

from src.plotting.style import ARCH_COLORS, ARCH_LABELS
from src.plotting.policy_base import (
    apply_feature_range_xticks,
    compute_pdp,
    resolve_feature_names,
    resolve_output_names,
)


def _sigmoid_scaled_leaf(policy, leaf_idx: int) -> np.ndarray:
    """Return the leaf's output in physical (post-sigmoid, post-scale) units."""
    raw = policy._leaf_values[leaf_idx]
    sig = 1.0 / (1.0 + np.exp(-raw))
    return policy._output_min + sig * (policy._output_max - policy._output_min)


def _draw_tree(ax, policy, feature_names: list[str], out_names: list[str]) -> None:
    """Draw the oblique tree on `ax` using matplotlib primitives (no graphviz)."""
    ax.set_axis_off()
    depth = policy._depth
    n_internal = policy._n_internal
    n_leaves = policy._n_leaves

    # Node positions: depth 0 at top, leaves at bottom; evenly spaced within level.
    positions: dict[int, tuple[float, float]] = {}
    level_heights = np.linspace(1.0, 0.0, depth + 1)
    for d in range(depth + 1):
        n_at_level = 2 ** d
        xs = np.linspace(0.05, 0.95, n_at_level + 2)[1:-1]
        for k in range(n_at_level):
            idx = (2 ** d - 1) + k
            positions[idx] = (xs[k], level_heights[d])

    # Draw edges (parent -> child).
    for i in range(n_internal):
        px, py = positions[i]
        left = 2 * i + 1
        right = 2 * i + 2
        for child, label in ((left, "p_L"), (right, "p_R")):
            cx, cy = positions[child]
            ax.plot([px, cx], [py, cy], color="lightgray", lw=0.8, zorder=1)
            ax.text(
                0.5 * (px + cx), 0.5 * (py + cy),
                label,
                fontsize=6, color="gray", ha="center", va="center",
                zorder=2,
            )

    # Leaf value color ramp.
    leaf_values = np.array([_sigmoid_scaled_leaf(policy, j)[0] for j in range(n_leaves)])
    vmin, vmax = float(leaf_values.min()), float(leaf_values.max())
    if vmax - vmin < 1e-9:
        vmax = vmin + 1.0
    cmap = plt.get_cmap("viridis")

    # Draw internal nodes (summarize the oblique split).
    # Max label width shrinks with depth so nodes at the wide bottom row stay
    # within their allotted horizontal slot.
    max_chars_by_depth = {0: 18, 1: 14, 2: 9}
    for i in range(n_internal):
        x, y = positions[i]
        w = policy._split_weights[i]
        b = policy._split_biases[i]
        dom = int(np.argmax(np.abs(w)))
        name = feature_names[dom]
        w_dom = w[dom]
        # Oblique split: w·x + b < 0 -> left. Project onto the dominant axis
        # holding others at 0.5 to get an approximate univariate threshold.
        other_contrib = float(np.dot(w, np.full_like(w, 0.5))) - w_dom * 0.5
        if abs(w_dom) > 1e-6:
            thr = -(b + other_contrib) / w_dom
            thr_str = f"≷ {thr:.2f}"
        else:
            thr_str = "(deg.)"
        node_depth = int(np.floor(np.log2(i + 1)))
        max_chars = max_chars_by_depth.get(node_depth, 8)
        short = name if len(name) <= max_chars else name[: max_chars - 1] + "…"
        label = f"{short}\n{thr_str}"
        # Box width also shrinks with depth.
        box_half_w = {0: 0.10, 1: 0.08, 2: 0.055}.get(node_depth, 0.05)
        ax.add_patch(FancyBboxPatch(
            (x - box_half_w, y - 0.045), 2 * box_half_w, 0.09,
            boxstyle="round,pad=0.01",
            facecolor="white",
            edgecolor="dimgray",
            lw=1.0,
            zorder=3,
        ))
        fs = {0: 8, 1: 7, 2: 6}.get(node_depth, 6)
        ax.text(x, y, label, fontsize=fs, ha="center", va="center", zorder=4)

    # Draw leaves (rectangles colored by output).
    for j in range(n_leaves):
        node_idx = n_internal + j
        x, y = positions[node_idx]
        val = leaf_values[j]
        frac = (val - vmin) / (vmax - vmin)
        color = cmap(frac)
        ax.add_patch(Rectangle(
            (x - 0.055, y - 0.035), 0.11, 0.07,
            facecolor=color, edgecolor="black", lw=0.8, zorder=3,
        ))
        ax.text(
            x, y,
            f"{val:.0f}",
            fontsize=7, ha="center", va="center",
            color="white" if frac < 0.55 else "black",
            zorder=4,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.1)
    ax.set_title(
        f"Soft oblique tree (depth={depth}; γ={policy._gamma:.1f}; {out_names[0]})",
        fontsize=10,
    )


def plot_tree_policy(
    policy,
    feature_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    feature_ranges: Optional[dict] = None,
    baseline: float = 0.5,
    title: Optional[str] = None,
) -> Figure:
    """Plot a parameterized `SoftTreePolicy`.

    Layout: PDP grid on the left, tree diagram on the right. PDP curves are
    drawn as continuous lines (soft gating — no piecewise-constant steps).
    """
    names = resolve_feature_names(policy, feature_names)
    out_names = resolve_output_names(policy, output_names)
    color = ARCH_COLORS.get("tree", "mediumseagreen")

    n = policy.n_inputs
    ncols_pdp = min(3, max(1, int(np.ceil(np.sqrt(n)))))
    nrows_pdp = int(np.ceil(n / ncols_pdp))

    fig = plt.figure(
        figsize=(ncols_pdp * 2.6 + 5.8, max(nrows_pdp * 2.2, 4.0) + 0.6),
        constrained_layout=True,
    )
    outer = fig.add_gridspec(1, 2, width_ratios=[ncols_pdp * 2.6, 5.8], wspace=0.15)
    pdp_grid = outer[0, 0].subgridspec(nrows_pdp, ncols_pdp, hspace=0.55, wspace=0.35)
    pdp_axes = [fig.add_subplot(pdp_grid[r // ncols_pdp, r % ncols_pdp]) for r in range(n)]
    tree_ax = fig.add_subplot(outer[0, 1])

    for i, ax in enumerate(pdp_axes):
        x, y = compute_pdp(policy, i, baseline=baseline)
        ax.plot(x, y, color=color, lw=2.0)
        ax.set_title(names[i], fontsize=9)
        ax.set_xlim(0.0, 1.0)
        y_range = float(y.max() - y.min())
        ax.text(
            0.98, 0.03, f"Δ={y_range:.0f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="dimgray",
        )
        apply_feature_range_xticks(ax, names[i], feature_ranges)

    pdp_axes[0].set_ylabel(out_names[0])

    _draw_tree(tree_ax, policy, names, out_names)

    suptitle = title or ARCH_LABELS.get("tree", "Soft Oblique Tree Policy")
    fig.suptitle(
        f"{suptitle}  |  baseline={baseline}  |  γ={policy._gamma:.2f}",
        fontsize=11,
    )
    return fig
