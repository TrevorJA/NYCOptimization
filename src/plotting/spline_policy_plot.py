"""
src/plotting/spline_policy_plot.py - Operational visualization for
`SplineAdditivePolicy`.

Because the spline policy is additive (output is a sigmoid of
sum_i phi_i(x_i) + bias), each input's PDP curve is the univariate
contribution exactly — no baseline approximation. The architecture-specific
detail is the set of knot locations drawn as dotted vertical ticks.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.plotting.style import ARCH_COLORS, ARCH_LABELS
from src.plotting.policy_base import (
    apply_feature_range_xticks,
    compute_pdp,
    make_grid_axes,
    resolve_feature_names,
    resolve_output_names,
)


def plot_spline_policy(
    policy,
    feature_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    feature_ranges: Optional[dict] = None,
    baseline: float = 0.5,
    title: Optional[str] = None,
) -> Figure:
    """Plot a parameterized `SplineAdditivePolicy` as a small-multiples PDP grid.

    Each subplot shows output vs. input i with other inputs at `baseline`.
    Knot locations are drawn as dotted vertical lines.
    """
    names = resolve_feature_names(policy, feature_names)
    out_names = resolve_output_names(policy, output_names)
    color = ARCH_COLORS.get("spline", "goldenrod")

    fig, axes = make_grid_axes(policy.n_inputs)

    knots_unique = np.unique(policy._knots)

    for i, ax in enumerate(axes):
        x, y = compute_pdp(policy, i, baseline=baseline)
        ax.plot(x, y, color=color, lw=2.0)

        for k in knots_unique:
            if 0.0 <= k <= 1.0:
                ax.axvline(k, color="gray", lw=0.5, ls=":", alpha=0.7)

        y_range = float(y.max() - y.min())
        ax.text(
            0.98, 0.03,
            f"Δ={y_range:.0f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8, color="dimgray",
        )

        ax.set_title(names[i])
        ax.set_xlim(0.0, 1.0)
        apply_feature_range_xticks(ax, names[i], feature_ranges)

    for ax in axes[::max(1, len(axes))]:
        ax.set_ylabel(out_names[0])
    axes[0].set_ylabel(out_names[0])
    for ax in axes:
        ax.set_xlabel("normalized input" if not feature_ranges else "")

    bias_str = ", ".join(f"{b:+.2f}" for b in policy._biases)
    suptitle = title or ARCH_LABELS.get("spline", "Spline Policy")
    fig.suptitle(
        f"{suptitle}\nbaseline={baseline}  |  biases=[{bias_str}]",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
