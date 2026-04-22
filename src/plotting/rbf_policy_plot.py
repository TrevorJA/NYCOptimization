"""
src/plotting/rbf_policy_plot.py - Operational visualization for `RBFPolicy`.

PDP grid showing output vs. each input with others at baseline. On each
subplot, RBF centers are overlaid as thick vertical segments along the
x-axis: one per basis function, positioned at the center's coordinate for
that input, colored by the *sign* of its weight toward the plotted output
and with alpha scaled by the weight magnitude. The segment length encodes
the RBF width — wider RBFs extend further up the subplot.
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


def plot_rbf_policy(
    policy,
    feature_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    feature_ranges: Optional[dict] = None,
    baseline: float = 0.5,
    title: Optional[str] = None,
    output_idx: int = 0,
) -> Figure:
    """Plot a parameterized `RBFPolicy` as a PDP grid with RBF-center overlays."""
    names = resolve_feature_names(policy, feature_names)
    out_names = resolve_output_names(policy, output_names)
    color = ARCH_COLORS.get("rbf", "darkorange")

    fig, axes = make_grid_axes(policy.n_inputs)

    weights_to_out = policy._weights[:, output_idx]
    max_abs_w = float(np.max(np.abs(weights_to_out))) or 1.0

    for i, ax in enumerate(axes):
        x, y = compute_pdp(policy, i, baseline=baseline)
        ax.plot(x, y, color=color, lw=2.0)

        y_lo, y_hi = ax.get_ylim()
        y_span = y_hi - y_lo if y_hi > y_lo else 1.0
        # Draw each RBF as a short vertical segment along the x-axis of this
        # subplot. Position = center on axis i; length ∝ width; alpha ∝ |weight|;
        # color red (negative) / blue (positive) weight contribution.
        for j in range(policy._n_rbf):
            cx = float(policy._centers[j, i])
            if cx < 0.0 or cx > 1.0:
                continue
            w_j = float(weights_to_out[j])
            alpha = min(1.0, 0.15 + 0.85 * (abs(w_j) / max_abs_w))
            width = float(policy._widths[j])
            seg_h = 0.08 * y_span * min(2.0, width)
            seg_color = "#1f4ea1" if w_j >= 0 else "#b22222"
            ax.plot(
                [cx, cx],
                [y_lo, y_lo + seg_h],
                color=seg_color, alpha=alpha, lw=3.0, solid_capstyle="round",
            )

        y_range = float(y.max() - y.min())
        ax.text(
            0.98, 0.97, f"Δ={y_range:.0f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="dimgray",
        )

        ax.set_title(names[i])
        ax.set_xlim(0.0, 1.0)
        apply_feature_range_xticks(ax, names[i], feature_ranges)

    axes[0].set_ylabel(out_names[output_idx])

    # Legend proxy: RBF center markers.
    handles = [
        plt.Line2D([0], [0], color="#1f4ea1", lw=3, label="RBF (w > 0)"),
        plt.Line2D([0], [0], color="#b22222", lw=3, label="RBF (w < 0)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2,
        frameon=False, bbox_to_anchor=(0.5, -0.02),
    )

    suptitle = title or ARCH_LABELS.get("rbf", "RBF Policy")
    fig.suptitle(
        f"{suptitle}\nn_rbf={policy._n_rbf}  |  baseline={baseline}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return fig
