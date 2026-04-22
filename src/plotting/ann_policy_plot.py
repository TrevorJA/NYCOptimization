"""
src/plotting/ann_policy_plot.py - Operational visualization for `ANNPolicy`.

PDP grid plus an ICE (Individual Conditional Expectation) overlay: for each
input, 20 faint curves are drawn with all *other* inputs sampled uniformly
from [0, 1], and one bold curve is the mean. The spread between ICE curves
reveals interactions that the single PDP hides.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.figure import Figure

from src.plotting.style import ARCH_COLORS, ARCH_LABELS
from src.plotting.policy_base import (
    apply_feature_range_xticks,
    compute_ice_curves,
    make_grid_axes,
    resolve_feature_names,
    resolve_output_names,
)


def plot_ann_policy(
    policy,
    feature_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    feature_ranges: Optional[dict] = None,
    baseline: float = 0.5,  # unused (ICE samples its own baselines); kept for API symmetry
    title: Optional[str] = None,
    n_ice: int = 20,
    seed: int = 0,
) -> Figure:
    """Plot a parameterized `ANNPolicy` as a PDP grid with ICE overlay."""
    del baseline  # ICE uses its own random baselines
    names = resolve_feature_names(policy, feature_names)
    out_names = resolve_output_names(policy, output_names)
    color = ARCH_COLORS.get("ann", "mediumpurple")

    rng = np.random.default_rng(seed)
    fig, axes = make_grid_axes(policy.n_inputs)

    for i, ax in enumerate(axes):
        x, Y = compute_ice_curves(policy, i, n_ice=n_ice, rng=rng)
        # Faint individual ICE curves.
        for k in range(Y.shape[0]):
            ax.plot(x, Y[k], color="gray", lw=0.6, alpha=0.35)
        # Bold mean curve (Monte-Carlo PDP).
        y_mean = Y.mean(axis=0)
        ax.plot(x, y_mean, color=color, lw=2.0, label="mean")

        ice_range = float(Y.max() - Y.min())
        mean_range = float(y_mean.max() - y_mean.min())
        ax.text(
            0.98, 0.03,
            f"Δmean={mean_range:.0f}\nΔICE={ice_range:.0f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="dimgray",
        )

        ax.set_title(names[i])
        ax.set_xlim(0.0, 1.0)
        apply_feature_range_xticks(ax, names[i], feature_ranges)

    axes[0].set_ylabel(out_names[0])

    suptitle = title or ARCH_LABELS.get("ann", "ANN Policy")
    fig.suptitle(
        f"{suptitle}\nhidden={policy._h1}×{policy._h2}  |  ICE n={n_ice}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
