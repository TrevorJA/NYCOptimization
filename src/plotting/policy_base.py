"""
src/plotting/policy_base.py - Shared utilities for per-policy operational plots.

Every `plot_<arch>_policy` function shares a common skeleton — a Partial
Dependence (PDP) grid, one subplot per input feature, showing how the output
responds as that input sweeps [0, 1] with all others held at a baseline.
Architecture-specific details (RBF centers, tree diagram, ICE curves, spline
knots) are layered on top in the per-architecture modules.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def resolve_feature_names(
    policy, feature_names: Optional[Sequence[str]] = None
) -> list[str]:
    """Return a list of length `policy.n_inputs` to use as subplot labels.

    If `feature_names` is given it must match `policy.n_inputs`. Otherwise,
    default to `config.STATE_FEATURES` + ["sin_doy", "cos_doy"] when the lengths
    line up, else fall back to `["x_0", "x_1", ...]`.
    """
    if feature_names is not None:
        names = list(feature_names)
        if len(names) != policy.n_inputs:
            raise ValueError(
                f"feature_names has length {len(names)}, "
                f"policy has n_inputs={policy.n_inputs}"
            )
        return names

    try:
        from config import STATE_FEATURES
        candidate = list(STATE_FEATURES) + ["sin_doy", "cos_doy"]
        if len(candidate) == policy.n_inputs:
            return candidate
    except Exception:
        pass
    return [f"x_{i}" for i in range(policy.n_inputs)]


def resolve_output_names(
    policy, output_names: Optional[Sequence[str]] = None
) -> list[str]:
    """Return a list of length `policy.n_outputs` to use as output labels."""
    if output_names is not None:
        names = list(output_names)
        if len(names) != policy.n_outputs:
            raise ValueError(
                f"output_names has length {len(names)}, "
                f"policy has n_outputs={policy.n_outputs}"
            )
        return names
    if policy.n_outputs == 1:
        return ["release (MGD)"]
    return [f"out_{i}" for i in range(policy.n_outputs)]


def make_grid_axes(
    n_inputs: int,
    per_cell: tuple[float, float] = (3.0, 2.3),
    max_cols: int = 4,
) -> tuple[Figure, np.ndarray]:
    """Create a near-square subplot grid sized for `n_inputs` panels.

    Unused trailing cells are hidden. Returns (figure, flat axes array of
    length n_inputs).
    """
    ncols = min(max_cols, max(1, int(np.ceil(np.sqrt(n_inputs)))))
    nrows = int(np.ceil(n_inputs / ncols))
    figsize = (ncols * per_cell[0], nrows * per_cell[1])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[n_inputs:]:
        ax.set_visible(False)
    return fig, axes_flat[:n_inputs]


def compute_pdp(
    policy,
    feature_idx: int,
    baseline: Union[float, np.ndarray] = 0.5,
    n_samples: int = 100,
    output_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial Dependence Plot data.

    Sweep input `feature_idx` over [0, 1] with all other inputs held at
    `baseline`. Returns (x_grid, y_grid) of length `n_samples`.
    """
    x_grid = np.linspace(0.0, 1.0, n_samples)
    if np.isscalar(baseline):
        base_vec = np.full(policy.n_inputs, float(baseline))
    else:
        base_vec = np.asarray(baseline, dtype=np.float64)
        if base_vec.shape != (policy.n_inputs,):
            raise ValueError(
                f"baseline must be scalar or shape ({policy.n_inputs},), "
                f"got {base_vec.shape}"
            )
    y = np.empty(n_samples)
    for j, v in enumerate(x_grid):
        s = base_vec.copy()
        s[feature_idx] = v
        y[j] = float(policy(s)[output_idx])
    return x_grid, y


def compute_ice_curves(
    policy,
    feature_idx: int,
    n_samples: int = 100,
    n_ice: int = 20,
    output_idx: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Individual Conditional Expectation curves.

    For each of `n_ice` random baselines drawn from Uniform[0, 1]^n_inputs,
    sweep `feature_idx` over [0, 1]. Returns (x_grid, Y) where Y has shape
    (n_ice, n_samples). The per-sample mean Y.mean(axis=0) is a Monte-Carlo
    approximation of the true marginal PDP.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    x_grid = np.linspace(0.0, 1.0, n_samples)
    Y = np.empty((n_ice, n_samples))
    baselines = rng.uniform(0.0, 1.0, size=(n_ice, policy.n_inputs))
    for k in range(n_ice):
        s = baselines[k].copy()
        for j, v in enumerate(x_grid):
            s[feature_idx] = v
            Y[k, j] = float(policy(s)[output_idx])
    return x_grid, Y


def plot_policy_from_theta(arch_name: str, theta: np.ndarray, **kwargs) -> Figure:
    """Instantiate the named architecture, load `theta`, and dispatch to its plot.

    Useful for plotting rows straight out of a Borg refset file:

        row = np.loadtxt(set_path, comments="#")[0]
        fig = plot_policy_from_theta("spline", row[:49])
    """
    from src.formulations.external import get_architecture
    policy = get_architecture(arch_name)
    policy.set_params(np.asarray(theta, dtype=np.float64))

    if arch_name == "spline":
        from src.plotting.spline_policy_plot import plot_spline_policy
        return plot_spline_policy(policy, **kwargs)
    if arch_name == "tree":
        from src.plotting.tree_policy_plot import plot_tree_policy
        return plot_tree_policy(policy, **kwargs)
    if arch_name == "rbf":
        from src.plotting.rbf_policy_plot import plot_rbf_policy
        return plot_rbf_policy(policy, **kwargs)
    if arch_name == "ann":
        from src.plotting.ann_policy_plot import plot_ann_policy
        return plot_ann_policy(policy, **kwargs)

    raise ValueError(
        f"No plot function registered for architecture '{arch_name}'. "
        f"Supported: spline, tree, rbf, ann."
    )


def apply_feature_range_xticks(
    ax, feature_name: str, feature_ranges: Optional[dict] = None
) -> None:
    """Relabel xticks in physical units if a range is provided for this feature.

    `feature_ranges` maps feature_name -> (lo, hi). If present, xtick positions
    in [0, 1] are relabeled as lo + x*(hi - lo). No-op otherwise.
    """
    if not feature_ranges or feature_name not in feature_ranges:
        return
    lo, hi = feature_ranges[feature_name]
    ticks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{lo + t * (hi - lo):.2g}" for t in ticks])
