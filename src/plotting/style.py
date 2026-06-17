"""
src/plotting/style.py - Shared matplotlib style and label dictionaries.

All figure scripts import from here to ensure consistent aesthetics across
manuscript figures and diagnostic plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import Rectangle

from config import FFMP_VR_N_SWEEP

# ---------------------------------------------------------------------------
# Architecture metadata
# ---------------------------------------------------------------------------

#: Human-readable name for each formulation.
ARCH_LABELS: dict[str, str] = {
    "ffmp":   "Parameterized FFMP",
}
for _n in FFMP_VR_N_SWEEP:
    ARCH_LABELS[f"ffmp_{_n}"] = f"FFMP (N={_n} zones)"

#: Distinct color per formulation for overlaid Pareto front comparisons.
ARCH_COLORS: dict[str, str] = {
    "ffmp":   "steelblue",
}
# N-zone variants get a sequential viridis-family ramp so higher N reads
# "deeper" complexity at a glance.
_vr_cmap = cm.get_cmap("viridis")
for _i, _n in enumerate(FFMP_VR_N_SWEEP):
    # Sample away from the extremes so the colors print well.
    _t = 0.15 + 0.70 * (_i / max(1, len(FFMP_VR_N_SWEEP) - 1))
    ARCH_COLORS[f"ffmp_{_n}"] = _vr_cmap(_t)

# ---------------------------------------------------------------------------
# Objective labels
# ---------------------------------------------------------------------------

#: Short per-objective labels for scatter-plot axes (ordered to match _DEFAULT_OBJECTIVES).
OBJ_SHORT: list[str] = [
    "NYC Rel.",
    "NYC Deficit %\n(CVaR90)",
    "Montague\nRel.",
    "Montague\nDeficit %\n(CVaR90)",
    "Trenton\nRel.",
    "Flood Days\n(minor)",
    "Stor. p5 %",
]

#: Compact single-line labels for **every** registry objective (recommended set
#: plus the diagnostics they replace). Shared by the objective-sensitivity
#: diagnostic figure scripts; ``label_for`` falls back to the raw name.
OBJECTIVE_LABELS: dict[str, str] = {
    "nyc_delivery_reliability_weekly": "NYC delivery rel.",
    "nyc_delivery_deficit_cvar90_pct": "NYC deficit CVaR90",
    "nyc_delivery_deficit_max_pct": "NYC deficit max",
    "nj_delivery_reliability_weekly": "NJ delivery rel.",
    "montague_flow_reliability_weekly": "Montague rel.",
    "montague_flow_deficit_cvar90_pct": "Montague deficit CVaR90",
    "montague_flow_deficit_max_pct": "Montague deficit max",
    "trenton_flow_reliability_weekly": "Trenton rel.",
    "trenton_flow_deficit_cvar90_pct": "Trenton deficit CVaR90",
    "downstream_flood_days_minor": "Flood days (minor)",
    "downstream_flood_days_action": "Flood days (action)",
    "downstream_flood_days_major": "Flood days (major)",
    "nyc_storage_p5_pct": "Storage p5",
    "nyc_storage_min_pct": "Storage min",
    "salt_front_intrusion_max_rm": "Salt front max RM",
    "lordville_temp_exceedance_days": "Lordville temp days",
}


def label_for(name: str) -> str:
    """Compact display label for an objective (or any) name; falls back to it."""
    return OBJECTIVE_LABELS.get(name, name)


#: Parallel-coordinates axis labels (multi-line, direction hint on third line).
OBJ_AXIS_LABELS: dict[str, str] = {
    "nyc_delivery_reliability_weekly":   "NYC\nReliability\n(max)",
    "nyc_delivery_deficit_cvar90_pct":   "NYC Deficit %\nCVaR90\n(min)",
    "montague_flow_reliability_weekly":  "Montague\nReliability\n(max)",
    "montague_flow_deficit_cvar90_pct":  "Montague Deficit %\nCVaR90\n(min)",
    "trenton_flow_reliability_weekly":   "Trenton\nReliability\n(max)",
    "downstream_flood_days_minor":       "Flood\nDays\n(min)",
    "nyc_storage_p5_pct":                "Storage\np5 %\n(max)",
}

# ---------------------------------------------------------------------------
# Scatter pair definitions
# ---------------------------------------------------------------------------

#: Six pairwise scatter pairs (0-based objective indices) for SI diagnostic plots.
SCATTER_PAIRS: list[tuple[int, int]] = [
    (0, 2),   # NYC Rel. vs Montague Rel.
    (1, 3),   # NYC Deficit vs Montague Deficit
    (4, 5),   # Salt Front vs Flood Days
    (0, 6),   # NYC Rel. vs Min Storage
    (2, 4),   # Montague Rel. vs Salt Front
    (1, 5),   # NYC Deficit vs Flood Days
]

# ---------------------------------------------------------------------------
# Figure size presets (width, height) in inches
# ---------------------------------------------------------------------------

FIGSIZE_SINGLE = (7, 5)
FIGSIZE_WIDE   = (13, 5)
FIGSIZE_GRID_2X3 = (12.6, 7.0)

# ---------------------------------------------------------------------------
# Shared rcParams
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply project-wide matplotlib rcParams.

    Call once at the top of each figure script's ``if __name__ == "__main__"``
    block before any plotting calls.
    """
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Shared figure helpers
# ---------------------------------------------------------------------------

#: Output formats for diagnostic figures. PNG only for now (drop vector copies).
FIGURE_FORMATS: tuple = ("png",)


def save_figure(fig, out_stub) -> None:
    """Save ``fig`` to ``{out_stub}.{ext}`` for each format in ``FIGURE_FORMATS``.

    Args:
        fig: Matplotlib figure.
        out_stub: Path or str without an extension (any existing suffix is replaced).
    """
    stub = Path(out_stub)
    for ext in FIGURE_FORMATS:
        fig.savefig(stub.with_suffix(f".{ext}"))


def annotated_corr_heatmap(ax, data, labels, *, label_fn=label_for,
                           box_threshold=None, fontsize: int = 6,
                           vmin: float = -1.0, vmax: float = 1.0):
    """Draw an annotated correlation/agreement heatmap on ``ax``.

    Shared by the redundancy (Spearman) and operator-agreement (Kendall tau_b)
    diagnostics. NaN cells render grey; cells with ``|value| > box_threshold``
    (off-diagonal) are outlined.

    Args:
        ax: Target axes.
        data: Square 2-D array of correlation/agreement values.
        labels: Row/column names (length matches ``data``).
        label_fn: Maps a name to its tick label (default :func:`label_for`).
        box_threshold: If set, outline off-diagonal cells exceeding it.
        fontsize: Tick-label font size (cell annotations use ``fontsize - 1``).
        vmin: Colour-scale minimum.
        vmax: Colour-scale maximum.

    Returns:
        The ``AxesImage`` (for an external colorbar).
    """
    arr = np.asarray(data, dtype=float)
    m = len(labels)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels([label_fn(n) for n in labels], rotation=45, ha="right",
                       fontsize=fontsize)
    ax.set_yticklabels([label_fn(n) for n in labels], fontsize=fontsize)
    for i in range(m):
        for j in range(m):
            v = arr[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=max(5, fontsize - 1),
                    color="white" if abs(v) > 0.55 else "black")
            if box_threshold is not None and i != j and abs(v) > box_threshold:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=1.6))
    return im
