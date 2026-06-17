"""
src/plotting/style.py - Shared matplotlib style and label dictionaries.

All figure scripts import from here to ensure consistent aesthetics across
manuscript figures and diagnostic plots.
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
