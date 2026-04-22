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

#: Human-readable name for each policy architecture.
ARCH_LABELS: dict[str, str] = {
    "ffmp":   "Parameterized FFMP",
    "rbf":    "RBF Policy (6 centers, 15 inputs)",
    "tree":   "Soft Oblique Tree (depth 3, optimized γ)",
    "ann":    "ANN (2×8 hidden, 15 inputs)",
    "spline": "Spline Additive Policy (G=5, k=3)",
}
for _n in FFMP_VR_N_SWEEP:
    ARCH_LABELS[f"ffmp_{_n}"] = f"FFMP (N={_n} zones)"

#: Distinct color per architecture for overlaid Pareto front comparisons.
ARCH_COLORS: dict[str, str] = {
    "ffmp":   "steelblue",
    "rbf":    "darkorange",
    "tree":   "mediumseagreen",
    "ann":    "mediumpurple",
    "spline": "goldenrod",
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

#: Short per-objective labels for scatter-plot axes (ordered to match DEFAULT_OBJECTIVES).
OBJ_SHORT: list[str] = [
    "NYC Rel.",
    "NYC Vuln. %",
    "NJ Rel.",
    "Montague\nRel.",
    "Trenton\nRel.",
    "Flood Days",
    "Min Stor. %",
]

#: Parallel-coordinates axis labels (multi-line, direction hint on third line).
OBJ_AXIS_LABELS: dict[str, str] = {
    "nyc_reliability_weekly":          "NYC\nReliability\n(max)",
    "nyc_vulnerability":               "NYC\nVuln. %\n(min)",
    "nj_reliability_weekly":           "NJ\nReliability\n(max)",
    "montague_reliability_weekly":     "Montague\nReliability\n(max)",
    "trenton_reliability_weekly":      "Trenton\nReliability\n(max)",
    "flood_risk_downstream_flow_days": "Flood\nDays\n(min)",
    "storage_min_combined_pct":        "Min\nStorage\n(max)",
}

# ---------------------------------------------------------------------------
# Scatter pair definitions
# ---------------------------------------------------------------------------

#: Six pairwise scatter pairs (0-based objective indices) for SI diagnostic plots.
SCATTER_PAIRS: list[tuple[int, int]] = [
    (0, 3),   # NYC Rel. vs Montague
    (0, 5),   # NYC Rel. vs Flood Days
    (3, 5),   # Montague vs Flood Days
    (0, 6),   # NYC Rel. vs Min Storage
    (3, 6),   # Montague vs Min Storage
    (2, 5),   # NJ Rel. vs Flood Days
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
