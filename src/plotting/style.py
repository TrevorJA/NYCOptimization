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
# plt.get_cmap: matplotlib.cm.get_cmap was removed in matplotlib 3.9.
_vr_cmap = plt.get_cmap("viridis")
for _i, _n in enumerate(FFMP_VR_N_SWEEP):
    # Sample away from the extremes so the colors print well.
    _t = 0.15 + 0.70 * (_i / max(1, len(FFMP_VR_N_SWEEP) - 1))
    ARCH_COLORS[f"ffmp_{_n}"] = _vr_cmap(_t)

# ---------------------------------------------------------------------------
# Objective labels
# ---------------------------------------------------------------------------

# Objective display labels name the METRIC, its statistic, AND its timescale /
# aggregation, in the language of the water-resources DMDU literature
# (objective_definitions.md §0-§2, §6): Hashimoto (1982) reliability; CVaR90
# deficit (Rockafellar & Uryasev 2000; Quinn et al. 2017); count of flood days
# over the NWS minor-flood stage (Quinn et al. 2017); low-percentile storage as a
# vulnerability proxy (Quinn et al. 2017). Each objective is keyed by BOTH names:
# the whole-trace (§1) metric carries its native timescale (weekly reliability /
# CVaR90, whole-record flood count, daily-storage percentile); the annual-unit
# (§2) search metric is marked "annual" (per-water-year metric). Labels therefore
# differ between the two reductions exactly where the timescale/statistic differs.

#: Short per-objective labels for scatter-plot axes (ordered to match the
#: default whole-trace objective set).
OBJ_SHORT: list[str] = [
    "NYC Rel.\n(weekly)",
    "NYC Deficit\n(wk CVaR90)",
    "Montague Rel.\n(weekly)",
    "Montague Def.\n(wk CVaR90)",
    "Trenton Rel.\n(weekly)",
    "Flood Exceedance\n(ft·d, minor)",
    "Storage\n(daily P5)",
    "NJ Rel.\n(weekly)",
]

#: Compact single-line objective labels; ``label_for`` falls back to the raw name.
OBJECTIVE_LABELS: dict[str, str] = {
    # NYC delivery: satisficing reliability; CVaR90 of the deficit (% of Decree)
    "nyc_delivery_reliability_weekly":  "NYC Delivery Reliability (weekly)",
    "nyc_delivery_reliability_annual":  "NYC Delivery Reliability (annual)",
    "nyc_delivery_deficit_cvar90_pct":  "NYC Delivery Deficit (weekly CVaR90, %)",
    "nyc_delivery_deficit_p99_pct":     "NYC Delivery Deficit (P99 of annual CVaR90, %)",
    "nyc_delivery_deficit_max_pct":     "NYC Delivery Deficit (weekly max, %)",
    # NJ delivery
    "nj_delivery_reliability_weekly":   "NJ Delivery Reliability (weekly)",
    "nj_delivery_reliability_annual":   "NJ Delivery Reliability (annual)",
    # Montague Decree flow
    "montague_flow_reliability_weekly": "Montague Flow Reliability (weekly)",
    "montague_flow_reliability_annual": "Montague Flow Reliability (annual)",
    "montague_flow_deficit_cvar90_pct": "Montague Flow Deficit (weekly CVaR90, %)",
    "montague_flow_deficit_p99_pct":    "Montague Flow Deficit (P99 of annual CVaR90, %)",
    "montague_flow_deficit_max_pct":    "Montague Flow Deficit (weekly max, %)",
    # Trenton Decree flow
    "trenton_flow_reliability_weekly":  "Trenton Flow Reliability (weekly)",
    "trenton_flow_reliability_annual":  "Trenton Flow Reliability (annual)",
    "trenton_flow_deficit_cvar90_pct":  "Trenton Flow Deficit (weekly CVaR90, %)",
    # Downstream flooding: ft·days above the NWS minor flood stage at the
    # worst-affected gauge (exceedance, active); day counts are diagnostics.
    "downstream_flood_exceedance_minor":  "Flood Exceedance (NWS minor, ft·d/yr)",
    "downstream_flood_exceedance_annual": "Flood Exceedance (NWS minor, annual mean)",
    "downstream_flood_days_minor":      "Flood Days (NWS minor, days/yr)",
    "downstream_flood_days_annual":     "Flood Days (NWS minor, annual mean)",
    "downstream_flood_days_annual_p99": "Flood Days (NWS minor, annual P99)",
    "downstream_flood_days_action":     "Flood Days (NWS action, days/yr)",
    "downstream_flood_days_major":      "Flood Days (NWS major, days/yr)",
    # NYC storage: low-percentile storage (vulnerability proxy)
    "nyc_storage_p5_pct":               "NYC Storage (daily 5th pctile, %)",
    "nyc_storage_min_p01_pct":          "NYC Storage (annual-min 1st pctile, %)",
    "nyc_storage_min_pct":              "NYC Storage (whole-record min, %)",
    # Other registered diagnostics
    "salt_front_intrusion_max_rm":      "Salt Front (max, river mi)",
    "lordville_temp_exceedance_days":   "Lordville Temp Exceedance (days)",
}


def label_for(name: str) -> str:
    """Compact display label for an objective (or any) name; falls back to it."""
    return OBJECTIVE_LABELS.get(name, name)


#: Very short per-objective labels for dense layouts (scatter-matrix edges).
OBJ_SHORT_LABELS: dict[str, str] = {
    "nyc_delivery_reliability_weekly":  "NYC Rel. (wk)",
    "nyc_delivery_reliability_annual":  "NYC Rel. (ann)",
    "nyc_delivery_deficit_cvar90_pct":  "NYC Def. CVaR90 %",
    "nyc_delivery_deficit_p99_pct":     "NYC Def. P99 %",
    "montague_flow_reliability_weekly": "Montague Rel. (wk)",
    "montague_flow_reliability_annual": "Montague Rel. (ann)",
    "montague_flow_deficit_cvar90_pct": "Montague Def. CVaR90 %",
    "montague_flow_deficit_p99_pct":    "Montague Def. P99 %",
    "trenton_flow_reliability_weekly":  "Trenton Rel. (wk)",
    "trenton_flow_reliability_annual":  "Trenton Rel. (ann)",
    "downstream_flood_exceedance_minor":  "Flood Exc. (ft·d/yr)",
    "downstream_flood_exceedance_annual": "Flood Exc. (ann)",
    "downstream_flood_days_minor":      "Flood Days",
    "downstream_flood_days_annual":     "Flood Days (ann)",
    "nyc_storage_p5_pct":               "Storage P5 %",
    "nyc_storage_min_p01_pct":          "Storage P1 %",
    "nj_delivery_reliability_weekly":   "NJ Rel. (wk)",
    "nj_delivery_reliability_annual":   "NJ Rel. (ann)",
}


def short_label_for(name: str) -> str:
    """Very short display label for an objective name; falls back to label_for."""
    return OBJ_SHORT_LABELS.get(name, label_for(name))


#: Parallel-coordinates axis labels (multi-line: metric, timescale + statistic +
#: unit, optimization direction).
OBJ_AXIS_LABELS: dict[str, str] = {
    "nyc_delivery_reliability_weekly":   "NYC Delivery\nReliability (weekly)\n(max)",
    "nyc_delivery_reliability_annual":   "NYC Delivery\nReliability (annual)\n(max)",
    "nyc_delivery_deficit_cvar90_pct":   "NYC Delivery Deficit\nweekly CVaR90 %\n(min)",
    "nyc_delivery_deficit_p99_pct":      "NYC Delivery Deficit\nP99 of ann. CVaR90 %\n(min)",
    "montague_flow_reliability_weekly":  "Montague Flow\nReliability (weekly)\n(max)",
    "montague_flow_reliability_annual":  "Montague Flow\nReliability (annual)\n(max)",
    "montague_flow_deficit_cvar90_pct":  "Montague Flow Deficit\nweekly CVaR90 %\n(min)",
    "montague_flow_deficit_p99_pct":     "Montague Flow Deficit\nP99 of ann. CVaR90 %\n(min)",
    "trenton_flow_reliability_weekly":   "Trenton Flow\nReliability (weekly)\n(max)",
    "trenton_flow_reliability_annual":   "Trenton Flow\nReliability (annual)\n(max)",
    "downstream_flood_exceedance_minor":   "Flood Exceedance\nNWS minor, ft·d/yr\n(min)",
    "downstream_flood_exceedance_annual":  "Flood Exceedance\nNWS minor, ann. mean\n(min)",
    "downstream_flood_days_minor":       "Flood Days\nNWS minor, days/yr\n(min)",
    "downstream_flood_days_annual":      "Flood Days\nNWS minor, annual mean\n(min)",
    "nyc_storage_p5_pct":                "NYC Storage\ndaily 5th pctile %\n(max)",
    "nyc_storage_min_p01_pct":           "NYC Storage\nP1 of ann. min %\n(max)",
    "nj_delivery_reliability_weekly":    "NJ Delivery\nReliability (weekly)\n(max)",
    "nj_delivery_reliability_annual":    "NJ Delivery\nReliability (annual)\n(max)",
}

# ---------------------------------------------------------------------------
# Scatter pair definitions
# ---------------------------------------------------------------------------

#: Six pairwise scatter pairs (0-based indices into the 8-objective active
#: set: 0 NYC Rel, 1 NYC Def, 2 Montague Rel, 3 Montague Def, 4 Trenton Rel,
#: 5 Flood Days, 6 Storage, 7 NJ Rel) for SI diagnostic plots.
SCATTER_PAIRS: list[tuple[int, int]] = [
    (0, 2),   # NYC Rel. vs Montague Rel.
    (1, 3),   # NYC Deficit vs Montague Deficit
    (4, 5),   # Trenton Rel. vs Flood Days
    (0, 6),   # NYC Rel. vs Storage
    (0, 7),   # NYC Rel. vs NJ Rel. (the two Decree delivery parties)
    (1, 5),   # NYC Deficit vs Flood Days
]

# ---------------------------------------------------------------------------
# Figure size presets (width, height) in inches
# ---------------------------------------------------------------------------

FIGSIZE_SINGLE = (7, 5)
FIGSIZE_WIDE   = (13, 5)
FIGSIZE_GRID_2X3 = (12.6, 7.0)

#: Main-manuscript 2x2 panel grid. Square-ish so each panel can be forced square
#: via ``ax.set_box_aspect(1)`` without the layout squeezing the tick labels.
FIGSIZE_MANUSCRIPT_2X2 = (11.0, 11.0)

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


#: Smallest type size permitted in a main-manuscript figure (points). Journal
#: figures are reduced on the page, so this is a floor on the RENDERED size, and
#: every entry in :func:`apply_manuscript_style` sits at or above it.
MANUSCRIPT_MIN_FONTSIZE: int = 12


def apply_manuscript_style() -> None:
    """Apply the main-manuscript figure style (>= 12 pt, no bold weights).

    Separate from :func:`apply_style`, which the ~20 SI/diagnostic scripts rely
    on at its smaller 10 pt sizing. Call once at the top of a main-figure
    script, before any plotting call.
    """
    fs = MANUSCRIPT_MIN_FONTSIZE
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          fs,
        "font.weight":        "normal",
        "axes.titlesize":     fs + 1,
        "axes.titleweight":   "normal",
        "axes.labelsize":     fs,
        "axes.labelweight":   "normal",
        "xtick.labelsize":    fs,
        "ytick.labelsize":    fs,
        "legend.fontsize":    fs,
        "legend.title_fontsize": fs,
        "figure.titlesize":   fs + 2,
        "figure.titleweight": "normal",
        "figure.dpi":         150,
        "savefig.dpi":        400,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        # Vector text stays editable text (not outlines) in the PDF, which
        # journal production systems require.
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
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


#: Output formats for main-manuscript figures: a raster copy to look at and a
#: vector copy to submit.
MANUSCRIPT_FIGURE_FORMATS: tuple = ("png", "pdf")


def save_manuscript_figure(fig, out_stub) -> list:
    """Save ``fig`` as both PNG and PDF; return the paths written.

    Args:
        fig: Matplotlib figure.
        out_stub: Path or str without an extension (any existing suffix is replaced).

    Returns:
        The written paths, in :data:`MANUSCRIPT_FIGURE_FORMATS` order.
    """
    stub = Path(out_stub)
    stub.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in MANUSCRIPT_FIGURE_FORMATS:
        path = stub.with_suffix(f".{ext}")
        fig.savefig(path)
        written.append(path)
    return written


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
