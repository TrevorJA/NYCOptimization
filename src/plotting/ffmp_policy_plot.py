"""
src/plotting/ffmp_policy_plot.py - Operational visualization for FFMP policy formulations.

Supports standard FFMP (6 zone levels, level1b–level5) and variable-resolution
FFMP_N (N zone boundaries, zone_1–zone_N).

Designed for Pareto-set comparison: pass a single DV vector or an array of
vectors and control alpha/color to distinguish solutions.

Public API
----------
    plot_ffmp_policy(dv_vectors, formulation_name, ...)        # composite figure
    plot_ffmp_storage_zones(config, ax, ...)                   # storage zone panel
    plot_ffmp_mrf_flow_targets(config, ax, ...)                # MRF baseline panel
    plot_ffmp_drought_delivery_factors(config, ax, ...)        # delivery factor panel
    plot_ffmp_mrf_seasonal_profiles(config, ax, ...)           # seasonal MRF panel
    plot_ffmp_flood_limits(config, ax, ...)                    # flood limit panel
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

from src.plotting.style import apply_style, ARCH_COLORS, ARCH_LABELS


# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

_ZONE_LABELS_STD = {
    "level1b": "L1b  (flood warning)",
    "level1c": "L1c  (flood watch)",
    "level2":  "L2   (normal)",
    "level3":  "L3   (drought watch)",
    "level4":  "L4   (drought warning)",
    "level5":  "L5   (drought emergency)",
}

# RdYlGn-inspired ramp: green (mild/flood) → dark red (severe/emergency)
_ZONE_COLORS_STD = [
    "#1a9641",  # L1b
    "#78c679",  # L1c
    "#d9ef8b",  # L2
    "#fdae61",  # L3
    "#d73027",  # L4
    "#7b0000",  # L5
]

_RESERVOIR_LABELS = {
    "cannonsville": "Cannonsville",
    "pepacton":     "Pepacton",
    "neversink":    "Neversink",
}

# Season windows (1-indexed DOY) matching _apply_mrf_profile_scaling in simulation.py
_SEASONS = {
    "Winter": (list(range(335, 367)) + list(range(1, 60)), "#c6dbef"),
    "Spring": (list(range(60, 152)),                        "#a1d99b"),
    "Summer": (list(range(152, 244)),                       "#fdcc8a"),
    "Fall":   (list(range(244, 335)),                       "#fc8d59"),
}

_MONTH_STARTS_DOY = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
_MONTH_NAMES      = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_INITIALS   = ["J", "F", "M", "A", "M", "J",
                     "J", "A", "S", "O", "N", "D"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_nzone(config) -> bool:
    return getattr(config, "n_drought_levels", 7) != 7


def _get_zone_rows(config) -> list[str]:
    """Return the ordered storage zone row names for this config."""
    df = config.storage_zones_df
    std = ["level1b", "level1c", "level2", "level3", "level4", "level5"]
    if all(lv in df.index for lv in std):
        return std
    n_zones = config.n_drought_levels - 1
    return [f"zone_{i+1}" for i in range(n_zones)]


def _zone_colors(n: int) -> list:
    if n == 6:
        return _ZONE_COLORS_STD
    return [mcolors.to_hex(plt.cm.RdYlGn_r(t)) for t in np.linspace(0.05, 0.95, n)]


def _zone_array(config, zone: str) -> np.ndarray:
    return config.storage_zones_df.loc[zone].values.astype(float)


def _mrf_array(config, level: str, reservoir: str) -> np.ndarray:
    key = f"{level}_factor_mrf_{reservoir}"
    return config.mrf_factors_daily_df.loc[key].values.astype(float)


def _level_sort_key(name: str):
    """Stable sort key handling both standard FFMP (level1a..level5)
    and VR-FFMP (zone_0..zone_N, with N possibly ≥ 10)."""
    if name.startswith("zone_"):
        return (1, int(name.split("_")[1]))
    order = {"level1a": 0, "level1b": 1, "level1c": 2,
             "level2": 3, "level3": 4, "level4": 5, "level5": 6}
    return (0, order.get(name, 99))


def _delivery_factors(config, system: str) -> dict[str, float]:
    """Return {level_name: factor} for all delivery levels, capping 1e5+ to 1.0.

    Large values (> 1e4) represent "unconstrained" levels; they are mapped to
    1.0 so they appear correctly on the [0, 1] delivery-factor axis.
    """
    out = {}
    for k, v in config.constants.items():
        if f"_factor_delivery_{system}" in k:
            fv = float(v)
            level = k.replace(f"_factor_delivery_{system}", "")
            out[level] = min(fv, 1.0)
    return dict(sorted(out.items(), key=lambda kv: _level_sort_key(kv[0])))


def _flow_target_factors(config, target: str) -> dict[str, float]:
    """Return {level_name: factor} for Montague or Trenton flow target step-downs.

    Reads from config.mrf_factors_monthly_df rows named
    ``<level>_factor_mrf_<target>`` and averages across the 12 months. Most
    months have identical factors, so the mean is a faithful single-value
    summary.

    Parameters
    ----------
    target : str
        Either "delMontague" or "delTrenton".
    """
    df = config.mrf_factors_monthly_df
    suffix = f"_factor_mrf_{target}"
    out = {}
    for row in df.index:
        if row.endswith(suffix):
            level = row[: -len(suffix)]
            out[level] = float(df.loc[row].values.astype(float).mean())
    return dict(sorted(out.items(), key=lambda kv: _level_sort_key(kv[0])))


def _doy_xaxis(ax: plt.Axes, short: bool = False) -> None:
    """Apply month-start xticks to a day-of-year axis (0-indexed).

    short=True uses single-letter month initials (J F M A M J J A S O N D),
    suitable for narrow panels.
    """
    ax.set_xlim(0, 365)
    ax.set_xticks([t - 1 for t in _MONTH_STARTS_DOY])
    ax.set_xticklabels(_MONTH_INITIALS if short else _MONTH_NAMES, fontsize=8)


def _make_config(dv_vector, formulation_name: str):
    from src.simulation import dvs_to_config
    return dvs_to_config(np.asarray(dv_vector, dtype=float), formulation_name)


def _season_shading(ax: plt.Axes) -> None:
    """Light seasonal background tinting on a DOY x-axis (0-indexed)."""
    ylim = ax.get_ylim()
    for doys, color in _SEASONS.values():
        # Convert to 0-indexed ranges; handle winter wrap
        segs = []
        if max(doys) > min(doys) + 10:  # non-wrap or clear segment
            segs = [(min(doys) - 1, max(doys) - 1)]
        else:
            # wrap-around: split at year boundary
            hi = [d for d in doys if d >= 335]
            lo = [d for d in doys if d < 60]
            if hi:
                segs.append((min(hi) - 1, 365))
            if lo:
                segs.append((0, max(lo) - 1))
        for s, e in segs:
            ax.axvspan(s, e, alpha=0.035, color=color, zorder=0)
    ax.set_ylim(ylim)


def _mrf_levels_for(config) -> list[str]:
    """Return the ordered drought-level prefixes used for MRF rows in this config.

    Standard FFMP: ['level1a', 'level1b', 'level1c', 'level2', 'level3', 'level4', 'level5']
    VR-FFMP_N:     ['zone_0', 'zone_1', ..., 'zone_N']
    Detected from the rows of config.mrf_factors_daily_df that match
    ``<level>_factor_mrf_cannonsville``.
    """
    df = config.mrf_factors_daily_df
    rows = [r for r in df.index if r.endswith("_factor_mrf_cannonsville")]
    levels = [r.replace("_factor_mrf_cannonsville", "") for r in rows]
    return sorted(levels, key=_level_sort_key)


# ---------------------------------------------------------------------------
# Individual panel functions
# ---------------------------------------------------------------------------

def plot_ffmp_storage_zones(
    config,
    ax: plt.Axes,
    alpha: float = 1.0,
    lw: float = 2.0,
    zorder: int = 2,
    show_labels: bool = True,
) -> None:
    """Draw seasonal storage zone threshold curves on *ax*.

    Parameters
    ----------
    config : NYCOperationsConfig
    ax : matplotlib Axes
    alpha, lw, zorder : rendering parameters for multi-solution overlays
    show_labels : add legend labels (disable for subsequent overlay draws)
    """
    zone_rows = _get_zone_rows(config)
    colors = _zone_colors(len(zone_rows))
    doy = np.arange(366)
    is_std = not _is_nzone(config)

    for zone, color in zip(zone_rows, colors):
        profile = _zone_array(config, zone) * 100.0
        if show_labels:
            label = _ZONE_LABELS_STD.get(zone, zone.replace("_", " ").title()) if is_std else zone
        else:
            label = "_nolegend_"
        ax.plot(doy, profile, color=color, lw=lw, alpha=alpha,
                label=label, zorder=zorder, solid_capstyle="round")

    ax.set_ylabel("Storage (% of capacity)")
    ax.set_ylim(0, 110)
    ax.set_xlabel("")
    _doy_xaxis(ax)
    _season_shading(ax)
    ax.grid(True, alpha=0.2, lw=0.5)


def plot_ffmp_mrf_flow_targets(
    config,
    ax: plt.Axes,
    color: str = "steelblue",
    alpha: float = 1.0,
    annotate: bool = True,
) -> None:
    """Horizontal dot chart of MRF baseline flows for the 3 NYC reservoirs.

    The Montague / Trenton / Max-NYC-delivery scalars are no longer shown
    here — they appear in the figure-level annotation block.
    """
    c = config.constants
    reservoirs = ["cannonsville", "pepacton", "neversink"]
    values = [float(c[f"mrf_baseline_{r}"]) for r in reservoirs]
    labels = ["Cannonsville", "Pepacton", "Neversink"]
    y = np.arange(len(reservoirs))

    ax.scatter(values, y, color=color, alpha=alpha, s=60, zorder=3)
    if annotate:
        for i, v in enumerate(values):
            ax.text(v + 2, i, f"{v:.0f}", va="center", ha="left",
                    fontsize=8, alpha=alpha, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Min release (MGD)", fontsize=9)
    ax.set_title("MRF baselines", fontsize=10, loc="left")
    ax.set_xlim(0, max(values) * 1.35)
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)


#: Visual style for each "factor curve" plotted in the drought-factor panel.
#: name → (color, marker, linestyle, label)
DROUGHT_FACTOR_STYLES = {
    "nyc":      ("#1f77b4", "o", "-",  "NYC delivery"),
    "nj":       ("#2ca02c", "s", "-",  "NJ delivery"),
    "montague": ("#d62728", "^", "--", "Montague flow target"),
    "trenton":  ("#ff7f0e", "D", "--", "Trenton flow target"),
}


def plot_ffmp_drought_delivery_factors(
    config,
    ax: plt.Axes,
    alpha: float = 1.0,
) -> None:
    """Connected-dot plot of all 4 drought-zone-driven multipliers:

    - NYC delivery factor   (demand reduction; solid blue circles)
    - NJ delivery factor    (demand reduction; solid green squares)
    - Montague flow-target  (downstream MRF target; dashed red triangles)
    - Trenton flow-target   (downstream MRF target; dashed orange diamonds)

    All four use the same fractional [0, 1] y-axis so they are directly comparable.
    """
    series = {
        "nyc":      _delivery_factors(config, "nyc"),
        "nj":       _delivery_factors(config, "nj"),
        "montague": _flow_target_factors(config, "delMontague"),
        "trenton":  _flow_target_factors(config, "delTrenton"),
    }

    # Union of all levels, sorted by drought severity
    all_levels = sorted(
        set().union(*[s.keys() for s in series.values()]),
        key=_level_sort_key,
    )
    x = np.arange(len(all_levels))

    for name, data in series.items():
        color, marker, ls, _label = DROUGHT_FACTOR_STYLES[name]
        vals = [data.get(lv, 1.0) for lv in all_levels]
        ax.plot(x, vals, color=color, marker=marker, linestyle=ls,
                lw=1.8, ms=5.5, alpha=alpha, label="_nolegend_",
                markeredgewidth=0)

    ax.axhline(1.0, color="gray", lw=0.75, ls=":", alpha=0.55)
    short = [lv.replace("level", "L").replace("zone_", "Z") for lv in all_levels]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8, rotation=35, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Factor", fontsize=9)
    ax.set_title("Storage-zone factors\n(delivery & flow adjustments)",
                 fontsize=10, loc="left")
    ax.grid(True, alpha=0.25)


def plot_ffmp_mrf_seasonal_profiles(
    config,
    ax: plt.Axes,
    reservoir: str = "cannonsville",
    color: str = "steelblue",
    alpha: float = 1.0,
) -> None:
    """Daily MRF factor profiles by drought level for one reservoir.

    Shows how the minimum release fraction varies through the year at each
    drought level. Each line = one drought level; color from green to red.
    Reads from config.mrf_factors_daily_df (post-scaling). Auto-detects level
    naming for both standard FFMP (level1a–level5) and VR-FFMP (zone_0–zone_N).
    """
    levels = _mrf_levels_for(config)
    level_colors = [plt.cm.RdYlGn_r(t)
                    for t in np.linspace(0.05, 0.95, len(levels))]
    doy = np.arange(366)
    mrf_baseline = float(config.constants[f"mrf_baseline_{reservoir}"])

    for level, lcol in zip(levels, level_colors):
        try:
            profile = _mrf_array(config, level, reservoir)
        except KeyError:
            continue
        release = profile * mrf_baseline
        ax.plot(doy, release, color=lcol, lw=1.6, alpha=alpha,
                label="_nolegend_", solid_capstyle="round")

    ax.set_ylabel("Min release (MGD)", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_title(f"MRF profiles — {_RESERVOIR_LABELS.get(reservoir, reservoir)}",
                 fontsize=10, loc="left")
    _doy_xaxis(ax, short=True)
    _season_shading(ax)
    ax.grid(True, alpha=0.2, lw=0.5)


def plot_ffmp_flood_limits(
    config,
    ax: plt.Axes,
    color: str = "#d62728",
    alpha: float = 1.0,
    annotate: bool = True,
) -> None:
    """Horizontal bar chart of peak flood release limits (CFS)."""
    c = config.constants
    reservoirs = ["cannonsville", "pepacton", "neversink"]
    limits = [float(c[f"flood_max_release_{r}_cfs"]) for r in reservoirs]
    labels = ["Cannonsville", "Pepacton", "Neversink"]
    y = np.arange(len(reservoirs))

    ax.barh(y, limits, color=color, alpha=alpha * 0.65, height=0.45, edgecolor=color,
            linewidth=0.75)
    if annotate:
        for i, v in enumerate(limits):
            ax.text(v + 50, i, f"{v:,.0f}", va="center", ha="left",
                    fontsize=8, alpha=alpha, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Max release (CFS)", fontsize=9)
    ax.set_title("Flood release limits", fontsize=10, loc="left")
    ax.set_xlim(0, max(limits) * 1.3)
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)


# ---------------------------------------------------------------------------
# Composite figure
# ---------------------------------------------------------------------------

#: Constant color for MRF-baseline / flood-limit markers across all solutions.
#: The figure illustrates operational range, not performance, so per-solution
#: colormaps are intentionally omitted.
_NEUTRAL_COLOR = "#4a6f99"


def plot_ffmp_policy(
    dv_vectors,
    formulation_name: str = "ffmp",
    highlight_idx: Optional[int] = None,
    title: Optional[str] = None,
    figsize: tuple = (14, 9),
    show_baseline: bool = True,
    mrf_reservoir: str = "cannonsville",
) -> Figure:
    """
    Comprehensive FFMP policy portrait, designed for Pareto-set comparison.

    Panels
    ------
    Top (full width)  : Seasonal storage zone threshold curves
    Bottom left       : MRF baseline flows (3 reservoirs)
    Bottom center-left: Storage-zone factors (delivery & flow target)
    Bottom center-right: MRF factor seasonal profiles (one reservoir)
    Bottom right      : Flood release limits (3 reservoirs)

    Parameters
    ----------
    dv_vectors : array-like
        Single DV vector (shape n_dvs) OR array of vectors (shape n_sols × n_dvs).
    formulation_name : str
        "ffmp", "ffmp_8", "ffmp_10", "ffmp_12", etc.
    highlight_idx : int or None
        Index of the solution to emphasize; all others drawn faintly.
    title : str or None
        Figure suptitle override.
    figsize : (width, height) in inches.
    show_baseline : bool
        Overlay FFMP default parameter values in black as reference.
    mrf_reservoir : str
        Reservoir shown in the MRF seasonal profiles panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from pywrdrb.parameters.nyc_operations_config import NYCOperationsConfig

    apply_style()

    dv_arr = np.atleast_2d(np.asarray(dv_vectors, dtype=float))
    n_sols = len(dv_arr)
    configs = [_make_config(row, formulation_name) for row in dv_arr]

    # All solutions share a single neutral color — the figure is about
    # operational range, not per-solution performance.
    sol_colors = [_NEUTRAL_COLOR] * n_sols

    # Alpha per solution
    if n_sols == 1:
        sol_alphas = [1.0]
        sol_lws    = [2.0]
    elif highlight_idx is not None:
        sol_alphas = [0.12] * n_sols
        sol_lws    = [0.8] * n_sols
        sol_alphas[highlight_idx] = 1.0
        sol_lws[highlight_idx]    = 2.5
    else:
        sol_alphas = [min(1.0, max(0.05, 3.0 / n_sols))] * n_sols
        sol_lws    = [1.0] * n_sols

    # --- Layout ---
    # Reserve generous bottom margin for the two side-by-side legend blocks
    # plus the scalar annotation. The left legend grows with the zone count,
    # so VR-FFMP with N >= 8 needs more headroom than standard FFMP.
    n_zones = len(_get_zone_rows(configs[0]))
    bottom_margin = 0.20 + 0.014 * max(0, n_zones - 6)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 4,
        height_ratios=[1.5, 1.0],
        hspace=0.32,
        wspace=0.50,
        left=0.07, right=0.97, top=0.92, bottom=bottom_margin,
    )
    ax_zones = fig.add_subplot(gs[0, :])
    ax_mrf   = fig.add_subplot(gs[1, 0])
    ax_del   = fig.add_subplot(gs[1, 1])
    ax_seas  = fig.add_subplot(gs[1, 2])
    ax_flood = fig.add_subplot(gs[1, 3])

    # --- Baseline reference (only meaningful for standard FFMP; skip for N-zone) ---
    cfg_base = None
    is_nzone_form = formulation_name.startswith("ffmp_") and formulation_name != "ffmp"
    if show_baseline and not is_nzone_form:
        cfg_base = NYCOperationsConfig.from_defaults()
        plot_ffmp_storage_zones(cfg_base, ax_zones, alpha=0.85, lw=2.8,
                                zorder=1, show_labels=False)

    # --- Draw all solutions ---
    _annot_idx = highlight_idx if highlight_idx is not None else 0
    for i, (cfg, col, alp, lw_i) in enumerate(
        zip(configs, sol_colors, sol_alphas, sol_lws)
    ):
        annotate  = (i == _annot_idx) and n_sols <= 3
        plot_ffmp_storage_zones(cfg, ax_zones, alpha=alp, lw=lw_i,
                                zorder=2, show_labels=False)
        plot_ffmp_mrf_flow_targets(cfg, ax_mrf, color=col, alpha=alp,
                                   annotate=annotate)
        plot_ffmp_drought_delivery_factors(cfg, ax_del, alpha=alp)
        plot_ffmp_mrf_seasonal_profiles(cfg, ax_seas, reservoir=mrf_reservoir,
                                        color=col, alpha=alp)
        plot_ffmp_flood_limits(cfg, ax_flood, color=col, alpha=alp,
                               annotate=annotate)

    ax_zones.set_title("Seasonal storage zone thresholds",
                       fontsize=11, loc="left")

    # --- Bottom-left scalar annotation block ---
    _draw_scalar_annotation(fig, configs[0])

    # --- Two side-by-side legend blocks below the figure ---
    _draw_grouped_legends(
        fig,
        configs[0],
        has_baseline=(cfg_base is not None),
        has_highlight=(highlight_idx is not None and n_sols > 1),
    )

    label = ARCH_LABELS.get(formulation_name, formulation_name)
    fig.suptitle(
        title or f"{label}  —  policy portrait",
        fontsize=12,
    )
    return fig


def _draw_scalar_annotation(fig, sample_config) -> None:
    """Place the Montague / Trenton / Max-NYC-delivery scalars in the
    bottom-left of the figure (next to the legends)."""
    c = sample_config.constants
    montague = float(c["mrf_baseline_delMontague"])
    trenton  = float(c["mrf_baseline_delTrenton"])
    max_del  = float(c.get("max_flow_baseline_delivery_nyc", 800.0))
    fig.text(
        0.02, 0.04,
        (f"Montague target: {montague:.0f} MGD\n"
         f"Trenton target:  {trenton:.0f} MGD\n"
         f"Max NYC delivery: {max_del:.0f} MGD"),
        fontsize=8.5, va="bottom", ha="left", linespacing=1.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffbe6",
                  edgecolor="#d4b97c", alpha=0.9),
    )


def _draw_grouped_legends(fig, sample_config, has_baseline: bool,
                          has_highlight: bool) -> None:
    """Two side-by-side fig-level legends below the panels.

    Left block  → storage zone color swatches (one per drought zone).
    Right block → drought-zone factors (NYC, NJ, Montague, Trenton)
                   plus baseline / highlight reference styles.
    """
    from matplotlib.lines import Line2D

    # --- Block 1: storage zones ---
    zone_rows = _get_zone_rows(sample_config)
    is_std = not _is_nzone(sample_config)
    colors = _zone_colors(len(zone_rows))
    zone_handles = []
    for zone, c in zip(zone_rows, colors):
        if is_std:
            lab = _ZONE_LABELS_STD.get(zone, zone)
        else:
            lab = zone.replace("_", " ")
        zone_handles.append(Line2D([0], [0], color=c, lw=3, label=lab))

    leg1 = fig.legend(
        handles=zone_handles,
        loc="lower left",
        bbox_to_anchor=(0.30, 0.005),
        title="Drought zones",
        title_fontsize=9,
        ncol=1,
        fontsize=8.5,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6,
        alignment="left",
    )

    # --- Block 2: factors + reference styles ---
    factor_handles = []
    for name in ("nyc", "nj", "montague", "trenton"):
        color, marker, ls, label = DROUGHT_FACTOR_STYLES[name]
        factor_handles.append(
            Line2D([0], [0], color=color, marker=marker, linestyle=ls,
                   lw=1.8, ms=6, label=label, markeredgewidth=0)
        )
    if has_baseline:
        factor_handles.append(Line2D([0], [0], color="black", lw=2.8,
                                     alpha=0.85, label="Baseline reference"))
    if has_highlight:
        factor_handles.append(Line2D([0], [0], color="dimgray", lw=2.5,
                                     label="Highlighted solution"))

    fig.legend(
        handles=factor_handles,
        loc="lower left",
        bbox_to_anchor=(0.62, 0.005),
        title="Storage-zone factors  &  references",
        title_fontsize=9,
        ncol=1,
        fontsize=8.5,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.6,
        alignment="left",
    )

    # Re-add legend 1 (matplotlib drops the first when multiple legends share fig).
    fig.add_artist(leg1)
