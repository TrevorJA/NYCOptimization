"""Render a decision-variable vector as the NYC operational rules it encodes.

One figure, five panels, organized by decision-variable grouping and covering
all 69 FFMP decision variables:

    - Storage-zone rule curves vs day of year    (24 zone_vshift_* +
      24 zone_tshift_* per-breakpoint DVs)
    - Zone-dependent diversion limits            (5 drought-factor DVs; the
      NYC cap itself is Decree-fixed)
    - Time-dependent minimum-release schedules   (4 mrf_profile_scale_* DVs;
      the mrf_{res} baselines are fixed FFMP constants)
    - Zone-dependent downstream flow
      targets at Montague and Trenton            (6 mrf_target_scale_* DVs)
    - Flood-zone (L1a/L1b) spill-mitigation
      releases                                   (6 flood_release_scale_* DVs;
      the Table 5 max-release caps are fixed constants, drawn as a
      reference line)

Rules are always derived through :func:`src.simulation.dvs_to_config` so the
figure shows exactly what the optimizer's evaluation model sees (zone-shift
clipping, monotonicity enforcement, seasonal profile scaling, and the 1.0 cap
on adjusted flow-target factors included). Panel (d) shows the *effective*
targets: the Decree-fixed baseline target times the per-level monthly factor
tables scaled by the DVs; the dotted line marks the Decree target itself.

Encoding contract, identical in every panel: linestyle separates baseline
(dashed) from candidate (solid; gray vs accent bars in the flood panel); hue
carries storage zone or actor. Storage-zone colors are keyed once by a
shared figure-level legend at the bottom (with the FFMP zone names, since
not every zone is a drought zone). All values are in MGD.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config
from src.formulations import get_baseline_values, get_var_names
from src.plotting.style import save_figure
from src.simulation import dvs_to_config, _CFS_TO_MGD

# ---------------------------------------------------------------------------
# Palette and calendar constants
# ---------------------------------------------------------------------------

#: Drought-severity ramp (wet -> dry), validated for lightness band, chroma,
#: contrast on white, and CVD separation. Zone identity is additionally
#: encoded by vertical band order and direct labels, never color alone.
LEVEL_COLORS = {
    "level1b": "#3b6bc9",
    "level1c": "#0d9aa3",
    "level2": "#4f9648",
    "level3": "#7d6608",
    "level4": "#e2711d",
    "level5": "#a51d24",
}

#: Descriptive FFMP storage-zone names for the shared legend (not every zone
#: is a drought zone — L1a-L2 are flood/elevated/normal operations).
LEVEL_LONG = {
    "level1a": "L1a: Flood",
    "level1b": "L1b: Elevated",
    "level1c": "L1c: Elevated",
    "level2": "L2: Normal",
    "level3": "L3: Drought Watch",
    "level4": "L4: Drought Warning",
    "level5": "L5: Drought Emergency",
}

#: Accent color for candidate-policy bars in the flood panel.
NYC_COLOR = "#1f77b4"

BASELINE_GRAY = "0.35"
BASELINE_BAR = "0.65"

#: The seven FFMP drought levels, mildest to most severe.
DROUGHT_LEVELS = ["level1a", "level1b", "level1c", "level2",
                  "level3", "level4", "level5"]
#: Abbreviated zone tick labels for the per-subplot x-axes (capital "L"
#: prefix on every zone, matching the descriptive legend labels).
ZONE_TICK_LABELS = ["L1a", "L1b", "L1c", "L2", "L3", "L4", "L5"]

#: The six storage-zone boundary curves (level1a has no curve of its own).
ZONE_LEVELS = ["level1b", "level1c", "level2", "level3", "level4", "level5"]

#: Representative levels for the release-schedule panel: normal operations
#: (level2) and drought emergency (level5), colored per the shared legend.
#: level1a/1b MRF factors are flat spill-regime multipliers that would
#: dominate the y-range without informing the seasonal DVs.
RULE_LEVELS = ["level2", "level5"]

#: Levels drawn in the flow-target panel (d): the Normal level (L2, at the
#: full Decree target) plus the drought levels with optimizable factors
#: (L3/L4/L5, which step the target below Decree).
FLOW_TARGET_LEVELS = ["level2", "level3", "level4", "level5"]

#: Flood zones with optimizable spill-mitigation releases, panel (e).
FLOOD_ZONE_LEVELS = ["level1a", "level1b"]

#: Downstream flow-target locations (pywrdrb keys and display names).
FLOW_TARGET_LOCATIONS = [("delMontague", "Montague"), ("delTrenton", "Trenton")]

#: First day of each month in a leap year (profiles have 366 columns).
MONTH_START_DOY = [1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336]
MONTH_LETTERS = list("JFMAMJJASOND")
MONTH_LENGTHS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

#: FFMP season windows (Tables 4a-4g headers), matching
#: simulation._SEASON_DOY_RANGES: winter Dec 1-Mar 31, spring Apr 1-May 31,
#: summer Jun 1-Aug 31, fall Sep 1-Nov 30 (leap-year day-of-year).
SEASON_MIDPOINTS = {"Win": 46, "Spr": 122, "Sum": 199, "Fall": 290}
SEASON_SPANS = [(92, 152), (245, 335)]  # shade spring + fall to mark seasons

_DOY = np.arange(1, 367)


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------

def _extract_rules(cfg) -> dict:
    """Read plotted rule structures off a NYCOperationsConfig.

    The only function in this module that touches the pywrdrb config object;
    all panel functions consume the plain dict returned here.

    Args:
        cfg: NYCOperationsConfig produced by ``dvs_to_config``.

    Returns:
        Dict with keys ``zones`` (level -> 366 values, % of capacity),
        ``releases`` (reservoir -> level -> 366 daily MGD), ``nyc_delivery``
        and ``nj_delivery`` (7 allowed-diversion values, MGD),
        ``flow_targets`` (location -> level -> 366 daily MGD, from the
        monthly factor tables), ``flood_releases`` (reservoir -> flood zone
        -> 366 daily effective MGD), and ``flood_caps`` (reservoir -> CFS,
        the fixed Table 5 constants).
    """
    zones = {
        lvl: np.asarray(cfg.get_storage_zone_profile(lvl), dtype=float) * 100.0
        for lvl in ZONE_LEVELS
    }
    releases = {
        res: {
            lvl: float(cfg.constants[f"mrf_baseline_{res}"])
            * np.asarray(
                cfg.get_mrf_factor_profile(f"{lvl}_factor_mrf_{res}", daily=True),
                dtype=float,
            )
            for lvl in RULE_LEVELS
        }
        for res in config.NYC_RESERVOIRS
    }
    nyc_cap = float(cfg.constants["max_flow_baseline_delivery_nyc"])
    nj_cap = float(cfg.constants["max_flow_baseline_monthlyAvg_delivery_nj"])
    nyc_delivery = [
        min(float(cfg.constants[f"{lvl}_factor_delivery_nyc"]), 1.0) * nyc_cap
        for lvl in DROUGHT_LEVELS
    ]
    nj_delivery = [
        float(cfg.constants[f"{lvl}_factor_delivery_nj"]) * nj_cap
        for lvl in DROUGHT_LEVELS
    ]
    flow_targets = {
        loc: {
            lvl: float(cfg.constants[f"mrf_baseline_{loc}"])
            * np.repeat(
                np.asarray(
                    cfg.get_mrf_factor_profile(f"{lvl}_factor_mrf_{loc}",
                                               daily=False),
                    dtype=float,
                ),
                MONTH_LENGTHS,
            )
            for lvl in FLOW_TARGET_LEVELS
        }
        for loc, _ in FLOW_TARGET_LOCATIONS
    }
    flow_target_baselines = {
        loc: float(cfg.constants[f"mrf_baseline_{loc}"])
        for loc, _ in FLOW_TARGET_LOCATIONS
    }
    flood_releases = {
        res: {
            lvl: float(cfg.constants[f"mrf_baseline_{res}"])
            * np.asarray(
                cfg.get_mrf_factor_profile(f"{lvl}_factor_mrf_{res}", daily=True),
                dtype=float,
            )
            for lvl in FLOOD_ZONE_LEVELS
        }
        for res in config.NYC_RESERVOIRS
    }
    flood_caps = {
        res: float(cfg.constants[f"flood_max_release_{res}_cfs"])
        for res in config.NYC_RESERVOIRS
    }
    return {
        "zones": zones,
        "releases": releases,
        "nyc_delivery": nyc_delivery,
        "nj_delivery": nj_delivery,
        "flow_targets": flow_targets,
        "flow_target_baselines": flow_target_baselines,
        "flood_releases": flood_releases,
        "flood_caps": flood_caps,
    }


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def _month_axis(ax) -> None:
    """Month-start ticks with single-letter labels on a day-of-year axis."""
    ax.set_xlim(1, 366)
    ax.set_xticks(MONTH_START_DOY)
    ax.set_xticklabels(MONTH_LETTERS, fontsize=8)
    ax.tick_params(axis="x", length=2.5)


def _season_shading(ax, labels: bool = False) -> None:
    """Shade spring/fall and optionally print tiny season initials."""
    for lo, hi in SEASON_SPANS:
        ax.axvspan(lo, hi, color="0.5", alpha=0.07, lw=0, zorder=0)
    if labels:
        marks = dict(SEASON_MIDPOINTS)
        marks["Win "] = 352  # winter wraps the year end
        for name, mid in marks.items():
            ax.text(mid, 0.97, name.strip(), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, color="0.45")


def _rule_level_lines(ax, levels: list, primary_by_level: dict,
                      reference_by_level) -> float:
    """Draw per-level rule curves (candidate solid, baseline dashed).

    Shared by panels (c) and (d) so both time-varying rule families use the
    identical visual idiom.

    Returns:
        The maximum plotted y value (for headroom scaling).
    """
    y_top = 0.0
    for lvl in levels:
        ax.plot(_DOY, primary_by_level[lvl], color=LEVEL_COLORS[lvl], lw=1.8)
        if reference_by_level is not None:
            ax.plot(_DOY, reference_by_level[lvl], color=LEVEL_COLORS[lvl],
                    ls="--", lw=1.1, alpha=0.65)
            y_top = max(y_top, float(np.max(reference_by_level[lvl])))
        y_top = max(y_top, float(np.max(primary_by_level[lvl])))
    return y_top


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def _panel_storage_zones(ax, primary: dict, reference: dict | None) -> None:
    """Panel (a): storage-zone rule curves with faintly tinted zone bands."""
    zones = primary["zones"]
    # Zone bands: the band bounded above by curve k is drought level k.
    lower_edges = [zones[lvl] for lvl in ZONE_LEVELS[1:]] + [np.zeros(366)]
    for lvl, lower in zip(ZONE_LEVELS, lower_edges):
        ax.fill_between(_DOY, lower, zones[lvl], color=LEVEL_COLORS[lvl],
                        alpha=0.05, lw=0, zorder=1)
    # Tapered linewidths keep coincident curves visible: when clipping or
    # monotonicity pins one boundary onto another, the wider stroke of the
    # milder level peeks out around the narrower severe one.
    widths = np.linspace(2.6, 1.4, len(ZONE_LEVELS))
    for lvl, lw in zip(ZONE_LEVELS, widths):
        ax.plot(_DOY, zones[lvl], color=LEVEL_COLORS[lvl], lw=lw, zorder=3)
    if reference is not None:
        # Dashed baseline drawn above the solid candidate so coincident
        # segments still show both encodings.
        for lvl in ZONE_LEVELS:
            ax.plot(_DOY, reference["zones"][lvl], color=LEVEL_COLORS[lvl],
                    ls="--", lw=1.1, alpha=0.65, zorder=3.5)
    _month_axis(ax)
    # Headroom above 100 keeps curves clipped to full storage visible.
    ax.set_ylim(0, 102.5)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Combined NYC storage\n(% of capacity)")


def _panel_delivery(ax_nyc, ax_nj, primary: dict, reference: dict | None) -> None:
    """Panel (b): allowed NYC / NJ diversions stepping down with drought level."""
    # Black lines: these curves are keyed by their own y-axes, and black
    # keeps them visually distinct from the drought-level color scheme.
    x = np.arange(len(DROUGHT_LEVELS))
    for ax, key, color, cap_lim, ticks, label in (
        (ax_nyc, "nyc_delivery", "0.15", 960, [0, 300, 600, 900],
         "NYC diversion\n(MGD)"),
        (ax_nj, "nj_delivery", "0.15", 112, [0, 50, 100],
         "NJ diversion\n(MGD)"),
    ):
        ax.step(x, primary[key], where="mid", color=color, lw=1.8)
        if reference is not None:
            ax.step(x, reference[key], where="mid", color=color, ls="--",
                    lw=1.2, alpha=0.65)
        ax.set_ylim(0, cap_lim)
        ax.set_yticks(ticks)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(x)
        ax.tick_params(axis="x", length=2.5)
    ax_nyc.set_xticklabels([])
    ax_nj.set_xticklabels(ZONE_TICK_LABELS, fontsize=8)
    ax_nj.set_xlabel("Storage zone")


def _panel_release_schedules(axes, primary: dict, reference: dict | None) -> None:
    """Panel (c): daily minimum-release schedules per reservoir."""
    for i, (ax, res) in enumerate(zip(axes, config.NYC_RESERVOIRS)):
        _season_shading(ax, labels=(i == 0))
        y_top = _rule_level_lines(
            ax, RULE_LEVELS, primary["releases"][res],
            reference["releases"][res] if reference is not None else None,
        )
        _month_axis(ax)
        # Headroom keeps season initials clear of tall summer plateaus.
        ax.set_ylim(0, 1.18 * y_top)
        ax.set_title(res.capitalize(), fontsize=9.5, pad=3)
        if i == 0:
            ax.set_ylabel("Min. release (MGD)")


def _panel_flow_targets(axes, primary: dict, reference: dict | None) -> None:
    """Panel (d): zone- and season-dependent downstream flow targets.

    Effective targets for the Normal level (L2, green — always the full
    Decree target) and the drought levels with optimizable factors
    (L3/L4/L5, which step below it), same idiom and hues as the other
    level-keyed panels. The dotted line marks the Decree-fixed baseline
    target (the L1a-L2 target, which the L2 line traces). The y-axis is
    zoomed to the plotted range — the seasonal factor structure is the
    point, not distance to zero.
    """
    for i, (ax, (loc, name)) in enumerate(zip(axes, FLOW_TARGET_LOCATIONS)):
        decree = primary["flow_target_baselines"][loc]
        ax.axhline(decree, color="0.35", ls=":", lw=1.0, zorder=2)
        y_top = _rule_level_lines(
            ax, FLOW_TARGET_LEVELS, primary["flow_targets"][loc],
            reference["flow_targets"][loc] if reference is not None else None,
        )
        y_top = max(y_top, decree)
        curves = [primary["flow_targets"][loc][lvl] for lvl in FLOW_TARGET_LEVELS]
        if reference is not None:
            curves += [reference["flow_targets"][loc][lvl]
                       for lvl in FLOW_TARGET_LEVELS]
        y_lo = min(float(np.min(c)) for c in curves)
        span = max(y_top - y_lo, 1.0)
        _month_axis(ax)
        ax.set_ylim(y_lo - 0.15 * span, y_top + 0.15 * span)
        ax.set_title(name, fontsize=9.5, pad=3)
        if i == 0:
            ax.set_ylabel("Flow target (MGD)")
            ax.text(0.99, decree, "Decree", color="0.35", fontsize=7,
                    va="bottom", ha="right",
                    transform=ax.get_yaxis_transform())


def _panel_flood_releases(ax, primary: dict, reference: dict | None,
                          primary_is_baseline: bool) -> None:
    """Panel (e): flood-zone (L1a/L1b) spill-mitigation releases, in MGD.

    Bars show the annual maximum effective release per reservoir and flood
    zone (the seasonal DV structure raises or lowers these plateaus).
    Single bars for one policy; paired baseline (gray) + candidate (accent)
    bars in overlay mode. The dotted line marks the fixed FFMP Table 5
    maximum combined discharge cap for each reservoir.
    """
    color = BASELINE_BAR if primary_is_baseline else NYC_COLOR
    sub_offsets = {"level1a": -0.21, "level1b": 0.21}
    tick_pos, tick_labels = [], []
    for i, res in enumerate(config.NYC_RESERVOIRS):
        cap_mgd = primary["flood_caps"][res] * _CFS_TO_MGD
        ax.hlines(cap_mgd, i - 0.45, i + 0.45, color="0.35", ls=":",
                  lw=1.0, zorder=2)
        if i == 0:
            ax.text(i - 0.42, cap_mgd, "Max", color="0.35",
                    fontsize=7, va="bottom", ha="left")
        for lvl in FLOOD_ZONE_LEVELS:
            xc = i + sub_offsets[lvl]
            prim = float(np.max(primary["flood_releases"][res][lvl]))
            if reference is not None:
                ref = float(np.max(reference["flood_releases"][res][lvl]))
                ax.bar(xc - 0.09, ref, 0.16, color=BASELINE_BAR)
                ax.bar(xc + 0.09, prim, 0.16, color=color)
            else:
                ax.bar(xc, prim, 0.30, color=color)
            tick_pos.append(xc)
            tick_labels.append(lvl.replace("level", "L"))
        ax.text(i, -0.16, res.capitalize(), fontsize=8.5,
                ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, len(config.NYC_RESERVOIRS) - 0.45)
    top_cap = max(primary["flood_caps"][r] for r in config.NYC_RESERVOIRS)
    ax.set_ylim(0, 1.08 * top_cap * _CFS_TO_MGD)
    ax.set_ylabel("Flood-zone release (MGD)")


def _figure_legend(fig, mode: str, candidate_label: str) -> None:
    """Two shared figure-level legend blocks at the bottom.

    Storage-zone block: keys the zone colors used across the zone, release,
    and flow-target panels with their FFMP zone names, grouped into the three
    operational regimes as columns — above-normal (L1a-L1c), normal (L2), and
    drought (L3-L5). matplotlib fills columns top-to-bottom, so the normal
    column is padded with blank spacers to center L2 between its neighbors.
    Policy block: each entry pairs the line style used in the curve panels
    with the bar fill used in the flood panel (side by side, via HandlerTuple).
    """
    from matplotlib.legend_handler import HandlerTuple

    def _zone_line(lvl):
        return Line2D([], [], color=LEVEL_COLORS[lvl], lw=3, label=LEVEL_LONG[lvl])

    def _spacer():
        return Line2D([], [], linestyle="none", marker="none", label=" ")

    # Column-major order: above-normal | normal (centered) | drought.
    zone_handles = [
        Patch(facecolor="white", edgecolor="0.7", label=LEVEL_LONG["level1a"]),
        _zone_line("level1b"), _zone_line("level1c"),
        _spacer(), _zone_line("level2"), _spacer(),
        _zone_line("level3"), _zone_line("level4"), _zone_line("level5"),
    ]
    leg = fig.legend(handles=zone_handles, loc="lower center", ncol=3,
                     frameon=True, fancybox=False, framealpha=1.0,
                     edgecolor="0.7", facecolor="white", borderpad=0.7,
                     fontsize=8.5, handlelength=1.5,
                     columnspacing=2.4, title="Storage zone",
                     bbox_to_anchor=(0.5, 0.02))
    leg.get_title().set_fontsize(8.5)
    leg.get_title().set_color("0.3")
    leg.get_frame().set_linewidth(0.8)

    handles, labels = [], []
    if mode in ("baseline", "overlay"):
        handles.append((
            Line2D([], [], color=BASELINE_GRAY,
                   ls="--" if mode == "overlay" else "-", lw=1.5),
            Patch(facecolor=BASELINE_BAR),
        ))
        labels.append("Baseline FFMP")
    if mode in ("candidate", "overlay"):
        handles.append((
            Line2D([], [], color="0.15", lw=1.8),
            Patch(facecolor=NYC_COLOR),
        ))
        labels.append(candidate_label)
    fig.legend(handles=handles, labels=labels, loc="lower center",
               ncol=len(handles), frameon=False, fontsize=9,
               handlelength=2.4, bbox_to_anchor=(0.5, -0.035),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_policy_rules(
    dv_vector: np.ndarray | None = None,
    formulation: str = "ffmp",
    show_baseline: bool = True,
    candidate_label: str | None = None,
    output_file: Path | None = None,
    figsize: tuple = (11.0, 9.2),
) -> Figure:
    """Render a decision-variable vector as NYC operational rule panels.

    Args:
        dv_vector: Candidate decision variables. ``None`` plots the baseline
            FFMP policy alone.
        formulation: Formulation name (panel structure is tuned for ``ffmp``).
        show_baseline: When a candidate is given, also draw the baseline
            rules dashed underneath for comparison.
        candidate_label: Legend label for the candidate policy.
        output_file: Optional path stub (no extension); saved via
            ``style.save_figure``.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure.
    """
    baseline_rules = _extract_rules(
        dvs_to_config(get_baseline_values(formulation), formulation)
    )
    if dv_vector is None:
        mode = "baseline"
        primary, reference = baseline_rules, None
    else:
        primary = _extract_rules(dvs_to_config(np.asarray(dv_vector, float),
                                               formulation))
        mode = "overlay" if show_baseline else "candidate"
        reference = baseline_rules if show_baseline else None

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.45, 1.0, 1.0],
                  hspace=0.50, wspace=0.36,
                  left=0.075, right=0.97, top=0.965, bottom=0.155)
    ax_zones = fig.add_subplot(gs[0, 0:2])
    gs_del = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], hspace=0.18)
    ax_nyc = fig.add_subplot(gs_del[0])
    ax_nj = fig.add_subplot(gs_del[1])
    axes_rel = [fig.add_subplot(gs[1, i]) for i in range(3)]
    axes_tgt = [fig.add_subplot(gs[2, i]) for i in range(2)]
    ax_flood = fig.add_subplot(gs[2, 2])

    _panel_storage_zones(ax_zones, primary, reference)
    _panel_delivery(ax_nyc, ax_nj, primary, reference)
    _panel_release_schedules(axes_rel, primary, reference)
    _panel_flow_targets(axes_tgt, primary, reference)
    _panel_flood_releases(ax_flood, primary, reference,
                          primary_is_baseline=(mode == "baseline"))
    _figure_legend(fig, mode, candidate_label or "Candidate policy")

    if output_file is not None:
        save_figure(fig, output_file)
    return fig
