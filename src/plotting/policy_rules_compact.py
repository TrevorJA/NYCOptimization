"""Compact FFMP operating-rule figures for the main manuscript.

Two public builders share one panel set:

    :func:`plot_policy_rules_compact`  one policy (or the baseline alone) in a
                                       two-row, five-panel figure
    :func:`plot_policy_rules_grid`     several policies as columns, the five
                                       rule families as rows, one legend
                                       column on the right

Panels (lettered in this order in both builders):

    (a) storage-zone boundary curves vs month, zone washes, zone names on a
        right-hand axis
    (b) combined NYC scheduled minimum release in the normal zone (L2) vs
        month -- the four seasonal release scales apply to every reservoir
        and every conservation zone, so one combined line is the honest view
    (c) NYC and NJ diversions as % of allocation, stepped over zones L2-L5
    (d) Montague and Trenton flow targets as % of the Decree target, stepped
        over zones L2-L5 (annual mean of the monthly factors)
    (e) combined NYC spill-mitigation release in the flood zones L1a and
        L1b, paired bars

Encoding contract: the candidate policy is solid dark ink (dark bar), the
baseline FFMP is dashed grey with hollow markers (light grey bar). Series
identity in (c) and (d) is carried by marker shape on both policies and
keyed in the legend. Hue is reserved for storage-zone identity and appears
only in panel (a). Nothing is labelled by free-floating text: every label is
a tick, an axis label, a panel title, or a legend entry.

The full-detail per-reservoir figure remains :mod:`src.plotting.policy_rules`
(SI and internal use).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config
from src.formulations import get_baseline_values
from src.plotting import policy_rules as pr
from src.plotting.layout import WIDTH_DOUBLE_COL
from src.plotting.style import save_manuscript_figure
from src.simulation import dvs_to_config

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

INK = "0.12"          #: candidate policy lines, markers, bars
BASE = "0.55"         #: baseline FFMP lines and marker edges
BASE_FILL = "0.74"    #: baseline FFMP bars
GRID = "0.88"
LW_CAND = 2.2
LW_BASE = 1.6
DASH = (0, (4.0, 2.2))
MS = 6.0

#: Light step of the L1b blue for the L1a sliver (above the level1b curve).
L1A_COLOR = "#8aa6d6"

ZONES7 = pr.DROUGHT_LEVELS
ZONE_TAG = {z: z.replace("level", "L") for z in ZONES7}
CURT_ZONES = ["level2", "level3", "level4", "level5"]
CURT_TICKS = [ZONE_TAG[z] for z in CURT_ZONES]
SPILL_TICKS = [ZONE_TAG[z] for z in pr.FLOOD_ZONE_LEVELS]

#: FFMP zone names (manuscript terminology) for the legend, long and short.
ZONE_NAME = {
    "level2": "L2 Normal",
    "level3": "L3 Drought watch",
    "level4": "L4 Drought warning",
    "level5": "L5 Drought emergency",
}
ZONE_NAME_SHORT = {
    "level2": "L2 Normal",
    "level3": "L3 Watch",
    "level4": "L4 Warning",
    "level5": "L5 Emergency",
}

#: Series identity in the curtailment panels: ``name -> (marker, getter)``.
ACTORS = {
    "NYC": ("o", lambda r: r["nyc_pct"]),
    "NJ": ("s", lambda r: r["nj_pct"]),
    "Montague": ("^", lambda r: r["target_pct"]["delMontague"].mean(axis=1)),
    "Trenton": ("D", lambda r: r["target_pct"]["delTrenton"].mean(axis=1)),
}
DIVERSION_ACTORS = ["NYC", "NJ"]
TARGET_ACTORS = ["Montague", "Trenton"]

PANEL_KEYS = ["zones", "release", "div", "tgt", "spill"]
TITLES = {
    "zones": "Storage zone boundaries",
    "release": "NYC minimum daily release",
    "div": "Diversion curtailments",
    "tgt": "Flow target curtailments",
    "spill": "Spill-mitigation releases",
}
#: Two-line titles for panels narrower than their title (single layout).
TITLES_WRAPPED = {
    "zones": "Storage zone boundaries",
    "release": "NYC minimum\ndaily release",
    "div": "Diversion\ncurtailments",
    "tgt": "Flow target\ncurtailments",
    "spill": "Spill-mitigation\nreleases",
}
YLABELS = {
    "zones": "NYC storage (% of capacity)",
    "release": "Combined L2 release (MGD)",
    "div": "% of allocation",
    "tgt": "% of Decree target",
    "spill": "Spill release (MGD)",
}
#: Two-line y labels for the shallow rows of the grid layout.
YLABELS_GRID = {
    "zones": "NYC storage\n(% capacity)",
    "release": "L2 release\n(MGD)",
    "div": "Diversion\n(% allocation)",
    "tgt": "Flow target\n(% Decree)",
    "spill": "Spill release\n(MGD)",
}

_DOY = pr._DOY


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------

def extract_rules(dv_vector: Optional[np.ndarray],
                  formulation: str = "ffmp") -> dict:
    """Rule structures of one policy, in the forms the compact panels plot.

    Extends :func:`src.plotting.policy_rules.extract_rules` with the percent
    forms of the curtailment rules and the combined NYC release schedules.

    Args:
        dv_vector: Decision variables; ``None`` means the baseline FFMP.
        formulation: Formulation name.

    Returns:
        The :func:`~src.plotting.policy_rules.extract_rules` dict plus
        ``nyc_pct`` / ``nj_pct`` (7 per-zone values, % of allocation),
        ``target_pct`` (location -> (7, 12) monthly table, % of Decree),
        ``combined_release_l2`` (366 daily MGD) and ``combined_spill`` (one
        plateau MGD value per flood zone).
    """
    if dv_vector is None:
        dv_vector = get_baseline_values(formulation)
    cfg = dvs_to_config(np.asarray(dv_vector, dtype=float), formulation)
    rules = pr.extract_rules(cfg)
    rules["nyc_pct"] = np.array([
        min(float(cfg.constants[f"{z}_factor_delivery_nyc"]), 1.0) * 100
        for z in ZONES7])
    rules["nj_pct"] = np.array([
        float(cfg.constants[f"{z}_factor_delivery_nj"]) * 100 for z in ZONES7])
    rules["target_pct"] = {
        loc: np.array([
            100 * np.asarray(cfg.get_mrf_factor_profile(
                f"{z}_factor_mrf_{loc}", daily=False), dtype=float)
            for z in ZONES7])
        for loc, _ in pr.FLOW_TARGET_LOCATIONS}
    rules["combined_release_l2"] = sum(
        rules["releases"][res]["level2"] for res in config.NYC_RESERVOIRS)
    rules["combined_spill"] = np.array([
        sum(float(np.max(rules["flood_releases"][res][z]))
            for res in config.NYC_RESERVOIRS)
        for z in pr.FLOOD_ZONE_LEVELS])
    return rules


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def _despine(ax: Axes) -> None:
    """Recessive axes: no top/right spines, no grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.35")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors="0.2", length=3.5, width=0.9)
    ax.grid(False)


#: Minimum contrast ratio of a zone-coloured tick label against white.
_TEXT_CONTRAST = 3.5


def zone_text_color(zone: str) -> tuple:
    """The zone's hue, darkened just enough to read as text on white.

    Tick labels on the zone axes wear their zone's colour so the axes key
    to panel (a) without a lookup. The L1a wash and the L4 orange are too
    light for 12 pt text, so each colour is scaled down until it meets
    :data:`_TEXT_CONTRAST` (WCAG relative luminance).
    """
    rgb = np.array(mcolors.to_rgb(
        L1A_COLOR if zone == "level1a" else pr.LEVEL_COLORS[zone]))

    def contrast(c):
        lin = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return 1.05 / (0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] + 0.05)

    while contrast(rgb) < _TEXT_CONTRAST:
        rgb = rgb * 0.93
    return tuple(rgb)


def _color_zone_ticks(ax: Axes, zones: Sequence[str], axis: str = "x") -> None:
    """Colour each tick label of ``axis`` by its zone, in tick order."""
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label, zone in zip(labels, zones):
        label.set_color(zone_text_color(zone))


def _repel(values: Sequence[float], min_sep: float, lo: float,
           hi: float) -> list:
    """Push tick positions apart to at least ``min_sep`` inside ``[lo, hi]``."""
    order = np.argsort(values)
    y = np.asarray(values, dtype=float)[order]
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + min_sep)
    y = np.minimum(y, hi)
    for i in range(len(y) - 2, -1, -1):
        y[i] = min(y[i], y[i + 1] - min_sep)
    y = np.maximum(y, lo)
    out = np.empty_like(y)
    out[order] = y
    return list(out)


def _month_axis(ax: Axes, labels: bool = True, compact: bool = False) -> None:
    """Day-of-year x axis: month letters, or quarter starts when compact."""
    ax.set_xlim(1, 366)
    if compact:
        ax.set_xticks([pr.MONTH_START_DOY[i] for i in (0, 3, 6, 9)])
        ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"] if labels else [])
        ax.tick_params(axis="x", length=3.5)
        return
    mids = [d + n / 2 for d, n in zip(pr.MONTH_START_DOY, pr.MONTH_LENGTHS)]
    ax.set_xticks(mids)
    ax.set_xticklabels(pr.MONTH_LETTERS if labels else [])
    ax.tick_params(axis="x", length=0)
    ax.set_xticks(pr.MONTH_START_DOY[1:], minor=True)
    ax.tick_params(axis="x", which="minor", length=3.5)


def _title(ax: Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=plt.rcParams["font.size"],
                 fontweight="normal", color="0.1", pad=7)


def _policy_pairs(primary: dict, reference: Optional[dict]) -> list:
    """``(rules, color, lw, ls, zorder)`` for the baseline then the candidate."""
    pairs = []
    if reference is not None:
        pairs.append((reference, BASE, LW_BASE, DASH, 2.0))
    pairs.append((primary, INK, LW_CAND, "-", 4.0))
    return pairs


def _markers(ax: Axes, x, y, marker: str, candidate: bool, zorder: float) -> None:
    """Filled ink markers for the candidate, hollow grey for the baseline."""
    if candidate:
        ax.plot(x, y, ls="none", marker=marker, ms=MS, color=INK,
                markeredgecolor="white", markeredgewidth=1.0, zorder=zorder)
    else:
        ax.plot(x, y, ls="none", marker=marker, ms=MS, markerfacecolor="white",
                markeredgecolor=BASE, markeredgewidth=1.4, zorder=zorder)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def panel_zones(ax: Axes, primary: dict, reference: Optional[dict], *,
                xlabels: bool = True, compact: bool = False,
                ylabel: bool = True, zone_axis: bool = True) -> None:
    """(a) Storage-zone boundary curves with washes and a zone-name axis."""
    zones = primary["zones"]
    lower = [zones[z] for z in pr.ZONE_LEVELS[1:]] + [np.zeros(366)]
    ax.fill_between(_DOY, zones["level1b"], 100.0, color=L1A_COLOR, alpha=0.12,
                    lw=0, zorder=1)
    for z, lo in zip(pr.ZONE_LEVELS, lower):
        ax.fill_between(_DOY, lo, zones[z], color=pr.LEVEL_COLORS[z],
                        alpha=0.06, lw=0, zorder=1)
    # Tapered widths keep coincident curves visible (the milder level's
    # wider stroke peeks out around the narrower severe one).
    widths = np.linspace(2.6, 1.9, len(pr.ZONE_LEVELS))
    for z, lw in zip(pr.ZONE_LEVELS, widths):
        ax.plot(_DOY, zones[z], color=pr.LEVEL_COLORS[z], lw=lw, zorder=3,
                solid_capstyle="round")
    if reference is not None:
        for z in pr.ZONE_LEVELS:
            ax.plot(_DOY, reference["zones"][z], color=pr.LEVEL_COLORS[z],
                    ls=DASH, lw=1.3, alpha=0.7, zorder=3.5)
    _month_axis(ax, xlabels, compact)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    if ylabel:
        ax.set_ylabel(YLABELS["zones"])
    _despine(ax)
    if not zone_axis:
        return
    # Zone identity as a right-hand axis: one tick per zone at the band's
    # December mid-height, pushed apart to one text height (0.19 in).
    edges = [100.0] + [float(zones[z][-1]) for z in pr.ZONE_LEVELS] + [0.0]
    mids = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    height_in = ax.get_position().height * ax.figure.get_figheight()
    mids = _repel(mids, 102 * 0.19 / height_in, 4.0, 98.5)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(mids)
    ax2.set_yticklabels([ZONE_TAG[z] for z in ZONES7])
    ax2.tick_params(axis="y", length=0, pad=5)
    ax2.grid(False)
    _color_zone_ticks(ax2, ZONES7, axis="y")
    for spine in ax2.spines.values():
        spine.set_visible(False)


def panel_release(ax: Axes, primary: dict, reference: Optional[dict], *,
                  xlabels: bool = True, compact: bool = False,
                  ylabel: bool = True, y_top: Optional[float] = None) -> None:
    """(b) Combined NYC scheduled minimum release in the normal zone (L2)."""
    top = 0.0
    for rules, color, lw, ls, z0 in _policy_pairs(primary, reference):
        y = rules["combined_release_l2"]
        ax.plot(_DOY, y, color=color, lw=lw, ls=ls, zorder=z0)
        top = max(top, float(y.max()))
    ax.set_ylim(0, y_top if y_top is not None else 1.15 * top)
    _month_axis(ax, xlabels, compact)
    if ylabel:
        ax.set_ylabel(YLABELS["release"])
    _despine(ax)


def panel_curtailment(ax: Axes, primary: dict, reference: Optional[dict],
                      actors: Sequence[str], *, ylabel: Optional[str],
                      xlabels: bool = True, ylim: tuple = (45, 104)) -> None:
    """(c)/(d) Curtailment stepped over zones L2-L5, one marker per actor."""
    idx = [ZONES7.index(z) for z in CURT_ZONES]
    x = np.arange(len(CURT_ZONES))
    for name in actors:
        marker, getter = ACTORS[name]
        for rules, color, lw, ls, z0 in _policy_pairs(primary, reference):
            v = np.asarray(getter(rules), dtype=float)[idx]
            ax.step(x, v, where="mid", color=color, lw=lw, ls=ls, zorder=z0,
                    solid_capstyle="round")
            _markers(ax, x, v, marker, candidate=(rules is primary),
                     zorder=z0 + 1)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CURT_TICKS if xlabels else [])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(*ylim)
    ax.set_yticks([50, 75, 100])
    if ylabel:
        ax.set_ylabel(ylabel)
    _despine(ax)
    _color_zone_ticks(ax, CURT_ZONES)


def panel_spill(ax: Axes, primary: dict, reference: Optional[dict], *,
                xlabels: bool = True, ylabel: Optional[str] = None,
                y_top: Optional[float] = None) -> None:
    """(e) Combined NYC spill-mitigation release per flood zone, paired bars."""
    x = np.arange(len(pr.FLOOD_ZONE_LEVELS))
    top = float(primary["combined_spill"].max())
    if reference is not None:
        ax.bar(x - 0.19, reference["combined_spill"], width=0.34,
               color=BASE_FILL, zorder=2)
        ax.bar(x + 0.19, primary["combined_spill"], width=0.34, color=INK,
               zorder=3)
        top = max(top, float(reference["combined_spill"].max()))
    else:
        ax.bar(x, primary["combined_spill"], width=0.5, color=INK, zorder=3)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(SPILL_TICKS if xlabels else [])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, y_top if y_top is not None else 1.12 * top)
    if ylabel:
        ax.set_ylabel(ylabel)
    _despine(ax)
    _color_zone_ticks(ax, pr.FLOOD_ZONE_LEVELS)


# ---------------------------------------------------------------------------
# Legends
# ---------------------------------------------------------------------------

def _policy_handles(reference: bool, candidate_label: str) -> tuple:
    """Line-plus-bar handle pairs keying the baseline and the candidate."""
    handles, labels = [], []
    if reference:
        handles.append((Line2D([], [], color=BASE, ls=DASH, lw=LW_BASE),
                        Patch(facecolor=BASE_FILL)))
        labels.append("Baseline FFMP")
    handles.append((Line2D([], [], color=INK, lw=LW_CAND),
                    Patch(facecolor=INK)))
    labels.append(candidate_label)
    return handles, labels


def _actor_handles(names: Sequence[str]) -> list:
    return [Line2D([], [], ls="none", marker=ACTORS[a][0], ms=MS, color=INK)
            for a in names]


def _zone_handles(short: bool = False) -> tuple:
    spill = tuple(Line2D([], [], color=c, lw=3.0) for c in (
        L1A_COLOR, pr.LEVEL_COLORS["level1b"], pr.LEVEL_COLORS["level1c"]))
    names = ZONE_NAME_SHORT if short else ZONE_NAME
    handles = [spill] + [Line2D([], [], color=pr.LEVEL_COLORS[z], lw=3.0)
                         for z in names]
    labels = ["L1a–L1c Spill" if short else "L1a–L1c Spill mitigation"]
    return handles, labels + list(names.values())


_TUPLE_HANDLER = {tuple: HandlerTuple(ndivide=None, pad=0.0)}


def _bottom_legend(fig: Figure, reference: bool, candidate_label: str) -> None:
    """Three frameless rows below the figure: policies, actors, zones."""
    fs = plt.rcParams["font.size"]
    row = 0.055
    handles, labels = _policy_handles(reference, candidate_label)
    fig.add_artist(fig.legend(
        handles=handles, labels=labels, loc="lower center",
        bbox_to_anchor=(0.5, 3 * row), ncol=len(handles), frameon=False,
        fontsize=fs, handlelength=2.6, columnspacing=2.5,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}))
    names = DIVERSION_ACTORS + TARGET_ACTORS
    fig.add_artist(fig.legend(
        handles=_actor_handles(names), labels=names, loc="lower center",
        bbox_to_anchor=(0.5, 2 * row), ncol=len(names), frameon=False,
        fontsize=fs, handlelength=1.2, handletextpad=0.5, columnspacing=2.0))
    handles, labels = _zone_handles()
    fig.legend(handles=handles, labels=labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=fs,
               handlelength=2.0, columnspacing=1.8, handletextpad=0.6,
               handler_map=_TUPLE_HANDLER)


def _side_legend(lax: Axes, handles: Sequence, labels: Sequence[str],
                 **kwargs) -> None:
    """One frameless legend block, vertically centred in an invisible axes."""
    lax.set_axis_off()
    kwargs.setdefault("handlelength", 1.8)
    kwargs.setdefault("handletextpad", 0.6)
    kwargs.setdefault("labelspacing", 0.55)
    lax.legend(handles=handles, labels=labels, loc="center left",
               bbox_to_anchor=(0.12, 0.5), frameon=False, borderaxespad=0,
               fontsize=plt.rcParams["font.size"],
               handler_map=_TUPLE_HANDLER, **kwargs)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def plot_policy_rules_compact(
    dv_vector: Optional[np.ndarray] = None,
    formulation: str = "ffmp",
    candidate_label: str = "Pareto policy",
    output_file: Optional[Path] = None,
) -> Figure:
    """One policy (or the baseline FFMP alone) in a two-row compact figure.

    Row 1 holds the two month-axis panels (a, b); row 2 the three zone-axis
    panels (c, d, e). Legend rows sit below. ``dv_vector=None`` draws the
    baseline FFMP as the only policy (solid ink, no dashed reference), the
    form intended for the Section 2 rule-description figure.

    Args:
        dv_vector: Candidate decision variables, or ``None`` for the baseline.
        formulation: Formulation name.
        candidate_label: Legend label of the candidate policy.
        output_file: Optional path stub; saved via ``save_manuscript_figure``.

    Returns:
        The matplotlib Figure.
    """
    baseline = extract_rules(None, formulation)
    if dv_vector is None:
        primary, reference = baseline, None
        candidate_label = "Baseline FFMP"
    else:
        primary, reference = extract_rules(dv_vector, formulation), baseline

    fig = plt.figure(figsize=(WIDTH_DOUBLE_COL, 6.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0],
                          width_ratios=[1.0, 1.0, 0.2, 1.05],
                          left=0.09, right=0.98, top=0.92, bottom=0.30,
                          hspace=0.95, wspace=0.5)
    axes = {
        "zones": fig.add_subplot(gs[0, 0:2]),
        "release": fig.add_subplot(gs[0, 3]),
        "div": fig.add_subplot(gs[1, 0]),
        "tgt": fig.add_subplot(gs[1, 1]),
        "spill": fig.add_subplot(gs[1, 3]),
    }
    panel_zones(axes["zones"], primary, reference)
    panel_release(axes["release"], primary, reference, compact=True)
    panel_curtailment(axes["div"], primary, reference, DIVERSION_ACTORS,
                      ylabel=YLABELS["div"])
    panel_curtailment(axes["tgt"], primary, reference, TARGET_ACTORS,
                      ylabel=YLABELS["tgt"])
    panel_spill(axes["spill"], primary, reference, ylabel=YLABELS["spill"])
    for letter, key in zip("abcde", PANEL_KEYS):
        _title(axes[key], f"({letter}) {TITLES_WRAPPED[key]}")
    _bottom_legend(fig, reference is not None, candidate_label)
    if output_file is not None:
        save_manuscript_figure(fig, output_file)
    return fig


def plot_policy_rules_grid(
    policies: Sequence[tuple],
    formulation: str = "ffmp",
    output_file: Optional[Path] = None,
    legend: str = "side",
) -> Figure:
    """Several policies as columns, the five rule families as rows.

    Every row shares its y range across policies; y tick labels appear on the
    first column only and the zone-name axis on the last. With
    ``legend="side"`` each legend block sits in a right-hand column centred
    on the row it keys (zones on a, the policy key on b, NYC/NJ on c,
    Montague/Trenton on d); ``"bottom"`` stacks the three rows below.

    Args:
        policies: ``(label, dv_vector)`` pairs, one column each.
        formulation: Formulation name.
        output_file: Optional path stub; saved via ``save_manuscript_figure``.
        legend: ``"side"`` or ``"bottom"``.

    Returns:
        The matplotlib Figure.
    """
    if legend not in ("side", "bottom"):
        raise ValueError(f"legend must be 'side' or 'bottom', got {legend!r}")
    side = legend == "side"
    reference = extract_rules(None, formulation)
    rules = [(label, extract_rules(dv, formulation)) for label, dv in policies]
    n = len(rules)

    fig = plt.figure(figsize=(WIDTH_DOUBLE_COL, 8.4 if side else 9.6))
    gs = fig.add_gridspec(
        len(PANEL_KEYS), n + (1 if side else 0),
        height_ratios=[1.45, 1, 1, 1, 1],
        width_ratios=[1.0] * n + ([0.92] if side else []),
        left=0.115, right=0.985 if side else 0.955, top=0.925,
        bottom=0.045 if side else 0.2, hspace=0.68, wspace=0.16)

    everyone = [r for _, r in rules] + [reference]
    rel_top = 1.15 * max(float(r["combined_release_l2"].max()) for r in everyone)
    spill_top = 1.12 * max(float(r["combined_spill"].max()) for r in everyone)

    for j, (label, primary) in enumerate(rules):
        first, last = j == 0, j == n - 1
        col = {k: fig.add_subplot(gs[i, j]) for i, k in enumerate(PANEL_KEYS)}
        panel_zones(col["zones"], primary, reference, compact=True,
                    ylabel=False, zone_axis=last)
        panel_release(col["release"], primary, reference, compact=True,
                      y_top=rel_top, ylabel=False)
        panel_curtailment(col["div"], primary, reference, DIVERSION_ACTORS,
                          ylabel=None)
        panel_curtailment(col["tgt"], primary, reference, TARGET_ACTORS,
                          ylabel=None)
        panel_spill(col["spill"], primary, reference, y_top=spill_top)
        if first:
            for letter, key in zip("abcde", PANEL_KEYS):
                _title(col[key], f"({letter}) {TITLES[key]}")
                col[key].set_ylabel(YLABELS_GRID[key])
        else:
            for ax in col.values():
                ax.set_yticklabels([])
        pos = col["zones"].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + (0.048 if side else 0.042),
                 label, ha="center", va="bottom",
                 fontsize=plt.rcParams["font.size"], color="0.1")

    if side:
        lax = [fig.add_subplot(gs[i, n]) for i in range(len(PANEL_KEYS))]
        _side_legend(lax[0], *_zone_handles(short=True))
        handles, _ = _policy_handles(True, "Pareto policy")
        _side_legend(lax[1], handles, ["Baseline\nFFMP", "Pareto\npolicy"],
                     handlelength=2.2, labelspacing=0.9)
        for ax, names in zip(lax[2:4], (DIVERSION_ACTORS, TARGET_ACTORS)):
            _side_legend(ax, _actor_handles(names), names, handlelength=1.2)
        lax[4].set_axis_off()
    else:
        _bottom_legend(fig, True, "Pareto policy")

    if output_file is not None:
        save_manuscript_figure(fig, output_file)
    return fig
