"""
src/plotting/style.py - Shared matplotlib style and label dictionaries.

All figure scripts import from here to ensure consistent aesthetics across
manuscript figures and diagnostic plots.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.text import Text

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
# Scenario-design metadata
# ---------------------------------------------------------------------------

#: Campaign scenario designs in canonical display order (ensemble designs first,
#: the historic reference trace last).
DESIGN_ORDER: tuple = ("monte_carlo", "hazard_filling_stationary", "historic")

#: Per-design plotting identity: Okabe-Ito colors keyed to the DESIGN (never to
#: plot order; the palette is validated for CVD + normal-vision separation) plus
#: the display name and reference-trace flag.
DESIGN_STYLE: dict[str, dict] = {
    "monte_carlo": {
        "color": "#0072B2", "label": "Monte Carlo Sampling (MC)",
        "reference": False},
    "hazard_filling_stationary": {
        "color": "#D55E00", "label": "Hazard Filling (HF)",
        "reference": False},
    "historic": {
        "color": "#B0B0B0", "label": "Historical (HIST)",
        "reference": True},
}

#: Fallback colors for designs outside the campaign trio (assigned by sorted
#: name so the mapping is deterministic across runs, not by plot order).
DESIGN_FALLBACK_COLORS: list[str] = ["#56B4E9", "#CC79A7", "#E69F00"]

#: Reserved non-design colors: the FFMP incumbent (status quo) polyline/marker
#: and the satisficing-threshold reference line. Never reuse for a design.
INCUMBENT_COLOR: str = "firebrick"
THRESHOLD_COLOR: str = "crimson"

#: Typeset name of the held-out test ensemble. One spelling everywhere: a raw
#: "E_test" in a label renders the underscore literally in some panels and as a
#: subscript in others, which reads as two different quantities.
ETEST: str = r"$E_\mathrm{test}$"


def overlap_style(rank: int) -> dict:
    """Line kwargs that keep EXACTLY coincident series individually visible.

    When several designs share an identical trace (e.g. every design pinned at
    a low-regret frequency of 1.0), plain solid lines hide all but the one drawn
    last, and a reader cannot distinguish "they agree" from "the other series
    are missing". Staggering dash phase, marker and width shows every series at
    its true position -- never offset the DATA to fake separation.

    Args:
        rank: Index of the series within the overlapping group.

    Returns:
        Kwargs for ``Axes.plot``.
    """
    dashes = [(None, None), (5, 2), (1, 2)][rank % 3]
    return {
        "dashes": dashes,
        "marker": ["o", "s", "^"][rank % 3],
        "markersize": [7.0, 5.0, 3.2][rank % 3],
        "markerfacecolor": "none",
        "lw": [2.6, 1.8, 1.1][rank % 3],
    }


def design_style(design: str, fallback_rank: int = 0) -> dict:
    """Entity-stable style dict for ``design`` (deterministic fallback if unknown)."""
    if design in DESIGN_STYLE:
        return DESIGN_STYLE[design]
    return {"color": DESIGN_FALLBACK_COLORS[fallback_rank % len(DESIGN_FALLBACK_COLORS)],
            "label": design.replace("_", " "), "reference": False}


def design_color(design: str) -> str:
    """Okabe-Ito color for ``design``."""
    return design_style(design)["color"]


def design_label(design: str) -> str:
    """Display name for ``design``."""
    return design_style(design)["label"]


# ---------------------------------------------------------------------------
# Objective labels
# ---------------------------------------------------------------------------

# Objective display labels name the metric, its statistic, and its timescale.
# Each objective is keyed by both names: the whole-trace (§1) metric carries its
# native timescale; the annual-unit (§2, FFMP-year) search metric is marked
# "annual". Labels differ between the two reductions only where the
# timescale/statistic differs.

#: Compact single-line objective labels; ``label_for`` falls back to the raw name.
OBJECTIVE_LABELS: dict[str, str] = {
    # NYC diversion: satisficing reliability; CVaR90 of the deficit (% of Decree)
    "nyc_delivery_reliability_weekly":  "NYC Diversion Reliability (weekly)",
    "nyc_delivery_reliability_annual":  "NYC Diversion Reliability (annual)",
    "nyc_delivery_deficit_cvar90_pct":  "NYC Diversion Deficit (weekly CVaR90, %)",
    "nyc_delivery_deficit_p99_pct":     "NYC Diversion Deficit (P99 of annual CVaR90, %)",
    "nyc_delivery_deficit_max_pct":     "NYC Diversion Deficit (weekly max, %)",
    # NJ diversion
    "nj_delivery_reliability_weekly":   "NJ Diversion Reliability (weekly)",
    "nj_delivery_reliability_annual":   "NJ Diversion Reliability (annual)",
    # Montague Decree flow
    "montague_flow_reliability_weekly": "Montague Flow Reliability (weekly)",
    "montague_flow_reliability_annual": "Montague Flow Reliability (annual)",
    "montague_flow_deficit_cvar90_pct": "Montague Flow Deficit (weekly CVaR90, %)",
    "montague_flow_deficit_p99_pct":    "Montague Flow Deficit (P99 of annual CVaR90, %)",
    "montague_flow_deficit_max_pct":    "Montague Flow Deficit (weekly max, %)",
    # Trenton flow target
    "trenton_flow_deficit_cvar90_pct":  "Trenton Flow Deficit (weekly CVaR90, %)",
    "trenton_flow_deficit_p99_pct":     "Trenton Flow Deficit (P99 of annual CVaR90, %)",
    "trenton_flow_reliability_weekly":  "Trenton Flow Reliability (weekly)",
    "trenton_flow_reliability_annual":  "Trenton Flow Reliability (annual)",
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
    "trenton_flow_deficit_cvar90_pct":  "Trenton Def. CVaR90 %",
    "trenton_flow_deficit_p99_pct":     "Trenton Def. P99 %",
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


# Two objective naming conventions only: the long form (`label_for` /
# OBJECTIVE_LABELS) and the abbreviation (`short_label_for` / OBJ_SHORT_LABELS).
# Any other rendering (e.g. the multi-line parallel-axis label) is derived from
# the long form.

def objective_direction(name: str) -> str:
    """The optimization direction of an objective, by name.

    Every registered objective follows the naming rule "reliability and
    storage are maximized; deficits, floods, and exceedances are minimized",
    so the direction is derivable from the name alone -- which lets label
    helpers work for both the weekly (whole-trace) and annual (search)
    metric families without importing either registry.
    """
    return ("maximize" if ("reliability" in name or "storage" in name)
            else "minimize")


def axis_label_for(name: str, direction: str = None) -> str:
    """Multi-line parallel-axis label derived from the long-form label.

    Splits the long form at its "(...)" qualifier and appends the
    optimization direction, e.g. ``NYC Diversion Reliability\n(annual)\n(max)``.
    The SOLE parallel-axis label convention -- derived from ``label_for``,
    never a third hand-written set.

    Args:
        name: Objective name.
        direction: "maximize" or "minimize"; defaults to
            :func:`objective_direction`.
    """
    long = label_for(name)
    direction = direction or objective_direction(name)
    arrow = "(max)" if direction == "maximize" else "(min)"
    if "(" in long:
        head, _, tail = long.partition("(")
        return f"{head.strip()}\n({tail}\n{arrow}"
    return f"{long}\n{arrow}"

# ---------------------------------------------------------------------------
# Robustness / factor-map color tokens
# ---------------------------------------------------------------------------

#: Sequential colormap for robustness MAGNITUDE (fraction of SOWs satisficing)
#: wherever robustness colors a mark: one hue family, light -> dark, CVD-safe.

#: Diverging colormap for success/failure probability surfaces (factor maps):
#: red (fail) -> neutral at P = 0.5 -> blue (success); the P = 0.5 contour is
#: drawn explicitly so the boundary never relies on hue.
FACTOR_MAP_CMAP = "RdBu"

#: Factor-map SOW scatter marks: luminance- and shape-separated, each with a
#: contrasting outline so it reads over both ends of the diverging field.
FACTOR_MAP_MARKS = {
    "success": {"marker": "o", "facecolor": "white", "edgecolor": "0.25"},
    "failure": {"marker": "X", "facecolor": "0.10", "edgecolor": "white"},
}

# ---------------------------------------------------------------------------
# Scatter pair definitions
# ---------------------------------------------------------------------------

#: Six pairwise scatter pairs (0-based indices into the 8-objective active
#: set: 0 NYC Rel, 1 NYC Def, 2 Montague Rel, 3 Montague Def, 4 Trenton Def,
#: 5 Flood Days, 6 Storage, 7 NJ Rel) for SI diagnostic plots.
SCATTER_PAIRS: list[tuple[int, int]] = [
    (0, 2),   # NYC Rel. vs Montague Rel.
    (1, 3),   # NYC Deficit vs Montague Deficit
    (4, 5),   # Trenton Deficit vs Flood Days
    (0, 6),   # NYC Rel. vs Storage
    (0, 7),   # NYC Rel. vs NJ Rel. (the two Decree diversion parties)
    (1, 5),   # NYC Deficit vs Flood Days
]

# ---------------------------------------------------------------------------
# Figure size presets (width, height) in inches
# ---------------------------------------------------------------------------

FIGSIZE_SINGLE = (7, 5)
FIGSIZE_WIDE   = (13, 5)

# ---------------------------------------------------------------------------
# Shared rcParams
# ---------------------------------------------------------------------------

# The ONE figure style of the project. Every figure, in every tier (manuscript,
# SI, diagnostic), takes its type sizes, weights, spines and resolution from
# :func:`apply_style` and is written by :func:`save_figure` or
# :func:`save_manuscript_figure`. Figure code never passes a font size: text
# inherits these rcParams, and a figure that needs larger type than the rest
# raises the floor for its whole body with ``plt.rc_context(font_rc(14))``.
# ``tests/test_figure_style.py`` enforces both rules on the source.

#: Smallest type size permitted in any figure (points), at the figure's own size.
MIN_FONTSIZE: int = 12


def font_rc(min_fontsize: int = MIN_FONTSIZE) -> dict:
    """Type-size rcParams of the project style with ``min_fontsize`` as the floor.

    :func:`apply_style` applies these at :data:`MIN_FONTSIZE`. A figure that
    needs larger type wraps its plotting (through the save) in
    ``plt.rc_context(font_rc(14))``: the floor is raised in one place, never by
    individual ``fontsize`` arguments.

    Args:
        min_fontsize: Smallest type size in the figure (points).

    Returns:
        rcParams mapping for every text class matplotlib sizes.

    Raises:
        ValueError: If ``min_fontsize`` is below :data:`MIN_FONTSIZE`.
    """
    if min_fontsize < MIN_FONTSIZE:
        raise ValueError(
            f"min_fontsize={min_fontsize} is below the project floor of {MIN_FONTSIZE} pt")
    fs = min_fontsize
    return {
        "font.size":             fs,
        "axes.titlesize":        fs + 1,
        "axes.labelsize":        fs,
        "xtick.labelsize":       fs,
        "ytick.labelsize":       fs,
        "legend.fontsize":       fs,
        "legend.title_fontsize": fs,
        "figure.titlesize":      fs + 2,
    }


#: Raster resolution (dpi) of saved figures: every tier, and the finer setting
#: :func:`save_manuscript_figure` writes the main-text figures at.
FIGURE_DPI: int = 300
MANUSCRIPT_FIGURE_DPI: int = 400


def apply_style() -> None:
    """Apply the project figure style (type >= 12 pt, no bold weights).

    Call once before any plotting call: at the top of a figure script's
    ``main()``, or in the figure runner for registry-built figures.
    """
    plt.rcParams.update({
        **font_rc(),
        "font.family":        "sans-serif",
        "font.weight":        "normal",
        "axes.titleweight":   "normal",
        "axes.labelweight":   "normal",
        "figure.titleweight": "normal",
        "figure.dpi":         150,
        "savefig.dpi":        FIGURE_DPI,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        # Vector text stays editable text (not outlines) in the PDF, which
        # journal production systems require.
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


# ---------------------------------------------------------------------------
# Axis furniture
# ---------------------------------------------------------------------------

#: Recessive axis furniture: spines and ticks are thin and grey so the data
#: carries the ink.
AXIS_COLOR: str = "0.35"
TICK_COLOR: str = "0.2"
AXIS_LINEWIDTH: float = 0.9
TICK_LENGTH: float = 3.5


def style_axes(ax, box: bool = False) -> None:
    """Recessive axes: thin grey spines and ticks, no grid.

    Resets the major tick length on both axes, so call it BEFORE any per-axis
    ``tick_params`` override.

    Args:
        ax: Target axes.
        box: Draw the full frame; ``False`` keeps the left and bottom spines only.
    """
    for side, spine in ax.spines.items():
        spine.set_visible(box or side in ("left", "bottom"))
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(colors=TICK_COLOR, length=TICK_LENGTH, width=AXIS_LINEWIDTH)
    ax.grid(False)


def wrap_label(text: str) -> str:
    """Break ``text`` into two balanced lines at a space outside parentheses.

    Ties go to the later break, which leaves a trailing unit on its own line.
    """
    depth, breaks = 0, []
    for i, ch in enumerate(text):
        depth += (ch == "(") - (ch == ")")
        if ch == " " and depth == 0:
            breaks.append(i)
    if not breaks:
        return text
    i = min(breaks, key=lambda b: (max(b, len(text) - b - 1), -b))
    return f"{text[:i]}\n{text[i + 1:]}"


def fit_ylabel(ax, text: str) -> None:
    """Set the y label, on two lines when one line is longer than the axis.

    Measures the rendered label against the axes box, so call it once the
    axes position and the type size are final.
    """
    label = ax.set_ylabel(text)
    if label.get_window_extent().height > ax.get_window_extent().height:
        label.set_text(wrap_label(text))


# ---------------------------------------------------------------------------
# Shared figure helpers
# ---------------------------------------------------------------------------

def enforce_min_fontsize(fig) -> int:
    """Raise every visible text in ``fig`` that is below :data:`MIN_FONTSIZE` to it.

    The save helpers call this, so the floor holds for a figure whatever
    rcParams were active when it was built (a module run without
    :func:`apply_style`, or text created by a third-party helper). A raise
    is reported as a warning, because it marks figure code to correct.

    Args:
        fig: Matplotlib figure.

    Returns:
        Number of text artists whose size was raised.
    """
    # Tick and offset labels get their text at draw time, so draw first.
    fig.canvas.draw()
    raised = 0
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis, *([ax.zaxis] if hasattr(ax, "zaxis") else [])):
            labels = axis.get_ticklabels(which="both")
            if any(t.get_text() and t.get_fontsize() < MIN_FONTSIZE for t in labels):
                # Through tick_params, so ticks created by a later draw keep the floor.
                axis.set_tick_params(which="both", labelsize=MIN_FONTSIZE)
                raised += 1
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text() and text.get_fontsize() < MIN_FONTSIZE:
            text.set_fontsize(MIN_FONTSIZE)
            raised += 1
    if raised:
        warnings.warn(
            f"{raised} text artist(s) below {MIN_FONTSIZE} pt were raised to the floor; "
            "call style.apply_style() before plotting and pass no font sizes.",
            stacklevel=3)
    return raised


def _save(fig, out_stub, formats: tuple, **savefig_kwargs) -> list:
    """Write ``fig`` to ``{out_stub}.{ext}`` per format after enforcing the type floor."""
    stub = Path(out_stub)
    stub.parent.mkdir(parents=True, exist_ok=True)
    enforce_min_fontsize(fig)
    written = []
    for ext in formats:
        path = stub.with_suffix(f".{ext}")
        fig.savefig(path, **savefig_kwargs)
        written.append(path)
    return written


#: Output formats for SI and diagnostic figures. PNG only for now (drop vector copies).
FIGURE_FORMATS: tuple = ("png",)

#: Output formats for main-manuscript figures. PNG only until the
#: manuscript-final pass extends this tuple to ("png", "pdf").
MANUSCRIPT_FIGURE_FORMATS: tuple = ("png",)


def save_figure(fig, out_stub) -> list:
    """Save an SI or diagnostic figure in every :data:`FIGURE_FORMATS` format.

    The single way a figure is written outside the main-manuscript tier: it
    enforces :data:`MIN_FONTSIZE` and takes resolution and bounding box from
    the rcParams of :func:`apply_style`.

    Args:
        fig: Matplotlib figure.
        out_stub: Path or str without an extension (any existing suffix is replaced).

    Returns:
        The written paths, in :data:`FIGURE_FORMATS` order.
    """
    return _save(fig, out_stub, FIGURE_FORMATS)


def save_manuscript_figure(fig, out_stub) -> list:
    """Save a main-manuscript figure in every :data:`MANUSCRIPT_FIGURE_FORMATS` format.

    As :func:`save_figure`, at :data:`MANUSCRIPT_FIGURE_DPI`.

    Args:
        fig: Matplotlib figure.
        out_stub: Path or str without an extension (any existing suffix is replaced).

    Returns:
        The written paths, in :data:`MANUSCRIPT_FIGURE_FORMATS` order.
    """
    return _save(fig, out_stub, MANUSCRIPT_FIGURE_FORMATS, dpi=MANUSCRIPT_FIGURE_DPI)


def criterion_condition(name: str, threshold: float, kind: str) -> str:
    """One satisficing condition as compact text, e.g. ``NYC Rel. (ann) >= 0.65``.

    A non-finite threshold (the axis-disabling convention of
    ``src.results_data.relax_axes``) renders as "no requirement".
    """
    label = short_label_for(name)
    if threshold is None or not np.isfinite(threshold):
        return f"{label}: no requirement"
    op = "≥" if kind == "ge" else "≤"
    unit = ""
    if name.endswith("_pct"):
        unit = "%"
        label = label.removesuffix(" %")
    elif "flood_exceedance" in name:
        unit = " ft·d/yr"
    return f"{label} {op} {threshold:g}{unit}"


def criteria_lines(thresholds: dict, kinds: dict, obj_order=None,
                   header: str = "Satisficing criteria (all must hold):") -> list:
    """The criterion vector as an explicit bulleted text block.

    Every results figure that depends on a satisficing criterion carries this
    in its footer (project rule: the exact thresholds are stated on the figure
    as one bullet per condition, never just a shorthand criterion name).

    Axes with a non-finite/absent threshold (the non-binding convention of
    subset criterion sets) are collapsed into one trailing "unconstrained"
    line rather than bulleted individually, so a 2-axis criterion reads as
    two conditions, not two conditions and six "no requirement" bullets.
    """
    names = list(obj_order) if obj_order is not None else list(thresholds)
    bound = [n for n in names
             if thresholds.get(n) is not None and np.isfinite(thresholds[n])]
    lines = [header] + [
        f"  •  {criterion_condition(n, thresholds[n], kinds[n])}" for n in bound
    ]
    unbound = [n for n in names if n not in bound]
    if unbound:
        lines.append(f"  •  other axes unconstrained "
                     f"({len(unbound)} of {len(names)})")
    return lines


def add_figure_footer(fig, lines, *, x: float = 0.5, y: float = 0.0,
                      ha: str = "center") -> None:
    """Attach a boxed provenance/criteria footer below a figure, at the rcParams type size.

    Args:
        fig: The figure.
        lines: Text lines (policy provenance first, then the bulleted
            criterion block from :func:`criteria_lines`).
        x: Figure-fraction x of the box anchor (with ``ha``); several boxes
            can sit side by side for multi-criterion figures.
        y: Figure-fraction y of the box top; tune per figure so it clears
            axis labels and legends (``savefig.bbox='tight'`` keeps it in
            frame even at negative y).
        ha: Horizontal anchor of the box at ``x``.
    """
    fig.text(x, y, "\n".join(lines), ha=ha, va="top",
             linespacing=1.55, color="0.15",
             multialignment="left",
             bbox=dict(boxstyle="round,pad=0.55", facecolor="0.965",
                       edgecolor="0.75", lw=0.8))


def annotated_corr_heatmap(ax, data, labels, *, label_fn=label_for,
                           box_threshold=None,
                           vmin: float = -1.0, vmax: float = 1.0):
    """Draw an annotated correlation/agreement heatmap on ``ax``, at the rcParams type size.

    Shared by the redundancy (Spearman) and operator-agreement (Kendall tau_b)
    diagnostics. NaN cells render grey; cells with ``|value| > box_threshold``
    (off-diagonal) are outlined.

    Args:
        ax: Target axes.
        data: Square 2-D array of correlation/agreement values.
        labels: Row/column names (length matches ``data``).
        label_fn: Maps a name to its tick label (default :func:`label_for`).
        box_threshold: If set, outline off-diagonal cells exceeding it.
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
    ax.set_xticklabels([label_fn(n) for n in labels], rotation=45, ha="right")
    ax.set_yticklabels([label_fn(n) for n in labels])
    for i in range(m):
        for j in range(m):
            v = arr[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.55 else "black")
            if box_threshold is not None and i != j and abs(v) > box_threshold:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=1.6))
    return im
