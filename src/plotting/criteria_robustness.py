"""
criteria_robustness.py - Manuscript figure 6: robustness rankings under each
satisficing criterion set.

One row per named criterion set (Quinn et al. 2017, Fig. 8 idiom): every
design's Pareto-approximate policies sorted by their satisficing robustness
under that set -- rank on x, the percentage of held-out E_test SOWs meeting
the set on y. Rank is the ABSOLUTE position within each design's own set, so
a curve ends where that design's Pareto set ends and the differently sized
fronts are read directly off the x-axis. A framed box beside each panel
states the set's criteria in words; the FFMP incumbent's E_test robustness
under the same set is the dashed horizontal reference.

Data: the per-design ``robustness_scorecard_criteria.csv`` companions written
by ``robustness.run`` (no raw cubes needed for the curves) plus the
``reeval_raw_meta.json`` threshold snapshot for the criteria text. The
incumbent line needs the cubes and is omitted silently when they are absent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import config
from src import results_data as rd
from src.plotting.layout import shared_legend
from src.plotting.pareto_parallel import DESIGN_TITLES
from src.plotting.style import (DESIGN_ORDER, INCUMBENT_COLOR, ETEST,
                                design_color, save_manuscript_figure,
                                short_label_for)
from src.satisficing_criteria import NAMED_SETS

#: Every character on the figure is drawn at >= this size (style guide).
FONTSIZE = 14

#: Canvas width in inches; the criteria box takes the right ~30%.
FIG_WIDTH = 13.5

#: Height per criterion row, inches.
ROW_HEIGHT = 3.0

#: Legend label for the incumbent reference line (E_test-evaluated, so it is
#: the same value for every design within a row).
INCUMBENT_LEGEND = f"Current FFMP policy evaluated on {ETEST}"

#: Two-line y-axis names per criterion set, in the manuscript's words.
_SET_YLABELS = {
    "nyc_supply": "NYC Supply\nRobustness (%)",
    "downstream_flows": "Downstream Flow\nRobustness (%)",
    "flood": "Flood Exposure\nRobustness (%)",
    "compromise": "All-Parties\nRobustness (%)",
}


def _criteria_scorecards(ctx) -> dict[str, pd.DataFrame]:
    """Each design's per-set criteria scorecard, keyed by design."""
    out = {}
    for design in DESIGN_ORDER:
        path = (config.OUTPUTS_DIR / design / ctx.slug / "reeval" / ctx.tag
                / "robustness_scorecard_criteria.csv")
        if path.exists():
            out[design] = pd.read_csv(path, index_col="solution_id")
    if not out:
        raise FileNotFoundError(
            f"no robustness_scorecard_criteria.csv found for tag '{ctx.tag}' "
            f"-- run `python -m src.robustness` per design first (Anvil)."
        )
    return out


def _incumbent_by_set(ctx) -> dict[str, float]:
    """Incumbent per-set Starr fraction on E_test, when the cubes are local."""
    try:
        results = ctx.results()
    except FileNotFoundError:
        return {}
    res = next((r for r in results.values() if r.incumbent is not None), None)
    if res is None:
        return {}
    values = {}
    for cset in NAMED_SETS:
        sat = rd.incumbent_satisfaction(
            res, thresholds=rd.criterion_thresholds(res, cset))
        values[cset.key] = float(sat.all(axis=1).mean())
    return values


def _condition(name: str, threshold: float, kind: str) -> str:
    """One condition in the style guide's number convention.

    Unit-scale thresholds print as x.xx, percent-scale and ft·d/yr ones to
    the hundredth only if needed (``:g``); the abbreviation convention
    (``short_label_for``) names the axis.
    """
    label = short_label_for(name)
    op = "≥" if kind == "ge" else "≤"
    if name.endswith("_pct"):
        return f"{label.removesuffix(' %')} {op} {threshold:g}%"
    if "flood_exceedance" in name:
        return f"{label} {op} {threshold:.2f} ft·d/yr"
    return f"{label} {op} {threshold:.2f}"


def _criteria_text(cset, kinds: dict) -> str:
    """The framed box body: a heading plus one condition per member axis."""
    lines = ["Satisficing criteria:"]
    lines += [_condition(name, thr, kinds[name])
              for name, thr in cset.criteria.items()]
    return "\n".join(lines)


def fig_criteria_robustness(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Sorted robustness-rank curves, one row per named criterion set.

    Rows (a)-(d): the four named sets in registry order. Within a row each
    design's policies are sorted by ``sat_set__{key}`` descending; x is the
    absolute rank (1 = most robust), y the percentage of E_test SOWs meeting
    the set. The incumbent's value is a dashed horizontal line.
    """
    cards = _criteria_scorecards(ctx)
    incumbent = _incumbent_by_set(ctx)
    designs = [d for d in DESIGN_ORDER if d in cards]
    sets = [c for c in NAMED_SETS
            if any(f"sat_set__{c.key}" in cards[d].columns for d in designs)]
    _, _, kinds = rd.load_threshold_snapshot(ctx.tag, ctx.slug)

    n_max = max(len(cards[d]) for d in designs)
    xmax = int(np.ceil(n_max / 100.0) * 100)

    fig, axes = plt.subplots(
        len(sets), 2, figsize=(FIG_WIDTH, ROW_HEIGHT * len(sets)),
        gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.05,
                     "hspace": 0.32})
    axes = np.atleast_2d(axes)

    rows = []
    for i, (cset, (ax, box)) in enumerate(zip(sets, axes)):
        col = f"sat_set__{cset.key}"
        for d in designs:
            v = cards[d][col].dropna().sort_values(ascending=False).to_numpy()
            if not len(v):
                continue
            rank = np.arange(1, len(v) + 1)
            ax.plot(rank, 100.0 * v, color=design_color(d), lw=2.4, zorder=3,
                    solid_capstyle="round")
            rows += [{"criterion": cset.key, "design": d, "rank": int(r),
                      "rank_frac": float(r / len(v)),
                      "robustness_pct": float(100.0 * x)}
                     for r, x in zip(rank, v)]
        if cset.key in incumbent:
            ax.axhline(100.0 * incumbent[cset.key], color=INCUMBENT_COLOR,
                       lw=2.0, ls=(0, (5, 3)), zorder=2)
            rows.append({"criterion": cset.key, "design": "incumbent",
                         "rank": -1, "rank_frac": np.nan,
                         "robustness_pct": 100.0 * incumbent[cset.key]})

        ax.set_title(f"({chr(ord('a') + i)}) {cset.label}", loc="left",
                     fontsize=FONTSIZE)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(np.arange(0, xmax + 1, 200))
        ax.tick_params(labelsize=FONTSIZE)
        ax.set_ylabel(_SET_YLABELS.get(cset.key, f"Robustness (%)"),
                      fontsize=FONTSIZE)
        ax.grid(axis="y", color="0.90", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
        for side in ax.spines.values():
            side.set_linewidth(1.2)
        if i == len(sets) - 1:
            ax.set_xlabel("Policy rank (most to least robust)",
                          fontsize=FONTSIZE)
        else:
            ax.tick_params(labelbottom=False)

        # The criteria box: a framed axes with no ticks, text top-left.
        box.set_xticks([])
        box.set_yticks([])
        for side in box.spines.values():
            side.set_visible(True)
            side.set_linewidth(1.2)
        box.text(0.04, 0.5, _criteria_text(cset, kinds), fontsize=FONTSIZE,
                 ha="left", va="center", transform=box.transAxes,
                 linespacing=1.6)

    handles = [Line2D([], [], color=design_color(d), lw=2.4,
                      label=f"{DESIGN_TITLES.get(d, d)} "
                            f"(n = {len(cards[d])} policies)")
               for d in designs]
    if incumbent:
        handles.append(Line2D([], [], color=INCUMBENT_COLOR, lw=2.0,
                              ls=(0, (5, 3)), label=INCUMBENT_LEGEND))
    # Legend top anchored a fixed gap below the bottom panel's x-label.
    y0 = axes[-1, 0].get_position().y0
    shared_legend(fig, handles, ncol=2, y=y0 - 0.06, fontsize=FONTSIZE)

    save_manuscript_figure(fig, out_stub)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(table_dir / "criteria_robustness_curves.csv",
                              index=False)
    pd.DataFrame([{"criterion": c.key, "objective": n, "kind": kinds[n],
                   "threshold": t}
                  for c in sets for n, t in c.criteria.items()]).to_csv(
        table_dir / "criteria_robustness_thresholds.csv", index=False)
    return {"criteria": [c.key for c in sets], "designs": designs,
            "incumbent_included": bool(incumbent)}
