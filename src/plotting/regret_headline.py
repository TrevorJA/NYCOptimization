"""
regret_headline.py - Manuscript Figure 7: reoptimization vs the incumbent.

One panel, the RQ1 headline: each policy's All-Parties satisficing robustness
(the criterion set of figure 6 panel (d)) against its regret frequency -- the
share of SOWs in which it leaves the FFMP incumbent worse off beyond tolerance
on those same axes (``1 - no_harm_freq_tau``). Both are percentages of the
held-out E_test SOWs; the lower-right corner (robust, zero regret) is the
target. A faint cloud per scenario
design with the per-design non-dominated frontier drawn on top, and the
marginal distribution of each axis as a KDE per design (seaborn) along the
top and right edges, each curve scaled to unit peak (shape, not mass).

Data: the per-design ``robustness_scorecard_criteria.csv`` companions -- no
raw cubes needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

import config
from src.plotting.layout import shared_legend
from src.plotting.pareto_parallel import DESIGN_TITLES
from src.plotting.regret_summary import pareto_frontier
from src.plotting.style import (DESIGN_ORDER, design_color,
                                save_manuscript_figure)
from src.satisficing_criteria import criterion_by_key

#: Every character on the figure is drawn at >= this size (style guide).
FONTSIZE = 14

#: Canvas size in inches; the joint panel plus marginals, sized so 14 pt
#: text fits cleanly.
FIG_SIZE = (10.5, 8.5)

#: Marginal KDE strip height/width relative to the joint panel.
MARGINAL_RATIO = 0.22

#: The criterion set: the All-Parties set of figure 6 panel (d).
CRITERION_KEY = "compromise"


def _scale_last_line(ax, *, axis: str):
    """Rescale the most recently drawn KDE line to unit peak along ``axis``.

    Returns the rescaled ``(x, y)`` arrays so the caller can fill under it.
    """
    line = ax.get_lines()[-1]
    x, y = (np.asarray(v, dtype=float) for v in line.get_data())
    if axis == "y":
        y = y / np.nanmax(y)
    else:
        x = x / np.nanmax(x)
    line.set_data(x, y)
    return x, y


def fig_regret_vs_incumbent(ctx, out_stub: Path, table_dir: Path) -> dict:
    """All-Parties robustness vs regret frequency against the incumbent.

    x = ``sat_set__compromise``, y = ``1 - no_harm_freq_tau__compromise`` per
    policy, both as percentages of E_test SOWs; faint cloud per design,
    non-dominated frontier (max x, min y) on top, marginal KDEs per design
    along the top (robustness) and right (regret) edges. The lower-right
    corner is the RQ1 target: robust AND never leaves the Decree parties
    worse off than the status quo.
    """
    cset = criterion_by_key(CRITERION_KEY)
    xcol, ycol = f"sat_set__{cset.key}", f"no_harm_freq_tau__{cset.key}"

    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, MARGINAL_RATIO],
                          height_ratios=[MARGINAL_RATIO, 1.0],
                          wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    clouds: dict = {}

    rows = []
    designs_drawn = []
    n_policies: dict = {}
    for d in DESIGN_ORDER:
        path = (config.OUTPUTS_DIR / d / ctx.slug / "reeval" / ctx.tag
                / "robustness_scorecard_criteria.csv")
        if not path.exists():
            continue
        card = pd.read_csv(path, index_col="solution_id")
        if xcol not in card.columns or ycol not in card.columns:
            continue
        pts = 100.0 * card[[xcol, ycol]].dropna()
        pts[ycol] = 100.0 - pts[ycol]   # no-harm -> regret frequency
        if pts.empty:
            continue
        designs_drawn.append(d)
        n_policies[d] = len(pts)
        clouds[d] = pts
        color = design_color(d)
        ax.scatter(pts[xcol], pts[ycol], s=14, color=color, alpha=0.30,
                   lw=0, zorder=2)
        # pareto_frontier maximizes both axes and returns INDICES, not a
        # boolean mask; regret is minimized, so negate it.
        front_idx = pareto_frontier(pts[xcol].to_numpy(),
                                    -pts[ycol].to_numpy())
        front = pts.iloc[front_idx].sort_values(xcol)
        # Most-robust policies also sit at zero regret, so the frontier is
        # often a single point: draw its markers large enough to read.
        ax.plot(front[xcol], front[ycol], color=color, lw=2.4, marker="o",
                ms=10, mec="white", mew=1.0, zorder=4, solid_capstyle="round")
        on_front = np.zeros(len(pts), dtype=bool)
        on_front[front_idx] = True
        rows += [{"design": d, "solution_id": int(sid),
                  "robustness_pct": float(r.iloc[0]),
                  "regret_pct": float(r.iloc[1]),
                  "on_frontier": bool(f)}
                 for (sid, r), f in zip(pts.iterrows(), on_front)]

    if not designs_drawn:
        raise FileNotFoundError(
            f"no robustness_scorecard_criteria.csv with columns "
            f"{xcol}/{ycol} found for tag '{ctx.tag}' -- score the cubes "
            f"first (Anvil, `python -m src.robustness`)."
        )

    # Marginal KDEs: both metrics are bounded on [0, 100], so the density is
    # clipped there (no mass leaking past the edges). Each curve is rescaled
    # to unit peak: Historical piles up at 0% robustness and its raw density
    # spike would flatten the other designs' shapes to the baseline.
    for d in designs_drawn:
        kw = dict(color=design_color(d), lw=2.4, clip=(0.0, 100.0), cut=0,
                  common_norm=False, warn_singular=False)
        sns.kdeplot(x=clouds[d][xcol], ax=ax_top, **kw)
        x, y = _scale_last_line(ax_top, axis="y")
        ax_top.fill_between(x, 0, y, color=design_color(d), alpha=0.15, lw=0)
        sns.kdeplot(y=clouds[d][ycol], ax=ax_right, **kw)
        x, y = _scale_last_line(ax_right, axis="x")
        ax_right.fill_betweenx(y, 0, x, color=design_color(d), alpha=0.15,
                               lw=0)
    ax_top.set_ylim(0, 1.05)
    ax_right.set_xlim(0, 1.05)
    for m in (ax_top, ax_right):
        m.set_xlabel("")
        m.set_ylabel("")
        m.grid(False)
        for side in m.spines.values():
            side.set_visible(False)
    ax_top.set_yticks([])
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_right.set_xticks([])
    ax_right.tick_params(labelleft=False, left=False)

    ax.set_xlabel("All-Parties Robustness (%)", fontsize=FONTSIZE)
    ax.set_ylabel("Regret Frequency Relative to Current FFMP (%)",
                  fontsize=FONTSIZE)
    # x scaled to the DATA (rounded up to the next 10%): padding to 100%
    # crushes the cloud and the frontiers into a thin strip.
    xmax = max(r["robustness_pct"] for r in rows)
    xmax = int(np.ceil(max(xmax, 10.0) / 10.0) * 10)
    # Small symmetric padding keeps edge values (0% robustness, 0% regret)
    # off the spines; ticks still read 0 .. 100.
    ax.set_xlim(-0.02 * xmax, 1.015 * xmax)
    ax.set_ylim(-1.5, 101.5)
    ax.set_xticks(np.arange(0, xmax + 1, 10))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=FONTSIZE)
    ax.grid(color="0.90", lw=0.8)
    ax.set_axisbelow(True)
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_linewidth(1.2)

    handles = [Line2D([], [], color=design_color(d), lw=2.4, marker="o",
                      ms=10, mec="white", mew=1.0, label=f"{DESIGN_TITLES.get(d, d)} "
                                  f"(n = {n_policies[d]} policies)")
               for d in designs_drawn]
    y0 = ax.get_position().y0
    shared_legend(fig, handles, ncol=3, y=y0 - 0.09, fontsize=FONTSIZE)

    save_manuscript_figure(fig, out_stub)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(
        table_dir / f"regret_vs_incumbent_{cset.key}.csv", index=False)
    pd.DataFrame([{"criterion": cset.key, "objective": n, "threshold": t}
                  for n, t in cset.criteria.items()]).to_csv(
        table_dir / f"regret_vs_incumbent_{cset.key}_thresholds.csv",
        index=False)
    return {"criterion": cset.key, "designs": designs_drawn}
