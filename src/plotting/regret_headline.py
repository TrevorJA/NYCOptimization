"""
regret_headline.py - Manuscript Figure 7: reoptimization vs the incumbent.

The RQ1 headline in two panels: (a) the robustness / no-harm plane -- each
policy's focal-set satisficing fraction against the frequency with which it
avoids harming the FFMP incumbent beyond tolerance on the focal axes, with
per-design non-dominated frontiers highlighted -- and (b) the same no-harm
frequency traced across the tolerance ladder ``k`` (how the RQ1 verdict
depends on what counts as harm).

Data: the per-design ``robustness_scorecard_criteria.csv`` companions (panel
a) and the cross-design ``design_regret_tolerance_sweep.csv`` (panel b) --
no raw cubes needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import results_data as rd
from src.plotting.layout import (WIDTH_DOUBLE_COL, criteria_footer,
                                 design_legend_handles, shared_legend)
from src.plotting.regret_summary import pareto_frontier
from src.plotting.style import (DESIGN_ORDER, ETEST, design_color,
                                design_label, overlap_style,
                                save_manuscript_figure)
from src.satisficing_criteria import focal_criterion


def fig_regret_vs_incumbent(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Robustness vs no-harm plane + the tolerance sweep (RQ1 headline).

    Panel (a): x = ``sat_set__{focal}``, y = ``no_harm_freq_tau__{focal}``
    per policy; faint cloud per design, non-dominated frontier bold. The
    upper-right corner is the RQ1 target: robust AND never leaves the Decree
    parties worse off than the status quo. Panel (b): best-policy
    ``no_harm_freq_tau`` per design across the tolerance rungs ``k``.
    """
    focal = focal_criterion()
    xcol, ycol = f"sat_set__{focal.key}", f"no_harm_freq_tau__{focal.key}"

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(WIDTH_DOUBLE_COL, WIDTH_DOUBLE_COL * 0.42),
        constrained_layout=True)

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
        pts = card[[xcol, ycol]].dropna()
        if pts.empty:
            continue
        designs_drawn.append(d)
        n_policies[d] = len(pts)
        color = design_color(d)
        ax_a.scatter(pts[xcol], pts[ycol], s=9, color=color, alpha=0.30,
                     lw=0, zorder=2)
        # pareto_frontier returns INDICES, not a boolean mask. Treating them as
        # a mask silently selected the wrong rows for the frontier line AND
        # truncated the companion CSV to the frontier's length, because
        # zip() stops at the shorter sequence.
        front_idx = pareto_frontier(pts[xcol].to_numpy(), pts[ycol].to_numpy())
        front = pts.iloc[front_idx].sort_values(xcol)
        ax_a.plot(front[xcol], front[ycol], color=color, lw=1.8, marker="o",
                  ms=4, zorder=4)
        on_front = np.zeros(len(pts), dtype=bool)
        on_front[front_idx] = True
        rows += [{"design": d, "solution_id": int(sid),
                  "robustness": float(r.iloc[0]), "no_harm": float(r.iloc[1]),
                  "on_frontier": bool(f)}
                 for (sid, r), f in zip(pts.iterrows(), on_front)]

    if not designs_drawn:
        raise FileNotFoundError(
            f"no robustness_scorecard_criteria.csv with columns "
            f"{xcol}/{ycol} found for tag '{ctx.tag}' -- score the cubes "
            f"first (Anvil, `python -m src.robustness`)."
        )

    # Axis labels stay SHORT: the full definitions live in the caption and the
    # criteria footer. Sentence-length labels on a 3.7-inch panel overflow the
    # axes and collide with the neighbouring panel's label.
    ax_a.set_xlabel("Focal-set robustness")
    ax_a.set_ylabel("No-harm frequency\nvs incumbent")
    # Scale x to the DATA, not to the metric's theoretical 0-1 range: every
    # policy sits below ~0.4, and padding to 1.0 crushes the whole cloud --
    # and the frontiers that carry the panel's message -- into a thin strip.
    xmax = max((row["robustness"] for row in rows), default=1.0)
    ax_a.set_xlim(-0.01, max(xmax, 0.05) * 1.10)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.grid(color="0.92", lw=0.6)
    ax_a.set_axisbelow(True)
    # Letter in the title, as in figs 5 and 8: both panels' upper-left corners
    # hold data (the cloud saturates at no-harm = 1.0), so an inside label
    # lands on the points.
    ax_a.set_title("(a) Robustness vs no-harm plane")

    # ---- panel (b): the tolerance sweep -----------------------------------
    sweep_path = ctx.comparison_dir() / "design_regret_tolerance_sweep.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        kcol = next((c for c in ("k", "tau_k", "regret_tau_k")
                     if c in sweep.columns), None)
        vcol = next((c for c in ("no_harm_tau_best", "best")
                     if c in sweep.columns), None)
        if kcol and vcol:
            traces = {}
            for rank, d in enumerate(designs_drawn):
                sub = (sweep[sweep["design"] == d]
                       .groupby(kcol)[vcol].max().sort_index())
                traces[d] = sub
                # The designs routinely coincide exactly here; stagger the
                # STYLE so every series stays visible at its true value.
                ax_b.plot(sub.index, sub.to_numpy(), color=design_color(d),
                          **overlap_style(rank))
            ax_b.set_xlabel("Tolerance rung  $k$")
            ax_b.set_ylabel("Best policy's no-harm frequency")
            ax_b.set_ylim(-0.02, 1.06)
            ax_b.grid(color="0.92", lw=0.6)
            ax_b.set_axisbelow(True)
            # The flat-at-one case IS the result; state it in the title rather
            # than leaving the reader to wonder whether series are missing.
            stacked = np.concatenate([s.to_numpy() for s in traces.values()])
            flat = stacked.size and np.allclose(stacked, stacked[0])
            ax_b.set_title(
                f"(b) All designs: {stacked[0]:.2f} at every rung" if flat
                else "(b) Tolerance sweep")
        else:
            ax_b.axis("off")
    else:
        ax_b.axis("off")

    shared_legend(fig, design_legend_handles(designs_drawn, incumbent=False),
                  y=-0.02)

    obj_names, thresholds, kinds = rd.load_threshold_snapshot(ctx.tag, ctx.slug)
    counts = ", ".join(f"{n_policies[d]} {design_label(d).split(' (')[0].lower()}"
                       for d in designs_drawn)
    criteria_footer(
        fig, focal, focal.thresholds(thresholds, kinds), kinds, obj_names,
        y=-0.14,
        provenance=(f"Every Pareto-set policy per design ({counts}), "
                    f"re-evaluated on held-out {ETEST} SOWs. Both axes in (a) "
                    f"are fractions of those SOWs; bold lines are per-design "
                    f"non-dominated frontiers."))
    save_manuscript_figure(fig, out_stub)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(
        table_dir / f"regret_vs_incumbent_{focal.key}.csv", index=False)
    return {"criterion": focal.key, "designs": designs_drawn}
