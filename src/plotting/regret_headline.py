"""
regret_headline.py - Manuscript Figure 8: reoptimization vs the incumbent.

The RQ2 headline in two panels: (a) the robustness / no-harm plane -- each
policy's focal-set satisficing fraction against the frequency with which it
avoids harming the FFMP incumbent beyond tolerance on the focal axes, with
per-design non-dominated frontiers highlighted -- and (b) the same no-harm
frequency traced across the tolerance ladder ``k`` (how the RQ2 verdict
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
from src.plotting.layout import (WIDTH_DOUBLE_COL, design_legend_handles,
                                 panel_label, shared_legend)
from src.plotting.regret_summary import pareto_frontier
from src.plotting.style import DESIGN_ORDER, design_color, save_manuscript_figure
from src.satisficing_criteria import focal_criterion


def fig_regret_vs_incumbent(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Robustness vs no-harm plane + the tolerance sweep (RQ2 headline).

    Panel (a): x = ``sat_set__{focal}``, y = ``no_harm_freq_tau__{focal}``
    per policy; faint cloud per design, non-dominated frontier bold. The
    upper-right corner is the RQ2 target: robust AND never leaves the Decree
    parties worse off than the status quo. Panel (b): best-policy
    ``no_harm_freq_tau`` per design across the tolerance rungs ``k``.
    """
    focal = focal_criterion()
    xcol, ycol = f"sat_set__{focal.key}", f"no_harm_freq_tau__{focal.key}"

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(WIDTH_DOUBLE_COL, WIDTH_DOUBLE_COL * 0.46),
        constrained_layout=True)

    rows = []
    designs_drawn = []
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
        color = design_color(d)
        ax_a.scatter(pts[xcol], pts[ycol], s=9, color=color, alpha=0.30,
                     lw=0, zorder=2)
        on_front = pareto_frontier(pts[xcol].to_numpy(), pts[ycol].to_numpy())
        front = pts.iloc[np.flatnonzero(on_front)].sort_values(xcol)
        ax_a.plot(front[xcol], front[ycol], color=color, lw=1.8, marker="o",
                  ms=4, zorder=4)
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

    ax_a.set_xlabel(f"Fraction of E_test SOWs meeting the focal set "
                    f"({focal.label})")
    ax_a.set_ylabel("Fraction of E_test SOWs with no harm vs the\n"
                    "incumbent beyond tolerance (focal axes)")
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.grid(color="0.92", lw=0.6)
    ax_a.set_axisbelow(True)
    panel_label(ax_a, "a")

    # ---- panel (b): the tolerance sweep -----------------------------------
    sweep_path = ctx.comparison_dir() / "design_regret_tolerance_sweep.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        kcol = next((c for c in ("k", "tau_k", "regret_tau_k")
                     if c in sweep.columns), None)
        vcol = next((c for c in ("no_harm_tau_best", "best")
                     if c in sweep.columns), None)
        if kcol and vcol:
            for d in designs_drawn:
                sub = (sweep[sweep["design"] == d]
                       .groupby(kcol)[vcol].max().sort_index())
                ax_b.plot(sub.index, sub.to_numpy(), color=design_color(d),
                          lw=1.6, marker="o", ms=4)
            ax_b.set_xlabel("Tolerance rung  $k$  "
                            r"($\tau_i = k\,\max(\varepsilon_i,$ floor$_i)$)")
            ax_b.set_ylabel("Best policy's no-harm frequency")
            ax_b.set_ylim(-0.02, 1.02)
            ax_b.grid(color="0.92", lw=0.6)
            ax_b.set_axisbelow(True)
        else:
            ax_b.axis("off")
    else:
        ax_b.axis("off")
    panel_label(ax_b, "b")

    shared_legend(fig, design_legend_handles(designs_drawn, incumbent=False))
    save_manuscript_figure(fig, out_stub)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(
        table_dir / f"regret_vs_incumbent_{focal.key}.csv", index=False)
    return {"criterion": focal.key, "designs": designs_drawn}
