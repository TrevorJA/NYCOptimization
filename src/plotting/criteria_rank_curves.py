"""
criteria_rank_curves.py - Manuscript Figure 6: robustness under criteria sets.

The RQ1 headline, in the Quinn et al. (2017, Fig. 8) idiom: one panel per
criterion set, each showing every design's policies SORTED by their
satisficing robustness under that set (rank on x, robustness on y). The read
is comparative and structural -- does the hazard-filling curve sit above the
i.i.d. control under every stakeholder framing (conclusion invariance), or
only under some? A final panel shows the cross-set Kendall tau_b ranking
agreement (the invariance check as a number), and the all-axes reference set
is drawn last, visually separated, as the degeneracy exhibit.

Data: the per-design ``robustness_scorecard_criteria.csv`` companions written
by ``robustness.run`` (no raw cubes needed), plus each design's
``robustness_criterion_stability.csv``. The incumbent line is added when the
cubes are locally available and silently omitted otherwise.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.plotting.layout import (WIDTH_DOUBLE_COL, design_legend_handles,
                                 panel_label, shared_legend)
from src.plotting.style import (DESIGN_ORDER, INCUMBENT_COLOR,
                                annotated_corr_heatmap, design_color,
                                save_manuscript_figure)
from src.satisficing_criteria import ALL_SETS


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
    """Incumbent per-set Starr fractions, when the cubes are local."""
    try:
        from src import results_data as rd
        results = ctx.results()
    except FileNotFoundError:
        return {}
    values: dict[str, float] = {}
    res = next((r for r in results.values() if r.incumbent is not None), None)
    if res is None:
        return {}
    for cset in ALL_SETS:
        sat = rd.incumbent_satisfaction(
            res, thresholds=rd.criterion_thresholds(res, cset))
        values[cset.key] = float(sat.all(axis=1).mean())
    return values


def fig_criteria_rank_curves(ctx, out_stub: Path, table_dir: Path) -> dict:
    """Sorted robustness-rank curves per criterion set + tau_b agreement.

    Panels (a)-(e): per criterion set, each design's policies sorted by
    ``sat_set__{key}`` descending; x = policy rank (fraction of that design's
    set, so differently-sized fronts are comparable), y = the satisficing
    fraction. The incumbent's value is a horizontal dashed line where
    available. The reference (all-axes) panel is shaded as the degeneracy
    exhibit. Final panel: mean cross-design Kendall tau_b between the
    rankings the sets induce.
    """
    cards = _criteria_scorecards(ctx)
    incumbent = _incumbent_by_set(ctx)
    designs = [d for d in DESIGN_ORDER if d in cards]
    sets = [c for c in ALL_SETS
            if any(f"sat_set__{c.key}" in cards[d].columns for d in designs)]

    n_panels = len(sets) + 1
    ncols = 3
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(WIDTH_DOUBLE_COL, WIDTH_DOUBLE_COL * 0.36 * nrows),
        constrained_layout=True)
    axes = np.atleast_2d(axes)

    rows = []
    for i, cset in enumerate(sets):
        ax = axes.flat[i]
        col = f"sat_set__{cset.key}"
        for d in designs:
            v = cards[d][col].dropna().sort_values(ascending=False).to_numpy()
            if not len(v):
                continue
            rank = np.arange(1, len(v) + 1) / len(v)
            ax.plot(rank, v, color=design_color(d), lw=1.4, zorder=3)
            rows += [{"criterion": cset.key, "design": d, "rank_frac": float(r),
                      "robustness": float(x)} for r, x in zip(rank, v)]
        if cset.key in incumbent:
            ax.axhline(incumbent[cset.key], color=INCUMBENT_COLOR, lw=1.2,
                       ls="--", zorder=2)
            rows.append({"criterion": cset.key, "design": "incumbent",
                         "rank_frac": np.nan,
                         "robustness": incumbent[cset.key]})
        if cset.reference:
            ax.set_facecolor("0.96")
            ax.set_title(f"{cset.label}", fontsize=9, style="italic")
        else:
            ax.set_title(cset.label, fontsize=9)
        panel_label(ax, chr(ord("a") + i))
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0, 1)
        ax.grid(color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        if i % ncols == 0:
            ax.set_ylabel("Fraction of E_test SOWs\nmeeting the set")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Policy rank (fraction of set)")

    # Cross-set ranking agreement: mean of the per-design tau_b matrices.
    ax_tau = axes.flat[len(sets)]
    taus = []
    for d in designs:
        path = (config.OUTPUTS_DIR / d / ctx.slug / "reeval" / ctx.tag
                / "robustness_criterion_stability.csv")
        if path.exists():
            taus.append(pd.read_csv(path, index_col=0))
    if taus:
        mean_tau = sum(t.to_numpy() for t in taus) / len(taus)
        keys = [c.replace("sat_set__", "") for c in taus[0].columns]
        annotated_corr_heatmap(ax_tau, mean_tau, keys,
                               label_fn=lambda k: k, fontsize=6,
                               vmin=-1, vmax=1)
        ax_tau.set_title("Ranking agreement between sets\n"
                         "(Kendall tau_b, mean over designs)", fontsize=8.5)
        pd.DataFrame(mean_tau, index=keys, columns=keys).to_csv(
            table_dir / "criteria_rank_agreement.csv")
    else:
        ax_tau.axis("off")
    panel_label(ax_tau, chr(ord("a") + len(sets)))
    for j in range(n_panels, axes.size):
        axes.flat[j].axis("off")

    shared_legend(fig, design_legend_handles(designs,
                                             incumbent=bool(incumbent)))
    save_manuscript_figure(fig, out_stub)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(table_dir / "criteria_rank_curves.csv",
                              index=False)
    return {"criteria": [c.key for c in sets], "designs": designs,
            "incumbent_included": bool(incumbent)}
