"""
satisficing_diagnostics.py - Phase-1 results figures: satisficing effects and
decomposition on the held-out E_test re-evaluation.

Four figures that make the joint (Starr) satisficing criterion's behavior
legible -- which thresholds bind, how satisficing decomposes by objective and
by design, how the joint fraction collapses as axes are conjoined, and why
individual SOWs are unattainable. All are pure post-processing on the persisted
per-SOW cubes (``src.results_data``); the adopted thresholds come from each
run's ``reeval_raw_meta.json`` snapshot, never live config.

Shared visual conventions (style.py): Okabe-Ito colors keyed to the DESIGN;
firebrick = FFMP incumbent (status quo); crimson dashed = adopted threshold;
solid = best across policies, dotted/open = median across policies. No in-panel
numeric annotations -- every figure writes a companion CSV with the numbers.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src import results_data as rd
from src.satisficing_criteria import criterion_by_key
from src.plotting.style import (
    DESIGN_ORDER,
    ETEST,
    INCUMBENT_COLOR,
    THRESHOLD_COLOR,
    add_figure_footer,
    criteria_lines,
    design_color,
    design_label,
    label_for,
    save_figure,
    short_label_for,
)

#: Display label for the status-quo policy in legends.
INCUMBENT_LABEL = "FFMP incumbent (status quo)"


def _reference_criterion():
    """The all-axes reference (adopted snapshot) these diagnostics decompose."""
    return criterion_by_key("reference_all8")


def _policies_line(results: dict) -> str:
    """Provenance line: which policies the results are based on."""
    designs = _designs(results)
    counts = " / ".join(
        f"{results[d].raw.cube.shape[0]} {design_label(d).split(' (')[0].lower()}"
        for d in designs)
    g = results[designs[0]].raw.n_sow
    return (f"Policies: every ε-refiltered Pareto-set policy per design "
            f"({counts}) and the FFMP incumbent, re-evaluated on {g} "
            f"held-out {ETEST} SOWs.")


def _add_footer(results: dict, fig, *, y: float,
                criteria: dict | None = None,
                criteria_header: str = "Satisficing criteria (all must hold):",
                policies: str | None = None) -> None:
    """Provenance + explicit-criteria footer shared by every results figure."""
    first = results[_designs(results)[0]].raw
    lines = [policies or _policies_line(results), ""]
    lines += criteria_lines(criteria or first.thresholds, first.kinds,
                            obj_order=first.obj_names, header=criteria_header)
    add_figure_footer(fig, lines, y=y)


def _designs(results: dict) -> list[str]:
    """Loaded designs in canonical display order."""
    return [d for d in DESIGN_ORDER if d in results]


def _design_legend(results: dict, incumbent: bool = True) -> list[Line2D]:
    handles = [Line2D([], [], color=design_color(d), lw=2.4, label=design_label(d))
               for d in _designs(results)]
    if incumbent:
        handles.append(Line2D([], [], color=INCUMBENT_COLOR, lw=2.4,
                              label=INCUMBENT_LABEL))
    return handles


###############################################################################
# P1.1 -- univariate satisficing decomposition
###############################################################################

def fig_satisficing_decomposition(results: dict, out_stub: Path,
                                  table_dir: Path) -> dict:
    """Per-axis satisficing fractions: policy range, best/median, incumbent.

    One row per objective; within it, one horizontal min-max span across each
    design's Pareto policies with markers at the best and median policy, plus
    the incumbent's per-axis fraction. Shows in one view which axes are
    saturated (non-discriminating), which are near-unsatisfiable, and where
    the designs differ.
    """
    crit = _reference_criterion()
    designs = _designs(results)
    obj_names = results[designs[0]].raw.obj_names
    m = len(obj_names)
    offsets = np.linspace(0.27, -0.27, num=len(designs))

    rows = []
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    for di, d in enumerate(designs):
        res = results[d]
        thr = rd.criterion_thresholds(res, crit)
        uni = rd.univariate_fraction(rd.satisfaction(res.raw, thresholds=thr))
        if thr == res.raw.thresholds:
            # Consistency guard against the persisted scorecard, valid only
            # when the analysis criterion IS the snapshot criterion.
            sc = res.scorecard[[f"sat_uni_sow__{n}" for n in obj_names]].to_numpy()
            if not np.allclose(np.sort(uni, axis=0), np.sort(sc, axis=0),
                               atol=1e-9, equal_nan=True):
                raise AssertionError(f"{d}: cube-derived satisficing != scorecard")
        inc_sat = rd.incumbent_satisfaction(res, thresholds=thr)
        inc_uni = inc_sat.mean(axis=0) if inc_sat is not None else [np.nan] * m

        color = design_color(d)
        for k, name in enumerate(obj_names):
            y = (m - 1 - k) + offsets[di]
            lo, hi = float(uni[:, k].min()), float(uni[:, k].max())
            med, best = float(np.median(uni[:, k])), hi
            ax.plot([lo, hi], [y, y], color=color, lw=2.0, alpha=0.45,
                    solid_capstyle="butt", zorder=2)
            ax.plot(best, y, "o", ms=6, color=color, zorder=4)
            ax.plot(med, y, "o", ms=6, mfc="white", mec=color, mew=1.4, zorder=4)
            ax.plot(float(inc_uni[k]), y, "D", ms=5.5, color=INCUMBENT_COLOR,
                    zorder=5)
            rows.append({"design": d, "objective": name,
                         "policy_min": lo, "policy_median": med,
                         "policy_best": best, "incumbent": float(inc_uni[k])})

    ax.set_yticks(range(m))
    ax.set_yticklabels([short_label_for(n) for n in reversed(obj_names)])
    ax.set_ylim(-0.6, m - 0.4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(f"Fraction of {ETEST} SOWs meeting the criterion "
                  "(single axis alone)")
    ax.grid(axis="x", color="0.9", lw=0.8)
    ax.set_axisbelow(True)

    handles = _design_legend(results)
    handles += [
        Line2D([], [], marker="o", ls="none", ms=6, color="0.25",
               label="best policy"),
        Line2D([], [], marker="o", ls="none", ms=6, mfc="white", mec="0.25",
               mew=1.4, label="median policy"),
        Line2D([], [], marker="D", ls="none", ms=5.5, color=INCUMBENT_COLOR,
               label=INCUMBENT_LABEL),
    ]
    # Deduplicate the incumbent entry (color line + marker say the same thing).
    handles = [h for h in handles
               if not (h.get_label() == INCUMBENT_LABEL and h.get_marker() == "None")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False)
    fig.tight_layout()
    _add_footer(results, fig, y=-0.15,
                criteria=rd.criterion_thresholds(results[designs[0]], crit),
                criteria_header=f"{crit.label} (all must hold):")

    save_figure(fig, out_stub)
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(table_dir / "satisficing_decomposition.csv", index=False)
    return {"n_rows": len(table)}


###############################################################################
# P1.2 -- conjunction collapse
###############################################################################

def fig_conjunction_collapse(results: dict, out_stub: Path,
                             table_dir: Path) -> dict:
    """Joint satisficing fraction as the eight criteria are conjoined in a
    fixed global order (easiest first), per design.

    Solid: the best single policy's joint SOW fraction at each depth. Dashed:
    the fraction of SOWs where ANY policy clears the conjunction (attainability
    ceiling); the gap between them is cross-SOW policy conflict. Firebrick: the
    incumbent. The right end of every solid curve is the primary Starr
    robustness under the full default criterion.
    """
    crit = _reference_criterion()
    designs = _designs(results)
    order = list(rd.COLLAPSE_ORDER)
    depths = np.arange(1, len(order) + 1)

    tables = []
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for d in designs:
        res = results[d]
        thr = rd.criterion_thresholds(res, crit)
        sat = rd.satisfaction(res.raw, thresholds=thr)
        curve = rd.collapse_curve(sat, res.raw.obj_names, order)
        curve.insert(0, "design", d)
        tables.append(curve)
        color = design_color(d)
        ax.plot(depths, curve["best_policy"], "-o", ms=4, color=color, zorder=4)
        ax.plot(depths, curve["any_policy"], "--", lw=1.4, color=color,
                alpha=0.85, zorder=3)

        inc_sat = rd.incumbent_satisfaction(res, thresholds=thr)
        if inc_sat is not None and d == designs[0]:
            inc_curve = rd.collapse_curve(inc_sat[np.newaxis], res.raw.obj_names,
                                          order)
            inc_curve.insert(0, "design", "ffmp_incumbent")
            tables.append(inc_curve)
            ax.plot(depths, inc_curve["best_policy"], "-D", ms=4,
                    color=INCUMBENT_COLOR, lw=1.8, zorder=5)

    ax.set_xticks(depths)
    ax.set_xticklabels([short_label_for(n) for n in order],
                       rotation=30, ha="right")
    ax.set_xlabel("Criteria conjoined left to right (cumulative)")
    ax.set_ylabel(f"Fraction of {ETEST} SOWs meeting\nall conjoined thresholds")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)

    handles = _design_legend(results, incumbent=False)
    handles += [
        Line2D([], [], color="0.25", ls="-", marker="o", ms=4,
               label="best single policy"),
        Line2D([], [], color="0.25", ls="--", label="any policy (ceiling)"),
        Line2D([], [], color=INCUMBENT_COLOR, ls="-", marker="D", ms=4,
               label=INCUMBENT_LABEL),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    _add_footer(results, fig, y=-0.06,
                criteria=rd.criterion_thresholds(results[designs[0]], crit),
                criteria_header=f"{crit.label} (all must hold):")

    save_figure(fig, out_stub)
    plt.close(fig)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(table_dir / "conjunction_collapse.csv", index=False)
    return {"final_best": {d: float(t[t.design == d].best_policy.iloc[-1])
                           for d in designs
                           for t in [table]}}


###############################################################################
# P1.3 -- threshold-response curves
###############################################################################

def _per_policy_response(values: np.ndarray, kind: str,
                         grid: np.ndarray) -> np.ndarray:
    """Satisficing fraction at each grid threshold for every policy ``(S, T)``."""
    s, g = values.shape
    out = np.empty((s, len(grid)))
    for i in range(s):
        out[i] = rd.threshold_response(values[i], kind, grid)
    return out


def fig_threshold_response(results: dict, out_stub: Path,
                           table_dir: Path) -> dict:
    """Satisficing fraction as a function of where each threshold is placed.

    One panel per objective; x is the objective value in natural units, y the
    fraction of E_test SOWs that would meet a threshold placed at x (survival
    curve for maximize-type axes, CDF for minimize-type, so up always = more
    SOWs passing). Solid: pointwise-best policy per design; dotted: pointwise
    median policy. Firebrick: incumbent. Crimson dashed vertical: the default
    criterion -- its intersection height IS the current satisficing fraction.
    """
    crit = _reference_criterion()
    designs = _designs(results)
    first = results[designs[0]].raw
    obj_names, kinds = first.obj_names, first.kinds
    thresholds = rd.criterion_thresholds(results[designs[0]], crit)

    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.6), sharey=True)
    for k, name in enumerate(obj_names):
        ax = axes.flat[k]
        pooled = np.concatenate(
            [results[d].raw.cube[:, :, k].ravel() for d in designs]
            + [results[d].incumbent[:, k].ravel() for d in designs
               if results[d].incumbent is not None])
        pooled = pooled[np.isfinite(pooled)]
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        thr = thresholds[name]
        if thr is not None:
            lo, hi = min(lo, thr), max(hi, thr)
        grid = np.linspace(lo, hi, 160)

        for d in designs:
            res = results[d]
            resp = _per_policy_response(res.raw.cube[:, :, k], kinds[name], grid)
            best, med = resp.max(axis=0), np.median(resp, axis=0)
            color = design_color(d)
            ax.plot(grid, best, "-", color=color, lw=1.7)
            ax.plot(grid, med, ":", color=color, lw=1.5)
            for stat, vals in (("best", best), ("median", med)):
                rows += [{"objective": name, "design": d, "statistic": stat,
                          "threshold": t, "satisficing": v}
                         for t, v in zip(grid, vals)]
        inc = results[designs[0]].incumbent
        if inc is not None:
            iv = rd.threshold_response(inc[:, k], kinds[name], grid)
            ax.plot(grid, iv, "-", color=INCUMBENT_COLOR, lw=1.7)
            rows += [{"objective": name, "design": "ffmp_incumbent",
                      "statistic": "incumbent", "threshold": t, "satisficing": v}
                     for t, v in zip(grid, iv)]
        if thr is not None:
            ax.axvline(thr, color=THRESHOLD_COLOR, ls="--", lw=1.3)

        ax.set_title(label_for(name), fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(color="0.92", lw=0.7)
        ax.set_axisbelow(True)
        if k % 4 == 0:
            ax.set_ylabel("Fraction of SOWs\nmeeting threshold")
        if k >= 4:
            ax.set_xlabel("Threshold placement\n(objective value, natural units)")

    handles = _design_legend(results)
    handles += [
        Line2D([], [], color="0.25", ls="-", label="best policy"),
        Line2D([], [], color="0.25", ls=":", label="median policy"),
        Line2D([], [], color=THRESHOLD_COLOR, ls="--", label="criterion threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _add_footer(results, fig, y=-0.10, criteria=thresholds, criteria_header=(
        f"{crit.label} (crimson dashed lines; all must hold):"))

    save_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(table_dir / "threshold_response.csv", index=False)
    return {"n_objectives": len(obj_names)}


###############################################################################
# P1.4 -- attainability blockers
###############################################################################

def _blocking_patterns(res: rd.DesignResults, thresholds: dict) -> Counter:
    """Count per-SOW attainability patterns under a criterion vector.

    Computed from the cube (the persisted attainability CSV holds only the
    snapshot criteria): per SOW, either some policy meets every criterion
    ("__attainable__"), or the sorted tuple of axes NO policy satisfies, or
    "__conjunction_only__" when every axis is individually satisfiable but no
    single policy clears them all at once.
    """
    sat = rd.satisfaction(res.raw, thresholds=thresholds)   # (S, G, M)
    anysat = sat.any(axis=0)                                # (G, M)
    attainable = sat.all(axis=2).any(axis=0)                # (G,)
    names = res.raw.obj_names
    patterns: Counter = Counter()
    for g in range(res.raw.n_sow):
        if attainable[g]:
            patterns[("__attainable__",)] += 1
            continue
        blocked = tuple(sorted(names[k] for k in range(len(names))
                               if not anysat[g, k]))
        patterns[blocked or ("__conjunction_only__",)] += 1
    return patterns


def _pattern_label(pattern: tuple) -> str:
    if pattern == ("__attainable__",):
        return "attainable\n(some policy meets all criteria)"
    if pattern == ("__conjunction_only__",):
        return "no single axis blocks\n(cross-axis conflict only)"
    return " and\n".join(short_label_for(n) for n in pattern)


def fig_attainability_blockers(results: dict, out_stub: Path,
                               table_dir: Path) -> dict:
    """Why E_test SOWs are unattainable: per-SOW blocking patterns.

    For each SOW, the set of axes that no policy in the design's Pareto set
    satisfies there ("blocked" axes); SOWs where every axis is individually
    satisfiable but no single policy clears every axis are the cross-axis
    conflict category; SOWs some policy fully satisfies count as attainable.
    Grouped horizontal bars over the shared pattern list.
    """
    crit = _reference_criterion()
    designs = _designs(results)
    counts = {d: _blocking_patterns(
        results[d], rd.criterion_thresholds(results[d], crit))
        for d in designs}
    pooled = Counter()
    for c in counts.values():
        pooled.update(c)
    patterns = [p for p, _ in pooled.most_common()]

    n_pat = len(patterns)
    offsets = np.linspace(0.28, -0.28, num=len(designs))
    fig, ax = plt.subplots(figsize=(8.8, 1.15 * n_pat + 1.8))
    rows = []
    for di, d in enumerate(designs):
        n_sow = results[d].raw.n_sow
        vals = [counts[d].get(p, 0) for p in patterns]
        y = np.arange(n_pat)[::-1] + offsets[di]
        ax.barh(y, vals, height=0.24, color=design_color(d),
                label=design_label(d))
        rows += [{"design": d,
                  "pattern": p[0].strip("_") if p[0].startswith("__")
                  else "+".join(p),
                  "n_sow": v, "frac_sow": v / n_sow}
                 for p, v in zip(patterns, vals)]

    n_sow = results[designs[0]].raw.n_sow
    ax.set_yticks(np.arange(n_pat)[::-1])
    ax.set_yticklabels([_pattern_label(p) for p in patterns], fontsize=8.5)
    ax.set_xlabel(f"Number of {ETEST} SOWs (of {n_sow})")
    ax.grid(axis="x", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    crit_thr = rd.criterion_thresholds(results[designs[0]], crit)
    _add_footer(results, fig, y=-0.05, criteria=crit_thr, criteria_header=(
        f"Blocking is judged against: {crit.label} (all must hold):"))

    save_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(table_dir / "attainability_blockers.csv",
                              index=False)
    return {"patterns": n_pat}


###############################################################################
# P1.5 (optional) -- pairwise co-satisfiability
###############################################################################

def fig_pairwise_cosatisficing(results: dict, out_stub: Path,
                               table_dir: Path) -> dict:
    """Best-policy joint satisficing for every PAIR of axes, per design.

    Lower-triangle heatmap (diagonal = the single-axis best fraction). Cells
    far below the minimum of their two diagonal entries mark pairwise conflict
    -- collapse beyond what individual hardness explains.
    """
    designs = _designs(results)
    obj_names = results[designs[0]].raw.obj_names
    m = len(obj_names)

    crit = _reference_criterion()
    rows = []
    fig, axes = plt.subplots(1, len(designs), figsize=(4.6 * len(designs), 4.9))
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("viridis")
    im = None
    for ax, d in zip(axes, designs):
        sat = rd.satisfaction(
            results[d].raw,
            thresholds=rd.criterion_thresholds(results[d], crit))
        mat = np.full((m, m), np.nan)
        for i in range(m):
            for j in range(i + 1):
                pair = (sat[:, :, i] & sat[:, :, j]).mean(axis=1).max()
                mat[i, j] = float(pair)
                rows.append({"design": d, "axis_a": obj_names[i],
                             "axis_b": obj_names[j], "best_policy_joint": pair})
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0)
        for i in range(m):
            for j in range(i + 1):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
        ax.set_xticks(range(m))
        ax.set_yticks(range(m))
        ax.set_xticklabels([short_label_for(n) for n in obj_names],
                           rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels([short_label_for(n) for n in obj_names] if ax is axes[0]
                           else [], fontsize=7)
        ax.set_title(design_label(d), fontsize=9)
    fig.colorbar(im, ax=list(axes), shrink=0.75, pad=0.02,
                 label="Best policy's joint SOW fraction (axis pair)")
    _add_footer(results, fig, y=-0.06,
                criteria=rd.criterion_thresholds(results[designs[0]], crit),
                criteria_header=(
                    f"Pairs are judged against: {crit.label} (all must hold):"))

    save_figure(fig, out_stub)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(table_dir / "pairwise_cosatisficing.csv",
                              index=False)
    return {"n_pairs": m * (m + 1) // 2}
