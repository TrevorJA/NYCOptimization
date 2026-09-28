"""transfer_evaluation_figures.py - Exploratory figures for the transfer matrix.

Pure post-processing: reads only the tables written by
``transfer_evaluation_run.py`` and never re-runs an analysis, so a redraw is
cheap and cannot change a number.

Exploratory tier, so the dense style applies. Following the project's figure
convention, exact values live in the companion tables and in panel titles
rather than in small in-panel annotations; the heatmaps are the one exception
the convention allows, because a matrix cell's number IS its position and a
heatmap without cell values cannot be read at all.

Figures:
  F1 tev_dominance_matrix     3x3 heatmap, fraction of each set dominating the
                              scenario-matched FFMP baseline under each ensemble.
  F2 tev_merged_composition   what each ensemble's merged reference set is made
                              of, plus the size-invariant contribution rate.
  F3 tev_hypervolume          each source set's hypervolume across the three
                              evaluation ensembles, on one shared reference.
  F4 tev_path_consistency     |delta| / epsilon per objective - the panel that
                              licenses mixing stored and simulated values in F2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_tev_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.plotting.layout import shared_legend  # noqa: E402
from src.plotting.parallel_coordinates import (  # noqa: E402
    custom_parallel_coordinates, minmaxs_from_directions,
)
from src.plotting.style import (  # noqa: E402
    INCUMBENT_COLOR, apply_style, design_color, design_label, save_figure,
    short_label_for,
)
from src.formulations import get_obj_directions, get_obj_names  # noqa: E402

#: Parallel-axes panels need more width than a column-true figure: eight axes
#: with end-value annotations do not fit at 7.48 in. Matches the manuscript
#: fig05 convention.
PANEL_WIDTH = 13.5
PANEL_HEIGHT = 4.1
#: The renderer draws its smallest text at ``fontsize - 2``.
PA_FONTSIZE = 13
_AXIS_LABEL_WRAP = 11

#: Plain-language design names. Internal slugs never appear on a figure.
DESIGN_LABEL = {
    "historic": "Historic",
    "monte_carlo": "Monte Carlo",
    "hazard_filling_stationary": "Hazard filling",
}

#: Target ensembles, labelled by the design whose search used them.
TARGET_LABEL = {
    "historic_single": "Historic trace",
    "fixprob_10yr_n100_d0": "Monte Carlo ensemble",
    "hazfill_stat_abs_10yr_n100_d0": "Hazard-filling ensemble",
}

INCUMBENT_LABEL = "FFMP incumbent (status quo)"

ROW_ORDER = ("historic", "monte_carlo", "hazard_filling_stationary")
COL_ORDER = ("historic_single", "fixprob_10yr_n100_d0",
             "hazfill_stat_abs_10yr_n100_d0")


def _axis_label(name: str) -> str:
    """Wrapped abbreviation; the preference arrow already carries direction."""
    import textwrap
    return textwrap.fill(short_label_for(name), _AXIS_LABEL_WRAP)


def _cell_objectives():
    """Tidy per-solution objectives with merged-set membership, or None."""
    return _table("tev_cell_objectives")


def _baselines():
    """Scenario-matched FFMP baselines keyed by target slug, or None."""
    frame = _table("tev_baseline")
    if frame is None:
        return None
    return {r["target_slug"]: r for _, r in frame.iterrows()}


def _table(name: str):
    """Read a persisted table, or None when the stage has not run."""
    path = scfg.tev_table_path(name)
    if not path.exists():
        print(f"[tev-fig] missing {path.name}; skipping the figures that need it")
        return None
    return pd.read_csv(path)


def _order(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot to source x target in the canonical order."""
    grid = frame.pivot(index="source", columns="target_slug", values=value)
    rows = [r for r in ROW_ORDER if r in grid.index]
    cols = [c for c in COL_ORDER if c in grid.columns]
    return grid.loc[rows, cols]


def _heatmap(ax, grid: pd.DataFrame, fmt: str, cmap: str):
    """Draw one matrix panel with the diagonal marked."""
    data = grid.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([TARGET_LABEL.get(c, c) for c in grid.columns],
                       rotation=20, ha="right")
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels([DESIGN_LABEL.get(r, r) for r in grid.index])
    ax.set_xlabel("Evaluated under")
    ax.set_ylabel("Optimized under")

    diag = {"historic": "historic_single",
            "monte_carlo": "fixprob_10yr_n100_d0",
            "hazard_filling_stationary": "hazfill_stat_abs_10yr_n100_d0"}
    for i, r in enumerate(grid.index):
        for j, c in enumerate(grid.columns):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            shade = im.cmap(im.norm(v))[:3]
            colour = "white" if sum(shade) / 3 < 0.5 else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center", color=colour)
            if diag.get(r) == c:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", linewidth=2.0))
    return im


def fig_dominance(frame: pd.DataFrame) -> None:
    """F1: fraction of each set dominating the FFMP baseline."""
    grid = _order(frame, "frac_dominating")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = _heatmap(ax, grid * 100.0, ".1f", "viridis")
    fig.colorbar(im, ax=ax, label="Solutions dominating the baseline (%)")
    ax.set_title("Share of each Pareto set that dominates the FFMP baseline\n"
                 "(outlined cells are on-design; read down a column to compare "
                 "designs)")
    fig.tight_layout()
    save_figure(fig, scfg.tev_figure_path("tev_dominance_matrix"))
    plt.close(fig)


def fig_merged(frame: pd.DataFrame) -> None:
    """F2: merged-reference-set composition and contribution rate."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    for ax, (value, title, ylab) in zip(axes, [
        ("composition_share",
         "What each ensemble's merged reference set is made of",
         "Share of the merged reference set"),
        ("contribution_rate",
         "How often a solution survives the merge (size-invariant)",
         "Contributed / source set size"),
    ]):
        grid = _order(frame, value)
        bottom = np.zeros(grid.shape[1])
        x = np.arange(grid.shape[1])
        for source in grid.index:
            vals = grid.loc[source].to_numpy(dtype=float)
            if value == "composition_share":
                ax.bar(x, vals, bottom=bottom, label=DESIGN_LABEL.get(source, source),
                       color=design_color(source), edgecolor="white", linewidth=0.6)
                bottom += np.nan_to_num(vals)
            else:
                off = (list(grid.index).index(source) - 1) * 0.27
                ax.bar(x + off, vals, width=0.26,
                       label=DESIGN_LABEL.get(source, source),
                       color=design_color(source))
        ax.set_xticks(x)
        ax.set_xticklabels([TARGET_LABEL.get(c, c) for c in grid.columns],
                           rotation=15, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    axes[0].legend(title="Optimized under", frameon=False, fontsize="small")
    fig.tight_layout()
    save_figure(fig, scfg.tev_figure_path("tev_merged_composition"))
    plt.close(fig)


def fig_hypervolume(frame: pd.DataFrame) -> None:
    """F3: hypervolume of each set under each ensemble."""
    grid = _order(frame, "hypervolume")
    if not np.isfinite(grid.to_numpy(dtype=float)).any():
        print("[tev-fig] hypervolume unavailable; skipping F3")
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(grid.shape[1])
    for i, source in enumerate(grid.index):
        ax.bar(x + (i - 1) * 0.27, grid.loc[source].to_numpy(dtype=float),
               width=0.26, label=DESIGN_LABEL.get(source, source),
               color=design_color(source))
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABEL.get(c, c) for c in grid.columns],
                       rotation=15, ha="right")
    ax.set_ylabel("Hypervolume (shared reference set)")
    ax.set_title("Hypervolume of each Pareto set under each evaluation ensemble")
    ax.legend(title="Optimized under", frameon=False, fontsize="small")
    fig.tight_layout()
    save_figure(fig, scfg.tev_figure_path("tev_hypervolume"))
    plt.close(fig)


def fig_path_consistency(frame: pd.DataFrame) -> None:
    """F4: path difference per objective, in epsilon units."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    objectives = list(dict.fromkeys(frame["objective"]))
    x = np.arange(len(objectives))
    designs = [d for d in ROW_ORDER if d in set(frame["design"])]
    for i, design in enumerate(designs):
        sub = frame[frame["design"] == design].set_index("objective")
        vals = [sub.loc[o, "max_diff_over_eps"] if o in sub.index else np.nan
                for o in objectives]
        ax.bar(x + (i - 1) * 0.27, vals, width=0.26,
               label=DESIGN_LABEL.get(design, design), color=design_color(design))
    ax.axhline(scfg.TEV_CHECK_EPS_FRAC, color="black", linestyle="--", linewidth=1.2,
               label=f"Reference magnitude ({scfg.TEV_CHECK_EPS_FRAC:g} eps)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", " ") for o in objectives], rotation=30,
                       ha="right", fontsize="small")
    ax.set_ylabel("Max |difference| / epsilon")
    worst = float(np.nanmax(frame["max_diff_over_eps"]))
    ax.set_title("Search path versus driver path, in epsilon units "
                 f"(worst = {worst:.1e} eps)")
    ax.legend(frameon=False, fontsize="small")
    fig.tight_layout()
    save_figure(fig, scfg.tev_figure_path("tev_path_consistency"))
    plt.close(fig)




def _normalized(values, axis_ranges, minmaxs, ideal_direction: str = "top"):
    """Reproduce the renderer's axis normalization for an overlay line.

    ``custom_parallel_coordinates`` min-max normalizes each axis over the rows
    it draws, widened by ``axis_ranges``, then flips the axes whose optimization
    direction disagrees with ``ideal_direction`` so up is better everywhere. An
    overlay drawn on the returned axes has to use the identical transform or it
    will not line up. Passing an ``axis_ranges`` that already spans every drawn
    row makes the transform deterministic, which is why the callers below
    compute it over the whole pool first.

    Args:
        values: ``(n_axes,)`` or ``(n_rows, n_axes)`` raw natural-unit values.
        axis_ranges: ``(2, n_axes)`` raw (lo, hi) per axis.
        minmaxs: Per-axis ``'max'``/``'min'``.
        ideal_direction: Preferred end, matching the renderer call.

    Returns:
        Array of the same shape, in normalized [0, 1] axis coordinates.
    """
    arr = np.atleast_2d(np.asarray(values, dtype=float))
    lo = np.asarray(axis_ranges, dtype=float)[0]
    hi = np.asarray(axis_ranges, dtype=float)[1]
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    base = (arr - lo) / rng
    flip = np.array([(ideal_direction == "top") != (mm == "max") for mm in minmaxs])
    out = np.where(flip, 1.0 - base, base)
    return out[0] if np.ndim(values) == 1 else out


def _draw_design_medians(ax, sub, order, obj_names, axis_ranges, minmaxs,
                         lw: float = 3.6):
    """Overlay each design's per-objective median as a bold polyline.

    A cloud of two thousand lines cannot be read: the design with the most
    solutions simply paints over the others, and the project's reference grey
    for ``historic`` disappears entirely. The median polyline is what makes the
    comparison legible - it states where each design's solutions typically sit
    on each axis, which is precisely the quantity the contribution shares are
    a consequence of. A white casing keeps every design's line readable
    wherever the three cross.
    """
    import matplotlib.patheffects as pe

    x = np.arange(len(obj_names))
    for d in order:
        rows = sub[sub["source"] == d][obj_names].to_numpy(dtype=float)
        if rows.size == 0:
            continue
        med = _normalized(np.nanmedian(rows, axis=0), axis_ranges, minmaxs)
        ax.plot(x, med, color=design_color(d), lw=lw, zorder=55,
                solid_capstyle="round",
                path_effects=[pe.withStroke(linewidth=lw + 2.4, foreground="white")])


def _panel_frames(cells, slug, obj_names):
    """Per-design natural-unit frames for one evaluation ensemble, in plot order."""
    sub = cells[cells["target_slug"] == slug]
    order = [d for d in ROW_ORDER if d in set(sub["source"])]
    return sub, order


def fig_parallel_by_design(cells, baselines, obj_names, directions) -> None:
    """F5: where each design's solutions sit, on one common evaluation ensemble.

    This is the figure the matrix summaries could not show. Every solution from
    all three optimizations is drawn on the same eight axes, colored by the
    design that produced it, with one panel per evaluation ensemble and a
    single shared axis range across panels so the panels are comparable. Axes
    are oriented so up is better on every one.

    It answers the question the contribution shares only gesture at: the three
    designs do not differ by being uniformly better or worse, they occupy
    different regions of objective space, and the separation is concentrated on
    particular axes.
    """
    minmaxs = minmaxs_from_directions(directions)
    labels = [_axis_label(n) for n in obj_names]

    stack = [cells[obj_names].to_numpy(dtype=float)]
    if baselines:
        stack.append(np.array([[b[n] for n in obj_names] for b in baselines.values()]))
    stacked = np.vstack(stack)
    axis_ranges = np.vstack([np.nanmin(stacked, axis=0), np.nanmax(stacked, axis=0)])

    slugs = [c for c in COL_ORDER if c in set(cells["target_slug"])]
    fig, axes = plt.subplots(len(slugs), 1,
                             figsize=(PANEL_WIDTH, PANEL_HEIGHT * len(slugs)))
    for letter, ax, slug in zip("abcdef", np.atleast_1d(axes), slugs):
        sub, order = _panel_frames(cells, slug, obj_names)
        # Draw the largest set first so the smallest is not painted over. The
        # renderer draws rows in frame order, and monte_carlo has three times
        # the members of historic.
        by_size = sorted(order, key=lambda d: -(sub["source"] == d).sum())
        sub = pd.concat([sub[sub["source"] == d] for d in by_size],
                        ignore_index=True)
        base = ([baselines[slug][n] for n in obj_names]
                if baselines and slug in baselines else None)
        custom_parallel_coordinates(
            sub[obj_names].reset_index(drop=True),
            columns_axes=obj_names, axis_labels=labels, minmaxs=minmaxs,
            color_by_categorical=sub["source"].to_numpy(),
            color_dict_categorical={d: design_color(d) for d in order},
            alpha_base=0.05, lw_base=0.7, fontsize=PA_FONTSIZE,
            baseline=base, baseline_label=INCUMBENT_LABEL,
            ax=ax, axis_ranges=axis_ranges,
            add_colorbar=False, add_legend=False,
        )
        _draw_design_medians(ax, sub, order, obj_names, axis_ranges, minmaxs)
        ax.set_title(f"({letter}) Evaluated under: {TARGET_LABEL.get(slug, slug)} "
                     f"— all {len(sub)} policies, bold lines are per-design medians",
                     loc="left", fontsize=PA_FONTSIZE)
        ax.add_patch(plt.Rectangle((0, 0), len(obj_names) - 1, 1, fill=False,
                                   edgecolor="black", lw=1.1, zorder=60,
                                   clip_on=False))
    fig.tight_layout()
    handles = [Line2D([], [], color=design_color(d), lw=3.2,
                      label=f"Optimized under {DESIGN_LABEL.get(d, d)} "
                            f"(median in bold)")
               for d in ROW_ORDER if d in set(cells["source"])]
    handles.append(Line2D([], [], color=INCUMBENT_COLOR, lw=2.5, marker="o",
                          markersize=5, label=INCUMBENT_LABEL))
    shared_legend(fig, handles, y=-0.035, fontsize=PA_FONTSIZE - 1)
    save_figure(fig, scfg.tev_figure_path("tev_parallel_axes_by_design"))
    plt.close(fig)


def fig_parallel_merged_set(cells, baselines, obj_names, directions) -> None:
    """F6: the members of each ensemble's merged reference set, on shared axes.

    Same axes and same shared ranges as F5, so the two figures can be read
    against each other, but only the solutions that enter the merged
    epsilon-nondominated reference set are drawn, colored by the optimization
    that produced them. The coloured lines here are exactly the members whose
    composition the contribution shares report, so a share and the region of
    objective space it comes from can be read off one picture.

    The excluded solutions are deliberately NOT drawn as a greyed background.
    This project colours `historic` in reference grey, which is nearly the same
    value as the renderer's screened-out grey, so a highlight-mask rendering
    made "in the merged set" and "not in it" indistinguishable in exactly the
    panel where it mattered most. F5 carries the full clouds; this figure
    answers only what survived.
    """
    minmaxs = minmaxs_from_directions(directions)
    labels = [_axis_label(n) for n in obj_names]
    stacked = cells[obj_names].to_numpy(dtype=float)
    axis_ranges = np.vstack([np.nanmin(stacked, axis=0), np.nanmax(stacked, axis=0)])

    slugs = [c for c in COL_ORDER if c in set(cells["target_slug"])]
    fig, axes = plt.subplots(len(slugs), 1,
                             figsize=(PANEL_WIDTH, PANEL_HEIGHT * len(slugs)))
    for letter, ax, slug in zip("abcdef", np.atleast_1d(axes), slugs):
        sub, order = _panel_frames(cells, slug, obj_names)
        counts = {d: int(((sub["source"] == d) & sub["in_merged_set"]).sum())
                  for d in order}
        members = sub[sub["in_merged_set"].to_numpy(dtype=bool)]
        by_size = sorted(order, key=lambda d: -counts[d])
        members = pd.concat([members[members["source"] == d] for d in by_size],
                            ignore_index=True)
        base = ([baselines[slug][n] for n in obj_names]
                if baselines and slug in baselines else None)
        custom_parallel_coordinates(
            members[obj_names].reset_index(drop=True),
            columns_axes=obj_names, axis_labels=labels, minmaxs=minmaxs,
            color_by_categorical=members["source"].to_numpy(),
            color_dict_categorical={d: design_color(d) for d in order},
            alpha_base=0.45, lw_base=1.3, fontsize=PA_FONTSIZE,
            baseline=base, baseline_label=INCUMBENT_LABEL,
            ax=ax, axis_ranges=axis_ranges,
            add_colorbar=False, add_legend=False,
        )
        _draw_design_medians(ax, members, order, obj_names, axis_ranges, minmaxs,
                             lw=3.2)
        made = ", ".join(f"{DESIGN_LABEL.get(d, d)} {counts[d]}" for d in order)
        ax.set_title(f"({letter}) Merged reference set for "
                     f"{TARGET_LABEL.get(slug, slug)} — {len(members)} members "
                     f"({made})", loc="left", fontsize=PA_FONTSIZE)
        ax.add_patch(plt.Rectangle((0, 0), len(obj_names) - 1, 1, fill=False,
                                   edgecolor="black", lw=1.1, zorder=60,
                                   clip_on=False))
    fig.tight_layout()
    handles = [Line2D([], [], color=design_color(d), lw=3.2,
                      label=f"Optimized under {DESIGN_LABEL.get(d, d)} "
                            f"(median in bold)")
               for d in ROW_ORDER if d in set(cells["source"])]
    handles.append(Line2D([], [], color=INCUMBENT_COLOR, lw=2.5, marker="o",
                          markersize=5, label=INCUMBENT_LABEL))
    shared_legend(fig, handles, y=-0.035, fontsize=PA_FONTSIZE - 1)
    save_figure(fig, scfg.tev_figure_path("tev_parallel_axes_merged_set"))
    plt.close(fig)


def fig_enrichment(enrich, loo) -> None:
    """F7: is a contribution share a design effect, or arithmetic?

    Panel (a) divides each source's share of the merged set by its share of the
    pooled input, so the no-effect null is 1.0 whatever the set sizes, and shows
    it for the plain-Pareto archive beside the epsilon-box archive. Panel (b)
    rebuilds the epsilon archive with each objective removed in turn; a source
    whose enrichment collapses when one axis is dropped owed its contribution
    to that axis.

    Together these say whether a contribution share reflects broadly better
    solutions or one objective doing all the work.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0),
                             gridspec_kw={"width_ratios": [1.0, 1.45]})

    ax = axes[0]
    archives = ["plain_pareto", "epsilon_box"]
    arch_label = {"plain_pareto": "Plain Pareto", "epsilon_box": "Epsilon box"}
    slugs = [c for c in COL_ORDER if c in set(enrich["target_slug"])]
    sources = [d for d in ROW_ORDER if d in set(enrich["source"])]
    x = np.arange(len(slugs) * len(archives))
    width = 0.26
    for i, d in enumerate(sources):
        vals = [float(enrich[(enrich.target_slug == s) & (enrich.archive == a)
                             & (enrich.source == d)]["enrichment"].iloc[0])
                for s in slugs for a in archives]
        ax.bar(x + (i - 1) * width, vals, width=width * 0.95,
               color=design_color(d), label=DESIGN_LABEL.get(d, d))
    ax.axhline(1.0, color="black", lw=1.4, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{TARGET_LABEL.get(s, s).replace(' ensemble', '')}\n{arch_label[a]}"
                        for s in slugs for a in archives],
                       rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Enrichment  (share of merged set / share of pool)")
    ax.set_title("(a) Contribution relative to pool share\n"
                 "(dashed line = no design effect)", loc="left")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    # Prefer an ensemble target over the single historic trace: the leave-one-out
    # collapse is cleanest where the enrichment is largest to begin with.
    ensembles = [s for s in slugs if s != "historic_single"]
    slug = ensembles[0] if ensembles else (slugs[-1] if slugs else None)
    sub = loo[loo.target_slug == slug] if slug else loo.iloc[:0]
    order = ["none"] + [d for d in sub["dropped"] if d != "none"]
    sub = sub.set_index("dropped").loc[order]
    xs = np.arange(len(sub))
    for i, d in enumerate(sources):
        ax.bar(xs + (i - 1) * width, sub[d].to_numpy(dtype=float),
               width=width * 0.95, color=design_color(d),
               label=DESIGN_LABEL.get(d, d))
    ax.axhline(1.0, color="black", lw=1.4, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(["none (all 8)"] + [short_label_for(d) for d in order[1:]],
                       rotation=35, ha="right", fontsize=9)
    ax.set_xlabel("Objective removed before rebuilding the merged set")
    ax.set_ylabel("Enrichment")
    ax.set_title(f"(b) Leave-one-out, {TARGET_LABEL.get(slug, slug)}\n"
                 "(a collapse means that objective carried the result)", loc="left")
    fig.tight_layout()
    save_figure(fig, scfg.tev_figure_path("tev_enrichment_diagnostic"))
    plt.close(fig)


def main() -> None:
    apply_style()
    scfg.TEV_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dominance = _table("tev_dominance_matrix")
    if dominance is not None:
        fig_dominance(dominance)
    merged = _table("tev_merged_contribution")
    if merged is not None:
        fig_merged(merged)
    hv = _table("tev_hypervolume_matrix")
    if hv is not None:
        fig_hypervolume(hv)
    path = _table("tev_path_consistency")
    if path is not None:
        fig_path_consistency(path)

    cells = _cell_objectives()
    if cells is not None:
        obj_names = [n for n in get_obj_names() if n in cells.columns]
        directions = get_obj_directions()
        baselines = _baselines()
        fig_parallel_by_design(cells, baselines, obj_names, directions)
        fig_parallel_merged_set(cells, baselines, obj_names, directions)
    enrich = _table("tev_enrichment")
    loo = _table("tev_leave_one_out")
    if enrich is not None and loo is not None:
        fig_enrichment(enrich, loo)

    print(f"[tev-fig] figures -> {scfg.TEV_FIGURES_DIR}")


if __name__ == "__main__":
    main()
