"""regret_summary.py - Figures for incumbent-relative regret (RQ2 + the hypothesis).

Three figures, each answering a question the satisficing figures cannot.

  **The plane** (:func:`plot_regret_robustness_plane`) -- every re-evaluated policy
    plotted as (satisficing robustness, no-harm frequency), with each design's
    non-dominated frontier drawn through it. This is the figure the working
    hypothesis lives or dies on: hazard filling is supposed to sit UP (more robust)
    without sitting LEFT (more regret against current operations). Bartholomew &
    Kwakkel (2020) report both halves of the expectation -- robustness bought in
    the search phase does survive re-evaluation, and it is normally paid for
    elsewhere, the "price of robustness" (Bertsimas & Sim 2004). A frontier that
    moves up and stays right is the result; one that moves up and shifts left is
    the price being paid, which is equally reportable.

    Both axes are unit-free by construction, which is exactly why the regret
    magnitudes are NOT on this plot: they are in each objective's own natural
    units and are never combined (see :func:`plot_regret_decomposition`).

  **The tolerance sweep** (:func:`plot_regret_tolerance_sweep`) -- the no-harm
    frequency against the tolerance ladder ``tau_i = k * eps_i``. A single
    tolerance could manufacture or hide the whole RQ2 answer, so the claim is
    reported as the tolerance at which it holds, mirroring the discipline applied
    to the satisficing criteria (Quinn et al. 2020).

  **The decomposition** (:func:`plot_regret_decomposition`) -- per objective, in
    NATURAL units, the tail regret and the mean gain against the status quo. This
    is where a reader learns WHICH party pays and HOW MUCH, which the frequency
    axes deliberately do not say. Gain is drawn alongside regret because a policy
    scores zero regret by BEING the incumbent: regret is never shown alone
    (Huang et al. 2025's degeneracy warning, already adopted repo-wide).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.style import label_for, save_figure


def _design_style(designs) -> tuple[dict, dict]:
    """Colour and label per design, derived from the registry, never hardcoded."""
    designs = list(dict.fromkeys(designs))
    cmap = plt.get_cmap("tab10" if len(designs) <= 10 else "tab20")
    colors = {d: cmap(i % cmap.N) for i, d in enumerate(designs)}
    try:
        from src.scenario_designs import SCENARIO_DESIGNS
        labels = {d: (SCENARIO_DESIGNS[d].name.replace("_", " ")
                      if d in SCENARIO_DESIGNS else str(d).replace("_", " "))
                  for d in designs}
    except Exception:                                          # noqa: BLE001
        labels = {d: str(d).replace("_", " ") for d in designs}
    return colors, labels


def pareto_frontier(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated points when BOTH axes are maximized.

    Used for the per-design frontier on the robustness/no-harm plane. Ties are
    kept (weak dominance only removes strictly dominated points), so a design is
    never advantaged by having found duplicates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return idx
    order = idx[np.lexsort((-y[idx], -x[idx]))]
    keep, best_y = [], -np.inf
    for i in order:
        if y[i] > best_y:
            keep.append(i)
            best_y = y[i]
    return np.array(keep, dtype=int)


def plot_regret_robustness_plane(points: pd.DataFrame, out_file,
                                 x: str = "sat_multivariate_sow",
                                 y: str = "no_harm_freq_tau",
                                 baseline_x: float | None = None,
                                 figsize: tuple = (7.6, 6.0)) -> Path:
    """Scatter every policy on the (robustness, no-harm) plane, frontier per design.

    Args:
        points: Tidy frame with columns ``design`` and the two metric columns; one
            row per re-evaluated policy. ``draw`` / ``seed`` are used for the
            caption count when present.
        out_file: Output stub (extension supplied by ``save_figure``).
        x: Robustness column (higher = better).
        y: No-harm column (higher = better, i.e. less regret).
        baseline_x: The status-quo policy's own robustness, drawn as a vertical
            reference. Its no-harm frequency against itself is 1.0 by definition,
            so it is annotated rather than plotted as a competing point.
        figsize: Figure size.

    Returns:
        The written figure path.
    """
    points = points.dropna(subset=[x, y])
    if points.empty:
        raise ValueError(f"no finite ({x}, {y}) pairs to plot")
    designs = sorted(points["design"].unique())
    colors, labels = _design_style(designs)

    fig, ax = plt.subplots(figsize=figsize)
    for d in designs:
        g = points[points["design"] == d]
        ax.scatter(g[x], g[y], s=14, color=colors[d], alpha=0.30,
                   edgecolor="none", zorder=2)
        front = pareto_frontier(g[x].to_numpy(), g[y].to_numpy())
        if front.size:
            f = g.iloc[front].sort_values(x)
            ax.plot(f[x], f[y], color=colors[d], lw=2.0, marker="o", ms=4.5,
                    zorder=4, label=f"{labels[d]} (n={len(g)})")

    if baseline_x is not None and np.isfinite(baseline_x):
        ax.axvline(baseline_x, color="firebrick", lw=2.0, ls="--", zorder=1)
        ax.annotate("status-quo FFMP\n(no-harm = 1 by definition)",
                    xy=(baseline_x, 1.0), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color="firebrick",
                    va="top")

    ax.set_xlabel("Satisficing robustness on the re-evaluation ensemble\n"
                  "(Starr domain criterion, SOW unit)")
    ax.set_ylabel("No-harm frequency vs current operations\n"
                  "fraction of SOWs degrading NO objective beyond tolerance")
    ax.set_title("Robustness bought, and what it cost\n"
                 "up = more robust; right = less regret vs the status quo",
                 fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", title="design frontier",
              title_fontsize=8)
    fig.tight_layout()
    out = Path(out_file)
    save_figure(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")


def plot_regret_tolerance_sweep(sweep: pd.DataFrame, out_file,
                                statistic: str = "best",
                                figsize: tuple = (7.2, 5.0)) -> Path:
    """No-harm frequency against the tolerance ladder, one line per design.

    Args:
        sweep: Output of ``compare_designs.regret_tolerance_sweep`` -- columns
            ``design``, ``tau_k``, ``best``, ``median``.
        out_file: Output stub.
        statistic: ``"best"`` (the policy a decision maker would deploy) or
            ``"median"`` (guards against one lucky policy carrying a design).
        figsize: Figure size.

    Returns:
        The written figure path.
    """
    if sweep.empty:
        raise ValueError("empty regret tolerance sweep")
    designs = sorted(sweep["design"].unique())
    colors, labels = _design_style(designs)

    fig, ax = plt.subplots(figsize=figsize)
    for d in designs:
        g = (sweep[sweep["design"] == d]
             .groupby("tau_k")[statistic].mean().sort_index())
        ax.plot(g.index, g.to_numpy(), color=colors[d], lw=2.0, marker="o",
                ms=5, label=labels[d])
        # Individual runs behind the design mean: with K = 3 draws the spread IS
        # the uncertainty, and averaging it away would overstate the result.
        for (_draw, _seed), h in sweep[sweep["design"] == d].groupby(
                ["draw", "seed"]):
            h = h.sort_values("tau_k")
            ax.plot(h["tau_k"], h[statistic], color=colors[d], lw=0.8,
                    alpha=0.35, zorder=1)

    ax.set_xlabel("Tolerance $k$  (no objective degraded by more than "
                  "$k$ just-noticeable differences)")
    ax.set_ylabel(f"No-harm frequency ({statistic} policy per run)")
    ax.set_title("How much tolerance does 'no degradation vs current operations'\n"
                 "need before it holds?", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = Path(out_file)
    save_figure(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")


def plot_regret_decomposition(reeval_dir, out_file, accepted_ids=None,
                              figsize: tuple = (12.5, 5.6)) -> Path:
    """Per-objective tail regret and mean gain vs the status quo, in NATURAL units.

    One panel per side, objectives on a shared axis, each objective on its OWN
    x-scale because the units are not commensurable and are never combined. This
    is the drill-down behind the unit-free plane: which party pays, and how much.

    Args:
        reeval_dir: Step-08 re-eval dir (holds ``reeval_raw*`` and ``baseline/``).
        out_file: Output stub.
        accepted_ids: Restrict to these ``solution_id``\\ s (e.g. the stakeholder
            screen from :mod:`src.pareto_filter`); None keeps every policy.
        figsize: Figure size.

    Returns:
        The written figure path.
    """
    from src.robustness import load_raw, regret_magnitudes

    reeval_dir = Path(reeval_dir)
    raw = load_raw(reeval_dir)
    base = load_raw(reeval_dir / "baseline")
    mags = regret_magnitudes(raw, base)
    if accepted_ids is not None:
        keep = [s for s in accepted_ids if s in mags.index]
        if not keep:
            raise ValueError("no accepted solutions found in the re-eval cube")
        mags = mags.loc[keep]

    names = list(raw.obj_names)
    # Order by how often the objective is harmed at all, worst first, so the
    # binding party is at the top of both panels.
    harm_rank = [np.nanmean(mags[f"regret_q90__{n}"].to_numpy()) > 0 for n in names]
    order = sorted(range(len(names)), key=lambda k: -float(harm_rank[k]))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, (prefix, title, color) in zip(axes, (
        ("regret_q90", "A. Tail regret vs current operations\n"
                       "90th-percentile shortfall across SOWs", "#c0392b"),
        ("gain_mean", "B. Mean gain vs current operations\n"
                      "the companion: zero regret can just mean 'unchanged'",
         "#2a7ab9"),
    )):
        data, ticks = [], []
        for k in order:
            v = mags[f"{prefix}__{names[k]}"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            # Each objective is scaled by its OWN maximum so eight incommensurable
            # units share one axis WITHOUT implying they are comparable; the
            # natural-unit maximum is printed on the tick so no magnitude is lost.
            hi = float(np.max(v)) if v.size and np.max(v) > 0 else 1.0
            data.append(v / hi if v.size else np.array([0.0]))
            ticks.append(f"{label_for(names[k])}\n(max {hi:.3g})")
        bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.4))
        for patch in bp["boxes"]:
            patch.set(facecolor=color, alpha=0.55)
        rng = np.random.default_rng(0)
        for i, v in enumerate(data):
            ax.scatter(v, np.full(v.size, i + 1) + rng.normal(0, 0.06, v.size),
                       s=6, color=color, alpha=0.30, zorder=3)
        ax.set_yticks(np.arange(len(order)) + 1)
        ax.set_yticklabels(ticks, fontsize=8)
        ax.set_xlabel("fraction of this objective's own maximum\n"
                      "(natural units; axes are NOT comparable across objectives)",
                      fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Regret against the status-quo FFMP policy, per objective "
                 "(SOW unit, natural units)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(out_file)
    save_figure(fig, out)
    plt.close(fig)
    return out.with_suffix(".png")
