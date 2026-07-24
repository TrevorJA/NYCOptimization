"""robustness_summary.py - DU robustness figure: multivariate satisficing + baseline.

Renders the held-out re-evaluation as the water-resources robustness literature
reports it (Herman et al. 2015; Trindade et al. 2017; Gold et al. 2022, 2023):

  Panel A -- the PRIMARY metric. Starr's (1962) multivariate domain criterion on
    the SOW unit: the fraction of deeply-uncertain states of the world in which a
    policy meets ALL seven objective thresholds jointly. Distribution across the
    acceptable Pareto policies, with the status-quo FFMP baseline drawn as a fixed
    external reference (Kasprzyk et al. 2013) and the most-robust policy marked.

  Panel B -- the per-objective decomposition of that same criterion (univariate
    satisficing on the SOW unit), which exposes the BINDING objective: the joint
    criterion can be no larger than its smallest component, so the objective with
    the lowest single-criterion satisficing is what caps robustness. Baseline
    overlaid per objective.

Both panels use the SOW unit consistently (the R realizations within each theta
collapsed first via ``within_sow_agg``), so Panel B decomposes exactly the Panel A
number rather than a differently-defined realization-unit quantity.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.style import label_for


def _sow_satisficing(raw, within_sow_agg: str = "mean"):
    """SOW-unit satisficing from a re-eval cube.

    Returns:
        ``(multivariate, univariate)`` where ``multivariate`` is ``(S,)`` -- the
        fraction of SOWs meeting ALL thresholds jointly -- and ``univariate`` is
        ``(S, M)`` -- the fraction of SOWs meeting each objective's threshold.
    """
    from src.robustness import collapse_within_sow, _satisfy
    cube_sow, _ = collapse_within_sow(raw, within_sow_agg)          # (S, n_sow, M)
    sat = _satisfy(cube_sow, raw.base_names, raw.thresholds, raw.kinds)
    univariate = sat.mean(axis=1)                                  # (S, M)
    multivariate = sat.all(axis=2).mean(axis=1)                    # (S,)
    return multivariate, univariate


def plot_du_robustness(reeval_dir, accepted_ids, out_file,
                       within_sow_agg: str = "mean",
                       most_robust_id: int | None = None,
                       figsize: tuple = (14, 5.6)) -> dict:
    """Draw the two-panel DU robustness summary and return headline numbers.

    Args:
        reeval_dir: Step-08 re-eval dir (holds ``reeval_raw*`` and ``baseline/``).
        accepted_ids: Re-eval ``solution_id``\\ s surviving the stakeholder screen
            (from :mod:`src.pareto_filter`). Only these are shown.
        out_file: PNG path.
        within_sow_agg: Within-SOW risk attitude (``"mean"`` risk-neutral, the
            Triangle-lineage default; ``"worst"`` risk-averse).
        most_robust_id: solution_id to highlight; if None, the max-robustness
            policy among ``accepted_ids`` is used.
        figsize: Figure size.

    Returns:
        Dict of headline numbers (baseline robustness, best/median among accepted,
        binding objective).
    """
    from src.robustness import load_raw

    reeval_dir = Path(reeval_dir)
    raw = load_raw(reeval_dir)
    base = load_raw(reeval_dir / "baseline")

    multi, uni = _sow_satisficing(raw, within_sow_agg)
    b_multi, b_uni = _sow_satisficing(base, within_sow_agg)
    b_multi = float(b_multi[0])
    b_uni = b_uni[0]

    names = list(raw.base_names)
    pos = {sid: i for i, sid in enumerate(raw.solution_ids)}
    keep = np.array([pos[s] for s in accepted_ids if s in pos], dtype=int)
    if keep.size == 0:
        raise ValueError("no accepted solutions found in the re-eval cube")

    a_multi = multi[keep]
    a_uni = uni[keep]

    if most_robust_id is None:
        most_robust_id = int(accepted_ids[int(np.argmax(a_multi))])
    mr_val = float(multi[pos[most_robust_id]])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=figsize)

    # ----- Panel A: primary metric distribution + baseline ------------------ #
    bins = np.linspace(0, max(0.05, float(a_multi.max()) * 1.05), 26)
    axA.hist(a_multi, bins=bins, color="#2a7ab9", alpha=0.85, edgecolor="white",
             label=f"acceptable Pareto policies (n={keep.size})")
    axA.axvline(b_multi, color="firebrick", lw=2.5, ls="--",
                label=f"FFMP baseline = {b_multi:.2f}")
    axA.axvline(mr_val, color="darkgreen", lw=2.5,
                label=f"most-robust policy = {mr_val:.2f}")
    axA.axvline(float(np.median(a_multi)), color="0.35", lw=1.4, ls=":",
                label=f"median = {np.median(a_multi):.2f}")
    axA.set_xlabel("Multivariate satisficing (SOW unit)\n"
                   "fraction of 50 SOWs meeting ALL 7 thresholds")
    axA.set_ylabel("number of Pareto policies")
    axA.set_title("A. Primary robustness — Starr domain criterion\n"
                  "(Herman 2015; Gold 2023)")
    axA.legend(fontsize=8, loc="upper right")
    axA.grid(True, alpha=0.3)

    # ----- Panel B: per-objective decomposition + baseline ------------------ #
    order = np.argsort(np.median(a_uni, axis=0))          # binding (lowest) first
    labels = [label_for(names[k]) for k in order]
    data = [a_uni[:, k] for k in order]
    y = np.arange(len(order))
    bp = axB.boxplot(data, vert=False, widths=0.6, patch_artist=True,
                     showfliers=False, medianprops=dict(color="navy", lw=1.5))
    for patch in bp["boxes"]:
        patch.set(facecolor="#9ecae1", alpha=0.8)
    # strip of raw policy points
    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        axB.scatter(a_uni[:, k], np.full(keep.size, i + 1)
                    + rng.normal(0, 0.06, keep.size),
                    s=6, color="#2a7ab9", alpha=0.25, zorder=2)
    # baseline markers
    axB.scatter([b_uni[k] for k in order], y + 1, marker="D", s=55,
                color="firebrick", edgecolor="white", zorder=5,
                label="FFMP baseline")
    axB.set_yticks(y + 1)
    axB.set_yticklabels(labels, fontsize=9)
    axB.set_xlim(-0.02, 1.02)
    axB.set_xlabel("Single-objective satisficing (SOW unit)\n"
                   "fraction of SOWs meeting this threshold")
    axB.set_title("B. Decomposition — the binding objective (red) caps joint robustness")
    axB.legend(fontsize=8, loc="lower right")
    axB.grid(True, axis="x", alpha=0.3)
    # The binding objective is the lowest-median row (bottom of the sorted axis);
    # flag it by reddening its tick label rather than a floating note.
    axB.get_yticklabels()[0].set_color("firebrick")
    axB.get_yticklabels()[0].set_fontweight("bold")

    fig.suptitle("DU robustness on held-out E_test (50 SOWs x 4 reps, wide DU box) — "
                 "acceptable policies vs status-quo FFMP", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    binding_k = int(order[0])
    return {
        "baseline_sat_sow": b_multi,
        "best_accepted_sat_sow": float(a_multi.max()),
        "median_accepted_sat_sow": float(np.median(a_multi)),
        "most_robust_id": int(most_robust_id),
        "binding_objective": names[binding_k],
        "binding_baseline_sat": float(b_uni[binding_k]),
        "binding_median_accepted_sat": float(np.median(a_uni[:, binding_k])),
    }
