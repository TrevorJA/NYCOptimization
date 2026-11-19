"""reeval_regret_figures.py - Regret of one design's Pareto set against the
current FFMP on the E_test re-evaluation.

Front-level regret figures of one design, rendered with the existing builders
and no plotting code of its own:

* robustness against regret frequency for every re-evaluated policy, with the
  non-dominated frontier and the marginal densities
  (:func:`src.plotting.regret_headline.fig_regret_vs_incumbent`, restricted to
  this design): x = the All-parties satisficing robustness, y = the share of
  E_test SOWs in which the policy is worse than the FFMP incumbent beyond
  tolerance on a member axis (``1 - no_harm_freq_tau__compromise``);
* per objective, in natural units, the tail regret and the mean gain against
  the incumbent across the policies
  (:func:`src.plotting.regret_summary.plot_regret_decomposition`).

The single-policy low/high-regret scenario-discovery scatters are rendered by
``robust_policy_figures.py``.

Outputs under ``outputs/figures/{design}/{slug}/robustness/regret_{tag}/``:
    regret_vs_robustness_compromise.png
    regret_decomposition.png
The plotted (robustness, regret) points go to the figure-tables tree
(``outputs/comparison/{slug}/{tag}/figure_tables/robustness/``).

Usage (repo root, venv active; light, one core):
    python scripts/supplemental/reeval_regret_figures.py [--design historic]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from src.figures.registry import FigureContext  # noqa: E402
from src.plotting import style  # noqa: E402
from src.plotting.regret_headline import (CRITERION_KEY,  # noqa: E402
                                          fig_regret_vs_incumbent)
from src.plotting.regret_summary import plot_regret_decomposition  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--design", default="historic")
    p.add_argument("--slug", default="ffmp_obj8")
    p.add_argument("--tag", default="etest_kn_50yr_n12500")
    a = p.parse_args()

    out_dir = (config.figure_dir_for(a.design, a.slug, "robustness")
               / f"regret_{a.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = FigureContext(a.tag, a.slug)
    table_dir = ctx.tables("robustness", frozenset({"criteria_scorecard"}))

    style.apply_style()
    style.FIGURE_FORMATS = ("png",)

    stem = f"regret_vs_robustness_{CRITERION_KEY}"
    fig_regret_vs_incumbent(ctx, out_dir / stem, table_dir, designs=[a.design])
    print(f"[figure] {stem}.png")

    reeval_dir = config.OUTPUTS_DIR / a.design / a.slug / "reeval" / a.tag
    plot_regret_decomposition(reeval_dir, out_dir / "regret_decomposition")
    print("[figure] regret_decomposition.png")
    print(f"figures -> {out_dir}\ntables  -> {table_dir}")


if __name__ == "__main__":
    main()
