"""policy_rules_compact_figures.py - Compact operating-rule figures of selected policies.

Renders :mod:`src.plotting.policy_rules_compact` for the highlighted policies
of a merged reference set: the policy grid (one column per policy) and one
single-policy figure for the first policy. Policies come from the
``selected_solutions.csv`` written by ``scripts/main/explore_results.py`` when
it exists under the explore figure directory; otherwise the balanced
compromise plus spread-out members are chosen here with
:mod:`src.solution_selection`.

Outputs under ``outputs/figures/{scenario}/{slug}/policy_inspection/`` (override with
``--out-dir``):

    rules_grid.png            the policy grid
    rules_{label}.png         the first policy alone
    rules_baseline.png        the baseline FFMP alone (``--baseline``)

Usage (from repo root, venv active):
    python scripts/supplemental/policy_rules_compact_figures.py --scenario monte_carlo
    python scripts/supplemental/policy_rules_compact_figures.py --set-file path/to/x.set
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import config
from src.plotting.policy_rules_compact import (
    plot_policy_rules_compact, plot_policy_rules_grid,
)
from src.plotting.style import apply_manuscript_style
from src.policy_simulation import read_selection_labels
from src.solution_selection import (
    compromise_scores, load_natural_front, select_diverse,
)


def _slugify(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def select_policies(set_file: Path, formulation: str, explore_dir: Path,
                    n_policies: int) -> list:
    """``(label, dv_vector)`` pairs for the grid columns.

    Prefers the highlighted rows of ``explore_dir/selected_solutions.csv``;
    falls back to the balanced (mean-scaled) compromise plus spread-out
    members of the front, mirroring ``explore_results._build_selections``
    without a baseline screen.
    """
    dv, natural, _, directions = load_natural_front(set_file, formulation)
    csv = explore_dir / "selected_solutions.csv"
    if csv.exists():
        labels = read_selection_labels(csv)
        print(f"[select] {len(labels)} highlighted policies from {csv}")
        return [(f"{names[0]} ({idx})", dv[idx])
                for idx, names in list(labels.items())[:n_policies]]
    balanced = compromise_scores(natural, directions, method="mean_scaled")
    seed = int(np.argmax(balanced))
    rows = select_diverse(natural, directions, n_policies, seed_indices=[seed])
    print(f"[select] no selection table under {explore_dir}; "
          f"balanced + diverse rows {list(map(int, rows))}")
    return [(("Balanced" if i == 0 else f"Diverse {i}") + f" ({int(idx)})",
             dv[idx]) for i, idx in enumerate(rows)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", default="historic")
    ap.add_argument("--slug", default="ffmp_obj8")
    ap.add_argument("--formulation", default="ffmp")
    ap.add_argument("--set-file", default=None,
                    help="default outputs/{scenario}/{slug}/sets/{slug}_merged.set")
    ap.add_argument("--out-dir", default=None,
                    help="default outputs/figures/{scenario}/{slug}/policy_inspection")
    ap.add_argument("--n-policies", type=int, default=3,
                    help="grid columns (highlighted policies, in order)")
    ap.add_argument("--legend", default="side", choices=["side", "bottom"],
                    help="legend placement of the grid figure")
    ap.add_argument("--baseline", action="store_true",
                    help="also render the baseline FFMP alone")
    args = ap.parse_args()

    set_file = Path(args.set_file) if args.set_file else (
        config.OUTPUTS_DIR / args.scenario / args.slug / "sets"
        / f"{args.slug}_merged.set")
    out_dir = Path(args.out_dir) if args.out_dir else config.figure_dir_for(
        args.scenario, args.slug, "policy_inspection")
    out_dir.mkdir(parents=True, exist_ok=True)
    explore_dir = config.figure_dir_for(args.scenario, args.slug, "explore")
    print(f"set file : {set_file}\nout dir  : {out_dir}")

    apply_manuscript_style()
    policies = select_policies(set_file, args.formulation, explore_dir,
                               args.n_policies)

    plot_policy_rules_grid(policies, args.formulation,
                           output_file=out_dir / "rules_grid",
                           legend=args.legend)
    plt.close("all")
    print("[figure] rules_grid.png")

    label, dv = policies[0]
    stem = f"rules_{_slugify(label)}"
    plot_policy_rules_compact(dv, args.formulation, candidate_label=label,
                              output_file=out_dir / stem)
    plt.close("all")
    print(f"[figure] {stem}.png")

    if args.baseline:
        plot_policy_rules_compact(None, args.formulation,
                                  output_file=out_dir / "rules_baseline")
        plt.close("all")
        print("[figure] rules_baseline.png")


if __name__ == "__main__":
    main()
