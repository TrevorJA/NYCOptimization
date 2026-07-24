"""Render the policy-rules figure for a baseline, random, or optimized policy.

Usage (identifiers only; all settings come from config/env):
    python scripts/main/plot_policy_rules.py --mode baseline
    python scripts/main/plot_policy_rules.py --mode random --seed 42
    python scripts/main/plot_policy_rules.py --mode solution [--set-file PATH]
                                             [--solution-index I]

Figures are written to figures/{scenario}/{slug}/policy_inspection/.
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.formulations import get_bounds, get_n_vars, get_n_objs, get_obj_directions
from src.load.reference_set import load_reference_set
from src.plotting.style import apply_style
from src.plotting.policy_rules import plot_policy_rules


def _compromise_index(objs: np.ndarray) -> int:
    """Index of the min-normalized-distance-to-ideal solution.

    Objectives are un-negated to raw orientation via the active objective
    directions, min-max normalized, and scored against the ideal point.

    Args:
        objs: Borg-orientation objective matrix, shape (n_solutions, n_objs).

    Returns:
        Row index of the compromise solution.
    """
    directions = np.array(get_obj_directions()[: objs.shape[1]])
    raw = objs * np.where(directions > 0, -1.0, 1.0)  # maximize objs stored negated
    span = raw.max(axis=0) - raw.min(axis=0)
    span[span == 0] = 1.0
    norm = (raw - raw.min(axis=0)) / span
    ideal = np.where(directions > 0, norm.max(axis=0), norm.min(axis=0))
    return int(np.argmin(np.linalg.norm(norm - ideal, axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "random", "solution"],
                        default="baseline")
    parser.add_argument("--formulation", default="ffmp")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for --mode random")
    parser.add_argument("--set-file", type=Path, default=None,
                        help="Solution set file for --mode solution")
    parser.add_argument("--solution-index", type=int, default=None,
                        help="Row in the set file; default = compromise solution")
    args = parser.parse_args()

    apply_style()
    scenario = config.active_scenario_name()
    slug = config.derive_slug(args.formulation)
    fig_dir = config.figure_dir_for(scenario, slug, "policy_inspection")

    if args.mode == "baseline":
        dv, label, stub = None, None, "policy_rules_baseline"
    elif args.mode == "random":
        lower, upper = get_bounds(args.formulation)
        rng = np.random.default_rng(args.seed)
        dv = lower + rng.uniform(size=lower.size) * (upper - lower)
        label = f"Random policy (seed {args.seed})"
        stub = f"policy_rules_random_s{args.seed}"
    else:
        set_file = args.set_file or (
            config.run_output_dir(scenario, slug, "sets") / f"{slug}_merged.set"
        )
        if not set_file.exists():
            sys.exit(f"Set file not found: {set_file}")
        dvs, objs = load_reference_set(set_file, get_n_vars(args.formulation),
                                       n_objs=get_n_objs())
        if dvs.shape[0] == 0:
            sys.exit(f"No solutions parsed from {set_file}")
        idx = (args.solution_index if args.solution_index is not None
               else _compromise_index(objs))
        dv = dvs[idx]
        label = "Optimized policy"
        stub = f"policy_rules_solution_{idx}"

    fig = plot_policy_rules(
        dv_vector=dv,
        formulation=args.formulation,
        candidate_label=label,
        output_file=fig_dir / stub,
    )
    plt.close(fig)
    print(f"Saved {fig_dir / stub}.png")


if __name__ == "__main__":
    main()
