"""
run_diagnostics.py - Compute runtime metrics via MOEAFramework.

Runs the MOEAFramework diagnostics pipeline (reference set merging +
MetricsEvaluator) on Borg runtime files. Produces .metrics files that
downstream figure scripts can read.

Usage:
    python scripts/main/run_diagnostics.py [--formulation ffmp] [--seed 1]
    python scripts/main/run_diagnostics.py --slug ffmp_obj7_sal   # explicit slug

Scenario design + moea slug come from the active config (env file); pass
--slug only to target a specific moea slug directly.

Produces (under the active scenario design):
    outputs/{scenario}/{slug}/sets/{slug}_seed{seed}_merged.set
    outputs/{scenario}/{slug}/metrics/seed_{seed}_{slug}_{island}.metrics
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import derive_slug
from src.diagnostics import run_diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Compute MOEA runtime metrics via MOEAFramework"
    )
    parser.add_argument("--slug", type=str, default=None,
                        help="Explicit moea slug (subdirectory + file prefix). "
                             "Defaults to derive_slug(--formulation).")
    parser.add_argument("--formulation", type=str, default="ffmp",
                        help="Formulation identifier; slug derived from it when "
                             "--slug is omitted.")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    slug = args.slug if args.slug else derive_slug(args.formulation)
    run_diagnostics(slug, args.seed)


if __name__ == "__main__":
    main()
