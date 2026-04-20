"""
run_diagnostics.py - Compute runtime metrics via MOEAFramework.

Runs the MOEAFramework diagnostics pipeline (reference set merging +
MetricsEvaluator) on Borg runtime files. Produces .metrics files that
can be plotted separately by plot_results.py.

Usage:
    python scripts/run_diagnostics.py [--slug ffmp] [--seed 1]
    python scripts/run_diagnostics.py --formulation ffmp    # alias for --slug

Produces:
    outputs/optimization/{slug}/sets/{slug}_seed{seed}_merged.set
    outputs/optimization/{slug}/metrics/seed_{seed}_{slug}_{island}.metrics
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics import run_diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Compute MOEA runtime metrics via MOEAFramework"
    )
    parser.add_argument("--slug", type=str, default=None,
                        help="Output slug (subdirectory + file prefix). "
                             "Defaults to --formulation if omitted.")
    parser.add_argument("--formulation", type=str, default="ffmp",
                        help="Fallback slug when --slug is not given.")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    slug = args.slug if args.slug else args.formulation
    run_diagnostics(slug, args.seed)


if __name__ == "__main__":
    main()
