"""
run_diagnostics.py - Compute runtime metrics via MOEAFramework.

Runs the MOEAFramework diagnostics pipeline (reference set merging +
MetricsEvaluator) on Borg runtime files. Produces .metrics files that
can be plotted separately by plot_results.py.

Usage:
    python scripts/run_diagnostics.py [--formulation ffmp] [--seed 1]

Produces:
    outputs/optimization/{formulation}/sets/{formulation}_seed{seed}_merged.set
    outputs/optimization/{formulation}/metrics/seed_{seed}_{formulation}_{island}.metrics
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
    parser.add_argument("--formulation", type=str, default="ffmp")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    run_diagnostics(args.formulation, args.seed)


if __name__ == "__main__":
    main()
