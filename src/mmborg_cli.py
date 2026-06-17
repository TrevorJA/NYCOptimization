"""
mmborg_cli.py - CLI entry point for MM Borg optimization.

Parses identifier-only command-line arguments and calls run_mmborg(). All
configuration values (algorithm settings, scenario design, objectives, physics)
come from config imports driven by the env file — never from value-carrying CLI
flags — so a run is fully reproducible from versioned config. The only CLI args
are identifiers: the seed (SLURM array index) and the formulation (VR sweep).

Separated from mmborg.py so the module can be imported without argparse.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ACTIVE_MOEA_CONFIG,
    active_scenario_name,
    derive_slug,
    run_output_dir,
)
from src.mmborg import run_mmborg


def main():
    parser = argparse.ArgumentParser(description="Run MM Borg optimization")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (typically the SLURM array index).")
    parser.add_argument("--formulation", type=str, default="ffmp",
                        help="Formulation identifier (e.g. ffmp, ffmp_8).")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--restore", type=str, default=None,
                        help="Path to checkpoint file to restore from")
    args = parser.parse_args()

    scenario = active_scenario_name()
    slug = derive_slug(args.formulation)

    checkpoint_base = None
    if args.checkpoint:
        ckpt_dir = run_output_dir(scenario, slug, "checkpoints")
        checkpoint_base = str(ckpt_dir / f"seed_{args.seed:02d}_{slug}")

    run_mmborg(
        formulation_name=args.formulation,
        seed=args.seed,
        scenario=scenario,
        moea_config=ACTIVE_MOEA_CONFIG,
        slug=slug,
        checkpoint_base=checkpoint_base,
        restore_checkpoint=args.restore,
    )


if __name__ == "__main__":
    main()
