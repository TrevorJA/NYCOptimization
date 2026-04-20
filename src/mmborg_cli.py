"""
mmborg_cli.py - CLI entry point for MM Borg optimization.

Parses command-line arguments and calls run_mmborg().
Separated from mmborg.py so the module can be imported without argparse.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BORG_SETTINGS, OUTPUTS_DIR
from src.mmborg import run_mmborg


def main():
    parser = argparse.ArgumentParser(description="Run MM Borg optimization")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--formulation", type=str, default="ffmp")
    parser.add_argument("--slug", type=str, default=None,
                        help="Output directory tag (default: formulation name). "
                             "Use distinct slugs when varying STATE_FEATURES/OBJECTIVES "
                             "so results don't collide under outputs/optimization/.")
    parser.add_argument("--islands", type=int, default=2)
    parser.add_argument("--nfe", type=int, default=None)
    parser.add_argument("--time", type=int, default=None,
                        help="Max wall time in seconds")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--restore", type=str, default=None,
                        help="Path to checkpoint file to restore from")
    parser.add_argument("--runtime-freq", type=int, default=None,
                        help="Runtime snapshot frequency (NFE interval). "
                             "Overrides BORG_SETTINGS['runtime_frequency'] when set.")
    args = parser.parse_args()

    slug = args.slug if args.slug else args.formulation

    checkpoint_base = None
    if args.checkpoint:
        ckpt_dir = OUTPUTS_DIR / "optimization" / slug / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_base = str(
            ckpt_dir / f"seed_{args.seed:02d}_{slug}"
        )

    runtime_freq = args.runtime_freq if args.runtime_freq is not None else BORG_SETTINGS["runtime_frequency"]

    run_mmborg(
        formulation_name=args.formulation,
        seed=args.seed,
        n_islands=args.islands,
        max_evaluations=args.nfe,
        max_time=args.time,
        runtime_frequency=runtime_freq,
        checkpoint_base=checkpoint_base,
        restore_checkpoint=args.restore,
        slug=slug,
    )


if __name__ == "__main__":
    main()
