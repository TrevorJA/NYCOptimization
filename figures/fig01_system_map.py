"""
figures/fig01_system_map.py - Figure 1: Study system and decision context.

Map of the NYC Delaware River Basin showing the three NYC reservoirs (Cannonsville,
Pepacton, Neversink), Montague and Trenton flow targets, NJ diversion point, and
NYC supply tunnels.  Inset: schematic of the 7-objective tradeoff space.

Data: static, hand-drawn or adapted from Hamilton et al. 2024.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_PATH = Path("outputs/manuscript_figures/fig01_system_map.png")


def make_figure() -> None:
    """Generate Figure 1 and save to OUTPUT_PATH."""
    raise NotImplementedError("fig01_system_map: not yet implemented")


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_figure()
    print(f"Saved: {OUTPUT_PATH}")
