"""
figures/fig05_robustness_degradation.py - Figure 5: Robustness degradation under OSST.

Two-panel figure.
  Panel A: boxplots of objective performance across 200 OSST traces, grouped by
           architecture.
  Panel B: scatter of historical hypervolume vs OSST hypervolume (one point per
           Pareto solution), colored by architecture.  Solutions near the 1:1 line
           are robust; solutions below are fragile.

Data: Phase 2–3 OSST evaluation output.
"""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plotting.style import ARCH_COLORS, ARCH_LABELS, apply_style

OUTPUT_PATH = Path("outputs/manuscript_figures/fig05_robustness_degradation.png")


def make_figure() -> None:
    """Generate Figure 5 and save to OUTPUT_PATH."""
    raise NotImplementedError("fig05_robustness_degradation: not yet implemented")


if __name__ == "__main__":
    apply_style()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_figure()
    print(f"Saved: {OUTPUT_PATH}")
