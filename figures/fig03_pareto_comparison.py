"""
figures/fig03_pareto_comparison.py - Figure 3: Pareto front comparison (main result).

Parallel coordinates plot showing the Pareto-approximate fronts from all architectures
overlaid.  Each architecture is assigned a distinct color from src/plotting/style.py.
The FFMP default-parameter baseline is marked as a reference.

Data: Phase 1 optimization output (reference set per architecture).
"""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plotting.style import ARCH_COLORS, OBJ_AXIS_LABELS, apply_style

OUTPUT_PATH = Path("outputs/manuscript_figures/fig03_pareto_comparison.png")


def make_figure() -> None:
    """Generate Figure 3 and save to OUTPUT_PATH."""
    raise NotImplementedError("fig03_pareto_comparison: not yet implemented")


if __name__ == "__main__":
    apply_style()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_figure()
    print(f"Saved: {OUTPUT_PATH}")
