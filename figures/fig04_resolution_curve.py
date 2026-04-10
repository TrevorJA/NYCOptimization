"""
figures/fig04_resolution_curve.py - Figure 4: Resolution-expressiveness curve.

X-axis: effective degrees of freedom (zone count for FFMP variants, DV count for
continuous policies).  Y-axis: hypervolume of the Pareto front normalized to the
combined reference set.  Points for FFMP-3, FFMP-6, FFMP-8, FFMP-10, FFMP-15,
RBF, Tree, ANN.  Reveals where FFMP saturates and whether continuous policies lie
above or on the asymptote.

Data: Phase 1 hypervolumes per architecture (outputs/diagnostics/{arch}/metrics/).
"""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plotting.style import ARCH_COLORS, ARCH_LABELS, apply_style

OUTPUT_PATH = Path("outputs/manuscript_figures/fig04_resolution_curve.png")


def make_figure() -> None:
    """Generate Figure 4 and save to OUTPUT_PATH."""
    raise NotImplementedError("fig04_resolution_curve: not yet implemented")


if __name__ == "__main__":
    apply_style()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_figure()
    print(f"Saved: {OUTPUT_PATH}")
