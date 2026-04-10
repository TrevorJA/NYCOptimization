"""
figures/figSI_lhs_diagnostics.py - SI Figure S1: LHS diagnostic objective spread.

Parallel coordinates and pairwise scatter plots showing the feasible objective space
under random LHS parameterization for each architecture.  Verifies that DV bounds
are reasonable and the objective landscape is non-degenerate.

Data: outputs/optimization/{arch}/sets/lhs_all_*.set
"""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plotting.style import ARCH_COLORS, ARCH_LABELS, OBJ_AXIS_LABELS, OBJ_SHORT, apply_style

OUTPUT_DIR = Path("outputs/manuscript_figures")


def make_figure() -> None:
    """Generate SI Figure S1 and save panels to OUTPUT_DIR."""
    raise NotImplementedError("figSI_lhs_diagnostics: not yet implemented")


if __name__ == "__main__":
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_figure()
