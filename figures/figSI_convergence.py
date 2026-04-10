"""
figures/figSI_convergence.py - SI Figure S2: Convergence diagnostics.

Hypervolume vs NFE for each architecture across all seeds.  Shows whether the
optimization budget (1M NFE) is sufficient and which architectures are harder to
optimize.  Seed variability (S3) is also computed here as a by-product.

Data: outputs/diagnostics/{arch}/runtime/*.metrics
"""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.plotting.style import ARCH_COLORS, ARCH_LABELS, apply_style

OUTPUT_DIR = Path("outputs/manuscript_figures")


def make_figure() -> None:
    """Generate SI Figure S2 (convergence) and save to OUTPUT_DIR."""
    raise NotImplementedError("figSI_convergence: not yet implemented")


if __name__ == "__main__":
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_figure()
