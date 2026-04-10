"""
figures/fig02_architecture_schematic.py - Figure 2: Policy architecture spectrum.

Schematic showing all five architectures (FFMP, RBF, Tree, ANN, PLMR) arranged
left-to-right by expressiveness.  Each panel shows a stylized representation of
the policy form.  Annotated with DV counts and interpretability ranking.

Data: conceptual, no simulation needed.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_PATH = Path("outputs/manuscript_figures/fig02_architecture_schematic.png")


def make_figure() -> None:
    """Generate Figure 2 and save to OUTPUT_PATH."""
    raise NotImplementedError("fig02_architecture_schematic: not yet implemented")


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_figure()
    print(f"Saved: {OUTPUT_PATH}")
