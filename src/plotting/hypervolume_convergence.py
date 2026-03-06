"""
hypervolume_convergence.py - Plot hypervolume vs NFE across seeds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_hypervolume_convergence(
    metrics_dir: Path,
    formulation: str,
    output_file: Path = None,
    figsize: tuple = (8, 5),
):
    """Plot hypervolume convergence across NFE for all seeds.

    Loads .metrics files produced by MOEAFramework MetricsEvaluator.
    Each file is tab/whitespace-delimited with columns including
    NFE and Hypervolume.

    Args:
        metrics_dir: Directory containing .metrics files.
        formulation: Formulation name (for title and file filtering).
        output_file: Path to save figure. If None, displays interactively.
        figsize: Figure size in inches.
    """
    metrics_files = sorted(metrics_dir.glob(f"*_{formulation}*.metrics"))
    if not metrics_files:
        print(f"No metrics files found in {metrics_dir}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for mf in metrics_files:
        seed_label = mf.stem.split("_")[1]
        try:
            df = pd.read_csv(mf, sep=r"\s+", comment="#")
        except Exception as e:
            print(f"Warning: Could not load {mf}: {e}")
            continue

        if "Hypervolume" in df.columns and "NFE" in df.columns:
            ax.plot(
                df["NFE"], df["Hypervolume"],
                alpha=0.6, linewidth=1.0,
                label=f"Seed {seed_label}",
            )

    ax.set_xlabel("Number of Function Evaluations (NFE)")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"Hypervolume Convergence ({formulation})")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
