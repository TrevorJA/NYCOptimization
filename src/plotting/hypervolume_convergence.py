"""
hypervolume_convergence.py - Plot hypervolume vs NFE across seeds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _load_metrics_file(path: Path):
    """Read a MOEAFramework v5 .metrics file.

    Format: one header line starting with '# ' naming the 6 indicators,
    followed by whitespace-separated numeric rows — one per runtime
    snapshot. Returns a DataFrame with the header names as columns or
    None on failure.
    """
    try:
        with open(path) as fh:
            first = fh.readline().strip()
        if first.startswith("#"):
            cols = first.lstrip("#").split()
        else:
            cols = None
        df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, names=cols)
        return df
    except Exception as e:
        print(f"Warning: could not parse {path}: {e}")
        return None


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
        df = _load_metrics_file(mf)
        if df is None or df.empty:
            print(f"Warning: empty metrics file {mf.name}")
            continue

        # MOEAFramework metrics files have no NFE column. Treat snapshot
        # index as an NFE proxy (runtime_frequency * (1+index)).
        x = df.index.values
        if "Hypervolume" in df.columns:
            ax.plot(
                x, df["Hypervolume"],
                alpha=0.6, linewidth=1.0,
                label=f"{mf.stem}",
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
