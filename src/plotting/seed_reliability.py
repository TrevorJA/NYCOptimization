"""
seed_reliability.py - Assess MOEA reliability across random seeds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_seed_reliability(
    metrics_dir: Path,
    formulation: str,
    output_file: Path = None,
    figsize: tuple = (5, 4),
):
    """Plot final hypervolume distribution across seeds.

    Shows a strip/box plot of the final hypervolume achieved by each
    seed, indicating algorithm reliability.
    """
    metrics_files = sorted(metrics_dir.glob(f"*_{formulation}*.metrics"))
    final_hvs = []
    seed_labels = []

    for mf in metrics_files:
        seed_label = mf.stem.split("_")[1]
        try:
            df = pd.read_csv(mf, sep=r"\s+", comment="#")
        except Exception:
            continue
        if "Hypervolume" in df.columns:
            final_hvs.append(df["Hypervolume"].iloc[-1])
            seed_labels.append(seed_label)

    if not final_hvs:
        print("No hypervolume data for reliability plot.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(final_hvs, vert=True, widths=0.4, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    ax.scatter(
        np.ones(len(final_hvs)), final_hvs,
        alpha=0.7, zorder=3, color="steelblue", s=30,
    )

    ax.set_ylabel("Final Hypervolume")
    ax.set_title(f"Seed Reliability ({formulation})")
    ax.set_xticks([1])
    ax.set_xticklabels([f"{formulation}\n({len(final_hvs)} seeds)"])

    fig.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
