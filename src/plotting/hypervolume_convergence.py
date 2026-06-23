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


def _read_runtime_nfe(metrics_file: Path, n_rows: int):
    """Return per-snapshot NFE for a .metrics file from its sibling .runtime.

    MOEAFramework .metrics files carry one row per runtime snapshot but no NFE
    column, so the snapshot index alone is a poor x-axis. The matching Borg
    .runtime file (same stem, under ``../runtime/``) lists one ``//NFE=<n>``
    marker per snapshot — these are the true (per-island) evaluation counts,
    and the final marker lands on ``max_evaluations`` rather than a clean
    multiple of the runtime frequency. Returns the list of ints when the marker
    count matches ``n_rows``; otherwise None (caller falls back to the index).
    """
    runtime_file = metrics_file.parent.parent / "runtime" / f"{metrics_file.stem}.runtime"
    if not runtime_file.exists():
        return None
    nfe = []
    with open(runtime_file) as fh:
        for line in fh:
            if line.startswith("//NFE="):
                try:
                    nfe.append(int(line.strip().split("=")[1]))
                except (IndexError, ValueError):
                    pass
    return nfe if len(nfe) == n_rows else None


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

    used_nfe_axis = False
    for mf in metrics_files:
        df = _load_metrics_file(mf)
        if df is None or df.empty:
            print(f"Warning: empty metrics file {mf.name}")
            continue
        if "Hypervolume" not in df.columns:
            continue

        # MOEAFramework .metrics files have no NFE column. Prefer the true
        # per-island NFE read from the sibling .runtime file; fall back to the
        # bare snapshot index only when those markers are unavailable.
        nfe = _read_runtime_nfe(mf, len(df))
        if nfe is not None:
            x = nfe
            used_nfe_axis = True
        else:
            x = df.index.values
        ax.plot(
            x, df["Hypervolume"],
            alpha=0.6, linewidth=1.0,
            label=f"{mf.stem}",
        )

    ax.set_xlabel(
        "Number of Function Evaluations (per island)"
        if used_nfe_axis else "Runtime snapshot index"
    )
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
