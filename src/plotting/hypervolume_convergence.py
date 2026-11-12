"""
hypervolume_convergence.py - Hypervolume vs total NFE, one line per seed.

Reads the per-seed coordinator-series metrics written by step 09
(``metrics/seed_SS_{slug}.metrics``, MOEAFramework MetricsEvaluator) and the
``//NFE=`` markers of the sibling coordinator series
(``runtime_coordinator/seed_SS_{slug}.runtime``), which carry total NFE
(per-island NFE x islands).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.plotting.style import save_figure

#: Per-seed line colors (CVD-safe as adjacent pairs).
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def seed_metrics_files(metrics_dir: Path, slug: str) -> list[tuple[int, Path]]:
    """``[(seed, path)]`` of the per-seed metrics files, sorted by seed.

    Matches ``seed_SS_{slug}.metrics`` exactly, so per-island files of the
    retired workflow (``seed_SS_{slug}_{i}.metrics``) are never picked up.
    """
    pattern = re.compile(rf"^seed_(\d+)_{re.escape(slug)}\.metrics$")
    found = []
    for f in Path(metrics_dir).glob("*.metrics"):
        m = pattern.match(f.name)
        if m:
            found.append((int(m.group(1)), f))
    return sorted(found)


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
    """Return per-snapshot NFE for a .metrics file from its sibling series.

    MOEAFramework .metrics files carry one row per entry but no NFE column.
    The matching series file (same stem) is looked up first under
    ``../runtime_coordinator/`` (total NFE across islands) and then under
    ``../runtime/`` (per-island NFE of a retired per-island metrics file).
    Returns the list of ints when the marker count matches ``n_rows``;
    otherwise None (caller falls back to the index).
    """
    base = Path(metrics_file).parent.parent
    for sub in ("runtime_coordinator", "runtime"):
        runtime_file = base / sub / f"{Path(metrics_file).stem}.runtime"
        if runtime_file.exists():
            break
    else:
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
    slug: str,
    output_file: Path = None,
    figsize: tuple = (8, 5),
    reporting_nfe: int | None = None,
):
    """Plot coordinator-archive hypervolume vs total NFE, one line per seed.

    Args:
        metrics_dir: Directory containing ``seed_SS_{slug}.metrics``.
        slug: Run moea slug (selects the files and titles the figure).
        output_file: Path to save figure. If None, displays interactively.
        figsize: Figure size in inches.
        reporting_nfe: Total NFE of the campaign reporting budget, drawn as a
            vertical reference line when given.
    """
    files = seed_metrics_files(metrics_dir, slug)
    if not files:
        print(f"No per-seed metrics files for {slug} in {metrics_dir}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    used_nfe_axis = False
    for i, (seed, mf) in enumerate(files):
        df = _load_metrics_file(mf)
        if df is None or df.empty:
            print(f"Warning: empty metrics file {mf.name}")
            continue
        if "Hypervolume" not in df.columns:
            continue
        nfe = _read_runtime_nfe(mf, len(df))
        if nfe is not None:
            x = nfe
            used_nfe_axis = True
        else:
            x = df.index.values
        ax.plot(x, df["Hypervolume"], color=SEED_COLORS[i % len(SEED_COLORS)],
                linewidth=1.6, label=f"seed {seed:02d}")

    if reporting_nfe is not None and used_nfe_axis:
        ax.axvline(reporting_nfe, color="0.4", lw=1.0, ls="--",
                   label="reporting budget")
    ax.set_xlabel("Function evaluations (total across islands)"
                  if used_nfe_axis else "Runtime snapshot index")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"Coordinator-archive hypervolume ({slug})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if output_file:
        save_figure(fig, output_file)
        plt.close(fig)
    else:
        plt.show()
