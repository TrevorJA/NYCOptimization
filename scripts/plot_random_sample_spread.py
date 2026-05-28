"""
plot_random_sample_spread.py - Visualize the objective-value spread from
the random-sample diagnostic.

Reads the CSV produced by `random_sample_mpi.py` (or its sequential
equivalent) and produces two figures:

1. Strip-plus-box plot showing each objective's spread across samples,
   with the FFMP baseline highlighted as a horizontal marker.
2. Parallel-coordinates plot showing each sample (and the baseline) as
   a polyline across normalized objective axes.

Reuses `OBJ_AXIS_LABELS` from src/plotting/style.py so axis labels match
the rest of the manuscript figures.

Usage:
    python scripts/plot_random_sample_spread.py \\
        --csv outputs/random_sample_tests/ffmp_obj7_sal/random_samples_seed42_n10.csv

If --csv is omitted, defaults to the most-recently-modified file under
outputs/random_sample_tests/<derived slug>/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import OUTPUTS_DIR, derive_slug, FIGURES_DIR
from src.formulations import get_objective_set
from src.plotting.style import OBJ_AXIS_LABELS


def _resolve_csv_for_slug(slug: str) -> Path | None:
    """Return the most-recently-modified CSV under outputs/.../<slug>/, if any."""
    sample_dir = OUTPUTS_DIR / "random_sample_tests" / slug
    if not sample_dir.exists():
        return None
    csvs = sorted(sample_dir.glob("random_samples_seed*_n*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def _resolve_csv(args) -> Path:
    if args.csv:
        return Path(args.csv).expanduser().resolve()
    csv = _resolve_csv_for_slug(derive_slug(args.formulation))
    if csv is None:
        sys.exit(f"No CSVs found for formulation '{args.formulation}'.")
    return csv


def _resolve_baseline(df: pd.DataFrame, args) -> tuple[pd.Series | None, str]:
    """Resolve the baseline reference series.

    Order of precedence:
      1. --baseline-csv path provided -> load its sample_id=-1 row.
      2. Current df has its own sample_id=-1 row -> use it.
      3. Fall back to the FFMP-family baseline CSV
         (outputs/random_sample_tests/ffmp_obj7_sal/...). Useful for external
         policies (ANN/RBF/tree/spline) that have no canonical baseline of
         their own — the FFMP baseline objective values are still meaningful
         as a reference even though the policy class differs.

    Returns (baseline_series_or_None, label).
    """
    if args.baseline_csv:
        bdf = pd.read_csv(Path(args.baseline_csv).expanduser().resolve()).set_index("sample_id")
        if -1 in bdf.index:
            return bdf.loc[-1], "FFMP baseline (external file)"
    if -1 in df.index:
        return df.loc[-1], "FFMP baseline"
    # Fallback: try the FFMP slug. Use the same env-derived slug suffix
    # (obj7_sal, obj7_sal_sfdv_mult, etc.) by deriving for "ffmp".
    fallback_slug = derive_slug("ffmp")
    fallback_csv = _resolve_csv_for_slug(fallback_slug)
    if fallback_csv is not None:
        bdf = pd.read_csv(fallback_csv).set_index("sample_id")
        if -1 in bdf.index:
            return bdf.loc[-1], f"FFMP baseline (from {fallback_slug})"
    return None, ""


def _short_label(obj_name: str) -> str:
    """Compact axis label: prefer style.OBJ_AXIS_LABELS; fall back to name."""
    return OBJ_AXIS_LABELS.get(obj_name, obj_name)


def plot_spread(df: pd.DataFrame, obj_names: list, directions: list,
                out_path: Path, title: str,
                baseline: pd.Series | None = None,
                baseline_label: str = "FFMP baseline"):
    """Per-objective box+strip plot, baseline highlighted."""
    n_obj = len(obj_names)
    samples = df.drop(index=-1, errors="ignore")

    fig, axes = plt.subplots(1, n_obj, figsize=(2.0 * n_obj, 4.5),
                             sharey=False)
    if n_obj == 1:
        axes = [axes]

    for i, (ax, name, sign) in enumerate(zip(axes, obj_names, directions)):
        if name not in samples.columns:
            ax.text(0.5, 0.5, f"{name}\n(missing)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        col = samples[name].dropna()
        if len(col) == 0:
            ax.text(0.5, 0.5, "all NaN", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Box plot
        bp = ax.boxplot(
            [col.values], widths=0.55, patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="lightsteelblue", alpha=0.6, edgecolor="steelblue"),
            medianprops=dict(color="navy", linewidth=2),
            whiskerprops=dict(color="steelblue"),
            capprops=dict(color="steelblue"),
        )

        # Strip plot (jittered points)
        rng = np.random.default_rng(7)
        jitter = rng.uniform(-0.12, 0.12, size=len(col))
        ax.scatter(np.ones(len(col)) + jitter, col.values,
                   s=22, color="darkorange", alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.5,
                   label=f"random samples (n={len(col)})")

        # Baseline marker
        if baseline is not None and pd.notna(baseline.get(name, np.nan)):
            base_val = float(baseline[name])
            ax.axhline(base_val, color="firebrick", linewidth=1.6, linestyle="--",
                       label=baseline_label, zorder=2)
            ax.plot(1.0, base_val, marker="*", markersize=14, color="firebrick",
                    markeredgecolor="white", zorder=4)

        # Direction arrow in title
        arrow = "↑ better" if sign == 1 else "↓ better"
        ax.set_title(f"{_short_label(name)}\n({arrow})", fontsize=8)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    # One shared legend at the bottom.
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange",
                   markeredgecolor="white", markersize=8, label="random sample"),
        plt.Line2D([0], [0], color="firebrick", linestyle="--", linewidth=1.6,
                   label=baseline_label),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(title, fontsize=11, y=1.07)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_parallel(df: pd.DataFrame, obj_names: list, directions: list,
                  out_path: Path, title: str,
                  baseline: pd.Series | None = None,
                  baseline_label: str = "FFMP baseline"):
    """Parallel-coordinates plot. Baseline highlighted, samples translucent.

    Each axis is normalized so 'up' = preferred direction (consistent with
    the manuscript Pareto plot convention).
    """
    samples = df.drop(index=-1, errors="ignore")
    cols = [c for c in obj_names if c in samples.columns]
    if not cols:
        print("(no plottable objective columns)")
        return

    raw_samples = samples[cols].dropna(how="any").to_numpy(dtype=float)
    raw_baseline = (
        np.array([float(baseline[c]) for c in cols], dtype=float)
        if baseline is not None and not baseline[cols].isna().any()
        else None
    )

    if raw_samples.shape[0] == 0 and raw_baseline is None:
        print("(no rows to plot in parallel coordinates)")
        return

    # Normalize using sample + baseline range so baseline always falls in [0,1].
    stack = raw_samples.copy()
    if raw_baseline is not None:
        stack = (np.vstack([stack, raw_baseline[None, :]])
                 if stack.size else raw_baseline[None, :])
    col_min = stack.min(axis=0)
    col_max = stack.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0

    def _norm(arr):
        out = (arr - col_min) / col_range
        # Flip minimization objectives so low raw -> high normalized (up = best)
        for i, sign in enumerate([directions[obj_names.index(c)] for c in cols]):
            if sign == -1:
                out[..., i] = 1.0 - out[..., i]
        return out

    normed_samples = _norm(raw_samples) if raw_samples.size else None
    normed_baseline = _norm(raw_baseline) if raw_baseline is not None else None

    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 2, 5))
    x = np.arange(len(cols))

    if normed_samples is not None:
        for row in normed_samples:
            ax.plot(x, row, color="steelblue", alpha=0.35, linewidth=1.0)

    if normed_baseline is not None:
        ax.plot(x, normed_baseline, color="firebrick", linewidth=2.4,
                marker="o", markersize=6, label=baseline_label, zorder=10)

    # Top label = best raw value, bottom = worst raw value.
    for i, c in enumerate(cols):
        sign = directions[obj_names.index(c)]
        if sign == 1:  # max: best = col_max, worst = col_min
            top_v, bot_v = col_max[i], col_min[i]
        else:  # min: best = col_min, worst = col_max
            top_v, bot_v = col_min[i], col_max[i]
        ax.text(i, 1.04, _fmt(top_v), ha="center", va="bottom",
                fontsize=8, color="0.25")
        ax.text(i, -0.04, _fmt(bot_v), ha="center", va="top",
                fontsize=8, color="0.25")

    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(c) for c in cols], rotation=0,
                       fontsize=8.5)
    ax.set_ylabel("Preference  (↑ better)", fontsize=9)
    ax.set_ylim(-0.13, 1.13)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    if normed_baseline is not None:
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None,
                        help="Path to random_samples_*.csv. If omitted, "
                             "uses most recent under outputs/random_sample_tests/<slug>/.")
    parser.add_argument("--formulation", default="ffmp")
    parser.add_argument("--outdir", default=None,
                        help="Output figure dir. Default: figures/random_samples/<slug>/")
    parser.add_argument("--baseline-csv", default=None,
                        help="Path to a CSV containing the FFMP baseline row "
                             "(sample_id=-1) to overlay. If omitted, falls back "
                             "to the FFMP slug's CSV when the current df has no "
                             "baseline row (useful for ANN/RBF/tree/spline).")
    args = parser.parse_args()

    csv_path = _resolve_csv(args)
    print(f"Reading: {csv_path}")
    df = pd.read_csv(csv_path).set_index("sample_id")

    obj_set = get_objective_set()
    obj_names = obj_set.names
    directions = obj_set.directions

    slug = derive_slug(args.formulation)
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = FIGURES_DIR / "random_samples" / slug
    outdir.mkdir(parents=True, exist_ok=True)

    baseline, baseline_label = _resolve_baseline(df, args)
    if baseline is None:
        print("(no baseline available — proceeding without overlay)")
    else:
        print(f"Baseline overlay: {baseline_label}")

    stem = csv_path.stem
    title = f"Random sample objective spread — {slug}\n({stem})"

    plot_spread(df, obj_names, directions,
                outdir / f"{stem}_spread.png", title,
                baseline=baseline, baseline_label=baseline_label or "FFMP baseline")
    plot_parallel(df, obj_names, directions,
                  outdir / f"{stem}_parallel.png", title,
                  baseline=baseline, baseline_label=baseline_label or "FFMP baseline")


if __name__ == "__main__":
    main()
