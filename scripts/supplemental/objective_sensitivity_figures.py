"""objective_sensitivity_figures.py - Tables + figures for the random-DV
objective-sensitivity diagnostic.

Pure post-processing of the per-sample CSV written by
``objective_sensitivity_run.py``; it never re-runs simulations, so figures can
be regenerated freely. Implements the two analyses of
the historic objective-sensitivity diagnostic:

  Step 2 - **Discrimination.** Per-objective spread across random policies
           (does the objective carry a Pareto gradient?).
  Step 3 - **Redundancy.** Spearman rank-correlation matrix over all evaluated
           objectives; flag ``|rho| > threshold`` (pairwise-correlation
           thresholding, Dormann et al. 2013).

Outputs (all under ``outputs/supplemental/objective_sensitivity/``):
  correlations/ : discrimination_summary, spearman_matrix, flagged_pairs (CSV)
  figures/      : discrimination (F1), redundancy_heatmap (F2)  [PNG]

Configuration and paths come from ``supplemental_config.py`` — no CLI flags.

Usage:
    python scripts/supplemental/objective_sensitivity_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402  (env-then-config contract)

scfg.configure_historic_env()  # set experiment env before config is imported

from src.objectives_ensemble import ENSEMBLE_OBJECTIVES  # noqa: E402

# The run scores ANNUAL-UNIT (§2) objectives, so columns resolve here.
_ALL_OBJ = ENSEMBLE_OBJECTIVES
from src.plotting.style import (  # noqa: E402
    annotated_corr_heatmap,
    apply_style,
    FIGSIZE_SINGLE,
    label_for as _label,
    save_figure,
)
from src.plotting.parallel_coordinates import custom_parallel_coordinates  # noqa: E402
from src.sensitivity_common import (  # noqa: E402
    resolve_objective_set,
    spearman_and_flagged,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Labels and ordering
# ---------------------------------------------------------------------------

#: Plot order placing each registered diagnostic next to its active
#: counterpart so the discrimination figure reads as a side-by-side comparison.
PREFERRED_ORDER: list[str] = [
    "nyc_delivery_reliability_annual",
    "nyc_delivery_deficit_p99_pct",
    "montague_flow_reliability_annual",
    "montague_flow_deficit_p99_pct",
    "trenton_flow_deficit_p99_pct",
    "trenton_flow_reliability_annual",
    "downstream_flood_exceedance_annual",
    "downstream_flood_days_annual",
    "downstream_flood_days_annual_p99",
    "nyc_storage_min_p01_pct",
    "nj_delivery_reliability_annual",
]

#: Names of the ACTIVE search objectives; every other scored column is a
#: registered diagnostic and is drawn as such.
ACTIVE_NAMES: frozenset = frozenset(resolve_objective_set("active").names)


def _ordered_objectives(columns) -> list:
    """Objective columns present in the data, in PREFERRED_ORDER then extras."""
    present = [c for c in columns if c in _ALL_OBJ]
    ordered = [n for n in PREFERRED_ORDER if n in present]
    ordered += [n for n in present if n not in ordered]
    return ordered


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def discrimination_summary(samples: pd.DataFrame, baseline: pd.Series | None,
                           obj_names: list) -> pd.DataFrame:
    """Per-objective discrimination statistics across random samples.

    Args:
        samples: Random-sample rows only (baseline excluded).
        baseline: Baseline objective values, or None if absent.
        obj_names: Objective columns to summarize, in display order.

    Returns:
        One row per objective with quantiles, IQR, range, NaN/saturation
        fractions, the baseline value, and a no_gradient flag.
    """
    rows = []
    n_total = len(samples)
    for name in obj_names:
        col = samples[name] if name in samples.columns else pd.Series(dtype=float)
        valid = col.dropna()
        n_valid = int(len(valid))
        rng = float(valid.max() - valid.min()) if n_valid else float("nan")
        # Saturation share: fraction of valid samples pinned at the observed
        # extreme (a degenerate, low-information signal even when not NaN).
        if n_valid:
            sat = float(((valid == valid.min()) | (valid == valid.max())).mean())
        else:
            sat = float("nan")
        rows.append({
            "objective": name,
            "direction": _ALL_OBJ[name].direction,
            "active": name in ACTIVE_NAMES,
            "n_valid": n_valid,
            "frac_nan": float(1.0 - n_valid / n_total) if n_total else float("nan"),
            "frac_saturated": sat,
            "min": float(valid.min()) if n_valid else float("nan"),
            "p5": float(valid.quantile(0.05)) if n_valid else float("nan"),
            "p25": float(valid.quantile(0.25)) if n_valid else float("nan"),
            "median": float(valid.median()) if n_valid else float("nan"),
            "p75": float(valid.quantile(0.75)) if n_valid else float("nan"),
            "p95": float(valid.quantile(0.95)) if n_valid else float("nan"),
            "max": float(valid.max()) if n_valid else float("nan"),
            "iqr": float(valid.quantile(0.75) - valid.quantile(0.25)) if n_valid else float("nan"),
            "range": rng,
            "baseline": float(baseline[name]) if baseline is not None and name in baseline else float("nan"),
            # No Pareto gradient: effectively no spread across random policies.
            "no_gradient": bool(n_valid < 2 or (np.isfinite(rng) and rng <= 1e-9)),
        })
    return pd.DataFrame(rows).set_index("objective")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_discrimination(samples: pd.DataFrame, baseline: pd.Series | None,
                       summary: pd.DataFrame, obj_names: list, out_stub: Path):
    """F1: per-objective spread on a shared min-max-normalized [0,1] axis."""
    n = len(obj_names)
    fig, ax = plt.subplots(figsize=(FIGSIZE_SINGLE[0] + 1.5, 0.45 * n + 1.5))

    box_data, positions, labels, valid_mask = [], [], [], []
    for i, name in enumerate(obj_names):
        y = n - i  # top-to-bottom in PREFERRED_ORDER
        col = samples[name].dropna() if name in samples.columns else pd.Series(dtype=float)
        arrow = "↑" if _ALL_OBJ[name].direction == "maximize" else "↓"
        labels.append((y, f"{_label(name)} {arrow}"))
        lo, hi = summary.loc[name, "min"], summary.loc[name, "max"]
        if len(col) >= 1 and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            norm = (col.values - lo) / (hi - lo)
            box_data.append(norm)
            positions.append(y)
            valid_mask.append((name, y, lo, hi))
        # NaN / degenerate annotation on the right margin.
        frac_nan = summary.loc[name, "frac_nan"]
        note = ""
        if frac_nan and frac_nan > 0:
            note = f"NaN {frac_nan:.0%}"
        elif summary.loc[name, "no_gradient"]:
            note = "no gradient"
        if note:
            ax.text(1.02, y, note, va="center", ha="left", fontsize=7,
                    color="firebrick", transform=ax.get_yaxis_transform())

    if box_data:
        bp = ax.boxplot(box_data, positions=positions, vert=False, widths=0.55,
                        patch_artist=True, showfliers=False)
        for patch, (name, *_) in zip(bp["boxes"], valid_mask):
            patch.set_facecolor("steelblue" if name in ACTIVE_NAMES else "0.75")
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")
        if any(name not in ACTIVE_NAMES for name, *_ in valid_mask):
            ax.plot([], [], marker="s", linestyle="none", color="0.75",
                    markersize=9, label="registered diagnostic (inactive)")

    # Baseline marker (normalized with each objective's own min/max).
    if baseline is not None:
        for name, y, lo, hi in valid_mask:
            if name in baseline and np.isfinite(baseline[name]):
                bn = (baseline[name] - lo) / (hi - lo)
                ax.plot(np.clip(bn, 0, 1), y, marker="D", color="darkorange",
                        markersize=6, zorder=5,
                        label="FFMP baseline" if name == valid_mask[0][0] else None)

    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([lab for _, lab in labels], fontsize=8)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Objective value, min–max normalized per objective")
    ax.set_title("Objective discrimination across random policies\n"
                 "(wider box = stronger Pareto gradient; ↑ maximize, "
                 "↓ minimize)", fontsize=10)
    # Leave a blank strip under the last row for the legend.
    ax.set_ylim(-0.7, n + 0.7)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    save_figure(fig, out_stub)
    plt.close(fig)


def fig_parallel_coordinates(samples: pd.DataFrame, baseline: pd.Series | None,
                             obj_names: list, out_png: Path):
    """Parallel-coordinates view of the random-DV objective spread.

    Complements the min-max-normalized discrimination boxplot: each axis keeps
    its NATIVE scale (raw min/max annotated at the ends), so the actual value
    range of every objective is legible while the axes stay aligned. Axes are
    oriented so "up" is the preferred direction. Reuses
    :func:`src.plotting.parallel_coordinates.custom_parallel_coordinates`.
    """
    s = samples[obj_names].dropna()
    if s.empty:
        return
    minmaxs = ["max" if _ALL_OBJ[n].direction == "maximize" else "min"
               for n in obj_names]
    labels = [_label(n) for n in obj_names]
    baseline_raw = None
    if baseline is not None and all(n in baseline for n in obj_names):
        baseline_raw = baseline[obj_names].to_numpy(dtype=float)
    custom_parallel_coordinates(
        s, axis_labels=labels, minmaxs=minmaxs,
        title=f"Objective spread across {len(s)} random policies (raw axes)",
        baseline=baseline_raw, alpha_base=0.35,
        figsize=(1.5 * len(obj_names) + 2, 5),
        save_fig_filename=out_png,
    )


def fig_redundancy_heatmap(spearman: pd.DataFrame, threshold: float,
                           out_stub: Path):
    """F2: annotated Spearman heatmap with |rho| > threshold cells boxed."""
    m = spearman.shape[0]
    fig, ax = plt.subplots(figsize=(0.62 * m + 2.5, 0.62 * m + 2.0))
    im = annotated_corr_heatmap(ax, spearman.values, list(spearman.columns),
                                box_threshold=threshold, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ")
    ax.set_title(f"Objective redundancy (Spearman ρ)\n"
                 f"boxed cells: |ρ| > {threshold}", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_stub)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv = scfg.samples_csv_path()
    if not csv.exists():
        sys.exit(f"ERROR: samples CSV not found: {csv}\n"
                 "Run objective_sensitivity_run.py first.")

    df = pd.read_csv(csv).set_index("sample_id")
    # Every scored objective column, active and diagnostic, in display order.
    obj_names = _ordered_objectives(df.columns)
    if not obj_names:
        sys.exit("ERROR: no objective columns found in the samples CSV.")

    baseline = df.loc[-1] if -1 in df.index else None
    samples = df.drop(index=-1, errors="ignore")

    scfg.CORRELATIONS_DIR.mkdir(parents=True, exist_ok=True)
    scfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- tables ---
    summary = discrimination_summary(samples, baseline, obj_names)
    summary.to_csv(scfg.discrimination_csv_path())

    spearman, flagged, excluded = spearman_and_flagged(
        samples, obj_names, scfg.RHO_FLAG_THRESHOLD)
    spearman.to_csv(scfg.spearman_csv_path())
    flagged.to_csv(scfg.flagged_pairs_csv_path(), index=False)

    apply_style()
    fig_discrimination(samples, baseline, summary, obj_names,
                       scfg.figure_path("discrimination", "pdf").with_suffix(""))
    fig_parallel_coordinates(samples, baseline, obj_names,
                             scfg.figure_path("parallel_coordinates", "png"))
    if spearman.shape[0] >= 2:
        fig_redundancy_heatmap(spearman, scfg.RHO_FLAG_THRESHOLD,
                               scfg.figure_path("redundancy_heatmap", "pdf").with_suffix(""))

    # --- console summary ---
    print(f"=== Objective-sensitivity figures ({csv.name}) ===")
    print(f"  objectives summarized: {len(obj_names)}  "
          f"(random samples: {len(samples)})")
    ng = summary.index[summary["no_gradient"]].tolist()
    if ng:
        print(f"  NO-GRADIENT (drop/reformulate): {ng}")
    if excluded:
        print(f"  excluded from correlation (too few valid / no variance): {excluded}")
    if len(flagged):
        print(f"  flagged |rho| > {scfg.RHO_FLAG_THRESHOLD}: {len(flagged)} pair(s)")
        for _, r in flagged.iterrows():
            print(f"    {r.obj_a} ~ {r.obj_b}: rho={r.rho:+.2f}  "
                  f"keep '{r.keep}'")
    else:
        print(f"  no objective pairs exceed |rho| > {scfg.RHO_FLAG_THRESHOLD}")
    print(f"  tables -> {scfg.CORRELATIONS_DIR}")
    print(f"  figures -> {scfg.FIGURES_DIR}")


if __name__ == "__main__":
    main()
