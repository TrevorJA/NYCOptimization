"""objective_sensitivity_figures.py - Tables + figures for the random-DV
objective-sensitivity diagnostic.

Pure post-processing of the per-sample CSV written by
``objective_sensitivity_run.py``; it never re-runs simulations, so figures can
be regenerated freely. Implements the two analyses of
``docs/notes/methods/objective_sensitivity_experiment.md``:

  Step 2 - **Discrimination.** Per-objective spread across random policies
           (does the objective carry a Pareto gradient?).
  Step 3 - **Redundancy** (Olden & Poff 2003 style). Spearman rank-correlation
           matrix over all evaluated objectives; flag ``|rho| > threshold``.

Outputs (all under ``outputs/supplemental/objective_sensitivity/``):
  correlations/ : discrimination_summary, spearman_matrix, flagged_pairs (CSV)
  figures/      : discrimination (F1), redundancy_heatmap (F2)  [PNG + PDF]

Configuration and paths come from ``supplemental_config.py`` — no CLI flags.

Usage:
    python scripts/supplemental/objective_sensitivity_figures.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402  (env-then-config contract)

scfg.configure_historic_env()  # set experiment env before config is imported

from src.objectives import OBJECTIVES  # noqa: E402
from src.plotting.style import apply_style, FIGSIZE_SINGLE  # noqa: E402
from src.sensitivity_common import spearman_and_flagged  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# ---------------------------------------------------------------------------
# Labels and ordering
# ---------------------------------------------------------------------------

#: Compact labels for every registry objective (style.OBJ_SHORT covers only the
#: 7 active ones; the diagnostics need labels too). Falls back to the name.
SHORT_LABELS: dict[str, str] = {
    "nyc_delivery_reliability_weekly": "NYC delivery rel.",
    "nyc_delivery_deficit_cvar90_pct": "NYC deficit CVaR90",
    "nyc_delivery_deficit_max_pct": "NYC deficit max",
    "nj_delivery_reliability_weekly": "NJ delivery rel.",
    "montague_flow_reliability_weekly": "Montague rel.",
    "montague_flow_deficit_cvar90_pct": "Montague deficit CVaR90",
    "montague_flow_deficit_max_pct": "Montague deficit max",
    "trenton_flow_reliability_weekly": "Trenton rel.",
    "trenton_flow_deficit_cvar90_pct": "Trenton deficit CVaR90",
    "downstream_flood_days_minor": "Flood days (minor)",
    "downstream_flood_days_action": "Flood days (action)",
    "downstream_flood_days_major": "Flood days (major)",
    "nyc_storage_p5_pct": "Storage p5",
    "nyc_storage_min_pct": "Storage min",
    "salt_front_intrusion_max_rm": "Salt front max RM",
    "lordville_temp_exceedance_days": "Lordville temp days",
}

#: Plot order grouping each replaced metric next to its stable replacement so
#: the discrimination figure reads as a side-by-side comparison.
PREFERRED_ORDER: list[str] = [
    "nyc_delivery_reliability_weekly",
    "nyc_delivery_deficit_cvar90_pct",
    "nyc_delivery_deficit_max_pct",
    "montague_flow_reliability_weekly",
    "montague_flow_deficit_cvar90_pct",
    "montague_flow_deficit_max_pct",
    "trenton_flow_reliability_weekly",
    "trenton_flow_deficit_cvar90_pct",
    "salt_front_intrusion_max_rm",
    "nj_delivery_reliability_weekly",
    "downstream_flood_days_minor",
    "downstream_flood_days_action",
    "downstream_flood_days_major",
    "nyc_storage_p5_pct",
    "nyc_storage_min_pct",
    "lordville_temp_exceedance_days",
]

def _label(name: str) -> str:
    """Compact display label for an objective name."""
    return SHORT_LABELS.get(name, name)


def _ordered_objectives(columns) -> list:
    """Objective columns present in the data, in PREFERRED_ORDER then extras."""
    present = [c for c in columns if c in OBJECTIVES]
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
            "direction": OBJECTIVES[name].direction,
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
        arrow = "↑" if OBJECTIVES[name].direction == "maximize" else "↓"
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
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
        for med in bp["medians"]:
            med.set_color("black")

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
    ax.set_xlabel("Objective value, min–max normalized per objective "
                  "(0 = sample min, 1 = sample max)")
    ax.set_title("Objective discrimination across random policies\n"
                 "(wider box = stronger Pareto gradient; ↑ maximize, "
                 "↓ minimize)", fontsize=10)
    ax.set_ylim(0.3, n + 0.7)
    if baseline is not None:
        ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stub.with_suffix(f".{ext}"))
    plt.close(fig)


def fig_redundancy_heatmap(spearman: pd.DataFrame, threshold: float,
                           out_stub: Path):
    """F2: annotated Spearman heatmap with |rho| > threshold cells boxed."""
    names = list(spearman.columns)
    m = len(names)
    fig, ax = plt.subplots(figsize=(0.62 * m + 2.5, 0.62 * m + 2.0))

    data = spearman.values.astype(float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels([_label(n) for n in names], rotation=45, ha="right",
                       fontsize=7)
    ax.set_yticklabels([_label(n) for n in names], fontsize=7)

    for i in range(m):
        for j in range(m):
            val = data[i, j]
            if not np.isfinite(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(val) > 0.55 else "black")
            if i != j and abs(val) > threshold:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=1.6))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ")
    ax.set_title(f"Objective redundancy (Spearman ρ)\n"
                 f"boxed cells: |ρ| > {threshold}", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_stub.with_suffix(f".{ext}"))
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
