"""satisfaction_factor_figures.py - Post-hoc reductions of the factor cubes.

Pure post-processing of the satisfaction-factor sweep cubes written by
``satisfaction_factor_run.py`` — never re-simulates. Per design and delivery
objective, reports across the feasible-policy population and the FFMP baseline:

  1. §1 weekly reliability vs factor (population median/IQR + baseline);
  2. failure-year frequency at the shipped k vs factor (same summary);
  3. Kendall tau_b of the policy ranking at each factor against the shipped
     0.99 ranking — the convention-sensitivity verdict.

Outputs -> ``outputs/supplemental/satisfaction_factor/{tables,figures}``.
Configuration lives in ``supplemental_config.py`` (SF_* section).

Usage:
    python scripts/supplemental/satisfaction_factor_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_epsilon_env()
os.environ.setdefault("NYCOPT_SCENARIO_DESIGN", "historic")

from src.objectives_ensemble import _DEFAULT_FAILURE_K  # noqa: E402
from src.sensitivity_common import kendall_tau_b  # noqa: E402
from src.plotting.style import apply_style, label_for, save_figure  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SHIPPED_FACTOR = 0.99


def _load(path: Path) -> dict:
    """Load one factor cube into pooled per-policy arrays."""
    with h5py.File(path, "r") as f:
        counts = f["failing_week_counts"][:]   # (n_dv, n_real, n_f, n_obj, n_u)
        rels = f["weekly_reliability"][:]      # (n_dv, n_real, n_f, n_obj)
        return {
            "design": f.attrs["design"],
            "counts": counts, "rels": rels,
            "factors": list(f["factors"][:]),
            "names": [n.decode() if isinstance(n, bytes) else str(n)
                      for n in f["objective_names"][:]],
            "sample_ids": f["sample_ids"][:],
        }


def summarize(cube: dict) -> pd.DataFrame:
    """Per (objective, factor): population + baseline summaries and tau_b."""
    counts, rels = cube["counts"], cube["rels"]
    n_dv, n_real, n_f, n_obj, n_units = counts.shape
    pooled = counts.reshape(n_dv, n_real, n_f, n_obj, n_units)
    pop = cube["sample_ids"] >= 0
    base = int(np.where(cube["sample_ids"] == -1)[0][0])
    fi_ship = cube["factors"].index(_SHIPPED_FACTOR)
    rows = []
    for j, nm in enumerate(cube["names"]):
        k = _DEFAULT_FAILURE_K[nm]
        # (n_dv, n_f, pooled units) and mean weekly reliability per policy
        cu = pooled[:, :, :, j, :].transpose(0, 2, 1, 3).reshape(n_dv, n_f, -1)
        freq = (cu < k).mean(axis=2)                    # reliability fraction
        wrel = rels[:, :, :, j].mean(axis=1)            # (n_dv, n_f)
        for fi, f in enumerate(cube["factors"]):
            rows.append({
                "design": cube["design"], "objective": nm, "factor": f,
                "k": k,
                "weekly_rel_pop_median": float(np.median(wrel[pop, fi])),
                "weekly_rel_baseline": float(wrel[base, fi]),
                "annual_rel_pop_median": float(np.median(freq[pop, fi])),
                "annual_rel_pop_iqr": float(
                    np.percentile(freq[pop, fi], 75)
                    - np.percentile(freq[pop, fi], 25)),
                "annual_rel_baseline": float(freq[base, fi]),
                "tau_vs_shipped": kendall_tau_b(freq[pop, fi],
                                                freq[pop, fi_ship]),
            })
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame) -> None:
    """One panel row per objective: reliability curves + tau_b vs factor."""
    names = sorted(df["objective"].unique())
    designs = sorted(df["design"].unique())
    fig, axes = plt.subplots(len(names), 2,
                             figsize=(7.0, 2.2 * len(names)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, nm in enumerate(names):
        ax_rel, ax_tau = axes[i]
        for design in designs:
            sub = df[(df["objective"] == nm)
                     & (df["design"] == design)].sort_values("factor")
            ln, = ax_rel.plot(sub["factor"], sub["annual_rel_pop_median"],
                              marker="o", ms=3.5, label=f"{design} (median)")
            ax_rel.fill_between(
                sub["factor"],
                sub["annual_rel_pop_median"] - sub["annual_rel_pop_iqr"] / 2,
                sub["annual_rel_pop_median"] + sub["annual_rel_pop_iqr"] / 2,
                alpha=0.18, color=ln.get_color())
            ax_rel.plot(sub["factor"], sub["annual_rel_baseline"], ls=":",
                        color=ln.get_color())
            ax_tau.plot(sub["factor"], sub["tau_vs_shipped"], marker="o",
                        ms=3.5, color=ln.get_color(), label=design)
        for ax in (ax_rel, ax_tau):
            ax.axvline(_SHIPPED_FACTOR, color="0.35", lw=0.9, ls=(0, (4, 3)))
        ax_rel.set_ylabel(label_for(nm), fontsize=7)
        ax_tau.set_ylim(-0.03, 1.03)
        if i == 0:
            ax_rel.set_title("Annual reliability at shipped k\n"
                             "(median ± IQR/2; dotted = baseline)", fontsize=8)
            ax_tau.set_title(r"$\tau_b$ of ranking vs factor 0.99", fontsize=8)
            ax_tau.legend(fontsize=6, frameon=False)
    axes[-1, 0].set_xlabel("weekly satisfaction factor")
    axes[-1, 1].set_xlabel("weekly satisfaction factor")
    save_figure(fig, scfg.sf_figure_path("satisfaction_factor_sweep"))
    plt.close(fig)


def main() -> None:
    apply_style()
    paths = sorted(scfg.SF_CUBE_DIR.glob(scfg.sf_cube_glob()))
    if not paths:
        sys.exit(f"[sf_figures] no cubes match {scfg.sf_cube_glob()} "
                 f"under {scfg.SF_CUBE_DIR}")
    frames = [summarize(_load(p)) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    scfg.SF_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    scfg.SF_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(scfg.sf_table_path("factor_sweep"), index=False)
    figure(df)
    print(df.to_string(index=False))
    print(f"[sf_figures] -> {scfg.SF_TABLES_DIR} / {scfg.SF_FIGURES_DIR}")


if __name__ == "__main__":
    main()
