"""ensemble_objective_sensitivity_figures.py - Diagnostics for the ensemble
objective-sensitivity experiment.

Pure post-processing of the per-realization base-metric matrix written by
``ensemble_objective_sensitivity_run.py`` (shape ``N_DV x N_realizations x
N_objectives``); never re-runs simulations, so figures regenerate freely.
Implements the diagnostics of
``docs/notes/methods/ensemble_objective_sensitivity_experiment.md``:

  (a) **Ensemble-size (K) ranking convergence** — Kendall tau_b(K) per objective,
      sub-sampling K realizations and ranking DVs by satisficing fraction, with
      the full-ensemble ranking as proxy-truth (Bonham et al. 2024).
  (b) **Across-realization operator agreement** — pairwise tau_b among the
      satisficing / mean / p90 / CVaR90 operators, per objective (McPhail 2020).
  (c) **Ensemble-objective redundancy** — Spearman screen over the full-ensemble
      satisficing fractions (Olden & Poff 2003), beside the historic result when
      its CSV is available.
  (d) **Threshold sensitivity** — tau_b of satisficing rankings vs the default
      threshold, across a multiplier grid (table).

Configuration and paths come from ``supplemental_config.py`` — no CLI flags.

Usage:
    python scripts/supplemental/ensemble_objective_sensitivity_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_ensemble_env()  # set experiment env before config is imported

from src.objectives import OBJECTIVES, _cvar_worst_mean  # noqa: E402
from src.objectives_ensemble import (  # noqa: E402
    SatisficingAgg,
    _REGISTRY_SPEC,
    _resolve_thresholds,
)
from src.plotting.style import (  # noqa: E402
    annotated_corr_heatmap,
    apply_style,
    label_for as _label,
    save_figure,
)
from src.sensitivity_common import kendall_tau_b, spearman_and_flagged  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Across-realization operators (display order); the figure axes use these keys.
_OPERATORS = ("satisficing", "mean", "p90", "cvar90")


def _threshold_kind_by_base() -> dict:
    """Map each base objective name -> (satisficing_threshold, kind)."""
    thresholds = _resolve_thresholds()
    return {base: (thresholds[ens], kind)
            for base, ens, kind, _ in _REGISTRY_SPEC}


# ---------------------------------------------------------------------------
# Matrix loading + per-DV operator reductions
# ---------------------------------------------------------------------------

def load_matrix(path: Path):
    """Load the stored matrix HDF5.

    Returns:
        ``(metrics, obj_names)`` where ``metrics`` is ``(n_dv, n_real, n_obj)``.
    """
    with h5py.File(path, "r") as f:
        metrics = f["metrics"][...]
        obj_names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f["objective_names"][...]]
    return metrics, obj_names


def _satisficing_scores(values: np.ndarray, threshold: float, kind: str) -> np.ndarray:
    """Per-DV satisficing fraction over the realization axis of ``values``.

    Args:
        values: ``(n_dv, n_real)`` per-realization base-metric values.
        threshold: Satisficing level.
        kind: ``"ge"`` or ``"le"``.

    Returns:
        ``(n_dv,)`` satisficing fractions (higher = better).
    """
    agg = SatisficingAgg(threshold=threshold, kind=kind)
    return np.array([agg(row) for row in values], dtype=float)


def operator_scores(values: np.ndarray, threshold: float, kind: str,
                    direction: str) -> dict:
    """Per-DV scalar under each operator, oriented so higher = better.

    The tail operators (p90, CVaR90) summarize the *unfavorable* decile: a
    "loss" is formed so that larger = worse (negated for maximize objectives),
    the tail of the loss is taken, then negated back to a higher-is-better score.
    This keeps all operators on a common orientation so tau_b reads as agreement.

    Args:
        values: ``(n_dv, n_real)`` per-realization base-metric values.
        threshold: Satisficing level (for the satisficing operator).
        kind: ``"ge"``/``"le"`` for the satisficing operator.
        direction: Base-objective direction, ``"maximize"`` or ``"minimize"``.

    Returns:
        Dict operator -> ``(n_dv,)`` higher-is-better score array.
    """
    sign = 1.0 if direction == "maximize" else -1.0
    loss = -sign * values  # larger loss = worse
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_score = sign * np.nanmean(values, axis=1)
        p90_loss = np.nanpercentile(loss, 90, axis=1)
        cvar = np.array([_cvar_worst_mean(row[np.isfinite(row)], 0.10)
                         if np.isfinite(row).any() else np.nan for row in loss])
    return {
        "satisficing": _satisficing_scores(values, threshold, kind),
        "mean": mean_score,
        "p90": -p90_loss,
        "cvar90": -cvar,
    }


# ---------------------------------------------------------------------------
# (a) K-convergence
# ---------------------------------------------------------------------------

def tau_vs_k(metrics: np.ndarray, obj_names: list, tk: dict) -> pd.DataFrame:
    """Kendall tau_b(K) of satisficing rankings vs the full-ensemble ranking."""
    n_real = metrics.shape[1]
    rng = np.random.default_rng(scfg.ENS_K_SUBSAMPLE_SEED)
    rows = []
    for o, name in enumerate(obj_names):
        threshold, kind = tk[name]
        vals = metrics[:, :, o]                       # (n_dv, n_real)
        truth = _satisficing_scores(vals, threshold, kind)
        for k in scfg.ENS_K_GRID:
            k = min(int(k), n_real)
            if k >= n_real:
                rows.append({"objective": name, "K": k, "tau_mean": 1.0,
                             "tau_min": 1.0, "tau_max": 1.0})
                continue
            taus = []
            for _ in range(scfg.ENS_K_SUBSAMPLE_REPEATS):
                idx = rng.choice(n_real, size=k, replace=False)
                sub = _satisficing_scores(vals[:, idx], threshold, kind)
                taus.append(kendall_tau_b(sub, truth))
            taus = np.array(taus, dtype=float)
            # All repeats can be NaN when rankings are constant (degenerate at
            # small N_DV / K) — report NaN without the empty-slice warnings.
            if not np.isfinite(taus).any():
                tau_mean = tau_min = tau_max = float("nan")
            else:
                tau_mean = float(np.nanmean(taus))
                tau_min = float(np.nanmin(taus))
                tau_max = float(np.nanmax(taus))
            rows.append({"objective": name, "K": k, "tau_mean": tau_mean,
                         "tau_min": tau_min, "tau_max": tau_max})
    return pd.DataFrame(rows)


def fig_tau_vs_k(df: pd.DataFrame, out_stub: Path) -> None:
    """F(a): tau_b(K) convergence, one line per objective with a min-max band."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("tab10")
    for i, (name, g) in enumerate(df.groupby("objective", sort=False)):
        g = g.sort_values("K")
        color = cmap(i % 10)
        ax.plot(g["K"], g["tau_mean"], marker="o", color=color, label=_label(name))
        ax.fill_between(g["K"], g["tau_min"], g["tau_max"], color=color, alpha=0.15)
    ax.axhline(0.9, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("Ensemble size K (realizations sub-sampled)")
    ax.set_ylabel(r"Kendall $\tau_b$ vs full-ensemble ranking")
    ax.set_title("Policy-ranking convergence with ensemble size\n"
                 "(band = min–max over sub-sample repeats)", fontsize=10)
    ax.set_ylim(min(0.0, df["tau_min"].min()) - 0.02, 1.02)
    ax.legend(loc="lower right", fontsize=7, frameon=True, ncol=2)
    fig.tight_layout()
    save_figure(fig, out_stub)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (a') split-half reliability  (N-independent companion to tau_vs_k)
# ---------------------------------------------------------------------------

def tau_split_half(metrics: np.ndarray, obj_names: list, tk: dict) -> pd.DataFrame:
    """Split-half reliability: Kendall tau_b between two *independent* K-subsample
    rankings, per objective.

    Unlike :func:`tau_vs_k` (which ranks a K-subsample against the full-ensemble
    "truth", so tau -> 1 trivially as K -> n_real and the apparent convergence
    point is anchored to whichever ``n_real`` was simulated), this draws two
    DISJOINT random sub-samples of size K and correlates the DV rankings they
    produce. It measures whether a K-realization ranking is *reproducible* — an
    absolute property that depends only on K and the signal/noise, not on the
    truth-ensemble size (requires ``2K <= n_real``). The K at which this plateaus
    is the honest, N-independent convergence point. Mirrors the split-sample
    reliability idea of Bonham et al. (2024).
    """
    n_real = metrics.shape[1]
    max_k = n_real // 2
    ks = sorted({int(k) for k in scfg.ENS_K_GRID if int(k) <= max_k} | {max_k})
    # Distinct RNG stream from tau_vs_k so the two diagnostics are independent.
    rng = np.random.default_rng(scfg.ENS_K_SUBSAMPLE_SEED + 101)
    rows = []
    for o, name in enumerate(obj_names):
        threshold, kind = tk[name]
        vals = metrics[:, :, o]                       # (n_dv, n_real)
        for k in ks:
            taus = []
            for _ in range(scfg.ENS_K_SUBSAMPLE_REPEATS):
                perm = rng.permutation(n_real)
                a = _satisficing_scores(vals[:, perm[:k]], threshold, kind)
                b = _satisficing_scores(vals[:, perm[k:2 * k]], threshold, kind)
                taus.append(kendall_tau_b(a, b))
            taus = np.array(taus, dtype=float)
            if not np.isfinite(taus).any():
                tau_mean = tau_min = tau_max = float("nan")
            else:
                tau_mean = float(np.nanmean(taus))
                tau_min = float(np.nanmin(taus))
                tau_max = float(np.nanmax(taus))
            rows.append({"objective": name, "K": k, "tau_mean": tau_mean,
                         "tau_min": tau_min, "tau_max": tau_max})
    return pd.DataFrame(rows)


def fig_tau_split_half(df: pd.DataFrame, out_stub: Path) -> None:
    """F(a'): split-half tau_b(K) between two independent K-subsamples."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("tab10")
    for i, (name, g) in enumerate(df.groupby("objective", sort=False)):
        g = g.sort_values("K")
        color = cmap(i % 10)
        ax.plot(g["K"], g["tau_mean"], marker="o", color=color, label=_label(name))
        ax.fill_between(g["K"], g["tau_min"], g["tau_max"], color=color, alpha=0.15)
    ax.axhline(0.9, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("Sub-sample size K (two disjoint K-samples; needs 2K ≤ n_real)")
    ax.set_ylabel(r"Kendall $\tau_b$ between independent K-subsamples")
    ax.set_title("Split-half ranking reliability (N-independent)\n"
                 "absolute reproducibility of a K-realization ranking", fontsize=10)
    ax.set_ylim(min(0.0, float(np.nanmin(df["tau_min"].values))) - 0.02, 1.02)
    ax.legend(loc="lower right", fontsize=7, frameon=True, ncol=2)
    fig.tight_layout()
    save_figure(fig, out_stub)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (b) operator agreement
# ---------------------------------------------------------------------------

def operator_agreement(metrics: np.ndarray, obj_names: list, tk: dict):
    """Pairwise tau_b among operators, per objective.

    Returns:
        ``(taus_by_obj, long_df)`` where ``taus_by_obj[name]`` is a 4x4 array in
        ``_OPERATORS`` order and ``long_df`` is the tidy table.
    """
    taus_by_obj, rows = {}, []
    n_op = len(_OPERATORS)
    for o, name in enumerate(obj_names):
        threshold, kind = tk[name]
        direction = OBJECTIVES[name].direction
        scores = operator_scores(metrics[:, :, o], threshold, kind, direction)
        mat = np.full((n_op, n_op), np.nan)
        for ia, a in enumerate(_OPERATORS):
            mat[ia, ia] = 1.0
            for ib in range(ia + 1, n_op):
                b = _OPERATORS[ib]
                tau = kendall_tau_b(scores[a], scores[b])
                mat[ia, ib] = mat[ib, ia] = tau
                rows.append({"objective": name, "op_a": a, "op_b": b,
                             "tau_b": tau})
        taus_by_obj[name] = mat
    return taus_by_obj, pd.DataFrame(rows)


def fig_operator_agreement(taus_by_obj: dict, out_stub: Path) -> None:
    """F(b): small-multiple 4x4 operator-agreement heatmaps, one per objective."""
    names = list(taus_by_obj)
    n = len(names)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 2.9 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    im = None
    for idx, name in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        ax.axis("on")
        # Operators are plotted verbatim (not objective names), so label_fn=str.
        im = annotated_corr_heatmap(ax, taus_by_obj[name], _OPERATORS,
                                    label_fn=str, fontsize=6)
        ax.set_title(_label(name), fontsize=8)
    fig.suptitle(r"Across-realization operator agreement (Kendall $\tau_b$ of "
                 "DV rankings)", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label=r"$\tau_b$")
    save_figure(fig, out_stub)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (c) redundancy
# ---------------------------------------------------------------------------

def full_satisficing_frame(metrics: np.ndarray, obj_names: list,
                           tk: dict) -> pd.DataFrame:
    """DataFrame of full-ensemble satisficing fractions (rows = DVs, cols = obj)."""
    data = {}
    for o, name in enumerate(obj_names):
        threshold, kind = tk[name]
        data[name] = _satisficing_scores(metrics[:, :, o], threshold, kind)
    return pd.DataFrame(data)


def fig_redundancy(spearman: pd.DataFrame, threshold: float, out_stub: Path) -> None:
    """F(c): ensemble-objective Spearman redundancy heatmap (single panel).

    Compare against the historic experiment's ``redundancy_heatmap`` figure
    rather than re-rendering it here (the historic panel is that experiment's
    deliverable; duplicating it would just repeat the same matrix).
    """
    m = spearman.shape[0]
    fig, ax = plt.subplots(figsize=(0.62 * m + 2.5, 0.62 * m + 2.0))
    im = annotated_corr_heatmap(ax, spearman.values, list(spearman.columns),
                                box_threshold=threshold, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Spearman $\rho$")
    ax.set_title(r"Ensemble-objective redundancy (Spearman $\rho$)" "\n"
                 rf"boxed cells: $|\rho| > {threshold}$", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_stub)
    plt.close(fig)


# ---------------------------------------------------------------------------
# (d) threshold sensitivity
# ---------------------------------------------------------------------------

def threshold_sensitivity(metrics: np.ndarray, obj_names: list,
                          tk: dict) -> pd.DataFrame:
    """tau_b of satisficing rankings vs the default-threshold ranking, per multiplier."""
    rows = []
    for o, name in enumerate(obj_names):
        base_thr, kind = tk[name]
        vals = metrics[:, :, o]
        ref = _satisficing_scores(vals, base_thr, kind)
        row = {"objective": name, "default_threshold": base_thr}
        for m in scfg.ENS_THRESHOLD_MULTIPLIERS:
            scores = _satisficing_scores(vals, base_thr * m, kind)
            row[f"tau_x{m:g}"] = kendall_tau_b(scores, ref)
        rows.append(row)
    return pd.DataFrame(rows).set_index("objective")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    matrix_path = scfg.ensemble_matrix_path()
    if not matrix_path.exists():
        sys.exit(f"ERROR: matrix not found: {matrix_path}\n"
                 "Run ensemble_objective_sensitivity_run.py first.")

    metrics, obj_names = load_matrix(matrix_path)
    tk = _threshold_kind_by_base()
    missing = [n for n in obj_names if n not in tk]
    if missing:
        sys.exit(f"ERROR: objectives without satisficing thresholds: {missing}")

    scfg.ENS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    scfg.ENS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    # (a) K-convergence (truth-anchored: tau vs full-ensemble ranking)
    k_df = tau_vs_k(metrics, obj_names, tk)
    k_df.to_csv(scfg.ensemble_table_path("tau_vs_k"), index=False)
    fig_tau_vs_k(k_df, scfg.ensemble_figure_path("tau_vs_k", "pdf").with_suffix(""))

    # (a') split-half reliability (N-independent: tau between two K-subsamples)
    sh_df = tau_split_half(metrics, obj_names, tk)
    sh_df.to_csv(scfg.ensemble_table_path("tau_split_half"), index=False)
    fig_tau_split_half(
        sh_df, scfg.ensemble_figure_path("tau_split_half", "pdf").with_suffix(""))

    # (b) operator agreement
    taus_by_obj, op_long = operator_agreement(metrics, obj_names, tk)
    op_long.to_csv(scfg.ensemble_table_path("operator_agreement"), index=False)
    fig_operator_agreement(
        taus_by_obj,
        scfg.ensemble_figure_path("operator_agreement", "pdf").with_suffix(""))

    # (c) redundancy
    sat_df = full_satisficing_frame(metrics, obj_names, tk)
    spearman, flagged, excluded = spearman_and_flagged(
        sat_df, obj_names, scfg.ENS_RHO_FLAG_THRESHOLD)
    spearman.to_csv(scfg.ensemble_table_path("redundancy_spearman"))
    flagged.to_csv(scfg.ensemble_table_path("redundancy_flagged"), index=False)
    if spearman.shape[0] >= 2:
        fig_redundancy(spearman, scfg.ENS_RHO_FLAG_THRESHOLD,
                       scfg.ensemble_figure_path("redundancy", "pdf").with_suffix(""))

    # (d) threshold sensitivity
    thr_df = threshold_sensitivity(metrics, obj_names, tk)
    thr_df.to_csv(scfg.ensemble_table_path("threshold_sensitivity"))

    # --- console summary ---
    print(f"=== Ensemble objective-sensitivity figures ({matrix_path.name}) ===")
    print(f"  DVs x realizations x objectives: {metrics.shape}")
    worst_k = k_df[k_df["K"] == min(scfg.ENS_K_GRID)]
    print(f"  [truth-anchored] tau_b at K={int(min(scfg.ENS_K_GRID))}: "
          + ", ".join(f"{_label(r.objective)}={r.tau_mean:.2f}"
                      for r in worst_k.itertuples()))
    # Split-half: smallest K reaching tau_mean>=0.9 (N-independent reliability),
    # and the value at the largest tested K (= n_real//2).
    sh_maxk = int(sh_df["K"].max())
    print(f"  [split-half] K@tau>=0.9 (max tested K={sh_maxk}):")
    for name, g in sh_df.groupby("objective", sort=False):
        g = g.sort_values("K")
        reached = g[g["tau_mean"] >= 0.9]
        k90 = int(reached["K"].iloc[0]) if len(reached) else None
        at_max = float(g["tau_mean"].iloc[-1])
        k90_str = f"K90={k90}" if k90 is not None else f"K90>{sh_maxk}"
        print(f"    {_label(name):28s} {k90_str:9s}  tau@{sh_maxk}={at_max:.2f}")
    if len(flagged):
        print(f"  redundancy flags |rho|>{scfg.ENS_RHO_FLAG_THRESHOLD}: "
              f"{len(flagged)} pair(s)")
        for _, r in flagged.iterrows():
            print(f"    {r.obj_a} ~ {r.obj_b}: rho={r.rho:+.2f}  keep '{r.keep}'")
    if excluded:
        print(f"  excluded from redundancy (no spread): {excluded}")
    print(f"  tables  -> {scfg.ENS_TABLES_DIR}")
    print(f"  figures -> {scfg.ENS_FIGURES_DIR}")


if __name__ == "__main__":
    main()
