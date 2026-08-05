"""robustness_threshold_figures.py - Satisficing-threshold placement diagnostics.

Reduces the baseline FFMP policy's persisted E_test re-eval cube (1,000
theta-SOWs x 25 realizations; step 05 ``--reeval``) into the manuscript-SI
evidence for placing the satisficing thresholds
(``objectives_ensemble._DEFAULT_THRESHOLDS``, shipped as placeholders):

Tables (outputs/supplemental/robustness_threshold_diagnostics/tables/):
  rtd_sow_mean_summary.csv           per-objective quantiles of the SOW-mean dist
  rtd_default_stringency.csv         where each current threshold sits (fraction
                                     satisficing + stringency coordinate, SOW and
                                     pooled-realization units)
  rtd_threshold_sweep.csv            tidy dense sweep: satisficing fraction vs
                                     threshold per objective, defaults/candidates
                                     marked
  rtd_candidate_placements.csv       candidate threshold menu with fractions
  rtd_theta_spearman.csv             (8+3)x(8+3) Spearman: SOW-mean objectives +
                                     DU factors (m, r1, r2)
  rtd_historic_anchor_comparison.csv historic-trace anchor vs cube distribution
                                     (base metrics; annual CSV as reference only)
  rtd_threshold_recommendation.csv   recommended vector + basis + headline flag

Figures (figures/):
  S_rtd_baseline_sow_cdfs        per-objective ECDFs (SOW-mean + pooled
                                 realizations) with thresholds/anchors overlaid
  S_rtd_threshold_sensitivity    satisficing fraction vs threshold (decision
                                 instrument; candidates annotated)
  S_rtd_theta_spearman           DU-factor attribution heatmap
  S_rtd_factor_maps_nyc          theta-plane failure maps for the NYC criteria

The satisfaction rule mirrors ``src.robustness._satisfy`` exactly (inclusive
comparison, non-finite = fail); thresholds/kinds come from the cube's own
``reeval_raw_meta.json`` snapshot (the moving-measuring-stick guard), never the
live registry. The historic anchor comes from the JSON cache written by
``robustness_threshold_anchor.py`` — this script never imports pywrdrb.

Configuration lives in supplemental_config.py (RTD_* section) — no CLI value
flags. Pass 2 (after filling RTD_RECOMMENDED_THRESHOLDS) reruns this script
unchanged.

Usage (never on a login node):
    sbatch workflow/supplemental/robustness_threshold_diagnostics.sh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_rtd_env()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import src.robustness as rob  # noqa: E402
from src.pareto_filter import DEFAULT_STAKEHOLDER_FLOORS  # noqa: E402
from src.plotting.style import (  # noqa: E402
    annotated_corr_heatmap, apply_style, label_for, save_figure,
)

# Okabe-Ito CVD-validated colors, matching the flood-figure conventions.
SOW_COLOR = "#0072B2"        # SOW-mean distribution / pass
FAIL_COLOR = "#D55E00"       # failing SOWs / historic anchor line
RESCUED_COLOR = "#009E73"    # fails current threshold, passes recommended
POOLED_COLOR = "0.72"        # pooled-realization underlay

#: DU-factor display labels (forcing_parameterization.md: log change-factor
#: harmonic with fixed CMIP6 phases).
THETA_LABELS = {
    "m": "θ m (annual mean, log)",
    "r1": "θ r$_1$ (seasonal amplitude)",
    "r2": "θ r$_2$ (semiannual shape)",
}

FLOOD_OBJ = "downstream_flood_exceedance_minor"


###############################################################################
# Pure computation helpers (unit-tested; no I/O)
###############################################################################

def sweep_fractions(values, grid, kind) -> np.ndarray:
    """Satisficing fraction at each grid threshold, mirroring ``_satisfy``.

    Inclusive comparison (``ge``: v >= t; ``le``: v <= t); non-finite values
    count as unsatisfied in the denominator.
    """
    v = np.asarray(values, dtype=float).ravel()
    t = np.asarray(grid, dtype=float).ravel()
    finite = np.isfinite(v)
    with np.errstate(invalid="ignore"):
        if kind == "ge":
            sat = finite[None, :] & (v[None, :] >= t[:, None])
        elif kind == "le":
            sat = finite[None, :] & (v[None, :] <= t[:, None])
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return sat.mean(axis=1)


def sweep_grid(values, extra_points, n_points) -> np.ndarray:
    """Dense threshold grid over the finite support, extended so every value in
    ``extra_points`` (defaults, candidates) lies exactly on a sample."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values to build a sweep grid from")
    extras = np.asarray([e for e in extra_points if np.isfinite(e)], dtype=float)
    lo = min(float(v.min()), *(extras.tolist() or [float(v.min())]))
    hi = max(float(v.max()), *(extras.tolist() or [float(v.max())]))
    grid = np.linspace(lo, hi, int(n_points))
    return np.union1d(grid, extras)


def stringency_coordinate(values, thr, kind) -> float:
    """Quantile position of a threshold: the fraction of finite values that fail
    the criterion marginally (``ge``: mean(v < t); ``le``: mean(v > t)) — the
    same convention as ``compare_designs.default_stringency``."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if kind == "ge":
        return float(np.mean(v < thr))
    if kind == "le":
        return float(np.mean(v > thr))
    raise ValueError(f"unknown kind {kind!r}")


def theta_for_sows(theta_params, theta_realization_ids, cube_realization_ids,
                   sow_ids, sow_labels) -> np.ndarray:
    """One DU-factor row per SOW, joined by realization id (never positional).

    Asserts every cube realization maps to a theta row and that theta is
    constant within each SOW block (25 realizations share one forcing profile).

    Returns:
        ``(n_sow, n_theta)`` array in ``sow_labels`` order.
    """
    theta_params = np.asarray(theta_params, dtype=float)
    row_of = {int(r): i for i, r in enumerate(theta_realization_ids)}
    by_sow: dict[int, list[int]] = {}
    for rid, sid in zip(cube_realization_ids, sow_ids):
        by_sow.setdefault(int(sid), []).append(int(rid))

    out = np.full((len(sow_labels), theta_params.shape[1]), np.nan)
    for g, s in enumerate(sow_labels):
        rids = by_sow.get(int(s))
        if not rids:
            raise ValueError(f"SOW {s} has no realizations in the cube")
        missing = [r for r in rids if r not in row_of]
        if missing:
            raise ValueError(
                f"realizations {missing[:5]} of SOW {s} missing from forcing npz")
        rows = theta_params[[row_of[r] for r in rids]]
        if not np.allclose(rows, rows[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"theta rows are not constant within SOW {s}")
        out[g] = rows[0]
    return out


def candidate_menu(name, kind, sow_values, default_thr, anchor_val, *,
                   floors=None, flood_anchors=None,
                   quantiles=scfg.RTD_CANDIDATE_QUANTILES) -> dict:
    """Candidate threshold placements for one objective.

    current + historic-trace anchor + SOW-mean distribution quantiles, plus the
    NYC stakeholder floor and the external flood anchors where they apply.
    """
    floors = DEFAULT_STAKEHOLDER_FLOORS if floors is None else floors
    flood_anchors = scfg.RTD_FLOOD_ANCHORS if flood_anchors is None else flood_anchors

    menu = {"current": float(default_thr)}
    if anchor_val is not None and np.isfinite(anchor_val):
        menu["historic_anchor"] = float(anchor_val)
    v = np.asarray(sow_values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    for q in quantiles:
        menu[f"sow_p{int(round(q * 100)):02d}"] = float(np.quantile(v, q))
    if name in floors:
        menu["stakeholder_floor"] = float(floors[name])
    if name == FLOOD_OBJ:
        for key, val in flood_anchors.items():
            menu[f"anchor_{key}"] = float(val)
    return menu


def build_recommendation_table(base_names, kinds, current_thresholds,
                               sow_values_by_name, recommended, basis,
                               headline_delta) -> pd.DataFrame:
    """Per-objective recommendation summary.

    ``headline_impact`` flags a recommendation that changes a headline result:
    |delta fraction| > ``headline_delta``, or a degenerate current fraction
    (<0.01 / >0.99) moving out of degeneracy. NaN recommendation columns on
    pass 1 (``recommended`` empty).
    """
    rows = []
    for name in base_names:
        kind = kinds[name]
        cur = float(current_thresholds[name])
        v = sow_values_by_name[name]
        frac_cur = float(sweep_fractions(v, [cur], kind)[0])
        rec = recommended.get(name)
        if rec is None:
            frac_rec, headline = np.nan, False
        else:
            frac_rec = float(sweep_fractions(v, [float(rec)], kind)[0])
            degen_cur = frac_cur < 0.01 or frac_cur > 0.99
            degen_rec = frac_rec < 0.01 or frac_rec > 0.99
            headline = (abs(frac_rec - frac_cur) > headline_delta
                        or (degen_cur and not degen_rec))
        rows.append({
            "objective": name,
            "kind": kind,
            "current_threshold": cur,
            "frac_sow_at_current": frac_cur,
            "stringency_of_current": stringency_coordinate(v, cur, kind),
            "recommended_threshold": np.nan if rec is None else float(rec),
            "basis": basis.get(name, ""),
            "frac_sow_at_recommended": frac_rec,
            "headline_impact": headline,
        })
    return pd.DataFrame(rows)


###############################################################################
# Loaders
###############################################################################

def load_cube():
    """Baseline cube + SOW-mean collapse: ``(raw, sow_means (n_sow, M), sow_labels)``."""
    raw = rob.load_raw(scfg.RTD_REEVAL_BASELINE_DIR)
    if raw.cube.shape[0] != 1:
        sys.exit(f"[rtd] expected the 1-solution baseline cube, got "
                 f"S={raw.cube.shape[0]}")
    cube_sow, sow_labels = rob.collapse_within_sow(raw, scfg.RTD_WITHIN_SOW_AGG)
    return raw, cube_sow[0], sow_labels


def load_theta(raw, sow_labels) -> tuple[np.ndarray, list]:
    """DU-factor matrix aligned to ``sow_labels``; returns ``(theta, names)``."""
    with np.load(scfg.RTD_FORCING_NPZ) as npz:
        names = [str(n) for n in npz["theta_param_names"]]
        theta = theta_for_sows(npz["theta_params"], npz["realization_ids"],
                               raw.realization_ids, raw.sow_ids, sow_labels)
    if tuple(names) != tuple(scfg.RTD_THETA_NAMES):
        sys.exit(f"[rtd] theta names {names} != expected {scfg.RTD_THETA_NAMES}")
    return theta, names


def load_anchor(base_names) -> dict:
    """Historic-trace base-metric anchor from the JSON cache (hard requirement)."""
    if not scfg.RTD_ANCHOR_CACHE.exists():
        sys.exit(
            f"[rtd] missing anchor cache {scfg.RTD_ANCHOR_CACHE} — run\n"
            "    sbatch workflow/supplemental/robustness_threshold_diagnostics.sh\n"
            "(or scripts/supplemental/robustness_threshold_anchor.py on a "
            "compute node) first.")
    with open(scfg.RTD_ANCHOR_CACHE) as f:
        payload = json.load(f)
    anchor = payload["anchor"]
    missing = [n for n in base_names if n not in anchor]
    if missing:
        sys.exit(f"[rtd] anchor cache missing objectives {missing} — rerun the "
                 "anchor script with NYCOPT_RTD_REFRESH=1")
    return anchor


###############################################################################
# Tables
###############################################################################

def write_sow_mean_summary(base_names, sow_values_by_name) -> pd.DataFrame:
    qs = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    rows = []
    for name in base_names:
        v = np.asarray(sow_values_by_name[name], dtype=float)
        v = v[np.isfinite(v)]
        row = {"objective": name, "n_sow": v.size,
               "min": v.min(), "mean": v.mean(), "max": v.max()}
        row.update({f"p{int(q * 100):02d}": np.quantile(v, q) for q in qs})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(scfg.rtd_table_path("rtd_sow_mean_summary"), index=False)
    return df


def write_default_stringency(raw, base_names, sow_values_by_name) -> pd.DataFrame:
    rows = []
    for k, name in enumerate(base_names):
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        sow_v = sow_values_by_name[name]
        pooled = raw.cube[0, :, k]
        rows.append({
            "objective": name, "kind": kind, "current_threshold": thr,
            "frac_sow": float(sweep_fractions(sow_v, [thr], kind)[0]),
            "frac_realization": float(sweep_fractions(pooled, [thr], kind)[0]),
            "stringency_sow": stringency_coordinate(sow_v, thr, kind),
            "stringency_realization": stringency_coordinate(pooled, thr, kind),
        })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.rtd_table_path("rtd_default_stringency"), index=False)
    return df


def write_threshold_sweep(raw, base_names, sow_values_by_name, menus) -> pd.DataFrame:
    frames = []
    for k, name in enumerate(base_names):
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        sow_v = sow_values_by_name[name]
        pooled = raw.cube[0, :, k]
        menu = menus[name]
        grid = sweep_grid(sow_v, list(menu.values()) + [thr],
                          scfg.RTD_SWEEP_POINTS)
        frac_sow = sweep_fractions(sow_v, grid, kind)
        frac_real = sweep_fractions(pooled, grid, kind)
        labels = ["" for _ in grid]
        for lab, val in menu.items():
            j = int(np.argmin(np.abs(grid - val)))
            labels[j] = f"{labels[j]}+{lab}" if labels[j] else lab
        frames.append(pd.DataFrame({
            "objective": name, "kind": kind, "threshold": grid,
            "frac_sow": frac_sow, "frac_realization": frac_real,
            "stringency_sow": [stringency_coordinate(sow_v, t, kind)
                               for t in grid],
            "is_default": np.isclose(grid, thr),
            "candidate_label": labels,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(scfg.rtd_table_path("rtd_threshold_sweep"), index=False)
    return df


def write_candidate_placements(raw, base_names, sow_values_by_name,
                               menus) -> pd.DataFrame:
    rows = []
    for k, name in enumerate(base_names):
        kind = raw.kinds[name]
        sow_v = sow_values_by_name[name]
        pooled = raw.cube[0, :, k]
        for lab, val in menus[name].items():
            rows.append({
                "objective": name, "kind": kind, "candidate": lab,
                "threshold": val,
                "frac_sow": float(sweep_fractions(sow_v, [val], kind)[0]),
                "frac_realization": float(sweep_fractions(pooled, [val], kind)[0]),
                "stringency_sow": stringency_coordinate(sow_v, val, kind),
            })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.rtd_table_path("rtd_candidate_placements"), index=False)
    return df


def write_theta_spearman(base_names, sow_means, theta, theta_names) -> pd.DataFrame:
    mat = np.hstack([sow_means, theta])          # (n_sow, M + n_theta)
    corr = spearmanr(mat).correlation            # (M+3, M+3)
    labels = list(base_names) + list(theta_names)
    df = pd.DataFrame(corr, index=labels, columns=labels)
    df.to_csv(scfg.rtd_table_path("rtd_theta_spearman"))
    return df


def write_anchor_comparison(raw, base_names, sow_values_by_name,
                            anchor) -> pd.DataFrame:
    rows = []
    for name in base_names:
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        val = float(anchor[name])
        v = np.asarray(sow_values_by_name[name], dtype=float)
        v = v[np.isfinite(v)]
        passes = val >= thr if kind == "ge" else val <= thr
        rows.append({
            "objective": name, "metric_space": "base_weekly_recomputed",
            "metric_name": name, "value": val,
            "current_threshold": thr, "kind": kind,
            "passes_current": bool(passes),
            "sow_mean_quantile_of_value": float(np.mean(v <= val)),
        })
    # Annual-unit search objectives: a DIFFERENT metric space, reference only.
    annual = pd.read_csv(scfg.RTD_BASELINE_ANNUAL_CSV)
    ens_names = raw.meta.get("ensemble_obj_names", [])
    for base, ann in zip(base_names, ens_names):
        if ann in annual.columns:
            rows.append({
                "objective": base, "metric_space": "search_annual_csv",
                "metric_name": ann, "value": float(annual[ann].iloc[0]),
                "current_threshold": np.nan, "kind": "",
                "passes_current": np.nan,
                "sow_mean_quantile_of_value": np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(scfg.rtd_table_path("rtd_historic_anchor_comparison"), index=False)
    return df


###############################################################################
# Figures
###############################################################################

def _ecdf(v):
    v = np.sort(np.asarray(v, dtype=float))
    v = v[np.isfinite(v)]
    return v, np.arange(1, v.size + 1) / v.size


def fig_sow_cdfs(raw, base_names, sow_values_by_name, anchor) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.6))
    for k, (name, ax) in enumerate(zip(base_names, axes.ravel())):
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        xs, ys = _ecdf(raw.cube[0, :, k])
        ax.step(xs, ys, where="post", color=POOLED_COLOR, lw=1.0)
        xs, ys = _ecdf(sow_values_by_name[name])
        ax.step(xs, ys, where="post", color=SOW_COLOR, lw=1.6)
        ax.axvline(thr, color="black", lw=1.2)
        ax.axvline(anchor[name], color=FAIL_COLOR, lw=1.2, ls="--")
        if name in DEFAULT_STAKEHOLDER_FLOORS:
            ax.axvline(DEFAULT_STAKEHOLDER_FLOORS[name], color="0.35",
                       lw=1.2, ls=":")
        if name == FLOOD_OBJ:
            for val in scfg.RTD_FLOOD_ANCHORS.values():
                ax.axvline(val, color="0.35", lw=1.0, ls=":")
        arrow = "pass →" if kind == "ge" else "← pass"
        ha = "left" if kind == "ge" else "right"
        ax.text(thr, 0.04, f" {arrow} ", ha=ha, transform=ax.get_xaxis_transform(),
                fontsize=8, color="black")
        ax.set_title(label_for(name), fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        if k % 4 == 0:
            ax.set_ylabel("fraction of SOWs ≤ x")
    handles = [
        Line2D([], [], color=SOW_COLOR, lw=1.6, label="SOW mean (n=1,000)"),
        Line2D([], [], color=POOLED_COLOR, lw=1.0,
               label="pooled realizations (n=25,000)"),
        Line2D([], [], color="black", lw=1.2, label="current threshold"),
        Line2D([], [], color=FAIL_COLOR, lw=1.2, ls="--",
               label="historic-trace anchor"),
        Line2D([], [], color="0.35", lw=1.2, ls=":",
               label="stakeholder floor / external anchor"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Baseline FFMP on $E_{test}$: per-objective distributions vs "
                 "satisficing thresholds", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(fig, scfg.rtd_figure_path("S_rtd_baseline_sow_cdfs"))
    plt.close(fig)


def fig_threshold_sensitivity(raw, base_names, sow_values_by_name, menus) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.6))
    for k, (name, ax) in enumerate(zip(base_names, axes.ravel())):
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        sow_v = sow_values_by_name[name]
        pooled = raw.cube[0, :, k]
        menu = menus[name]
        grid = sweep_grid(sow_v, list(menu.values()) + [thr],
                          scfg.RTD_SWEEP_POINTS)
        ax.plot(grid, sweep_fractions(pooled, grid, kind), color=POOLED_COLOR,
                lw=1.0, ls="--")
        ax.plot(grid, sweep_fractions(sow_v, grid, kind), color=SOW_COLOR,
                lw=1.6)
        for ref in (0.5, 0.9):
            ax.axhline(ref, color="0.85", lw=0.8, zorder=0)
        # Annotate in fraction order, alternating above/below to limit overlap.
        ordered = sorted(menu.items(),
                         key=lambda kv: sweep_fractions(sow_v, [kv[1]], kind)[0])
        for i, (lab, val) in enumerate(ordered):
            frac = float(sweep_fractions(sow_v, [val], kind)[0])
            is_cur = lab == "current"
            ax.plot(val, frac, marker="o" if is_cur else "D",
                    color="black" if is_cur else FAIL_COLOR,
                    ms=5 if is_cur else 4, zorder=5)
            offset = (3, 5) if i % 2 == 0 else (3, -13)
            ax.annotate(f"{lab}\n{frac:.2f}", xy=(val, frac),
                        xytext=offset, textcoords="offset points",
                        fontsize=5.5, color="0.25", clip_on=True)
        if kind == "le":
            ax.invert_xaxis()  # stringency increases rightward on every panel
        ax.set_title(label_for(name), fontsize=9)
        ax.set_ylim(-0.04, 1.04)
        if k % 4 == 0:
            ax.set_ylabel("satisficing fraction")
        if k >= 4:
            ax.set_xlabel("threshold (natural units)")
    handles = [
        Line2D([], [], color=SOW_COLOR, lw=1.6, label="SOW unit (within-SOW mean)"),
        Line2D([], [], color=POOLED_COLOR, lw=1.0, ls="--",
               label="realization unit (pooled)"),
        Line2D([], [], color="black", marker="o", ls="", label="current threshold"),
        Line2D([], [], color=FAIL_COLOR, marker="D", ls="", label="candidate placement"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Satisficing fraction vs threshold placement (stringency "
                 "increases rightward)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(fig, scfg.rtd_figure_path("S_rtd_threshold_sensitivity"))
    plt.close(fig)


def fig_theta_heatmap(corr_df) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = annotated_corr_heatmap(
        ax, corr_df.values, list(corr_df.index),
        label_fn=lambda n: THETA_LABELS.get(n, label_for(n)),
        box_threshold=0.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title("Baseline SOW-mean objectives vs DU forcing factors", fontsize=10)
    fig.tight_layout()
    save_figure(fig, scfg.rtd_figure_path("S_rtd_theta_spearman"))
    plt.close(fig)


def fig_factor_maps(raw, base_names, sow_values_by_name, theta,
                    theta_names) -> None:
    objectives = [n for n in scfg.RTD_FACTOR_MAP_OBJECTIVES if n in base_names]
    planes = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(len(objectives), len(planes),
                             figsize=(12.6, 3.6 * len(objectives)),
                             squeeze=False)
    rec = scfg.RTD_RECOMMENDED_THRESHOLDS
    for r, name in enumerate(objectives):
        kind, thr = raw.kinds[name], float(raw.thresholds[name])
        v = np.asarray(sow_values_by_name[name], dtype=float)
        ok = sweep_fractions_mask(v, thr, kind)
        rescued = np.zeros_like(ok)
        if name in rec:
            ok_rec = sweep_fractions_mask(v, float(rec[name]), kind)
            rescued = ~ok & ok_rec
        for c, (i, j) in enumerate(planes):
            ax = axes[r, c]
            ax.scatter(theta[~ok & ~rescued, i], theta[~ok & ~rescued, j],
                       s=9, c=FAIL_COLOR, alpha=0.65, lw=0, label="fail")
            if rescued.any():
                ax.scatter(theta[rescued, i], theta[rescued, j], s=9,
                           c=RESCUED_COLOR, alpha=0.8, lw=0,
                           label="pass at recommended only")
            ax.scatter(theta[ok, i], theta[ok, j], s=9, c=SOW_COLOR,
                       alpha=0.65, lw=0, label="pass")
            ax.set_xlabel(THETA_LABELS[theta_names[i]], fontsize=8)
            ax.set_ylabel(THETA_LABELS[theta_names[j]], fontsize=8)
            if c == 1:
                ax.set_title(f"{label_for(name)} — pass/fail at current "
                             "threshold", fontsize=9)
    handles = [
        Line2D([], [], color=SOW_COLOR, marker="o", ls="", label="pass"),
        Line2D([], [], color=FAIL_COLOR, marker="o", ls="", label="fail"),
    ]
    if scfg.RTD_RECOMMENDED_THRESHOLDS:
        handles.append(Line2D([], [], color=RESCUED_COLOR, marker="o", ls="",
                              label="pass at recommended only"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, scfg.rtd_figure_path("S_rtd_factor_maps_nyc"))
    plt.close(fig)


def sweep_fractions_mask(values, thr, kind) -> np.ndarray:
    """Boolean per-value satisfaction at one threshold (same rule as _satisfy)."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    with np.errstate(invalid="ignore"):
        return finite & (v >= thr) if kind == "ge" else finite & (v <= thr)


###############################################################################
# Verification against the shipped scorer
###############################################################################

def verify_against_summary(raw, base_names) -> None:
    """The realization-unit fraction at the current thresholds must reproduce
    objectives_summary.csv (written by the shipped SatisficingAgg at re-eval
    time) — end-to-end check that this script scores the same criterion."""
    path = scfg.RTD_REEVAL_BASELINE_DIR / "objectives_summary.csv"
    if not path.exists():
        print(f"[rtd] WARNING: {path} missing; skipping cross-check", flush=True)
        return
    summary = pd.read_csv(path)
    bad = []
    for k, name in enumerate(base_names):
        col = f"sat__{name}"
        if col not in summary.columns:
            continue
        expected = float(summary[col].iloc[0])
        got = float(sweep_fractions(raw.cube[0, :, k],
                                    [float(raw.thresholds[name])],
                                    raw.kinds[name])[0])
        status = "OK" if abs(got - expected) <= 1e-9 else "MISMATCH"
        print(f"[rtd] verify {name}: summary={expected:.5f} "
              f"recomputed={got:.5f} {status}", flush=True)
        if status == "MISMATCH":
            bad.append(name)
    if bad:
        sys.exit(f"[rtd] FAIL: satisficing recomputation mismatches shipped "
                 f"summary for {bad}")
    print("[rtd] PASS: realization-unit fractions reproduce "
          "objectives_summary.csv", flush=True)


###############################################################################
# Main
###############################################################################

def main() -> None:
    apply_style()
    for d in (scfg.RTD_TABLES_DIR, scfg.RTD_FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    raw, sow_means, sow_labels = load_cube()
    base_names = list(raw.base_names)
    sow_values_by_name = {n: sow_means[:, k] for k, n in enumerate(base_names)}
    anchor = load_anchor(base_names)
    theta, theta_names = load_theta(raw, sow_labels)

    verify_against_summary(raw, base_names)

    menus = {
        n: candidate_menu(n, raw.kinds[n], sow_values_by_name[n],
                          raw.thresholds[n], anchor.get(n))
        for n in base_names
    }

    write_sow_mean_summary(base_names, sow_values_by_name)
    stringency = write_default_stringency(raw, base_names, sow_values_by_name)
    write_threshold_sweep(raw, base_names, sow_values_by_name, menus)
    write_candidate_placements(raw, base_names, sow_values_by_name, menus)
    corr_df = write_theta_spearman(base_names, sow_means, theta, theta_names)
    write_anchor_comparison(raw, base_names, sow_values_by_name, anchor)
    recommendation = build_recommendation_table(
        base_names, raw.kinds, raw.thresholds, sow_values_by_name,
        scfg.RTD_RECOMMENDED_THRESHOLDS, scfg.RTD_RECOMMENDATION_BASIS,
        scfg.RTD_HEADLINE_IMPACT_DELTA)
    recommendation.to_csv(scfg.rtd_table_path("rtd_threshold_recommendation"),
                          index=False)

    fig_sow_cdfs(raw, base_names, sow_values_by_name, anchor)
    fig_threshold_sensitivity(raw, base_names, sow_values_by_name, menus)
    fig_theta_heatmap(corr_df)
    fig_factor_maps(raw, base_names, sow_values_by_name, theta, theta_names)

    print("\n[rtd] === summary (SOW unit, within-SOW "
          f"{scfg.RTD_WITHIN_SOW_AGG}) ===", flush=True)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(stringency.to_string(index=False), flush=True)
        stage = ("pass 2" if scfg.RTD_RECOMMENDED_THRESHOLDS
                 else "pass 1 — recommendations pending")
        print(f"\n[rtd] recommendation table ({stage}):", flush=True)
        print(recommendation.to_string(index=False), flush=True)
    print(f"\n[rtd] tables  -> {scfg.RTD_TABLES_DIR}", flush=True)
    print(f"[rtd] figures -> {scfg.RTD_FIGURES_DIR}", flush=True)


if __name__ == "__main__":
    main()
