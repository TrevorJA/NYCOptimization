"""diagnose_hazard_selectors.py - Selector + axis-set + sizing diagnostics for hazard filling.

Supplemental (SI) experiment: characterizes the hazard-filling selection machinery on a
real staged candidate pool, entirely at the selection level (no simulation), so the
campaign selector, its normalization bounds, the retained axis set, and the ensemble
size N are chosen from measured diagnostics rather than asserted. See
``docs/notes/methods/hazard_selector_diagnostics.md`` for the design and findings.

Analysis blocks:

  A. Retained-set report — the axis screen (degenerate drop + near-duplicate dedupe at
     |rho_S| >= 0.95) on the pool image, with the Spearman matrix + cluster tree as a
     redundancy diagnostic.
  1. Selector comparison at the campaign bounds — random / lhs_nn / lhs_assign /
     maximin / eps_cell on the full retained axis set, S seeds each, with a many-seed
     random null band.
  2. Normalization-bounds sweep — the designed selectors re-run under each
     (lo_pct, hi_pct) pair, isolating how the bounds choice moves tail enrichment and
     coverage.
  3. Sub-pool draw stability — block 1 re-run on disjoint random halves of the pool;
     between-half spread approximates construction variance.
  B. Per-axis marginal coverage + tail enrichment at the full retained set — the
     mechanism metric (LHS stratifies every axis regardless of dimension), vs the null.
  C. Snap behavior vs dimension — nested axis sets m4 ⊂ m6 ⊂ full: snap distances,
     minimum separation, distance-concentration ratio, and lhs_nn vs lhs_assign
     order-dependence at full m.
  D. N-sweep — N × axis-set decision surface: per-axis tail enrichment and
     stratification + joint L2-star vs the matched random null.
  E. Selection invariance / implicit weighting — leave-one-axis-out and
     add-one-axis-back Jaccard overlaps vs the full-set selection, plus per-axis (and
     dry/wet group) contributions to the snap distance.

All configuration is via environment variables (no CLI value flags):

    NYCOPT_SELDIAG_POOL_SLUG   staged pool slug   (default statpool_10yr_n4000_d0)
    NYCOPT_SELDIAG_N           ensemble size N    (default 100)
    NYCOPT_SELDIAG_SEEDS       selector seeds     (default 10)
    NYCOPT_SELDIAG_NULL_SEEDS  random-null seeds  (default 50)

Run after staging the pool hazard image (workflow step 02 with
``NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary``; locally use
``NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY=1`` so only ``hazard_image.npz`` is kept)::

    python scripts/supplemental/diagnose_hazard_selectors.py

Outputs -> ``outputs/supplemental/hazard_selector_diagnostics/{pool_slug}/``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from scengen import selector_diagnostics as sd  # noqa: E402
from scengen import subsample as ss  # noqa: E402
from scengen.diagnostics import load_hazard_image, spearman_clusters  # noqa: E402
from scengen.hazard_filling import screen_hazard_axes  # noqa: E402
from src.plotting.style import apply_style, save_figure  # noqa: E402

POOL_SLUG = os.environ.get("NYCOPT_SELDIAG_POOL_SLUG", "statpool_10yr_n4000_d0")
N_SELECT = int(os.environ.get("NYCOPT_SELDIAG_N", "100"))
N_SEEDS = int(os.environ.get("NYCOPT_SELDIAG_SEEDS", "10"))
N_NULL_SEEDS = int(os.environ.get("NYCOPT_SELDIAG_NULL_SEEDS", "50"))

#: Designed (non-null) selectors, in presentation order.
DESIGNED = ("lhs_nn", "lhs_assign", "maximin", "eps_cell")

#: (lo_pct, hi_pct) pairs for the normalization sweep. (1, 99) is the campaign
#: default; (0, 100) is the full-range sensitivity.
BOUNDS_SWEEP: tuple[tuple[float, float], ...] = ((0.0, 100.0), (0.5, 99.5), (1.0, 99.0), (2.0, 98.0))

#: Sub-pool halves for the draw-stability block.
N_SUBPOOLS = 2

#: Nested axis sets for the dimension / sizing sweeps. ``m4`` is the axis set the
#: superseded cluster-compress screen used to retain (kept as the benchmark for
#: continuity with the earlier battery); ``m6`` adds the next canonical-priority
#: member per tail; ``full`` is the retained set of the live screen.
M4_AXES: tuple[str, ...] = (
    "drought_deficit_volume", "drought_onset_rate",
    "flood_peak_magnitude", "flood_pulse_duration",
)
M6_AXES: tuple[str, ...] = M4_AXES + ("drought_duration", "flood_rise_rate")

#: Ensemble sizes for the sizing decision surface (block D).
N_SWEEP: tuple[int, ...] = (100, 150, 200, 300)

#: Adequacy criterion for the sizing decision: minimum per-axis tail share above
#: the pool P90 (unbiased rule = 0.10; criterion = >= ~3x the null on EVERY axis).
TAIL_CRITERION: float = 0.30

_COLORS = {
    "random": "0.55", "lhs_nn": "#1f6fb4", "lhs_assign": "#7db3d9",
    "maximin": "#2c8c5a", "eps_cell": "#c1272d",
}
_MSET_COLORS = {"m4": "#c9a227", "m6": "#2c8c5a", "full": "#1f6fb4"}


def _out_dir() -> Path:
    out = config.OUTPUTS_DIR / "supplemental" / "hazard_selector_diagnostics" / POOL_SLUG
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def _load_pool() -> tuple[np.ndarray, list[str], dict]:
    """Load the staged pool hazard image and run the axis screen on it."""
    path = config.STAGED_ENSEMBLE_DIR / POOL_SLUG / "hazard_image.npz"
    if not path.exists():
        raise SystemExit(
            f"[seldiag] pool hazard image not staged: {path}. Run workflow step 02 "
            f"(NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary, "
            f"NYCOPT_CANDIDATE_POOL_N=<P>, NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY=1) first."
        )
    img = load_hazard_image(path)
    H_full, candidate_axes = img["H"], list(img["hazard_axes"])
    screen = screen_hazard_axes(H_full, candidate_axes)
    return H_full, candidate_axes, screen


def _sub(H_full: np.ndarray, candidate_axes: list[str], axes: list[str]) -> np.ndarray:
    return H_full[:, [candidate_axes.index(a) for a in axes]]


def _seeds(k: int, offset: int = 0) -> list[int]:
    return [offset + i for i in range(k)]


def _axis_sets(retained: list[str]) -> dict[str, list[str]]:
    """Nested axis sets m4 ⊂ m6 ⊂ full, intersected with the retained set."""
    sets = {
        "m4": [a for a in M4_AXES if a in retained],
        "m6": [a for a in M6_AXES if a in retained],
        "full": list(retained),
    }
    return {k: v for k, v in sets.items() if len(v) >= 3}


def _main_comparison(H: np.ndarray, axes: list[str]) -> tuple[pd.DataFrame, dict]:
    """Block 1: all selectors at campaign bounds; wide random null."""
    table_d, details = sd.run_selector_comparison(
        H, axes, N_SELECT, seeds=_seeds(N_SEEDS), selectors=DESIGNED,
    )
    table_r, details_r = sd.run_selector_comparison(
        H, axes, N_SELECT, seeds=_seeds(N_NULL_SEEDS), selectors=("random",),
    )
    details.update(details_r)
    return pd.concat([table_r, table_d], ignore_index=True), details


def _bounds_sweep(H: np.ndarray, axes: list[str]) -> pd.DataFrame:
    """Block 2: designed selectors under each normalization-bounds pair."""
    frames = []
    for lo, hi in BOUNDS_SWEEP:
        t, _ = sd.run_selector_comparison(
            H, axes, N_SELECT, seeds=_seeds(N_SEEDS), selectors=DESIGNED,
            lo_pct=lo, hi_pct=hi,
        )
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _subpool_stability(H: np.ndarray, axes: list[str]) -> pd.DataFrame:
    """Block 3: block 1 re-run on disjoint random halves of the pool.

    A random partition of an i.i.d. pool yields independent i.i.d. pools, so
    between-half spread is an honest (cheap) stand-in for pool-re-roll variance.
    """
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(H))
    frames = []
    for h, part in enumerate(np.array_split(perm, N_SUBPOOLS)):
        Hh = H[np.sort(part)]
        t, _ = sd.run_selector_comparison(
            Hh, axes, N_SELECT, seeds=_seeds(N_SEEDS),
            selectors=("random",) + DESIGNED, pool_label=f"half{h}",
        )
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _per_axis_coverage(H: np.ndarray, axes: list[str]) -> pd.DataFrame:
    """Block B: per-axis marginal coverage + tail enrichment, lhs_nn vs null."""
    records = []
    for selector, seeds in (("lhs_nn", _seeds(N_SEEDS)), ("random", _seeds(N_NULL_SEEDS))):
        for seed in seeds:
            if selector == "lhs_nn":
                rows = ss.absolute_filling_subsample(H, N_SELECT, seed=seed)
            else:
                rows = ss.random_subsample(H, N_SELECT, seed=seed)
            for axis, m in sd.per_axis_selection_metrics(H, rows, axes).items():
                records.append({"selector": selector, "seed": seed, "axis": axis, **m})
    return pd.DataFrame.from_records(records)


def _dimension_sweep(
    H_full: np.ndarray, candidate_axes: list[str], axis_sets: dict[str, list[str]]
) -> pd.DataFrame:
    """Block C: snap behavior at nested axis sets, incl. order-dependence at full m."""
    records = []
    for mset, axes in axis_sets.items():
        H = _sub(H_full, candidate_axes, axes)
        X = ss.minmax_normalize(H)
        for seed in _seeds(N_SEEDS):
            res_nn = sd.select_lhs_nn(X, N_SELECT, seed=seed)
            res_as = sd.select_lhs_assign(X, N_SELECT, seed=seed)
            conc = sd.distance_concentration(X, res_nn.info["snap_distances"], seed=seed)
            lb, ub = np.zeros(X.shape[1]), np.ones(X.shape[1])
            cov = ss.coverage_metrics(X[res_nn.rows], lb, ub)
            records.append({
                "m_set": mset, "m": len(axes), "seed": seed,
                "snap_mean": float(np.mean(res_nn.info["snap_distances"])),
                "snap_p95": float(np.percentile(res_nn.info["snap_distances"], 95)),
                "nn_min_abs": float(cov.get("nn_min", 0.0)),
                "L2_star_abs": float(cov["L2_star_discrepancy"]),
                **conc,
                "jaccard_nn_vs_assign": sd.jaccard(res_nn.rows, res_as.rows),
            })
    return pd.DataFrame.from_records(records)


def _n_sweep(
    H_full: np.ndarray, candidate_axes: list[str], axis_sets: dict[str, list[str]]
) -> pd.DataFrame:
    """Block D: the (axis set × N) sizing decision surface, vs matched random nulls."""
    records = []
    for mset, axes in axis_sets.items():
        H = _sub(H_full, candidate_axes, axes)
        for n in N_SWEEP:
            for selector, seeds in (
                ("lhs_nn", _seeds(N_SEEDS)), ("random", _seeds(N_NULL_SEEDS)),
            ):
                for seed in seeds:
                    if selector == "lhs_nn":
                        rows = ss.absolute_filling_subsample(H, n, seed=seed)
                    else:
                        rows = ss.random_subsample(H, n, seed=seed)
                    per_axis = sd.per_axis_selection_metrics(H, rows, axes)
                    tails = [m["tail_share_p90"] for m in per_axis.values()]
                    kss = [m["ks_to_uniform"] for m in per_axis.values()]
                    X = ss.minmax_normalize(H)
                    lb, ub = np.zeros(X.shape[1]), np.ones(X.shape[1])
                    cov = ss.coverage_metrics(X[rows], lb, ub)
                    records.append({
                        "m_set": mset, "m": len(axes), "n": n,
                        "selector": selector, "seed": seed,
                        "tail_share_min": float(np.min(tails)),
                        "tail_share_mean": float(np.mean(tails)),
                        "ks_mean": float(np.mean(kss)),
                        "L2_star_abs": float(cov["L2_star_discrepancy"]),
                    })
    return pd.DataFrame.from_records(records)


def _invariance(
    H_full: np.ndarray, candidate_axes: list[str], axis_sets: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict]:
    """Block E: LOO / add-one-back selection overlap + per-axis snap contributions."""
    retained = axis_sets["full"]
    H_ret = _sub(H_full, candidate_axes, retained)
    full_rows = {s: ss.absolute_filling_subsample(H_ret, N_SELECT, seed=s)
                 for s in _seeds(N_SEEDS)}

    records = []
    for axis in retained:  # leave-one-axis-out
        axes = [a for a in retained if a != axis]
        H = _sub(H_full, candidate_axes, axes)
        for seed in _seeds(N_SEEDS):
            rows = ss.absolute_filling_subsample(H, N_SELECT, seed=seed)
            records.append({
                "variant": "loo", "axis": axis, "m": len(axes), "seed": seed,
                "jaccard_vs_full": sd.jaccard(rows, full_rows[seed]),
            })
    base = axis_sets.get("m4")
    if base:  # add-one-axis-back from the benchmark base set
        for axis in [a for a in retained if a not in base]:
            axes = base + [axis]
            H = _sub(H_full, candidate_axes, axes)
            for seed in _seeds(N_SEEDS):
                rows = ss.absolute_filling_subsample(H, N_SELECT, seed=seed)
                records.append({
                    "variant": "add_one", "axis": axis, "m": len(axes), "seed": seed,
                    "jaccard_vs_full": sd.jaccard(rows, full_rows[seed]),
                })

    # Per-axis contribution to the snap distance (and dry/wet group shares).
    X = ss.minmax_normalize(H_ret)
    shares_per_seed = [
        sd.snap_axis_contributions(X, full_rows[s], retained, seed=s)
        for s in _seeds(N_SEEDS)
    ]
    mean_shares = {a: float(np.mean([sh[a] for sh in shares_per_seed])) for a in retained}
    dry = sum(v for a, v in mean_shares.items() if a.startswith("drought"))
    contributions = {
        "per_axis": mean_shares,
        "dry_group": float(dry),
        "wet_group": float(1.0 - dry),
    }
    return pd.DataFrame.from_records(records), contributions


###############################################################################
# Figures
###############################################################################

def _fig_selection_scatter(H, axes, details, out) -> None:
    """F1: where each rule lands on the (dry, wet) magnitude plane (seed 0)."""
    dry = next((i for i, a in enumerate(axes) if a.startswith("drought")), 0)
    wet = next((i for i, a in enumerate(axes) if a.startswith("flood")), min(1, len(axes) - 1))
    names = ("random",) + DESIGNED
    fig, ax = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.3), sharex=True, sharey=True)
    for c, name in enumerate(names):
        a = ax[c]
        rows = details[name]["rows"][0]
        a.scatter(H[:, dry], H[:, wet], s=6, c="0.85", edgecolors="none", rasterized=True)
        a.scatter(H[rows, dry], H[rows, wet], s=22, c=_COLORS[name],
                  edgecolors="white", linewidths=0.3)
        a.set_title(name)
        a.set_xlabel(axes[dry])
        if c == 0:
            a.set_ylabel(axes[wet])
    fig.suptitle(f"Selected members by rule (N={N_SELECT}, pool P={len(H)}, seed 0)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, out / "figures" / "F1_selection_scatter")
    plt.close(fig)


def _fig_coverage_vs_null(table, out) -> None:
    """F2: L2-star per rule (both geometries) against the random-null band."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for p, metric in enumerate(("L2_star_abs", "L2_star_cdf")):
        a = ax[p]
        null = table.loc[table.selector == "random", metric]
        a.axhspan(null.mean() - 2 * null.std(), null.mean() + 2 * null.std(),
                  color="0.9", zorder=0, label="random null (±2σ)")
        a.axhline(null.mean(), color="0.6", lw=1, zorder=1)
        for i, name in enumerate(DESIGNED):
            vals = table.loc[table.selector == name, metric]
            a.scatter(np.full(len(vals), i), vals, s=18, c=_COLORS[name], zorder=3)
            a.scatter([i], [vals.mean()], marker="_", s=500, c="black", zorder=4)
        a.set_xticks(range(len(DESIGNED)))
        a.set_xticklabels(DESIGNED, rotation=20)
        a.set_ylabel("L2-star discrepancy")
        a.set_title({"L2_star_abs": "absolute (campaign) geometry",
                     "L2_star_cdf": "rank geometry"}[metric])
        if p == 0:
            a.legend(loc="upper right")
    fig.suptitle("Coverage vs the random null (lower = more uniform)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out / "figures" / "F2_coverage_vs_null")
    plt.close(fig)


def _fig_tail_and_atom(table, out) -> None:
    """F3: tail enrichment + zero-event atom per rule (seed spread)."""
    metrics = [
        ("tail_share_p90", "mean share above pool P90\n(unbiased ≈ 0.10)", 0.10),
        ("corner_share_p90", "share in any-axis P90 corner", None),
        ("zero_event_share_selected", "zero-drought-event share\n(pool line = atom mass)", None),
    ]
    pool_atom = table["zero_event_share_pool"].iloc[0] if "zero_event_share_pool" in table else None
    names = ("random",) + DESIGNED
    fig, ax = plt.subplots(1, len(metrics), figsize=(11.4, 3.6))
    for p, (metric, label, ref) in enumerate(metrics):
        a = ax[p]
        for i, name in enumerate(names):
            vals = table.loc[table.selector == name, metric]
            a.scatter(np.full(len(vals), i), vals, s=16, c=_COLORS[name])
            a.scatter([i], [vals.mean()], marker="_", s=450, c="black")
        if metric == "zero_event_share_selected" and pool_atom is not None:
            ref = pool_atom
        if ref is not None:
            a.axhline(ref, color="0.5", lw=1, ls="--")
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=25)
        a.set_ylabel(label)
    fig.suptitle("Tail enrichment and the dry zero-event atom, by selection rule")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out / "figures" / "F3_tail_and_atom")
    plt.close(fig)


def _fig_snap_and_separation(table, details, out) -> None:
    """F4: anchor snap distances (LHS rules) + realized minimum separation."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    a = ax[0]
    for name in ("lhs_nn", "lhs_assign"):
        snaps = np.concatenate([info["snap_distances"] for info in details[name]["info"]])
        a.hist(snaps, bins=40, histtype="step", lw=1.8, density=True,
               color=_COLORS[name], label=f"{name} (mean {snaps.mean():.3f})")
    a.set_xlabel("anchor→selected distance (unit box)")
    a.set_ylabel("density")
    a.set_title("Snap distances: off-manifold anchor cost")
    a.legend()
    a2 = ax[1]
    names = ("random",) + DESIGNED
    for i, name in enumerate(names):
        vals = table.loc[table.selector == name, "nn_min_abs"]
        a2.scatter(np.full(len(vals), i), vals, s=16, c=_COLORS[name])
        a2.scatter([i], [vals.mean()], marker="_", s=450, c="black")
    a2.set_xticks(range(len(names)))
    a2.set_xticklabels(names, rotation=25)
    a2.set_ylabel("min pairwise separation (abs geometry)")
    a2.set_title("Near-duplicate guard: minimum separation")
    fig.tight_layout()
    save_figure(fig, out / "figures" / "F4_snap_and_separation")
    plt.close(fig)


def _fig_bounds_sweep(sweep, out) -> None:
    """F5: how the normalization bounds move tail enrichment and coverage."""
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    x = [f"({lo:g},{hi:g})" for lo, hi in BOUNDS_SWEEP]
    for p, (metric, label) in enumerate((
        ("tail_share_p90", "mean share above pool P90"),
        ("L2_star_abs", "L2-star, absolute geometry"),
    )):
        a = ax[p]
        for name in DESIGNED:
            means = [
                sweep.loc[(sweep.selector == name) & (sweep.hi_pct == hi), metric].mean()
                for _, hi in BOUNDS_SWEEP
            ]
            a.plot(x, means, "o-", color=_COLORS[name], label=name)
        a.set_xlabel("(lo_pct, hi_pct) normalization bounds")
        a.set_ylabel(label)
        if p == 0:
            a.axhline(0.10, color="0.5", lw=1, ls="--")
            a.legend()
    fig.suptitle("Normalization-bounds sweep (campaign default = (1, 99))")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out / "figures" / "F5_bounds_sweep")
    plt.close(fig)


def _fig_axis_screen(screen, H_full, candidate_axes, out) -> None:
    """F6: Spearman |rho| heatmap + 1-|rho| cluster tree (redundancy diagnostic)."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    axes = screen["spearman_axes"]
    rho = np.asarray(screen["spearman_rho"])
    fig, (a, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                                gridspec_kw={"width_ratios": [1.15, 1]})
    im = a.imshow(np.abs(rho), vmin=0, vmax=1, cmap="magma_r")
    a.set_xticks(range(len(axes)))
    a.set_yticks(range(len(axes)))
    a.set_xticklabels(axes, rotation=45, ha="right", fontsize=7)
    a.set_yticklabels(axes, fontsize=7)
    for i in range(len(axes)):
        for j in range(len(axes)):
            a.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center",
                   fontsize=6, color="white" if abs(rho[i, j]) > 0.5 else "black")
    a.set_title("Spearman ρ between candidate hazard axes")
    fig.colorbar(im, ax=a, fraction=0.046)

    cl = spearman_clusters(_sub(H_full, candidate_axes, axes), axes)
    d = 1.0 - np.abs(np.atleast_2d(cl["rho"]))
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(np.clip((d + d.T) / 2.0, 0.0, None), checks=False),
                method="average")
    dendrogram(Z, labels=axes, ax=a2, color_threshold=0.0,
               above_threshold_color="0.3", leaf_rotation=45, leaf_font_size=7)
    thr = 1.0 - screen["dedupe_threshold"]
    a2.axhline(thr, color="#c1272d", lw=1.2, ls="--",
               label=f"near-duplicate cut (1−|ρ| = {thr:g})")
    a2.set_ylabel("1 − |ρ_S| (average linkage)")
    a2.set_title("Cluster tree (diagnostic only — no reduction below the cut)")
    a2.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save_figure(fig, out / "figures" / "F6_axis_screen")
    plt.close(fig)


def _fig_per_axis_coverage(per_axis, axes, out) -> None:
    """F7: per-axis KS-to-uniform + tail share, lhs_nn seeds vs the null band."""
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
    xs = np.arange(len(axes))
    for p, (metric, label, ref) in enumerate((
        ("ks_to_uniform", "KS distance to uniform (scaled coords)", None),
        ("tail_share_p90", "share above pool P90", 0.10),
    )):
        a = ax[p]
        for i, axis in enumerate(axes):
            null = per_axis.loc[(per_axis.selector == "random") & (per_axis.axis == axis), metric]
            a.errorbar([i - 0.12], [null.mean()], yerr=[2 * null.std()], fmt="o",
                       color=_COLORS["random"], ms=4, capsize=3,
                       label="random null (±2σ)" if i == 0 else None)
            vals = per_axis.loc[(per_axis.selector == "lhs_nn") & (per_axis.axis == axis), metric]
            a.scatter(np.full(len(vals), i + 0.12), vals, s=14, c=_COLORS["lhs_nn"],
                      label="lhs_nn (seeds)" if i == 0 else None)
            a.scatter([i + 0.12], [vals.mean()], marker="_", s=300, c="black")
        if ref is not None:
            a.axhline(ref, color="0.5", lw=1, ls="--")
            a.axhline(TAIL_CRITERION, color="#c1272d", lw=1, ls=":",
                      label=f"criterion ≥ {TAIL_CRITERION:g}")
        a.set_xticks(xs)
        a.set_xticklabels(axes, rotation=40, ha="right", fontsize=7)
        a.set_ylabel(label)
        a.legend(fontsize=7)
    fig.suptitle(f"Per-axis marginal coverage + tail enrichment at the full retained set "
                 f"(N={N_SELECT})")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out / "figures" / "F7_per_axis_coverage")
    plt.close(fig)


def _fig_dimension_sweep(dim, out) -> None:
    """F8: snap behavior vs dimension (nested axis sets)."""
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.6))
    msets = list(dict.fromkeys(dim["m_set"]))
    for p, (metric, label) in enumerate((
        ("snap_mean", "mean anchor→selected distance"),
        ("concentration_ratio", "snap / random-pair distance ratio"),
        ("nn_min_abs", "min pairwise separation"),
    )):
        a = ax[p]
        for i, mset in enumerate(msets):
            sel = dim.loc[dim.m_set == mset]
            vals = sel[metric]
            a.scatter(np.full(len(vals), i), vals, s=16, c=_MSET_COLORS[mset])
            a.scatter([i], [vals.mean()], marker="_", s=450, c="black")
        a.set_xticks(range(len(msets)))
        a.set_xticklabels([f"{m} (m={dim.loc[dim.m_set == m, 'm'].iloc[0]})" for m in msets])
        a.set_ylabel(label)
    jac = dim.groupby("m_set")["jaccard_nn_vs_assign"].mean()
    ax[0].set_title("snap cost")
    ax[1].set_title("distance concentration")
    ax[2].set_title(f"separation (nn vs assign Jaccard: "
                    f"{', '.join(f'{m}={jac[m]:.2f}' for m in msets)})", fontsize=8)
    fig.suptitle(f"Snap behavior vs hazard dimension (N={N_SELECT})")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out / "figures" / "F8_snap_vs_dimension")
    plt.close(fig)


def _fig_n_sweep(nsw, out) -> None:
    """F9: the sizing decision surface — worst-axis tail enrichment + coverage vs N."""
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.6))
    msets = list(dict.fromkeys(nsw["m_set"]))
    for p, (metric, label) in enumerate((
        ("tail_share_min", "min per-axis tail share"),
        ("ks_mean", "mean per-axis KS to uniform"),
        ("L2_star_abs", "joint L2-star (abs geometry)"),
    )):
        a = ax[p]
        for mset in msets:
            sel = nsw.loc[(nsw.m_set == mset) & (nsw.selector == "lhs_nn")]
            means = sel.groupby("n")[metric].mean()
            a.plot(means.index, means.values, "o-", color=_MSET_COLORS[mset],
                   label=f"lhs_nn {mset}")
            null = nsw.loc[(nsw.m_set == mset) & (nsw.selector == "random")]
            nmeans = null.groupby("n")[metric].mean()
            a.plot(nmeans.index, nmeans.values, "--", lw=1, color=_MSET_COLORS[mset],
                   alpha=0.55, label=f"null {mset}")
        if metric == "tail_share_min":
            a.axhline(TAIL_CRITERION, color="#c1272d", lw=1, ls=":",
                      label=f"criterion ≥ {TAIL_CRITERION:g}")
            a.axhline(0.10, color="0.5", lw=1, ls="--")
        a.set_xlabel("ensemble size N")
        a.set_ylabel(label)
        if p == 0:
            a.legend(fontsize=6.5)
    fig.suptitle("Sizing decision surface: N × axis set (dashed = matched random null)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out / "figures" / "F9_n_sweep")
    plt.close(fig)


def _fig_invariance(inv, contributions, out) -> None:
    """F10: selection invariance (LOO / add-one-back) + snap-distance weighting."""
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 3.9))
    a = ax[0]
    loo = inv.loc[inv.variant == "loo"].groupby("axis")["jaccard_vs_full"]
    order = list(loo.mean().sort_values().index)
    a.barh(range(len(order)), [loo.mean()[x] for x in order],
           xerr=[2 * loo.std()[x] for x in order], color="#1f6fb4", height=0.55,
           label="leave-one-axis-out")
    add = inv.loc[inv.variant == "add_one"].groupby("axis")["jaccard_vs_full"].mean()
    for i, axis in enumerate(order):
        if axis in add.index:
            a.plot([add[axis]], [i], "d", color="#c1272d", ms=6,
                   label="add-one-back (from m4)" if i == min(
                       j for j, x in enumerate(order) if x in add.index) else None)
    a.set_yticks(range(len(order)))
    a.set_yticklabels(order, fontsize=7)
    a.set_xlabel("Jaccard overlap with full-set selection")
    a.set_xlim(0, 1)
    a.legend(fontsize=7)
    a.set_title("Selection invariance to single axes")

    a2 = ax[1]
    per = contributions["per_axis"]
    names = list(per)
    colors = ["#8c5a2c" if n.startswith("drought") else "#1f6fb4" for n in names]
    a2.bar(range(len(names)), [per[n] for n in names], color=colors)
    a2.axhline(1.0 / len(names), color="0.5", lw=1, ls="--", label="equal weighting")
    a2.set_xticks(range(len(names)))
    a2.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    a2.set_ylabel("mean share of squared snap distance")
    a2.set_title(f"Implicit axis weighting (dry {contributions['dry_group']:.2f} / "
                 f"wet {contributions['wet_group']:.2f})")
    a2.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, out / "figures" / "F10_invariance")
    plt.close(fig)


###############################################################################
# Driver
###############################################################################

def main() -> None:
    """Run all analysis blocks and write tables, summary, and SI figures."""
    apply_style()
    out = _out_dir()
    H_full, candidate_axes, screen = _load_pool()
    retained = screen["retained"]
    H_ret = _sub(H_full, candidate_axes, retained)
    axis_sets = _axis_sets(retained)
    print(f"[seldiag] pool '{POOL_SLUG}': P={len(H_full)}, retained axes (m="
          f"{len(retained)})={retained}, dropped={list(screen['dropped'])}, "
          f"N={N_SELECT}, seeds={N_SEEDS} (+{N_NULL_SEEDS} null)")

    table, details = _main_comparison(H_ret, retained)
    sweep = _bounds_sweep(H_ret, retained)
    halves = _subpool_stability(H_ret, retained)
    per_axis = _per_axis_coverage(H_ret, retained)
    dim = _dimension_sweep(H_full, candidate_axes, axis_sets)
    nsw = _n_sweep(H_full, candidate_axes, axis_sets)
    inv, contributions = _invariance(H_full, candidate_axes, axis_sets)

    table.to_csv(out / "selector_comparison.csv", index=False)
    sweep.to_csv(out / "normalization_sweep.csv", index=False)
    halves.to_csv(out / "subpool_stability.csv", index=False)
    per_axis.to_csv(out / "per_axis_coverage.csv", index=False)
    dim.to_csv(out / "dimension_sweep.csv", index=False)
    nsw.to_csv(out / "n_sweep.csv", index=False)
    inv.to_csv(out / "selection_invariance.csv", index=False)

    # Criterion table: smallest N per axis set with min per-axis tail share >= criterion.
    adequacy = {}
    for mset in axis_sets:
        sel = nsw.loc[(nsw.m_set == mset) & (nsw.selector == "lhs_nn")]
        means = sel.groupby("n")["tail_share_min"].mean()
        passing = [int(n) for n in means.index if means[n] >= TAIL_CRITERION]
        adequacy[mset] = {
            "min_tail_share_by_n": {int(n): float(means[n]) for n in means.index},
            "smallest_passing_n": min(passing) if passing else None,
        }

    summary = {
        "pool_slug": POOL_SLUG, "P": int(len(H_full)), "n_select": N_SELECT,
        "seeds": N_SEEDS, "null_seeds": N_NULL_SEEDS,
        "axis_screen": {k: v for k, v in screen.items() if k != "spread"},
        "axis_sets": {k: list(v) for k, v in axis_sets.items()},
        "tail_criterion": TAIL_CRITERION,
        "adequacy": adequacy,
        "snap_contributions": contributions,
        "zero_event_share_pool": float(table["zero_event_share_pool"].iloc[0])
        if "zero_event_share_pool" in table else None,
        "jaccard_across_seeds": {
            name: details[name]["jaccard_across_seeds"] for name in details
        },
        "eps_cell": details["eps_cell"]["info"][0] if "eps_cell" in details else None,
        "selector_means_at_campaign_bounds": {
            name: table.loc[table.selector == name]
            .drop(columns=["pool", "selector"]).mean(numeric_only=True).to_dict()
            for name in ("random",) + DESIGNED
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    _fig_selection_scatter(H_ret, retained, details, out)
    _fig_coverage_vs_null(table, out)
    _fig_tail_and_atom(table, out)
    _fig_snap_and_separation(table, details, out)
    _fig_bounds_sweep(sweep, out)
    _fig_axis_screen(screen, H_full, candidate_axes, out)
    _fig_per_axis_coverage(per_axis, retained, out)
    _fig_dimension_sweep(dim, out)
    _fig_n_sweep(nsw, out)
    _fig_invariance(inv, contributions, out)

    print(f"[seldiag] wrote 7 tables + summary.json + 10 figures -> {out}")


if __name__ == "__main__":
    main()
