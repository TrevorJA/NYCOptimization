"""diagnose_hazard_selectors.py - Selector + normalization diagnostics for hazard filling.

Supplemental (SI) experiment: compares candidate SELECTION RULES for the
hazard-filling design on a real staged candidate pool, and sweeps the robust
normalization bounds, so the campaign selector and its bounds are chosen from
measured diagnostics rather than asserted. See
``docs/notes/methods/hazard_selector_diagnostics.md`` for the design.

Four analysis blocks, all selection-level (no simulation; laptop-scale on a
test pool, HPC-scale on a production pool unchanged):

  1. Selector comparison at the campaign bounds — random / lhs_nn / lhs_assign /
     maximin / eps_cell over ``scengen.selector_diagnostics.SELECTORS``, S seeds
     each, with a many-seed random null band.
  2. Normalization-bounds sweep — the designed selectors re-run under each
     (lo_pct, hi_pct) pair, isolating how the bounds choice moves tail
     enrichment and coverage.
  3. Sub-pool draw stability — the pool is randomly partitioned into disjoint
     halves (independent i.i.d. pools, since the pool is i.i.d.), and block 1 is
     re-run per half; between-half spread approximates construction variance.
  4. Zero-event atom + snap-distance detail for the SI figures.

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
from scengen.diagnostics import load_hazard_image  # noqa: E402
from scengen.hazard_filling import screen_axes  # noqa: E402
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

_COLORS = {
    "random": "0.55", "lhs_nn": "#1f6fb4", "lhs_assign": "#7db3d9",
    "maximin": "#2c8c5a", "eps_cell": "#c1272d",
}


def _out_dir() -> Path:
    out = config.OUTPUTS_DIR / "supplemental" / "hazard_selector_diagnostics" / POOL_SLUG
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def _load_screened_pool() -> tuple[np.ndarray, list[str], dict]:
    """Load the staged pool hazard image and screen it to the final axis set."""
    path = config.STAGED_ENSEMBLE_DIR / POOL_SLUG / "hazard_image.npz"
    if not path.exists():
        raise SystemExit(
            f"[seldiag] pool hazard image not staged: {path}. Run workflow step 02 "
            f"(NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary, "
            f"NYCOPT_CANDIDATE_POOL_N=<P>, NYCOPT_ENSEMBLE_MASTER_STREAM_ONLY=1) first."
        )
    img = load_hazard_image(path)
    H_full, candidate_axes = img["H"], list(img["hazard_axes"])
    screen = screen_axes(H_full, candidate_axes)
    axes = screen["representatives"]
    cols = [candidate_axes.index(a) for a in axes]
    return H_full[:, cols], axes, screen


def _seeds(k: int, offset: int = 0) -> list[int]:
    return [offset + i for i in range(k)]


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


###############################################################################
# Driver
###############################################################################

def main() -> None:
    """Run all four analysis blocks and write tables, summary, and SI figures."""
    apply_style()
    out = _out_dir()
    H, axes, screen = _load_screened_pool()
    print(f"[seldiag] pool '{POOL_SLUG}': P={len(H)}, screened axes={axes}, "
          f"N={N_SELECT}, seeds={N_SEEDS} (+{N_NULL_SEEDS} null)")

    table, details = _main_comparison(H, axes)
    sweep = _bounds_sweep(H, axes)
    halves = _subpool_stability(H, axes)

    table.to_csv(out / "selector_comparison.csv", index=False)
    sweep.to_csv(out / "normalization_sweep.csv", index=False)
    halves.to_csv(out / "subpool_stability.csv", index=False)

    summary = {
        "pool_slug": POOL_SLUG, "P": int(len(H)), "n_select": N_SELECT,
        "seeds": N_SEEDS, "null_seeds": N_NULL_SEEDS,
        "screened_axes": list(axes),
        "clusters": screen["clusters"],
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

    _fig_selection_scatter(H, axes, details, out)
    _fig_coverage_vs_null(table, out)
    _fig_tail_and_atom(table, out)
    _fig_snap_and_separation(table, details, out)
    _fig_bounds_sweep(sweep, out)

    print(f"[seldiag] wrote 3 tables + summary.json + 5 figures -> {out}")


if __name__ == "__main__":
    main()
