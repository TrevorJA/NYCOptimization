"""compare_scenario_designs_pilot.py — cross-design Pareto comparison for the
scenario-design go/no-go pilot.

Loads the final Pareto-approximate set of each pilot arm (same formulation,
objectives, MOEA config — only NYCOPT_SCENARIO_DESIGN differs), and compares
them IN-SEARCH (raw, on each design's own training ensemble):

  * per-objective ranges / medians (natural units & directions),
  * front sizes,
  * Monte-Carlo hypervolume vs a COMMON reference point (union nadir) in
    normalized minimization space — comparable across arms,
  * objective-space figures: 7-axis parallel coordinates + key 2-D scatters.

This is a "do they differ at all" signal, NOT a held-out robustness verdict.

Usage:
    python scripts/supplemental/compare_scenario_designs_pilot.py \
        --slug ffmp_obj7_pilot --seed 1 \
        --arms hazard_filling hazard_filling_absolute fixed_probabilistic_short
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import OUTPUTS_DIR  # noqa: E402
from src.formulations import get_objective_set, get_n_vars  # noqa: E402
from src.load.reference_set import load_reference_set  # noqa: E402

ARM_COLORS = {
    "hazard_filling": "#1f77b4",
    "hazard_filling_absolute": "#d62728",
    "fixed_probabilistic_short": "#2ca02c",
    "historic": "#7f7f7f",
}


def nondominated_min(objs: np.ndarray) -> np.ndarray:
    """Return the nondominated subset (all-minimization convention)."""
    if objs.shape[0] == 0:
        return objs
    keep = np.ones(objs.shape[0], dtype=bool)
    for i in range(objs.shape[0]):
        if not keep[i]:
            continue
        # j dominates i if j <= i in all and < in at least one
        dom = np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1)
        dom[i] = False
        if np.any(dom):
            keep[i] = False
    return objs[keep]


def find_set_file(scenario: str, slug: str, seed: int) -> Path | None:
    sets_dir = OUTPUTS_DIR / scenario / slug / "sets"
    candidates = [
        sets_dir / f"seed_{seed:02d}_{slug}.set",
        sets_dir / f"{slug}_seed{seed:02d}_merged.set",
    ]
    for c in candidates:
        if c.exists():
            return c
    # last resort: any .set in the dir
    hits = sorted(sets_dir.glob("*.set")) if sets_dir.exists() else []
    return hits[0] if hits else None


def load_arm_min_space(arm, slug, seed, source, reeval_tag, n_vars, names,
                       directions):
    """Return (min-space objective array, source_path) for one arm.

    ``source='set'``  -> the in-search Pareto .set (objectives already in Borg
    minimization space). This is the RAW, own-ensemble front — NOT comparable
    across designs.
    ``source='reeval'`` -> objectives_summary.csv from re-evaluation on a COMMON
    ensemble (natural units); converted here to min-space so the rest of the
    pipeline is identical. This is the valid cross-design comparison.
    """
    if source == "set":
        f = find_set_file(arm, slug, seed)
        if f is None:
            return None, None
        _, objs = load_reference_set(f, n_vars)
        return objs, f
    # source == "reeval"
    csv = OUTPUTS_DIR / arm / slug / "reeval" / reeval_tag / "objectives_summary.csv"
    if not csv.exists():
        return None, None
    df = pd.read_csv(csv, index_col="solution_id")
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise KeyError(f"re-eval CSV {csv} missing objective columns: {missing}")
    nat = df[names].to_numpy(dtype=float)
    nat = nat[~np.isnan(nat).any(axis=1)]   # drop failed solutions
    # natural -> min-space (stored): maximize objectives negated, minimize kept.
    stored = nat.copy()
    for j, d in enumerate(directions):
        if d == 1:
            stored[:, j] = -stored[:, j]
    return stored, csv


def mc_hypervolume(front_norm: np.ndarray, n_samples: int, rng) -> float:
    """Monte-Carlo dominated hypervolume in normalized [0,1]^m min-space, ref=1.

    Fraction of uniform samples in the unit box that are dominated by >=1 front
    point (front point <= sample componentwise) times the box volume (=1).
    """
    if front_norm.shape[0] == 0:
        return 0.0
    m = front_norm.shape[1]
    dominated = 0
    batch = 200_000
    done = 0
    while done < n_samples:
        k = min(batch, n_samples - done)
        u = rng.random((k, m))
        # sample dominated if any front point <= u in all dims
        # (k x n x m) memory guard: loop over front in chunks
        dom_mask = np.zeros(k, dtype=bool)
        for start in range(0, front_norm.shape[0], 256):
            fp = front_norm[start:start + 256]  # (c, m)
            # for each sample, any fp <= u
            le = np.all(fp[None, :, :] <= u[:, None, :], axis=2)  # (k, c)
            dom_mask |= np.any(le, axis=1)
        dominated += int(np.sum(dom_mask))
        done += k
    return dominated / n_samples


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slug", default="ffmp_obj7_pilot")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--formulation", default="ffmp")
    p.add_argument("--arms", nargs="+",
                   default=["hazard_filling", "hazard_filling_absolute",
                            "fixed_probabilistic_short"])
    p.add_argument("--mc-samples", type=int, default=2_000_000)
    p.add_argument("--source", choices=["set", "reeval"], default="set",
                   help="'set' = in-search Pareto .set (raw, each design's OWN "
                        "ensemble; NOT comparable across designs). 'reeval' = "
                        "objectives re-evaluated on a COMMON ensemble — the only "
                        "valid cross-design comparison.")
    p.add_argument("--reeval-tag", default=None,
                   help="Common re-eval ensemble preset name (the reeval/<tag>/ "
                        "subdir). Required when --source reeval.")
    p.add_argument("--drop-objectives", nargs="*", default=None,
                   help="Objective names to EXCLUDE from the comparison (e.g. "
                        "degenerate/non-discriminating objectives that distort "
                        "the hypervolume).")
    p.add_argument("--outdir", default=None)
    args = p.parse_args(argv)

    if args.source == "reeval" and not args.reeval_tag:
        p.error("--source reeval requires --reeval-tag <common_ensemble_preset>")

    n_vars = get_n_vars(args.formulation)
    obj_set = get_objective_set()
    full_names = list(obj_set.names)
    full_dirs = list(obj_set.directions)  # 1=maximize, -1=minimize
    drop = set(args.drop_objectives or [])
    bad = drop - set(full_names)
    if bad:
        p.error(f"--drop-objectives names not in objective set: {sorted(bad)}")
    keep_idx = [i for i, n in enumerate(full_names) if n not in drop]
    names = [full_names[i] for i in keep_idx]
    directions = [full_dirs[i] for i in keep_idx]
    m = len(names)
    if drop:
        print(f"[compare] dropping {len(drop)} objective(s) {sorted(drop)}; "
              f"comparing on {m} of {len(full_names)}.")

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        leaf = "in_search" if args.source == "set" else f"reeval_{args.reeval_tag}"
        outdir = OUTPUTS_DIR / "diagnostics" / "scenario_design_pilot" / leaf
    outdir.mkdir(parents=True, exist_ok=True)

    # --- load each arm's final nondominated front (min-space) ---
    fronts = {}     # arm -> (n, m) min-space objectives
    set_files = {}
    for arm in args.arms:
        # Load full objective set (so CSV columns + sign conversion are correct),
        # then keep only the compared objectives.
        objs, src = load_arm_min_space(arm, args.slug, args.seed, args.source,
                                       args.reeval_tag, n_vars, full_names,
                                       full_dirs)
        if objs is None:
            print(f"[compare] WARNING: no data for arm '{arm}' "
                  f"(source={args.source}, slug={args.slug}); skipping.")
            continue
        if objs.shape[0] == 0 or objs.shape[1] != len(full_names):
            print(f"[compare] WARNING: arm '{arm}' source {src} has objs shape "
                  f"{objs.shape}; skipping.")
            continue
        objs = objs[:, keep_idx]
        front = nondominated_min(objs)
        fronts[arm] = front
        set_files[arm] = src
        print(f"[compare] {arm}: {objs.shape[0]} raw -> {front.shape[0]} "
              f"nondominated  ({src})")

    if not fronts:
        print("[compare] No arms loaded. Nothing to do.")
        return 1

    # --- common normalization bounds (min-space) over the union ---
    allpts = np.vstack(list(fronts.values()))
    lo = allpts.min(axis=0)   # ideal (best) in min-space
    hi = allpts.max(axis=0)   # nadir (worst) in min-space
    span = np.where(hi > lo, hi - lo, 1.0)

    def to_natural(objs):
        # raw natural value: maximize -> -stored, minimize -> stored
        out = objs.copy()
        for j, d in enumerate(directions):
            if d == 1:
                out[:, j] = -out[:, j]
        return out

    # --- hypervolume vs common reference point (ref = normalized nadir = 1) ---
    rng = np.random.default_rng(0)
    hv = {}
    for arm, front in fronts.items():
        fn = (front - lo) / span     # in [0,1], lower=better
        fn = np.clip(fn, 0.0, 1.0)
        hv[arm] = mc_hypervolume(fn, args.mc_samples, rng)

    # --- write summary memo data (tables) ---
    lines = []
    lines.append("# Scenario-design pilot — cross-design Pareto comparison\n")
    if args.source == "set":
        lines.append(f"Slug `{args.slug}`, seed {args.seed}, formulation "
                     f"`{args.formulation}`. **Source: in-search (raw) .set** — each "
                     "design scored on its OWN ensemble. NOT comparable across "
                     "designs; shown only as a 'do the fronts differ' signal. "
                     "Single seed.\n")
    else:
        lines.append(f"Slug `{args.slug}`, seed {args.seed}, formulation "
                     f"`{args.formulation}`. **Source: re-evaluation on common "
                     f"ensemble `{args.reeval_tag}`** — every design's policies "
                     "scored on the SAME ensemble. This is the valid cross-design "
                     "comparison. Single seed.\n")
    lines.append("## Front size & hypervolume (common reference point)\n")
    lines.append("| arm | front size | MC hypervolume |")
    lines.append("|-----|-----------:|---------------:|")
    for arm in args.arms:
        if arm in fronts:
            lines.append(f"| {arm} | {fronts[arm].shape[0]} | {hv[arm]:.4f} |")
    lines.append("")
    lines.append("Hypervolume is the dominated fraction of the unit box in normalized "
                 "minimization space against the common union-nadir reference point "
                 f"(MC, {args.mc_samples:,} samples). Higher = better/larger front.\n")

    # per-objective ranges/medians (natural units & directions)
    lines.append("## Per-objective ranges & medians (natural units)\n")
    for j, nm in enumerate(names):
        d = "max" if directions[j] == 1 else "min"
        lines.append(f"### {nm}  ({d})")
        lines.append("| arm | min | median | max |")
        lines.append("|-----|----:|-------:|----:|")
        for arm in args.arms:
            if arm not in fronts:
                continue
            nat = to_natural(fronts[arm])[:, j]
            lines.append(f"| {arm} | {nat.min():.3f} | {np.median(nat):.3f} | {nat.max():.3f} |")
        lines.append("")

    memo_data = outdir / "comparison_tables.md"
    memo_data.write_text("\n".join(lines))
    print(f"[compare] wrote {memo_data}")

    # --- parallel coordinates (natural units, per-axis min-max normalized) ---
    nat_all = to_natural(allpts)
    nlo, nhi = nat_all.min(axis=0), nat_all.max(axis=0)
    nspan = np.where(nhi > nlo, nhi - nlo, 1.0)
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(m)
    for arm in args.arms:
        if arm not in fronts:
            continue
        nat = (to_natural(fronts[arm]) - nlo) / nspan
        c = ARM_COLORS.get(arm, None)
        for row in nat:
            ax.plot(xs, row, color=c, alpha=0.15, lw=0.6)
        ax.plot([], [], color=c, label=arm, lw=2)  # legend proxy
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n({'max' if directions[j]==1 else 'min'})"
                        for j, n in enumerate(names)], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("normalized (0=worst axis end .. 1=best across union)")
    _src_lbl = "in-search (raw)" if args.source == "set" else f"re-eval on {args.reeval_tag}"
    ax.set_title(f"Pareto fronts by scenario design — parallel coordinates "
                 f"({args.slug}; {_src_lbl})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    pc = outdir / "parallel_coordinates.png"
    fig.savefig(pc, dpi=130)
    plt.close(fig)
    print(f"[compare] wrote {pc}")

    # --- key 2-D scatter pairs (natural units) ---
    pairs = [(0, 2), (0, 6), (2, 4), (1, 5)]  # rel-nyc vs rel-mont, rel-nyc vs storage, etc.
    pairs = [(i, j) for (i, j) in pairs if i < m and j < m]
    ncol = 2
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (i, j) in zip(axes, pairs):
        for arm in args.arms:
            if arm not in fronts:
                continue
            nat = to_natural(fronts[arm])
            ax.scatter(nat[:, i], nat[:, j], s=12, alpha=0.6,
                       color=ARM_COLORS.get(arm), label=arm)
        ax.set_xlabel(f"{names[i]} ({'max' if directions[i]==1 else 'min'})", fontsize=8)
        ax.set_ylabel(f"{names[j]} ({'max' if directions[j]==1 else 'min'})", fontsize=8)
        ax.legend(fontsize=7)
    for ax in axes[len(pairs):]:
        ax.axis("off")
    fig.suptitle(f"Objective-space scatter by scenario design ({args.slug}; {_src_lbl})")
    fig.tight_layout()
    sc = outdir / "objective_scatter.png"
    fig.savefig(sc, dpi=130)
    plt.close(fig)
    print(f"[compare] wrote {sc}")

    print("\n[compare] HYPERVOLUME SUMMARY (common reference):")
    for arm in args.arms:
        if arm in fronts:
            print(f"  {arm:32s} front={fronts[arm].shape[0]:4d}  HV={hv[arm]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
