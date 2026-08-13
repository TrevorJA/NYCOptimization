"""
epsilon_refilter_sweep.py - Post-shakeout epsilon revision diagnostics.

Re-filters a completed MM-Borg run's Pareto-approximate sets under candidate
epsilon vectors to measure the archive-cardinality consequence of revising
the campaign epsilons, without re-running any search. Complements the
epsilon-calibration experiment (docs/notes/methods/
epsilon_calibration_experiment.md): that experiment measured the signal /
noise / granularity floors on random feasible policies; this diagnostic
measures the same ε-box filter on a CONVERGED front, where box occupancy —
and therefore reported-set cardinality — is far higher.

Parts:
  1. Per-axis structure of the current archives (natural units): range,
     distinct values, value lattice (reliability axes are quantized at
     1/n_units on the historic trace), occupied 1-D ε-boxes under the
     current vector, and members per occupied box (the density a parallel-
     axes plot shows).
  2. Archive-size sweep: one-at-a-time grids on the axes under revision plus
     combined candidate vectors, applied to each seed's final .set archive
     and the cross-seed merged set (`src.sensitivity_common.
     epsilon_nondominated`, Borg box convention). Re-filtering approximates
     the archive the search WOULD have kept — ε also steers Borg's selection
     and restarts, so a confirmatory search under the adopted vector is
     still required (disclosed).
  3. MOEAFramework comparison: ResultFileMerger re-merges the per-seed
     merged sets under one candidate vector. MEASURED: v5's
     merger applies PLAIN Pareto dominance regardless of --epsilon
     (identical 7,544-row output under the current and C1 vectors), so the
     step-07 {slug}_merged.set is a plain nondominated union, NOT an ε-box
     archive (2,678 of 7,544 rows survive the current-ε box filter). The
     Python filter's validation is stronger and internal: under the current
     vector it reproduces the Borg C archive membership of every seed's
     final .set EXACTLY (100.0% retained; part-2 'current' row).
     Consequence for step 08/09: ε-box-filter (or screen) the merged set
     before re-evaluation — its raw cardinality overstates the front.
  4. Figure: parallel axes of the cross-seed set with the preferred
     candidate's retained members highlighted (thins without collapsing any
     axis span — the same acceptance view as calibration figure F4).

Usage (from repo root, venv; submit via
workflow/supplemental/epsilon_refilter_sweep.sh):
    python3 scripts/supplemental/epsilon_refilter_sweep.py \
        --slug ffmp_obj8_mm_full --scenario historic
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.formulations import get_obj_names, get_obj_directions, get_n_vars, get_n_objs
from src.load.reference_set import load_reference_set
from src.sensitivity_common import epsilon_nondominated
from config import get_epsilons

# Candidate grids, keyed by objective name (active-set order preserved).
# Axes not listed keep the current adopted value. Grid choices: the flagged
# axes (deficit-P99 pair, flood exceedance, storage-P01
# too fine; Trenton reliability possibly too coarse), spanning current ->
# the calibration experiment's measured per-design recommendations
# (epsilon_recommendation_ffmp_combined: NYC-def measured 10.0 hazfill /
# 5.0 historic; Mont-def 5.0 historic; storage 10.0 historic).
OAT_GRIDS: dict[str, list[float]] = {
    "nyc_delivery_deficit_p99_pct":      [3.0, 5.0, 10.0],
    "montague_flow_deficit_p99_pct":     [3.0, 4.0, 5.0],
    "trenton_flow_reliability_annual":   [0.01, 0.02],
    "downstream_flood_exceedance_annual": [0.3, 0.5],
    "nyc_storage_min_p01_pct":           [7.5, 10.0],
}

# Combined candidates: overrides applied to the current vector.
CANDIDATES: dict[str, dict[str, float]] = {
    # storage-P01 epsilon stays at the current 5.0 (a 5%-of-capacity
    # distinction is significant); the NYC/Montague deficit-P99 epsilons are
    # PAIRED at 5.0 — matching the paired 0.02 on the two sites' reliability
    # axes.
    "C1_adopted": {
        "nyc_delivery_deficit_p99_pct": 5.0,
        "montague_flow_deficit_p99_pct": 5.0,
        "downstream_flood_exceedance_annual": 0.3,
    },
    "C1_moderate": {
        "nyc_delivery_deficit_p99_pct": 5.0,
        "montague_flow_deficit_p99_pct": 4.0,
        "downstream_flood_exceedance_annual": 0.3,
        "nyc_storage_min_p01_pct": 7.5,
    },
    "C2_measured": {
        "nyc_delivery_deficit_p99_pct": 10.0,
        "montague_flow_deficit_p99_pct": 5.0,
        "downstream_flood_exceedance_annual": 0.5,
        "nyc_storage_min_p01_pct": 10.0,
    },
    "C3_moderate_fine_trenton": {
        "nyc_delivery_deficit_p99_pct": 5.0,
        "montague_flow_deficit_p99_pct": 4.0,
        "downstream_flood_exceedance_annual": 0.3,
        "nyc_storage_min_p01_pct": 7.5,
        "trenton_flow_reliability_annual": 0.01,
    },
}

#: Candidate highlighted in the part-4 figure.
PREFERRED = "C1_adopted"


def _vector(base: np.ndarray, names: list, overrides: dict) -> np.ndarray:
    v = base.copy()
    for k, val in overrides.items():
        v[names.index(k)] = val
    return v


def _load_archives(run_dir: Path, slug: str, n_vars: int, n_objs: int) -> dict:
    """Borg-minimized objective arrays for each seed's final archive + merged."""
    archives = {}
    for f in sorted((run_dir / "sets").glob(f"seed_*_{slug}.set")):
        seed = f.stem.split("_")[1]
        _, objs = load_reference_set(f, n_vars, n_objs=n_objs)
        archives[f"seed{seed}"] = objs
    merged = run_dir / "sets" / f"{slug}_merged.set"
    if merged.exists():
        _, objs = load_reference_set(merged, n_vars, n_objs=n_objs)
        archives["merged"] = objs
    if not archives:
        sys.exit(f"no archives found under {run_dir / 'sets'}")
    return archives


def part1_axis_structure(archives, names, directions, eps, out_csv):
    rows = []
    for arch_name, objs in archives.items():
        natural = objs * np.where(directions == 1, -1.0, 1.0)
        for j, name in enumerate(names):
            vals = np.unique(natural[:, j])
            diffs = np.diff(vals)
            diffs = diffs[diffs > 1e-12]
            boxes = np.unique(np.floor(objs[:, j] / eps[j]))
            rows.append({
                "archive": arch_name, "objective": name,
                "n_members": len(natural),
                "min": f"{vals.min():.6g}", "max": f"{vals.max():.6g}",
                "range": f"{vals.max() - vals.min():.6g}",
                "n_distinct": len(vals),
                "lattice_min_step": f"{diffs.min():.6g}" if diffs.size else "",
                "lattice_median_step": f"{np.median(diffs):.6g}" if diffs.size else "",
                "epsilon": eps[j],
                "occupied_1d_boxes": len(boxes),
                "members_per_box": f"{len(natural) / len(boxes):.1f}",
            })
    _write_csv(out_csv, rows)
    return rows


def part2_size_sweep(archives, names, eps, out_csv):
    sweeps = [("current", {})]
    sweeps += [(f"OAT {n} -> {v:g}", {n: v}) for n, grid in OAT_GRIDS.items() for v in grid]
    sweeps += [(label, ov) for label, ov in CANDIDATES.items()]

    rows = []
    for label, overrides in sweeps:
        v = _vector(eps, names, overrides)
        sizes = {a: len(epsilon_nondominated(objs, v)) for a, objs in archives.items()}
        base = {a: len(objs) for a, objs in archives.items()}
        rows.append({
            "candidate": label,
            "epsilons": " ".join(f"{x:g}" for x in v),
            **{f"size_{a}": s for a, s in sizes.items()},
            **{f"pct_of_current_{a}": f"{100 * s / base[a]:.1f}"
               for a, s in sizes.items()},
        })
    _write_csv(out_csv, rows)
    return rows


def part3_moea_crosscheck(run_dir, slug, names, eps, scratch_dir):
    """ResultFileMerger under the preferred candidate vs the Python filter."""
    from src.diagnostics import get_cli_path, problem_name_for

    v = _vector(eps, names, CANDIDATES[PREFERRED])
    # Inputs must be MOEAFramework-written result files: the Borg C-written
    # seed .set files lack the '#' entry terminator and parse as zero
    # entries. The step-07 per-seed merged sets are the CLI-native twins.
    seed_sets = sorted((run_dir / "sets").glob(f"{slug}_seed*_merged.set"))
    out = Path(scratch_dir) / f"{slug}_crosscheck_{PREFERRED}.set"
    cmd = [get_cli_path(), "ResultFileMerger",
           "--problem", problem_name_for(slug),
           "--epsilon", ",".join(str(x) for x in v),
           "--output", str(out)] + [str(f) for f in seed_sets]
    subprocess.run(cmd, check=True)
    n_vars = get_n_vars(problem_name_for(slug).removeprefix("drb_"))
    _, objs_cli = load_reference_set(out, n_vars, n_objs=len(names))

    union = np.vstack([load_reference_set(f, n_vars, n_objs=len(names))[1]
                       for f in seed_sets])
    n_py = len(epsilon_nondominated(union, v))
    print(f"[merger comparison] {PREFERRED}: MOEAFramework plain-dominance "
          f"merge={len(objs_cli)} vs python ε-box archive={n_py} — expected "
          f"to differ (see docstring part 3); the box filter is validated "
          f"against Borg's own seed archives in the part-2 'current' row.")
    return len(objs_cli), n_py


def part4_figure(archives, names, directions, eps, out_png):
    import pandas as pd
    from src.plotting.parallel_coordinates import (custom_parallel_coordinates,
                                                   minmaxs_from_directions)
    from src.plotting.style import OBJ_AXIS_LABELS

    objs = archives["merged"] if "merged" in archives else next(iter(archives.values()))
    v = _vector(eps, names, CANDIDATES[PREFERRED])
    keep_idx = epsilon_nondominated(objs, v)
    mask = np.zeros(len(objs), dtype=bool)
    mask[keep_idx] = True

    natural = objs * np.where(directions == 1, -1.0, 1.0)
    custom_parallel_coordinates(
        pd.DataFrame(natural, columns=list(names)),
        axis_labels=[OBJ_AXIS_LABELS.get(n, n) for n in names],
        minmaxs=minmaxs_from_directions(directions),
        title=f"Re-filter under {PREFERRED}",
        highlight_mask=mask, figsize=(13, 5.5),
        highlight_label="retained", exclude_label="merged by coarser ε",
        save_fig_filename=out_png,
    )
    _two_panel_figure(natural, mask, names, directions,
                      out_png.with_name(out_png.stem + "_2panel.png"))


def _two_panel_figure(natural, mask, names, directions, out_png, label=None):
    """Full set (top) vs ε-filtered set (bottom), shared normalization.

    Both panels normalize on the FULL set's ranges so the filtered panel is
    directly comparable — any span loss would appear as untouched headroom.
    ``label`` names the candidate in the bottom-panel title (default: the
    module's PREFERRED candidate).
    """
    label = PREFERRED if label is None else label
    import matplotlib.pyplot as plt
    from src.plotting.style import apply_style, OBJ_AXIS_LABELS
    from src.plotting.parallel_coordinates import _format_value

    n_objs = len(names)
    lo, hi = natural.min(axis=0), natural.max(axis=0)
    rng_ = np.where(hi - lo == 0, 1.0, hi - lo)
    normed = (natural - lo) / rng_
    flip = directions == -1
    normed[:, flip] = 1.0 - normed[:, flip]

    apply_style()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = np.arange(n_objs)
    panels = [
        (ax_top, normed, 0.05, f"Full merged set (n={len(normed)})"),
        (ax_bot, normed[mask], 0.15,
         f"ε-box filtered, {label} (n={int(mask.sum())})"),
    ]
    for ax, rows, alpha, label in panels:
        for row in rows:
            ax.plot(x, row, alpha=alpha, color="steelblue", linewidth=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Preference Direction  (↑ better)")
        ax.set_ylim(-0.14, 1.12)
        ax.grid(True, alpha=0.3, axis="x")
        for i in range(n_objs):
            best, worst = (hi[i], lo[i]) if directions[i] == 1 else (lo[i], hi[i])
            ax.text(i, 1.04, _format_value(best), ha="center", va="bottom",
                    fontsize=7, color="0.3")
            ax.text(i, -0.04, _format_value(worst), ha="center", va="top",
                    fontsize=7, color="0.3")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([OBJ_AXIS_LABELS.get(n, n) for n in names], fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out_png}")


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[table] {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default="ffmp_obj8_mm_full")
    ap.add_argument("--scenario", default="historic")
    ap.add_argument("--figures-only", action="store_true",
                    help="Skip the tables and the MOEAFramework comparison; "
                         "re-render the part-4 figures only.")
    args = ap.parse_args()

    run_dir = Path("outputs") / args.scenario / args.slug
    out_dir = Path("outputs/supplemental/epsilon_refilter") / f"{args.scenario}_{args.slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = get_obj_names()
    directions = np.array(get_obj_directions())
    eps = np.array(get_epsilons(), dtype=float)
    formulation_nvars = get_n_vars("ffmp")
    archives = _load_archives(run_dir, args.slug, formulation_nvars, get_n_objs())
    print(f"archives: {', '.join(f'{k} (n={len(v)})' for k, v in archives.items())}")
    print(f"current epsilons: {dict(zip(names, eps))}")

    if not args.figures_only:
        part1_axis_structure(archives, names, directions, eps,
                             out_dir / "axis_structure.csv")
        part2_size_sweep(archives, names, eps, out_dir / "archive_size_sweep.csv")
        with tempfile.TemporaryDirectory() as td:
            part3_moea_crosscheck(run_dir, args.slug, names, eps, td)
    part4_figure(archives, names, directions, eps,
                 out_dir / f"refilter_parallel_axes_{PREFERRED}.png")
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
