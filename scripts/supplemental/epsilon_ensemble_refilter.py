"""
epsilon_ensemble_refilter.py - Grouped-epsilon re-filter of production reference sets.

Re-filters the production Pareto-approximate sets under grouped candidate
epsilon 4-tuples (eps_rel shared by the four *_reliability_annual axes,
eps_def by both *_deficit_p99_pct axes, flood exceedance and storage-P01 each
their own) with no re-simulation, and reports front size vs objective coverage
(epsilon_calibration_experiment.md).

Substrates per scenario, from outputs/{scenario}/{slug}/sets/:
  - seed_NN_{slug}.set        Borg's own final archive (the 'adopted' row must
                              reproduce its membership)
  - {slug}_merged_raw.set     the plain-dominance cross-seed union, the
                              filtering substrate (step 07 writes the
                              epsilon-box-filtered {slug}_merged.set alongside;
                              the 'adopted' union row must reproduce its count)

A static re-filter under-prices live-search cardinality (epsilon also steers
Borg's selection and restarts), so the recommendation table carries
inflation-adjusted sizes (INFLATE_MERGED / INFLATE_SEED).

Usage (from repo root, venv; submit via
workflow/supplemental/epsilon_ensemble_refilter.sh):
    python3 scripts/supplemental/epsilon_ensemble_refilter.py \
        --slug ffmp_obj8 --scenarios historic fixed_probabilistic \
        hazard_filling_stationary
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import epsilon_refilter_sweep as legacy  # noqa: E402  (sibling module)

from src.formulations import (get_obj_names, get_obj_directions,  # noqa: E402
                              get_n_vars, get_n_objs)
from src.load.reference_set import load_reference_set  # noqa: E402
from src.sensitivity_common import epsilon_nondominated  # noqa: E402
from config import get_epsilons  # noqa: E402

#: Objective-family membership, by registry-name pattern (order-robust).
GROUPS = {
    "rel":     lambda n: n.endswith("_reliability_annual"),
    "def":     lambda n: "deficit_p99" in n,
    "flood":   lambda n: n == "downstream_flood_exceedance_annual",
    "storage": lambda n: n == "nyc_storage_min_p01_pct",
}

#: Grouped reference point the ladders/OATs perturb: the adopted vector with
#: Trenton and NJ raised to the paired NYC/Montague reliability value.
G_BASE = (0.02, 5.0, 0.3, 5.0)

# Candidate grid (label -> 4-tuple; None = the adopted ungrouped 8-vector):
# a reliability ladder, per-group OATs, a joint ladder (box occupancy
# multiplies across axes, so OAT alone understates joint thinning), and mixed
# shapes at intermediate reliability resolution.
CANDIDATES: dict[str, tuple | None] = {
    "adopted":      None,
    "grouped_base": G_BASE,
    "rel_x2":       (0.04, 5.0, 0.3, 5.0),
    "rel_x3":       (0.06, 5.0, 0.3, 5.0),
    "rel_x4":       (0.08, 5.0, 0.3, 5.0),
    "rel_x5":       (0.10, 5.0, 0.3, 5.0),
    "def_7.5":      (0.02, 7.5, 0.3, 5.0),
    "def_10":       (0.02, 10.0, 0.3, 5.0),
    "flood_0.5":    (0.02, 5.0, 0.5, 5.0),
    "flood_0.6":    (0.02, 5.0, 0.6, 5.0),
    "stor_7.5":     (0.02, 5.0, 0.3, 7.5),
    "stor_10":      (0.02, 5.0, 0.3, 10.0),
    "joint_x1.5":   (0.03, 7.5, 0.45, 7.5),
    "joint_x2":     (0.04, 10.0, 0.6, 10.0),
    "joint_x3":     (0.06, 15.0, 1.0, 15.0),
    # Mixed shapes at intermediate reliability resolution.
    "rel_x2.5":     (0.05, 5.0, 0.3, 5.0),
    "mixed_r2":     (0.04, 7.5, 0.3, 5.0),
    "mixed_r2.5":   (0.05, 7.5, 0.5, 7.5),
    "mixed_r3":     (0.06, 7.5, 0.5, 7.5),
    "mixed_r2f":    (0.04, 7.5, 0.5, 5.0),
    "mixed_r2fs":   (0.04, 7.5, 0.5, 7.5),
    "mixed_r2.5f":  (0.05, 7.5, 0.5, 5.0),
    # Shapes that keep flood at 0.3 and take the size cut from the deficit group.
    "keepf_a":      (0.05, 10.0, 0.3, 5.0),
    "keepf_b":      (0.06, 7.5, 0.3, 5.0),
    "keepf_c":      (0.05, 7.5, 0.3, 5.0),
    "keepf_d":      (0.06, 10.0, 0.3, 5.0),
}

#: Re-eval sizing band (per design, cross-seed-adjusted retained count).
TARGET_BAND = (1000, 1200)

#: Live-search inflation factors (measured on a confirmatory search;
#: epsilon_calibration_experiment.md): cross-seed epsilon-front, per-seed archives.
INFLATE_MERGED, INFLATE_SEED = 1.10, 1.35

#: An axis is over-coarsened when its occupied 1-D box count falls below
#: min(MIN_AXIS_BOXES, the ADOPTED vector's count on the same substrate) —
#: relative to adopted because some axes are intrinsically narrow (the
#: fixed_probabilistic flood axis spans ~1.6 boxes under the adopted 0.3, so
#: an absolute floor would disqualify every candidate including 'adopted').
MIN_AXIS_BOXES = 8

#: Borg archives can carry a handful of box-dominated members from exact
#: box-boundary coordinates (write-precision / FP artifact — e.g. 2 of the
#: 2,685 historic production members, both with a coordinate exactly on a
#: boundary). The seed-reproduction check passes within this tolerance.
SEED_CHECK_RTOL = 1e-3


def fast_epsilon_nondominated(objs: np.ndarray,
                              epsilons: np.ndarray) -> np.ndarray:
    """Exact fast twin of src.sensitivity_common.epsilon_nondominated.

    Same semantics (Borg box convention, corner-closest representative,
    non-finite rows excluded, sorted indices returned) but the dominance pass
    sweeps boxes in ascending coordinate-sum order and compares each box only
    against the kept archive: a dominating box has every coordinate <= and one
    < , hence a strictly smaller sum, so it precedes its victims in the sweep;
    by transitivity a dominated box is always dominated by a KEPT one. The
    reference implementation compares all boxes pairwise (quadratic in
    occupied boxes). Cross-checked against the reference at runtime (see
    _self_check).
    """
    F = np.asarray(objs, dtype=float)
    eps = np.asarray(epsilons, dtype=float)
    if F.ndim != 2 or F.shape[1] != eps.size:
        raise ValueError(f"shape mismatch: objs {F.shape} vs {eps.size} epsilons")
    if not np.all(eps > 0.0):
        raise ValueError(f"epsilons must be positive, got {eps}")

    valid = np.flatnonzero(np.isfinite(F).all(axis=1))
    if valid.size == 0:
        return np.array([], dtype=int)
    scaled = F[valid] / eps
    boxes = np.floor(scaled)
    corner_dist = ((scaled - boxes) ** 2).sum(axis=1)

    rep_for_box: dict = {}
    for local, box in enumerate(map(tuple, boxes)):
        best = rep_for_box.get(box)
        if best is None or corner_dist[local] < corner_dist[best]:
            rep_for_box[box] = local
    box_arr = np.array(list(rep_for_box.keys()), dtype=float)
    reps = np.array(list(rep_for_box.values()), dtype=int)

    order = np.argsort(box_arr.sum(axis=1), kind="stable")
    sorted_boxes = box_arr[order]
    kept_arr = np.empty_like(sorted_boxes)
    kept_pos, n_kept = [], 0
    for i in range(len(order)):
        b = sorted_boxes[i]
        if n_kept:
            K = kept_arr[:n_kept]
            if ((K <= b).all(axis=1) & (K < b).any(axis=1)).any():
                continue
        kept_arr[n_kept] = b
        kept_pos.append(i)
        n_kept += 1
    return np.sort(valid[reps[order[np.array(kept_pos)]]])


def _self_check(objs: np.ndarray, vectors: list[np.ndarray]):
    """Assert the fast filter matches the reference on a small archive."""
    for v in vectors:
        fast = fast_epsilon_nondominated(objs, v)
        ref = epsilon_nondominated(objs, v)
        if not np.array_equal(fast, ref):
            sys.exit(f"fast filter mismatch vs reference under eps={v} "
                     f"({len(fast)} vs {len(ref)} members) — aborting")
    print(f"[check] fast filter == reference on n={len(objs)} archive "
          f"({len(vectors)} vectors) -> PASS", flush=True)


def grouped_vector(names, cand: tuple) -> np.ndarray:
    """Expand a (rel, def, flood, storage) tuple to the 8-objective vector."""
    rel, deficit, flood, storage = cand
    by_group = {"rel": rel, "def": deficit, "flood": flood, "storage": storage}
    v = np.empty(len(names), dtype=float)
    for j, n in enumerate(names):
        matches = [g for g, pred in GROUPS.items() if pred(n)]
        if len(matches) != 1:
            sys.exit(f"objective {n!r} matched groups {matches} — grid invalid")
        v[j] = by_group[matches[0]]
    return v


def fineness(cand: tuple) -> float:
    """Geometric-mean coarsening ratio vs G_BASE (1.0 = base resolution)."""
    r = [c / b for c, b in zip(cand, G_BASE)]
    return float(np.prod(r) ** (1.0 / len(r)))


def _load_archives(run_dir: Path, slug: str, n_vars: int, n_objs: int) -> dict:
    """seed archives + the raw cross-seed union, minimized objective arrays."""
    archives = {}
    for f in sorted((run_dir / "sets").glob(f"seed_*_{slug}.set")):
        seed = f.stem.split("_")[1]
        _, objs = load_reference_set(f, n_vars, n_objs=n_objs)
        archives[f"seed{seed}"] = objs
    raw = run_dir / "sets" / f"{slug}_merged_raw.set"
    if raw.exists():
        _, objs = load_reference_set(raw, n_vars, n_objs=n_objs)
        archives["union"] = objs
    return archives


def _coverage(objs: np.ndarray, keep: np.ndarray, v: np.ndarray) -> dict:
    """Per-axis coverage of the retained subset vs the full set (minimized).

    occupied 1-D boxes among retained members; retained-span fraction of the
    full span; gap between the full set's best and the retained best, in
    epsilon units (epsilon-dominance guarantees a representative within ~1 box
    of every axis extreme — the gap reports the realized loss).
    """
    kept = objs[keep]
    boxes = [len(np.unique(np.floor(kept[:, j] / v[j])))
             for j in range(objs.shape[1])]
    full_span = objs.max(axis=0) - objs.min(axis=0)
    full_span = np.where(full_span == 0, 1.0, full_span)
    span_frac = (kept.max(axis=0) - kept.min(axis=0)) / full_span
    gap_eps = (kept.min(axis=0) - objs.min(axis=0)) / v
    return {"boxes": boxes, "span_frac": span_frac, "gap_eps": gap_eps}


def sweep_scenario(archives: dict, names, eps_adopted: np.ndarray) -> dict:
    """Filter every candidate against every archive; coverage on the union."""
    results = {}
    for label, cand in CANDIDATES.items():
        v = (eps_adopted.copy() if cand is None
             else grouped_vector(names, cand))
        sizes, cov = {}, None
        for a, objs in archives.items():
            keep = fast_epsilon_nondominated(objs, v)
            sizes[a] = len(keep)
            if a == "union":
                cov = _coverage(objs, keep, v)
        results[label] = {"vector": v, "cand": cand, "sizes": sizes,
                          "coverage": cov}
        print(f"  {label}: " + ", ".join(f"{a}={s}" for a, s in sizes.items()),
              flush=True)
    return results


def write_scenario_tables(results: dict, archives: dict, names, out_dir: Path):
    base = {a: len(objs) for a, objs in archives.items()}
    rows, cov_rows = [], []
    for label, r in results.items():
        row = {"candidate": label,
               "epsilons": " ".join(f"{x:g}" for x in r["vector"])}
        for a in archives:
            row[f"size_{a}"] = r["sizes"][a]
            row[f"pct_of_full_{a}"] = f"{100 * r['sizes'][a] / base[a]:.1f}"
        cov = r["coverage"]
        if cov is not None:
            row["min_axis_boxes"] = int(min(cov["boxes"]))
            row["min_span_frac"] = f"{min(cov['span_frac']):.3f}"
            row["max_extreme_gap_eps"] = f"{max(cov['gap_eps']):.2f}"
            for j, n in enumerate(names):
                cov_rows.append({
                    "candidate": label, "objective": n,
                    "epsilon": f"{r['vector'][j]:g}",
                    "occupied_1d_boxes": cov["boxes"][j],
                    "span_frac": f"{cov['span_frac'][j]:.3f}",
                    "extreme_gap_eps": f"{cov['gap_eps'][j]:.2f}",
                })
        rows.append(row)
    legacy._write_csv(out_dir / "grouped_size_sweep.csv", rows)
    if cov_rows:
        legacy._write_csv(out_dir / "grouped_axis_coverage.csv", cov_rows)


def validate_scenario(scenario: str, results: dict, archives: dict,
                      run_dir: Path, slug: str, n_vars: int, n_objs: int):
    """The two built-in reproduction checks for the 'adopted' row."""
    adopted = results["adopted"]["sizes"]
    for a, objs in archives.items():
        if a.startswith("seed"):
            miss = len(objs) - adopted[a]
            ok = 0 <= miss <= max(1, int(SEED_CHECK_RTOL * len(objs)))
            note = f" ({miss} box-boundary member(s), see SEED_CHECK_RTOL)" \
                if ok and miss else ""
            print(f"[check:{scenario}] adopted on {a}: {adopted[a]} vs "
                  f"archive {len(objs)} -> {'PASS' if ok else 'FAIL'}{note}",
                  flush=True)
    merged = run_dir / "sets" / f"{slug}_merged.set"
    if merged.exists() and "union" in archives:
        _, objs = load_reference_set(merged, n_vars, n_objs=n_objs)
        ok = adopted["union"] == len(objs)
        print(f"[check:{scenario}] adopted on union: {adopted['union']} vs "
              f"step-07 merged.set {len(objs)} -> {'PASS' if ok else 'FAIL'}")


def combined_table(all_results: dict, out_dir: Path) -> tuple[list, str | None]:
    """Cross-design recommendation table; returns (rows, finalist label).

    The band test applies to the LARGEST design (the re-eval cost binder):
    front sizes differ ~2.5x across designs, so requiring every design in
    band is unsatisfiable — smaller designs simply re-evaluate cheaper.
    """
    scenarios = list(all_results)
    rows = []
    for label, cand in CANDIDATES.items():
        row = {"candidate": label}
        for g, val in zip(("eps_rel", "eps_def", "eps_flood", "eps_storage"),
                          cand if cand is not None else ("", "", "", "")):
            row[g] = val
        adjusted = []
        for sc in scenarios:
            r = all_results[sc][label]
            n = r["sizes"].get("union")
            if n is None:  # step 07 not run for this scenario yet
                seed_sizes = [v for a, v in r["sizes"].items()
                              if a.startswith("seed")]
                n = max(seed_sizes) if seed_sizes else None
                row[f"size_{sc}"] = n if n is not None else ""
                row[f"substrate_{sc}"] = "seed"
            else:
                row[f"size_{sc}"] = n
                row[f"substrate_{sc}"] = "union"
            if n is None:
                continue
            adj = int(round(n * INFLATE_MERGED))
            adjusted.append(adj)
            row[f"adj110_{sc}"] = adj
            row[f"adj135_{sc}"] = int(round(n * INFLATE_SEED))
            cov = r["coverage"]
            if cov is not None:
                row[f"min_axis_boxes_{sc}"] = int(min(cov["boxes"]))
                # over-coarsened relative to the adopted vector's own
                # resolution on the same substrate (some axes are
                # intrinsically narrow — see MIN_AXIS_BOXES comment)
                adopted_boxes = all_results[sc]["adopted"]["coverage"]["boxes"]
                row[f"overcoarse_{sc}"] = any(
                    b < min(MIN_AXIS_BOXES, ab)
                    for b, ab in zip(cov["boxes"], adopted_boxes))
        max_adj = max(adjusted) if adjusted else None
        row["max_adj110"] = max_adj if max_adj is not None else ""
        # overcoarse_* columns are ADVISORY (any real coarsening drops some
        # axis below the adopted resolution — the size band already excludes
        # degenerate vectors); the coverage table is the judgment substrate.
        in_band = (cand is not None and max_adj is not None
                   and TARGET_BAND[0] <= max_adj <= TARGET_BAND[1])
        row["fineness"] = f"{fineness(cand):.3f}" if cand is not None else ""
        row["in_band_max"] = in_band
        rows.append(row)
    legacy._write_csv(out_dir / "recommendation.csv", rows)

    eligible = [(float(r["fineness"]), r["candidate"])
                for r in rows if r["in_band_max"]]
    finalist = min(eligible)[1] if eligible else None
    return rows, finalist


def write_recommendation_md(rows: list, finalist: str | None, scenarios: list,
                            out_path: Path):
    cols = ["candidate", "eps_rel", "eps_def", "eps_flood", "eps_storage"]
    cols += [f"{p}_{sc}" for sc in scenarios for p in ("size", "adj110")]
    cols += ["max_adj110", "fineness", "in_band_max"]
    lines = [
        "# Grouped-epsilon re-assessment on the production ensemble fronts",
        "",
        f"Substrate: cross-seed raw unions (draw 0 / seed 1, 500k NFE) for "
        f"{', '.join(scenarios)}. `adj110` = retained size x{INFLATE_MERGED} "
        f"(measured cross-seed live-search inflation); the target band "
        f"{TARGET_BAND[0]}-{TARGET_BAND[1]} applies to the LARGEST design "
        "(the re-eval cost binder — front sizes differ ~2.5x across designs). "
        f"`overcoarse` flags (ADVISORY, per design) mark candidates whose "
        f"occupied 1-D box count on some axis falls below "
        f"min({MIN_AXIS_BOXES}, the adopted vector's count on that axis) — "
        "judge them against the per-axis coverage tables.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")
    if finalist:
        lines.append(f"Selection rule (finest candidate whose largest-design "
                     f"adjusted size is in band): **{finalist}**. Decision "
                     "deferred — epsilon steers the live search, so "
                     "adopted-vector archives will run ~10-35% larger than "
                     "this static re-filter predicts.")
    else:
        lines.append("No candidate puts the largest design in the band — "
                     "inspect `recommendation.csv` and extend the grid.")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[table] {out_path}")


def size_figure(all_results: dict, out_png: Path):
    """Retained union size per candidate, one panel per design (dot plot)."""
    import matplotlib.pyplot as plt
    from src.plotting.style import apply_style

    apply_style()
    scenarios = list(all_results)
    labels = list(CANDIDATES)
    fig, axes = plt.subplots(1, len(scenarios),
                             figsize=(4.2 * len(scenarios), 5.5), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(labels))[::-1]
    for ax, sc in zip(axes, scenarios):
        sizes = [all_results[sc][lb]["sizes"].get("union") for lb in labels]
        ax.axvspan(TARGET_BAND[0] / INFLATE_MERGED,
                   TARGET_BAND[1] / INFLATE_MERGED,
                   color="0.85", zorder=0,
                   label=f"target band / {INFLATE_MERGED:g}")
        ax.plot([s for s in sizes if s is not None],
                [yy for yy, s in zip(y, sizes) if s is not None],
                "o", color="steelblue", markersize=5)
        ax.set_title(sc, fontsize=10)
        ax.set_xlabel("retained solutions (raw union)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, axis="x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", default="ffmp_obj8")
    ap.add_argument("--scenarios", nargs="+",
                    default=["historic", "fixed_probabilistic",
                             "hazard_filling_stationary"])
    ap.add_argument("--figures-only", action="store_true",
                    help="Re-render figures only (sweeps are recomputed in "
                         "memory; tables are not rewritten).")
    args = ap.parse_args()

    names = get_obj_names()
    directions = np.array(get_obj_directions())
    eps_adopted = np.array(get_epsilons(), dtype=float)
    n_vars = get_n_vars("ffmp")
    n_objs = get_n_objs()
    root = Path("outputs/supplemental/epsilon_refilter")

    all_results, all_archives = {}, {}
    for scenario in args.scenarios:
        run_dir = Path("outputs") / scenario / args.slug
        archives = (_load_archives(run_dir, args.slug, n_vars, n_objs)
                    if (run_dir / "sets").is_dir() else {})
        if not archives:
            print(f"[skip] {scenario}: no archives under {run_dir / 'sets'}")
            continue
        print(f"[{scenario}] archives: "
              f"{', '.join(f'{k} (n={len(v)})' for k, v in archives.items())}",
              flush=True)
        out_dir = root / f"{scenario}_{args.slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        smallest = min(archives.values(), key=len)
        _self_check(smallest, [eps_adopted,
                               grouped_vector(names, CANDIDATES["joint_x2"])])
        results = sweep_scenario(archives, names, eps_adopted)
        if not args.figures_only:
            legacy.part1_axis_structure(archives, names, directions,
                                        eps_adopted,
                                        out_dir / "axis_structure.csv")
            write_scenario_tables(results, archives, names, out_dir)
            validate_scenario(scenario, results, archives, run_dir,
                              args.slug, n_vars, n_objs)
        all_results[scenario] = results
        all_archives[scenario] = archives

    if not all_results:
        sys.exit("no scenario had archives — nothing to do")

    combined_dir = root / f"combined_{args.slug}_grouped"
    combined_dir.mkdir(parents=True, exist_ok=True)
    rows, finalist = combined_table(all_results, combined_dir)
    write_recommendation_md(rows, finalist, list(all_results),
                            combined_dir / "recommendation.md")
    size_figure(all_results, combined_dir / "size_vs_coarsening.png")

    # Full-vs-filtered parallel axes for the finalist, per scenario.
    highlight = finalist or "joint_x2"
    for scenario, archives in all_archives.items():
        if "union" not in archives:
            continue
        objs = archives["union"]
        v = all_results[scenario][highlight]["vector"]
        keep = fast_epsilon_nondominated(objs, v)
        mask = np.zeros(len(objs), dtype=bool)
        mask[keep] = True
        natural = objs * np.where(directions == 1, -1.0, 1.0)
        legacy._two_panel_figure(
            natural, mask, names, directions,
            root / f"{scenario}_{args.slug}" /
            f"grouped_refilter_parallel_axes_{highlight}_2panel.png",
            label=highlight)
    print(f"DONE -> {root} (finalist: {finalist or 'none in band'})")


if __name__ == "__main__":
    main()
