"""epsilon_union_refilter.py - Archive size under candidate epsilons, from every archived row.

Estimates the Pareto-approximate set size that candidate epsilon vectors would
produce, using every solution one completed seed ever held in an island
archive (all snapshots of every island runtime file) rather than its final
``.set``. Re-filtering the final set alone cannot exceed its own size under a
finer epsilon, because the coarser search discarded the intermediate
solutions; the union of all island snapshots retains many of them.

The estimate is a lower bound in expectation: a search run under a finer
epsilon keeps more solutions in its archive, and therefore explores and
retains points that the coarser search never stored.

Parts:
  1. Per-axis structure of the union (distinct values, smallest positive gap
     between distinct values, range), natural units. On the historic trace the
     annual reliabilities lie on a 1/77 lattice, so an epsilon below the gap
     adds no resolution on that axis.
  2. Epsilon-box nondominated size (``src.sensitivity_common.epsilon_nondominated``,
     Borg box convention) under the adopted vector and per-family multiples of
     it (:data:`CANDIDATES`), for three layers:
     the final ``.set``, the island archives at their last snapshot, and the
     union of every island snapshot.

Outputs (``outputs/supplemental/epsilon_union_refilter/{scenario}/{slug}/``):

    axis_structure.csv
    archive_size_by_candidate.csv

Usage (from repo root, through the wrapper):
    sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
        workflow/supplemental/results_diagnostics/epsilon_union_refilter.sh \
        --scenario historic --slug ffmp_obj8 --seed 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import config
from src.formulations import (get_n_objs, get_n_vars, get_obj_directions,
                              get_obj_names)
from src.pareto_filter import to_natural
from src.runtime_archive import parse_result_file
from src.sensitivity_common import epsilon_nondominated

#: Objective families for the candidate multipliers, matched on the active
#: objective name.
FAMILIES: dict[str, str] = {
    "reliability": "reliability",
    "deficit": "deficit",
    "flood": "flood",
    "storage": "storage",
}

#: Candidates as per-family multipliers of the adopted vector
#: (``config.get_epsilons()``); families absent from a candidate keep the
#: adopted value.
CANDIDATES: dict[str, dict[str, float]] = {
    "adopted": {},
    "reliability_x0.5": {"reliability": 0.5},
    "deficit_x0.5": {"deficit": 0.5},
    "flood_x0.5": {"flood": 0.5},
    "storage_x0.5": {"storage": 0.5},
    "all_x0.5": {f: 0.5 for f in FAMILIES},
    "all_x2": {f: 2.0 for f in FAMILIES},
}


def candidate_vectors(names: list[str], adopted: list[float]) -> dict[str, list[float]]:
    """Expand :data:`CANDIDATES` into full epsilon vectors in objective order.

    Raises:
        ValueError: If an objective name matches no family.
    """
    family_of = []
    for nm in names:
        hits = [f for f, key in FAMILIES.items() if key in nm]
        if len(hits) != 1:
            raise ValueError(f"objective {nm!r} matches families {hits}")
        family_of.append(hits[0])
    return {label: [e * mult.get(f, 1.0) for e, f in zip(adopted, family_of)]
            for label, mult in CANDIDATES.items()}


def _entries(path: Path) -> list[np.ndarray]:
    """Row arrays of every non-empty entry of a runtime or set file."""
    return [rows for _, rows in parse_result_file(path.read_text())
            if rows.size]


def _dedupe(rows: np.ndarray, n_vars: int) -> np.ndarray:
    """Drop rows whose objective vector repeats an earlier row."""
    _, idx = np.unique(rows[:, n_vars:], axis=0, return_index=True)
    return rows[np.sort(idx)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="historic")
    ap.add_argument("--slug", default="ffmp_obj8")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--formulation", default="ffmp")
    args = ap.parse_args()

    n_vars = get_n_vars(args.formulation)
    n_objs = get_n_objs()
    names = get_obj_names()
    directions = get_obj_directions()
    vectors = candidate_vectors(names, list(config.get_epsilons()))

    run_dir = PROJECT_DIR / "outputs" / args.scenario / args.slug
    tag = f"seed_{args.seed:02d}_{args.slug}"
    set_file = run_dir / "sets" / f"{tag}.set"
    island_files = sorted((run_dir / "runtime").glob(f"{tag}_*.runtime"))
    if not set_file.exists():
        raise FileNotFoundError(set_file)
    if not island_files:
        raise FileNotFoundError(f"no island runtime files for {tag}")

    island_snaps = [_entries(p) for p in island_files]
    layers = {
        "final_set": _dedupe(np.vstack(_entries(set_file)), n_vars),
        "island_final": _dedupe(np.vstack([s[-1] for s in island_snaps]), n_vars),
        "island_all": _dedupe(np.vstack([r for s in island_snaps for r in s]), n_vars),
    }
    for layer, rows in layers.items():
        if rows.shape[1] != n_vars + n_objs:
            raise ValueError(f"{layer}: {rows.shape[1]} columns, expected "
                             f"{n_vars + n_objs}")

    out_dir = (PROJECT_DIR / "outputs" / "supplemental"
               / "epsilon_union_refilter" / args.scenario / args.slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    union = layers["island_all"]
    natural = to_natural(union[:, n_vars:], directions)
    axis_rows = []
    for j, nm in enumerate(names):
        vals = np.unique(np.round(natural[:, j], 9))
        gaps = np.diff(vals)
        gaps = gaps[gaps > 0]
        axis_rows.append({
            "objective": nm,
            "distinct_values": int(vals.size),
            "min_gap": float(gaps.min()) if gaps.size else float("nan"),
            "median_gap": float(np.median(gaps)) if gaps.size else float("nan"),
            "min": float(vals.min()),
            "max": float(vals.max()),
        })
    axes = pd.DataFrame(axis_rows)
    axes.to_csv(out_dir / "axis_structure.csv", index=False)
    print(f"union of island snapshots: {union.shape[0]} distinct rows\n")
    print("per-axis structure (natural units):")
    print(axes.to_string(index=False))

    size_rows = []
    for name, eps in vectors.items():
        rec = {"candidate": name, "epsilons": ",".join(f"{e:g}" for e in eps)}
        for layer, rows in layers.items():
            rec[layer] = int(epsilon_nondominated(rows[:, n_vars:], eps).size)
        size_rows.append(rec)
    sizes = pd.DataFrame(size_rows)
    sizes.to_csv(out_dir / "archive_size_by_candidate.csv", index=False)
    print("\nepsilon-box nondominated size per layer:")
    print(sizes.to_string(index=False))
    print(f"\n[table] {out_dir / 'axis_structure.csv'}")
    print(f"[table] {out_dir / 'archive_size_by_candidate.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
