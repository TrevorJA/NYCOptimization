"""baseline_dominance_layers.py - Baseline dominance at each archive layer of one search.

Tests whether policies that dominate the FFMP baseline were found during the
search and then removed by the epsilon-box archive, or were never found. The
same baseline comparison (``src.solution_selection.dominance_mask``, weak
Pareto dominance, exact comparison) is applied to four nested layers of one
seed:

    final_set          the controller ``seed_SS_{slug}.set``
    island_final       the union of the four island archives at their last snapshot
    island_all         the union of every snapshot of every island archive
    coordinator_all    the union of every coordinator-archive snapshot
                       (``runtime_coordinator/seed_SS_{slug}.runtime``, step 09)

Every layer is deduplicated on its objective vector. For each layer the table
reports the row count, the number of rows that dominate the baseline, the
nondominated subset of those dominators, the epsilon boxes they occupy, and
the distribution of the number of objectives on which rows beat the baseline.
A second table counts, per objective, the rows that are no worse than the
baseline, which locates the binding axes.

Outputs (``outputs/supplemental/baseline_dominance_layers/{scenario}/{slug}/``):

    dominance_by_layer.csv
    no_worse_by_objective.csv
    baseline_dominators.set   (every distinct dominating row, if any)

Usage (from repo root, through the wrapper):
    sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
        workflow/supplemental/results_diagnostics/baseline_dominance_layers.sh \
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
from src.runtime_archive import (epsilon_boxes, parse_result_file,
                                 write_set_file)
from src.solution_selection import (dominance_mask, n_objectives_beaten,
                                    nondominated_mask, orient_maximize)


def _all_snapshot_rows(path: Path) -> list[np.ndarray]:
    """Row arrays of every non-empty entry of a runtime file, in file order."""
    return [rows for _, rows in parse_result_file(path.read_text())
            if rows.size]


def _dedupe(rows: np.ndarray, n_vars: int) -> np.ndarray:
    """Drop rows whose objective vector repeats an earlier row."""
    if rows.size == 0:
        return rows
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
    eps = config.get_epsilons()

    run_dir = PROJECT_DIR / "outputs" / args.scenario / args.slug
    tag = f"seed_{args.seed:02d}_{args.slug}"
    set_file = run_dir / "sets" / f"{tag}.set"
    island_files = sorted((run_dir / "runtime").glob(f"{tag}_*.runtime"))
    coord_file = run_dir / "runtime_coordinator" / f"{tag}.runtime"
    for p in [set_file, coord_file, *island_files]:
        if not p.exists():
            raise FileNotFoundError(p)
    if not island_files:
        raise FileNotFoundError(f"no island runtime files for {tag}")

    base_csv = config.baseline_objectives_csv(args.formulation, args.scenario)
    base_df = pd.read_csv(base_csv)
    if list(base_df.columns) != list(names):
        raise ValueError(f"{base_csv}: columns {list(base_df.columns)} "
                         f"!= active objectives {names}")
    baseline = base_df.iloc[0][names].to_numpy(dtype=float)

    final_rows = np.vstack(_all_snapshot_rows(set_file))
    island_snaps = [_all_snapshot_rows(p) for p in island_files]
    layers = {
        "final_set": final_rows,
        "island_final": np.vstack([s[-1] for s in island_snaps]),
        "island_all": np.vstack([r for s in island_snaps for r in s]),
        "coordinator_all": np.vstack(_all_snapshot_rows(coord_file)),
    }

    out_dir = (PROJECT_DIR / "outputs" / "supplemental"
               / "baseline_dominance_layers" / args.scenario / args.slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"baseline : {base_csv}")
    print(f"epsilons : {list(eps)}")
    rows_out, no_worse_out = [], []
    dominators_all = []
    for layer, raw in layers.items():
        if raw.shape[1] != n_vars + n_objs:
            raise ValueError(f"{layer}: {raw.shape[1]} columns, expected "
                             f"{n_vars + n_objs}")
        rows = _dedupe(raw, n_vars)
        natural = to_natural(rows[:, n_vars:], directions)
        dom = dominance_mask(natural, baseline, directions)
        beaten = n_objectives_beaten(natural, baseline, directions)
        dom_rows = rows[dom]
        nd = (nondominated_mask(natural[dom], directions)
              if dom.any() else np.zeros(0, dtype=bool))
        dominators_all.append(dom_rows)
        rec = {
            "layer": layer,
            "rows_raw": int(raw.shape[0]),
            "rows_distinct": int(rows.shape[0]),
            "dominate_baseline": int(dom.sum()),
            "dominators_nondominated": int(nd.sum()),
            "dominator_eps_boxes": len(epsilon_boxes(dom_rows, n_vars, eps)),
            "eps_boxes_all_rows": len(epsilon_boxes(rows, n_vars, eps)),
        }
        for k in range(n_objs + 1):
            rec[f"beats_on_{k}"] = int((beaten == k).sum())
        rows_out.append(rec)

        z = orient_maximize(natural, directions)
        r = orient_maximize(baseline, directions)
        no_worse = {"layer": layer}
        for j, nm in enumerate(names):
            no_worse[nm] = int((z[:, j] >= r[j]).sum())
        no_worse_out.append(no_worse)

        print(f"\n[{layer}] {rec['rows_raw']} rows ({rec['rows_distinct']} distinct), "
              f"{rec['eps_boxes_all_rows']} eps boxes")
        print(f"  dominate baseline       : {rec['dominate_baseline']} "
              f"({rec['dominators_nondominated']} mutually nondominated, "
              f"{rec['dominator_eps_boxes']} eps boxes)")
        print(f"  objectives-beaten dist. : "
              f"{[rec[f'beats_on_{k}'] for k in range(n_objs + 1)]}")

    table = pd.DataFrame(rows_out)
    table.to_csv(out_dir / "dominance_by_layer.csv", index=False)
    nw = pd.DataFrame(no_worse_out)
    nw.to_csv(out_dir / "no_worse_by_objective.csv", index=False)
    print("\nrows no worse than the baseline, per objective:")
    print(nw.to_string(index=False))

    dom_union = _dedupe(np.vstack(dominators_all), n_vars) if any(
        d.size for d in dominators_all) else np.empty((0, n_vars + n_objs))
    if dom_union.size:
        from src.formulations import get_var_names
        write_set_file(out_dir / "baseline_dominators.set", dom_union,
                       get_var_names(args.formulation), names,
                       header_lines=[f"Formulation: {args.formulation}, Seed: {args.seed}",
                                     "Epsilons: " + ",".join(str(e) for e in eps),
                                     "Rows dominating the scenario-matched FFMP baseline, "
                                     "all archive layers"])
        print(f"\n[set] {out_dir / 'baseline_dominators.set'} "
              f"({dom_union.shape[0]} rows)")
    print(f"[table] {out_dir / 'dominance_by_layer.csv'}")
    print(f"[table] {out_dir / 'no_worse_by_objective.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
