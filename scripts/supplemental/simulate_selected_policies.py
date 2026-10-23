"""simulate_selected_policies.py - Full-model HDF5 runs of a selected policy set.

Thin command-line wrapper around :mod:`src.policy_simulation`. By default it
simulates the highlighted policies of an ``explore_results`` selection table
(``figures/{scenario}/{slug}/selected_solutions.csv``: the balanced compromise
and the diverse picks shown in ``explore_02_selected_policies.png``) from the
matching reference set, plus the incumbent FFMP policy, on the historic trace
with the full Pywr-DRB model, and keeps the complete HDF5 output of each run
under ``outputs/{scenario}/{slug}/timeseries/{tag}/``.

Other policy sets: ``--all-rules`` adds every single-objective and compromise
optimum of the table, ``--rows`` names explicit reference-set rows,
``--set-file`` / ``--selection-csv`` point at other searches, and ``--tag``
keeps several collections side by side. Load the results later with
``src.policy_simulation.load_policy_runs(run_dir)``.

Usage (from repo root, via sbatch; never on a login node):
    sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic.env \
        workflow/supplemental/sim_selected_policies.sh [options]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from config import OUTPUTS_DIR
from src.policy_simulation import (
    PolicySet, default_run_dir, read_selection_labels, simulate_policy_set,
)


def build_policy_set(args: argparse.Namespace) -> PolicySet:
    """Resolve the command-line options into a :class:`PolicySet`."""
    set_file = Path(args.set_file) if args.set_file else (
        OUTPUTS_DIR / args.scenario / args.slug / "sets"
        / f"{args.slug}_merged.set")
    selection_csv = Path(args.selection_csv) if args.selection_csv else (
        PROJECT_DIR / "figures" / args.scenario / args.slug
        / "selected_solutions.csv")
    if args.rows:
        labels = (read_selection_labels(selection_csv, highlighted_only=False)
                  if selection_csv.exists() else None)
        ps = PolicySet.from_reference_set(set_file, args.rows, labels,
                                          args.formulation)
    else:
        ps = PolicySet.from_selection_csv(
            set_file, selection_csv, args.formulation,
            highlighted_only=not args.all_rules)
    return ps if args.no_baseline else ps.with_baseline()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", default="historic")
    ap.add_argument("--slug", default="ffmp_obj8")
    ap.add_argument("--formulation", default="ffmp")
    ap.add_argument("--tag", default="selected_policies",
                    help="subdirectory under .../timeseries/ for this set")
    ap.add_argument("--set-file", default=None,
                    help="reference set; default outputs/{scenario}/{slug}/"
                         "sets/{slug}_merged.set")
    ap.add_argument("--selection-csv", default=None,
                    help="selection table (row_index, label); default "
                         "figures/{scenario}/{slug}/selected_solutions.csv")
    ap.add_argument("--rows", type=int, nargs="*", default=None,
                    help="explicit reference-set rows instead of the table")
    ap.add_argument("--all-rules", action="store_true",
                    help="every row of the table, not only the highlighted "
                         "(plotted) subset")
    ap.add_argument("--out-dir", default=None,
                    help="default outputs/{scenario}/{slug}/timeseries/{tag}")
    ap.add_argument("--no-baseline", action="store_true",
                    help="do not add the incumbent FFMP policy")
    ap.add_argument("--trimmed", action="store_true",
                    help="use the trimmed model (default: full model)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-simulate policies whose HDF5 already exists")
    ap.add_argument("--no-objectives", action="store_true",
                    help="skip recomputing the active objective set")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else default_run_dir(
        args.scenario, args.slug, args.tag)
    policy_set = build_policy_set(args)
    manifest = simulate_policy_set(
        policy_set, out_dir, use_trimmed=args.trimmed,
        overwrite=args.overwrite, compute_objectives=not args.no_objectives,
        log=lambda msg: print(msg, flush=True))
    print(f"\n{len(manifest)} policies recorded in {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
