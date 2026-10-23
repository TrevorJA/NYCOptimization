"""policy_dynamics_figures.py - System-dynamics panels of a simulated policy set.

Renders :func:`src.plotting.policy_dynamics.plot_policy_dynamics` for a
policy-set run directory written by ``simulate_selected_policies.py``
(``src.policy_simulation``), once per requested period. Figure-only: it reads
the HDF5 runs and never simulates.

Outputs ``policy_dynamics_{period}.png`` under ``figures/{scenario}/{slug}/
dynamics/`` (override with ``--out-dir``).

Usage (from repo root, venv active):
    python3 scripts/supplemental/policy_dynamics_figures.py
    python3 scripts/supplemental/policy_dynamics_figures.py --periods drought_2002 full
    python3 scripts/supplemental/policy_dynamics_figures.py --range 1980-01-01 1986-01-01
    python3 scripts/supplemental/policy_dynamics_figures.py --panels storage montague
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.plotting.policy_dynamics import (
    DEFAULT_PANELS, DROUGHT_PERIODS, PANELS, REQUIRED_RESULTS_SETS,
    plot_policy_dynamics,
)
from src.policy_simulation import default_run_dir, load_policy_runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", default="historic")
    ap.add_argument("--slug", default="ffmp_obj8")
    ap.add_argument("--tag", default="selected_policies",
                    help="policy-set tag under .../timeseries/")
    ap.add_argument("--run-dir", default=None,
                    help="policy-set run directory; default "
                         "outputs/{scenario}/{slug}/timeseries/{tag}")
    ap.add_argument("--out-dir", default=None,
                    help="default figures/{scenario}/{slug}/dynamics")
    ap.add_argument("--periods", nargs="*", default=["drought_1960s", "drought_2002"],
                    choices=list(DROUGHT_PERIODS),
                    help="named periods (src.plotting.policy_dynamics.DROUGHT_PERIODS)")
    ap.add_argument("--range", nargs=2, metavar=("START", "END"), default=None,
                    help="one explicit period instead of --periods")
    ap.add_argument("--panels", nargs="*", default=list(DEFAULT_PANELS),
                    choices=list(PANELS), help="panels, top to bottom")
    ap.add_argument("--policies", nargs="*", default=None,
                    help="subset of policy names from the manifest")
    ap.add_argument("--resample", default="W", help="pandas resample rule")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(
        args.scenario, args.slug, args.tag)
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_DIR / "figures" / args.scenario / args.slug / "dynamics")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, runs = load_policy_runs(run_dir, REQUIRED_RESULTS_SETS, args.policies)
    labels = dict(zip(manifest["policy"], manifest["labels"]))
    print(f"run dir  : {run_dir}\npolicies : {list(runs)}")

    periods = ({f"{args.range[0]}_{args.range[1]}": tuple(args.range)}
               if args.range else {p: DROUGHT_PERIODS[p] for p in args.periods})
    for name, date_range in periods.items():
        stub = out_dir / f"policy_dynamics_{name}"
        plot_policy_dynamics(runs, labels, panels=args.panels,
                             date_range=date_range, resample=args.resample,
                             output_file=stub)
        plt.close("all")
        print(f"[figure] {stub}.png")


if __name__ == "__main__":
    main()
