"""drought_curtailment_figures.py - Storage and FFMP curtailments through a drought.

Renders :func:`src.plotting.drought_curtailment.plot_drought_curtailment` for a
policy-set run directory written by ``simulate_selected_policies.py``: NYC
storage, NYC diversion (% of the 800 MGD Decree allocation) and the Montague
flow target (% of the 1,750 cfs Decree obligation) for the FFMP baseline and
the selected policies. Figure-only: it reads the HDF5 runs and never simulates.

The default run directory holds the baseline and solution 108 of the historic
merged set under a constant 800 MGD NYC demand:

    sbatch --export=ALL,HDF5_USE_FILE_LOCKING=FALSE,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj8_historic_production.env \
        workflow/supplemental/figures/sim_selected_policies.sh --slug ffmp_obj8 \
        --rows 108 --tag constant_max_s0108 --trimmed --nyc-demand constant_max

Outputs ``drought_curtailment_{tag}_{start}_{end}.png`` and the plotted series
as a ``.csv`` of the same stem under ``outputs/figures/{scenario}/{slug}/dynamics/``
(override with ``--out-dir``).

Usage (from repo root, venv active):
    python3 scripts/supplemental/drought_curtailment_figures.py
    python3 scripts/supplemental/drought_curtailment_figures.py --range 1961-01-01 1968-01-01
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

import config  # noqa: E402
from src.plotting.drought_curtailment import (  # noqa: E402
    REQUIRED_RESULTS_SETS, curtailment_table, plot_drought_curtailment,
)
from src.policy_simulation import (  # noqa: E402
    BASELINE_NAME, default_run_dir, load_policy_runs,
)

#: Subtitle wording per manifest ``nyc_demand`` value.
DEMAND_TEXT = {"constant_max": "constant 800 MGD NYC demand",
               "time_dynamic": "time-varying NYC demand"}


def _label(name: str) -> str:
    """Short legend label of a manifest policy name."""
    if name == BASELINE_NAME:
        return "FFMP baseline"
    if name.startswith("row_"):
        return f"Robust policy (#{int(name[4:])})"
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenario", default="historic")
    ap.add_argument("--slug", default="ffmp_obj8")
    ap.add_argument("--tag", default="constant_max_s0108",
                    help="policy-set tag under .../timeseries/")
    ap.add_argument("--run-dir", default=None,
                    help="policy-set run directory; default "
                         "outputs/{scenario}/{slug}/timeseries/{tag}")
    ap.add_argument("--out-dir", default=None,
                    help="default outputs/figures/{scenario}/{slug}/dynamics")
    ap.add_argument("--range", nargs=2, metavar=("START", "END"),
                    default=("1963-01-01", "1968-01-01"), help="plotted window")
    ap.add_argument("--policies", nargs="*", default=None,
                    help="subset of policy names from the manifest")
    ap.add_argument("--title", default="1960s drought of record")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(
        args.scenario, args.slug, args.tag)
    out_dir = Path(args.out_dir) if args.out_dir else (
        config.FIGURES_DIR / args.scenario / args.slug / "dynamics")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, runs = load_policy_runs(run_dir, REQUIRED_RESULTS_SETS, args.policies)
    labels = {name: _label(name) for name in runs}
    demand = (manifest["nyc_demand"].iloc[0] if "nyc_demand" in manifest
              else "time_dynamic")
    subtitle = f"Historic inflows, {DEMAND_TEXT.get(demand, demand)}"
    print(f"run dir  : {run_dir}\npolicies : {list(runs)}\ndemand   : {demand}")

    date_range = tuple(args.range)
    stub = out_dir / (f"drought_curtailment_{args.tag}_"
                      f"{date_range[0][:4]}_{date_range[1][:4]}")
    plot_drought_curtailment(runs, labels, date_range=date_range,
                             title=args.title, subtitle=subtitle,
                             output_file=stub)
    plt.close("all")
    curtailment_table(runs, date_range).to_csv(stub.with_suffix(".csv"),
                                               float_format="%.3f")
    print(f"[figure] {stub}.png\n[table]  {stub}.csv")


if __name__ == "__main__":
    main()
