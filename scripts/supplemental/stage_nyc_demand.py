"""stage_nyc_demand.py - Stage the per-realization NYC demand series only.

Adds ``diversion_nyc_extrapolated_mgd.hdf5`` -- the time-dynamic extrapolated
NYC demand every ensemble simulation reads
(``config.NYC_DEMAND_SOURCE_ENSEMBLE``) -- to an ensemble whose other step-04
inputs are already staged, without repeating the flood-node, STARFIT-presim and
predicted-inflow passes. Step 04 (``scripts/main/prep_pywrdrb_inputs.py``)
stages the series itself for new ensembles; this script exists for ensembles
staged before the series was part of the contract (the E_test chunks, the
N = 100 diagnostics ensembles).

The target is the active scenario design's search ensemble
(``config.SEARCH_ENSEMBLE_SPEC``) or, with ``--preset``, any staged ensemble
slug (e.g. an E_test chunk ``etest_kn_50yr_n12500__chunk007``). MPI is used
automatically when launched under more than one rank. Existing output is
rebuilt so the staged series always reflects the current pywrdrb code.

Usage (serial):
    python scripts/supplemental/stage_nyc_demand.py [--preset SLUG]
Usage (SLURM / MPI):
    mpirun -np 33 python scripts/supplemental/stage_nyc_demand.py --preset SLUG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src.ensemble_prep import stage_nyc_demand  # noqa: E402
from src.ensembles import get_ensemble_spec  # noqa: E402


def _get_mpi_context():
    """Return (comm, rank, size). Falls back to (None, 0, 1) without a runtime."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        return None, 0, 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", default=None,
        help="Ensemble preset / kn_{Y}yr_n{N} slug / staged dir to stage the NYC "
             "demand for (default: the active scenario design's search ensemble).",
    )
    args, _ = parser.parse_known_args()

    comm, rank, size = _get_mpi_context()
    use_mpi = size > 1

    spec = get_ensemble_spec(args.preset) if args.preset else config.SEARCH_ENSEMBLE_SPEC
    scenario = config.active_scenario_name()

    if spec is None:
        if rank == 0:
            print(f"[stage_nyc_demand] scenario design '{scenario}' has no search "
                  "ensemble wired (SEARCH_ENSEMBLE_SPEC is None); nothing to stage.",
                  flush=True)
        return
    if not spec.is_ensemble:
        if rank == 0:
            print(f"[stage_nyc_demand] '{spec.inflow_type}' is a single trace; its "
                  "NYC demand is pywrdrb's historic-flow series, nothing to stage.",
                  flush=True)
        return

    if rank == 0:
        print(f"[stage_nyc_demand] inflow_type='{spec.inflow_type}' "
              f"n_realizations={spec.n_realizations} ranks={size}", flush=True)

    out = stage_nyc_demand(spec, use_mpi=use_mpi, comm=comm, force=True)

    if rank == 0:
        print(f"[stage_nyc_demand] staged: {out}", flush=True)


if __name__ == "__main__":
    main()
