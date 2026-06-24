"""prep_pywrdrb_inputs.py - Workflow Step 3: format a generated streamflow
ensemble into the pywrdrb HDF5 inputs the trimmed optimization model reads.

For the active scenario design's search ensemble (``config.SEARCH_ENSEMBLE_SPEC``)
this stages, under ``STAGED_ENSEMBLE_DIR/{inflow_type}/``:

    catchment_inflow_with_flood_nodes_mgd.hdf5   (FlowEnsemble, flood ops on)
    presimulated_releases_mgd.hdf5  (+ _metadata.json)   (trimmed-model releases)
    predicted_inflows_mgd.hdf5                   (Montague/Trenton forecasts)

so pywrdrb's path navigator resolves them at simulation start (see
``src/ensembles.py::register_ensemble_path``). The base
``catchment_inflow_mgd.hdf5`` must already exist (Step 1 generator).

All settings come from ``config`` (scenario design, ensemble preset, demand
source, initial volume) — there are no CLI value flags. The actual staging is
delegated to the shared ``src.ensemble_prep.stage_pywrdrb_ensemble_inputs``,
which the supplemental ensemble experiment also uses. MPI is used automatically
when launched under more than one rank.

Usage (serial):
    python scripts/main/prep_pywrdrb_inputs.py
Usage (SLURM / MPI):
    srun python scripts/main/prep_pywrdrb_inputs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
from src.ensemble_prep import stage_pywrdrb_ensemble_inputs  # noqa: E402
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
    # --preset stages an ARBITRARY ensemble (e.g. a held-out common re-eval
    # ensemble) instead of the active scenario design's search ensemble. Use
    # parse_known_args so MPI launchers can pass through extra args harmlessly.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", default=None,
        help="Ensemble preset / kn_{Y}yr_n{N} slug / staged dir to prep "
             "(default: the active scenario design's search ensemble).",
    )
    args, _ = parser.parse_known_args()

    comm, rank, size = _get_mpi_context()
    use_mpi = size > 1

    if args.preset:
        spec = get_ensemble_spec(args.preset)
    else:
        spec = config.SEARCH_ENSEMBLE_SPEC
    scenario = config.active_scenario_name()

    if spec is None:
        if rank == 0:
            print(f"[prep_pywrdrb_inputs] scenario design '{scenario}' has no "
                  "search ensemble wired (SEARCH_ENSEMBLE_SPEC is None); "
                  "nothing to stage.", flush=True)
        return

    if not spec.is_ensemble:
        if rank == 0:
            print(f"[prep_pywrdrb_inputs] scenario design '{scenario}' uses a "
                  f"single trace ({spec.inflow_type}); no ensemble staging "
                  "needed.", flush=True)
        return

    if rank == 0:
        print(f"[prep_pywrdrb_inputs] scenario='{scenario}' "
              f"inflow_type='{spec.inflow_type}' "
              f"n_realizations={spec.n_realizations} "
              f"ranks={size}", flush=True)

    manifest = stage_pywrdrb_ensemble_inputs(spec, use_mpi=use_mpi, comm=comm)

    if rank == 0:
        print("[prep_pywrdrb_inputs] staged:", flush=True)
        for key, value in manifest.items():
            print(f"  {key}: {value}", flush=True)


if __name__ == "__main__":
    main()
