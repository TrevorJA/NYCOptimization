"""ensemble_objective_sensitivity_prep.py - Stage the experiment's ensemble.

One-time preparation for the ensemble objective-sensitivity experiment: generate
the fixed probabilistic Kirsch-Nowak ensemble (Step 1) at the sizes declared in
``supplemental_config.py`` and stage the pywrdrb HDF5 inputs the trimmed-model
ensemble simulation needs (Step 3). The DV sweep
(``ensemble_objective_sensitivity_run.py``) reads the staged ensemble.

This is the experiment's own driver (the ensemble is a supplemental
``kn_{Y}yr_n{N}`` slug, not the active scenario design's search ensemble that
``scripts/main/prep_pywrdrb_inputs.py`` stages). Generation is serial (rank 0);
the three staging preprocessors run under MPI when launched with multiple ranks.
All settings come from ``supplemental_config.py`` — no CLI value flags.

Usage (serial / laptop smoke):
    python scripts/supplemental/ensemble_objective_sensitivity_prep.py
Usage (SLURM / MPI):
    sbatch workflow/supplemental/ensemble_objective_sensitivity_prep.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_ensemble_env()  # set experiment env before config is imported

import config  # noqa: E402
from src.ensemble_generation import generate_kirsch_nowak_ensemble  # noqa: E402
from src.ensemble_prep import stage_pywrdrb_ensemble_inputs  # noqa: E402
from src.ensembles import get_ensemble_spec, staged_ensemble_dir  # noqa: E402


def _get_mpi_context():
    """Return (comm, rank, size). Falls back to (None, 0, 1) without a runtime."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        return None, 0, 1


def main() -> None:
    comm, rank, size = _get_mpi_context()
    use_mpi = size > 1

    inflow_type = scfg.ensemble_inflow_type()
    flows_dir = staged_ensemble_dir(inflow_type)
    base_inflow = flows_dir / "catchment_inflow_mgd.hdf5"

    # Step 1 (rank 0 only): generate the Kirsch-Nowak ensemble if not present.
    if rank == 0:
        if base_inflow.exists():
            print(f"[ensemble_prep] base inflows present: {base_inflow}", flush=True)
        else:
            flows_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ensemble_prep] generating {inflow_type} "
                  f"(N={scfg.ENS_N_REALIZATIONS}, {scfg.ENS_REALIZATION_YEARS}-yr, "
                  f"seed={scfg.ENS_KN_SEED}) ...", flush=True)
            generate_kirsch_nowak_ensemble(
                n_years=scfg.ENS_REALIZATION_YEARS,
                n_realizations=scfg.ENS_N_REALIZATIONS,
                seed=scfg.ENS_KN_SEED,
                output_dir=flows_dir,
                start_date=config.START_DATE,
            )
    if use_mpi:
        comm.Barrier()  # all ranks wait until the ensemble is on disk

    # Step 3 (all ranks): stage flood / presim / predicted-inflow HDF5s.
    spec = get_ensemble_spec(inflow_type)
    manifest = stage_pywrdrb_ensemble_inputs(
        spec, use_mpi=use_mpi, comm=comm,
        prediction_modes=scfg.ENS_PREDICTION_MODES,
    )

    if rank == 0:
        print("[ensemble_prep] staged:", flush=True)
        for key, value in manifest.items():
            print(f"  {key}: {value}", flush=True)


if __name__ == "__main__":
    main()
