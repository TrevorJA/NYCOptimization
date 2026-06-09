"""generate_stochastic_ensemble.py - Step 1 of the NYCOpt ensemble pipeline.

Builds a Kirsch-Nowak synthetic streamflow ensemble using SynHydro, sourcing
historical flows from pywrdrb's ``pub_nhmv10_BC_withObsScaled`` dataset.

All configuration comes from ``config.py`` (which honors ``NYCOPT_ENSEMBLE_KN_*``
env vars). Typical invocation:

    sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ensemble_kn_long.env \\
           workflow/01_generate_stochastic_ensemble.sh

Output (under ``outputs/synthetic_ensembles/kn_{Y}yr_n{N}/``):

    gage_flow_mgd.hdf5
    catchment_inflow_mgd.hdf5
    _meta.json
"""

from __future__ import annotations

import sys

from config import (
    STAGED_ENSEMBLE_DIR,
    ENSEMBLE_KN_YEARS,
    ENSEMBLE_KN_REALIZATIONS,
    ENSEMBLE_KN_SEED,
    ENSEMBLE_KN_FORCE,
)
from src.ensembles import kirsch_nowak_slug
from src.ensemble_generation import generate_kirsch_nowak_ensemble


def main() -> None:
    slug = kirsch_nowak_slug(ENSEMBLE_KN_YEARS, ENSEMBLE_KN_REALIZATIONS)
    out = STAGED_ENSEMBLE_DIR / slug
    if out.exists() and any(out.iterdir()) and not ENSEMBLE_KN_FORCE:
        print(
            f"[gen] Ensemble {slug} already staged at {out}. "
            f"Set NYCOPT_ENSEMBLE_KN_FORCE=1 to regenerate."
        )
        sys.exit(0)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[gen] Building {slug} "
        f"(years={ENSEMBLE_KN_YEARS}, reals={ENSEMBLE_KN_REALIZATIONS}, "
        f"seed={ENSEMBLE_KN_SEED})"
    )
    generate_kirsch_nowak_ensemble(
        n_years=ENSEMBLE_KN_YEARS,
        n_realizations=ENSEMBLE_KN_REALIZATIONS,
        seed=ENSEMBLE_KN_SEED,
        output_dir=out,
    )
    print(f"[gen] Done: {slug} -> {out}")


if __name__ == "__main__":
    main()
