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
    ACTIVE_SCENARIO_DESIGN,
    STAGED_ENSEMBLE_DIR,
    ENSEMBLE_KN_YEARS,
    ENSEMBLE_KN_REALIZATIONS,
    ENSEMBLE_KN_SEED,
    ENSEMBLE_KN_FORCE,
)
from src.ensembles import kirsch_nowak_slug
from src.ensemble_generation import generate_kirsch_nowak_ensemble


def main() -> None:
    # Prefer the active scenario design's (length, size) when it resolves to a
    # directly generated Kirsch-Nowak ensemble (the fixed probabilistic designs),
    # so `NYCOPT_SCENARIO_DESIGN=fixed_probabilistic_short` + this script stage
    # exactly the slug the design's search spec resolves to. Otherwise fall back
    # to the NYCOPT_ENSEMBLE_KN_* env vars (e.g. for a large standalone master).
    _staged_dims = ACTIVE_SCENARIO_DESIGN.kn_staged_dims()
    if _staged_dims is not None:
        # (n_years, n_reals) of the ensemble this design needs staged. For the
        # resampled design this is the master POOL (n_reals = master_pool_size),
        # not the per-evaluation draw size.
        n_years, n_reals = _staged_dims
        print(
            f"[gen] Sizing from scenario design "
            f"'{ACTIVE_SCENARIO_DESIGN.name}' (years={n_years}, reals={n_reals})."
        )
    else:
        n_years = ENSEMBLE_KN_YEARS
        n_reals = ENSEMBLE_KN_REALIZATIONS
        print(
            f"[gen] Sizing from NYCOPT_ENSEMBLE_KN_* env "
            f"(years={n_years}, reals={n_reals})."
        )

    slug = kirsch_nowak_slug(n_years, n_reals)
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
        f"(years={n_years}, reals={n_reals}, seed={ENSEMBLE_KN_SEED})"
    )
    generate_kirsch_nowak_ensemble(
        n_years=n_years,
        n_realizations=n_reals,
        seed=ENSEMBLE_KN_SEED,
        output_dir=out,
    )
    print(f"[gen] Done: {slug} -> {out}")


if __name__ == "__main__":
    main()
