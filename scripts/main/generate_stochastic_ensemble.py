"""generate_stochastic_ensemble.py - Step 1 of the NYCOpt ensemble pipeline.

Stages the master (pool) ensemble the active scenario design draws from, sourcing historical flows
from pywrdrb's ``pub_nhmv10_BC_withObsScaled`` dataset via SynHydro. Two master kinds:

  - ``stationary`` (direct Kirsch-Nowak, ``kn_{Y}yr_n{N}``): the fixed / resampled probabilistic
    designs. Sized from the design, else from ``NYCOPT_ENSEMBLE_KN_*``.
  - ``forcing`` (CMIP6-forced master, ``master_{L}yr_n{N_M}``): the shared master behind
    ``hazard_filling`` and ``input_stratified`` (methods §3.1-3.2). Sized from the design; forcing
    space + storage mode from ``NYCOPT_ENSEMBLE_FORCING_*`` / ``NYCOPT_ENSEMBLE_MASTER_*``.

All configuration comes from ``config.py`` / the scenario-design registry — no CLI value flags.
Typical invocation:

    sbatch --export=ALL,NYCOPT_ENV_FILE=slurm/envs/ensemble_kn_long.env \\
           workflow/01_generate_stochastic_ensemble.sh

Output under ``outputs/synthetic_ensembles/{slug}/``: ``gage_flow_mgd.hdf5``,
``catchment_inflow_mgd.hdf5``, ``_meta.json`` (both kinds), plus ``forcing_profiles.npz``,
``hazard_image.npz``, ``manifest.json`` (forcing master).
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
    ENSEMBLE_FORCING_MEAN_FRAC_CSV,
    ENSEMBLE_FORCING_MEAN_ABS_CSV,
    ENSEMBLE_FORCING_STD_CSV,
    ENSEMBLE_FORCING_VARIANCE_AXIS,
    ENSEMBLE_FORCING_BOUND_PCT,
    ENSEMBLE_FORCING_MARGIN,
    ENSEMBLE_MASTER_SEED,
    ENSEMBLE_MASTER_STREAM_ONLY,
    ENSEMBLE_MASTER_HAZARD_BLOCK,
    ENSEMBLE_MASTER_CHUNK_SIZE,
)
from src.ensembles import kirsch_nowak_slug
from src.ensemble_generation import generate_kirsch_nowak_ensemble


def _already_staged(out) -> bool:
    """True if ``out`` holds a staged ensemble and regeneration was not forced."""
    if out.exists() and any(out.iterdir()) and not ENSEMBLE_KN_FORCE:
        print(f"[gen] Ensemble {out.name} already staged at {out}. "
              f"Set NYCOPT_ENSEMBLE_KN_FORCE=1 to regenerate.")
        return True
    return False


def _build_forcing_master(design) -> None:
    """Generate the shared CMIP6-forced master for a forcing design (methods §3.2)."""
    from scengen.master_ensemble import MasterEnsembleConfig, generate_master_ensemble

    slug = design.master_slug()
    out = STAGED_ENSEMBLE_DIR / slug
    if _already_staged(out):
        return
    out.mkdir(parents=True, exist_ok=True)
    cfg = MasterEnsembleConfig(
        master_seed=ENSEMBLE_MASTER_SEED,
        n_forcing_profiles=design.n_forcing_profiles,
        realizations_per_profile=design.realizations_per_profile,
        realization_years=design.realization_years,
        output_dir=out,
        mean_frac_csv=ENSEMBLE_FORCING_MEAN_FRAC_CSV,
        variance_axis=ENSEMBLE_FORCING_VARIANCE_AXIS,
        mean_abs_csv=ENSEMBLE_FORCING_MEAN_ABS_CSV if ENSEMBLE_FORCING_VARIANCE_AXIS else None,
        std_csv=ENSEMBLE_FORCING_STD_CSV if ENSEMBLE_FORCING_VARIANCE_AXIS else None,
        bound_pct=ENSEMBLE_FORCING_BOUND_PCT,
        margin=ENSEMBLE_FORCING_MARGIN,
        store_daily=not ENSEMBLE_MASTER_STREAM_ONLY,
        hazard_block_size=ENSEMBLE_MASTER_HAZARD_BLOCK,
        chunk_size=ENSEMBLE_MASTER_CHUNK_SIZE,
    )
    print(f"[gen] Building forcing master {slug} "
          f"(N_forcing={cfg.n_forcing_profiles} x R={cfg.realizations_per_profile} "
          f"= {cfg.n_realizations}, L={cfg.realization_years}yr, seed={cfg.master_seed}, "
          f"store_daily={cfg.store_daily}, chunk_size={cfg.chunk_size}).")
    generate_master_ensemble(cfg)
    print(f"[gen] Done: {slug} -> {out}")


def _build_stationary(design) -> None:
    """Generate a stationary Kirsch-Nowak pool for a direct-KN / resampled design."""
    # Size from the design when it stages a direct-KN ensemble (for the resampled design this is the
    # master POOL, not the per-evaluation draw); else the NYCOPT_ENSEMBLE_KN_* env for a standalone
    # master.
    dims = design.kn_staged_dims()
    if dims is not None:
        n_years, n_reals = dims
        print(f"[gen] Sizing from scenario design '{design.name}' "
              f"(years={n_years}, reals={n_reals}).")
    else:
        n_years, n_reals = ENSEMBLE_KN_YEARS, ENSEMBLE_KN_REALIZATIONS
        print(f"[gen] Sizing from NYCOPT_ENSEMBLE_KN_* env (years={n_years}, reals={n_reals}).")

    slug = kirsch_nowak_slug(n_years, n_reals)
    out = STAGED_ENSEMBLE_DIR / slug
    if _already_staged(out):
        sys.exit(0)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[gen] Building {slug} (years={n_years}, reals={n_reals}, seed={ENSEMBLE_KN_SEED}).")
    generate_kirsch_nowak_ensemble(
        n_years=n_years, n_realizations=n_reals, seed=ENSEMBLE_KN_SEED, output_dir=out
    )
    print(f"[gen] Done: {slug} -> {out}")


def main() -> None:
    design = ACTIVE_SCENARIO_DESIGN
    if design.master_kind == "forcing":
        _build_forcing_master(design)
    else:
        _build_stationary(design)


if __name__ == "__main__":
    main()
