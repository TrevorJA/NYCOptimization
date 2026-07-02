"""Generate the input-spanning master ensembles for the input-vs-hazard coverage diagnostic.

Produces the two richness-ladder masters used by
``scripts/supplemental/diagnose_input_vs_hazard_coverage.py``:

  - ``kn_inputlhs_meanonly_5yr_n{N}`` -- impoverished baseline: monthly *mean* perturbed only,
    ``c_j = 1`` (absolute SD preserved). theta = 12-month mean factor ``a_j``.
  - ``kn_inputlhs_meanvar_5yr_n{N}``  -- enriched: independent CMIP6-derived CV-change axis,
    ``c_j = a_j * v_j``. theta = 24-dim ``(a_j, v_j)``.

Each master draws ``N`` LHS forcing profiles over the CMIP6 change-factor envelope and generates one
5-yr realization per profile (see ``src.ensemble_generation.generate_input_spanning_master``).

Env vars:
    NYCOPT_NPROFILES   master size (default 1000; the diagnostic uses N up to 128 subsets)
    NYCOPT_SMOKE=1     tiny run (n_profiles=24) to validate the path end to end
    NYCOPT_RUNG        'meanonly' | 'meanvar' | 'both' (default 'both')
    NYCOPT_FORCE=1     regenerate even if already staged

Run from the repo root::

    PYTHONPATH=$(pwd -W) venv/Scripts/python.exe scripts/supplemental/generate_input_spanning_master.py
"""

from __future__ import annotations

import os
from pathlib import Path

import config
from src.ensemble_generation import generate_input_spanning_master
from src.ensembles import staged_ensemble_dir

REALIZATION_YEARS = 5
SEED = 0

_CMIP6 = Path(config.PROJECT_DIR).parent / "CMIP6_multimodel_streamflow" / "stats"
MEAN_FRAC_CSV = (
    _CMIP6 / "diff_relative_to_dataset_baseline"
    / "nyc_inflow_monthly_mean_frac_by_dataset_ssp_and_period.csv"
)
MEAN_ABS_CSV = _CMIP6 / "datasets_nyc_inflow_monthly_means.csv"
STD_CSV = _CMIP6 / "datasets_nyc_inflow_monthly_stds.csv"


def _run_rung(*, variance_axis: bool, n_profiles: int, force: bool) -> None:
    rung = "meanvar" if variance_axis else "meanonly"
    slug = f"kn_inputlhs_{rung}_{REALIZATION_YEARS}yr_n{n_profiles}"
    out_dir = staged_ensemble_dir(slug)
    if (out_dir / "_meta.json").exists() and not force:
        print(f"[master] '{slug}' already staged -> {out_dir} (set NYCOPT_FORCE=1 to regen).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[master] Generating '{slug}' (variance_axis={variance_axis}, n={n_profiles})...")
    meta = generate_input_spanning_master(
        n_profiles=n_profiles,
        n_years=REALIZATION_YEARS,
        seed=SEED,
        output_dir=out_dir,
        mean_frac_csv=MEAN_FRAC_CSV,
        variance_axis=variance_axis,
        mean_abs_csv=MEAN_ABS_CSV if variance_axis else None,
        std_csv=STD_CSV if variance_axis else None,
    )
    print(f"[master] Done '{slug}': {meta['n_realizations']} realizations -> {out_dir}")


def main() -> None:
    for csv in (MEAN_FRAC_CSV, MEAN_ABS_CSV, STD_CSV):
        if not csv.exists():
            raise FileNotFoundError(f"CMIP6 table not found: {csv}")

    smoke = os.environ.get("NYCOPT_SMOKE") == "1"
    n_profiles = 24 if smoke else int(os.environ.get("NYCOPT_NPROFILES", "1000"))
    force = os.environ.get("NYCOPT_FORCE") == "1"
    rung = os.environ.get("NYCOPT_RUNG", "both").lower()

    rungs = {"meanonly": [False], "meanvar": [True], "both": [False, True]}[rung]
    for variance_axis in rungs:
        _run_rung(variance_axis=variance_axis, n_profiles=n_profiles, force=force)


if __name__ == "__main__":
    main()
