"""subsample_hazard_filling.py - Step 2 (hazard-filling) of the NYCOpt pipeline.

Computes the hazard-filling search ensemble for the active ``hazard_filling``
scenario design: loads the staged master pool (Step 1) and the historical
reference record, computes the hazard image, runs the space-filling selector,
and writes the subset manifest that ``ScenarioDesign.resolve_search_spec`` reads.

The heavy lifting (hazard metrics + selector) lives in the sibling
``NYCOptimization_scenario_generation`` package (``scengen``); this script is the
NYCOptimization-side orchestration that supplies pywrdrb-loaded inputs. Install
the sibling editable into this venv first::

    pip install -e ../NYCOptimization_scenario_generation

Run after Step 1 (which stages the master pool ``kn_{L}yr_n{master}``)::

    NYCOPT_SCENARIO_DESIGN=hazard_filling python scripts/main/subsample_hazard_filling.py
"""

from __future__ import annotations

import sys

import config
from src.ensembles import (
    hazard_filling_subset_filename,
    kirsch_nowak_slug,
    staged_ensemble_dir,
)
from src.load.historical_flows import load_historical_flows


def main() -> None:
    design = config.ACTIVE_SCENARIO_DESIGN
    if design.selection != "hazard_fill":
        print(
            f"[hazfill] Active design '{design.name}' is not a hazard-filling "
            f"design (selection={design.selection!r}). Set "
            f"NYCOPT_SCENARIO_DESIGN=hazard_filling."
        )
        sys.exit(1)

    # scengen is the sibling generation/diagnostics package (copied, not MOEA-FIND).
    from scengen.hazard_filling import (
        build_hazard_filling_subset,
        daily_to_monthly,
        load_ensemble_monthly_aggregate,
    )
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES

    n = design.n_realizations
    seed = design.subset_seed
    master_slug = kirsch_nowak_slug(design.realization_years, design.master_pool_size)
    pool_dir = staged_ensemble_dir(master_slug)
    pool_hdf5 = pool_dir / "catchment_inflow_mgd.hdf5"
    if not pool_hdf5.exists():
        print(
            f"[hazfill] Master pool not staged: {pool_hdf5}. Run Step 1 "
            f"(workflow/01) with NYCOPT_SCENARIO_DESIGN=hazard_filling first."
        )
        sys.exit(1)

    # Historical reference for the SSI fit: aggregate NYC inflow, monthly mean,
    # water-year aligned (the BC reconstruction starts 1945-10-01).
    hist = load_historical_flows(gage=False, period="full")
    reference_monthly = daily_to_monthly(
        hist.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1), agg="mean"
    )

    # Master pool: per-realization monthly aggregate inflow.
    scenario_monthly, realization_ids = load_ensemble_monthly_aggregate(
        pool_hdf5, DEFAULT_NYC_INFLOW_NODES, agg="mean"
    )
    print(
        f"[hazfill] master='{master_slug}' pool={scenario_monthly.shape[0]} "
        f"months={scenario_monthly.shape[1]}; selecting n={n} (seed={seed})."
    )

    result = build_hazard_filling_subset(
        scenario_monthly, reference_monthly, n, seed=seed,
    )
    selected_ids = [int(realization_ids[r]) for r in result["selected_rows"]]
    out_path = pool_dir / hazard_filling_subset_filename(n, seed)

    from scengen.hazard_filling import write_subset_manifest
    write_subset_manifest(
        out_path,
        master_slug=master_slug,
        selected_global_indices=selected_ids,
        hazard_axes=result["hazard_axes"],
        realization_years=design.realization_years,
        seed=seed,
        coverage=result["coverage"],
    )
    print(
        f"[hazfill] axes={result['hazard_axes']} coverage={result['coverage']}\n"
        f"[hazfill] Wrote subset manifest -> {out_path}"
    )


if __name__ == "__main__":
    main()
