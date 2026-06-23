"""subsample_hazard_filling.py - Step 2 (hazard-filling) of the NYCOpt pipeline.

Stages the hazard-filling search ensemble for the active ``hazard_filling``
scenario design: loads the staged master pool (Step 1) and the historical
reference record, computes the hazard image, runs the LHS+nearest-neighbor
space-filling selector, and **writes the final reduced ensemble** (its own
pywrdrb-format HDF5s + ``_meta.json``) that ``ScenarioDesign.resolve_search_spec``
loads directly by slug.

The heavy lifting (hazard metrics, selector, HDF5 slicing) lives in the sibling
``NYCOptimization_scenario_generation`` package (``scengen``); this script is the
NYCOptimization-side orchestration that supplies pywrdrb-loaded inputs and the
slug grammar. Install the sibling editable into this venv first::

    pip install -e ../NYCOptimization_scenario_generation

Run after Step 1 (which stages the master pool ``kn_{L}yr_n{master}``)::

    NYCOPT_SCENARIO_DESIGN=hazard_filling python scripts/main/subsample_hazard_filling.py
"""

from __future__ import annotations

import sys

import config
from src.ensembles import staged_ensemble_dir
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
        load_ensemble_daily_aggregate,
        load_ensemble_monthly_aggregate,
        stage_subset_ensemble,
    )
    from scengen.hazard_metrics import DEFAULT_NYC_INFLOW_NODES

    n = design.n_realizations
    seed = design.subset_seed or 0
    master_slug = design.kn_ensemble_slug()
    final_slug = design.hazard_filling_slug()
    pool_dir = staged_ensemble_dir(master_slug)
    pool_hdf5 = pool_dir / "catchment_inflow_mgd.hdf5"
    if not pool_hdf5.exists():
        print(
            f"[hazfill] Master pool not staged: {pool_hdf5}. Run Step 1 "
            f"(workflow/01) with NYCOPT_SCENARIO_DESIGN=hazard_filling first."
        )
        sys.exit(1)

    # Historical reference: aggregate NYC inflow, monthly mean (dry SSI fit) and
    # daily (POT flood threshold + mean), water-year aligned (BC record starts
    # 1945-10-01).
    hist = load_historical_flows(gage=False, period="full")
    reference_daily = hist.loc[:, list(DEFAULT_NYC_INFLOW_NODES)].sum(axis=1)
    reference_monthly = daily_to_monthly(reference_daily, agg="mean")
    reference_daily = reference_daily.to_numpy(dtype=float)

    # Master pool: per-realization monthly + daily aggregate NYC inflow.
    scenario_monthly, realization_ids = load_ensemble_monthly_aggregate(
        pool_hdf5, DEFAULT_NYC_INFLOW_NODES, agg="mean"
    )
    scenario_daily, _ = load_ensemble_daily_aggregate(pool_hdf5, DEFAULT_NYC_INFLOW_NODES)
    print(
        f"[hazfill] master='{master_slug}' pool={scenario_monthly.shape[0]} "
        f"(months={scenario_monthly.shape[1]}, days={scenario_daily.shape[1]}); "
        f"selecting n={n} (seed={seed})."
    )

    result = build_hazard_filling_subset(
        scenario_monthly, scenario_daily, reference_monthly, reference_daily, n,
        seed=seed, selector_space=design.selector_space,
    )
    selected_ids = [int(realization_ids[r]) for r in result["selected_rows"]]

    # Stage the final reduced ensemble (HDF5s + provenance _meta.json) under its
    # own slug; the optimizer resolves it directly via get_ensemble_spec.
    out_dir = staged_ensemble_dir(final_slug)
    meta = {
        "slug": final_slug,
        "design": design.name,
        "selector": design.selector,
        "selector_space": design.selector_space,
        "source_slug": master_slug,
        "n_realizations": n,
        "realization_years": design.realization_years,
        "subset_seed": seed,
        "source_kind": "synhydro_kn",
        "selected_source_ids": selected_ids,
        "chosen_axes": result["chosen_axes"],
        "candidate_axes": result["candidate_axes"],
        "redundancy_clusters": result["screen"]["clusters"],
        "coverage": result["coverage"],
    }
    stage_subset_ensemble(pool_dir, out_dir, selected_ids, meta=meta)

    # Persist the full candidate hazard image + chosen axes + selected rows so the
    # scengen diagnostics are reproducible and decoupled from pywrdrb (the
    # reference/SSI fit lives here, in NYCOptimization). Written beside the ensemble.
    from scengen.diagnostics import save_hazard_image
    save_hazard_image(
        out_dir / "hazard_image.npz",
        H=result["H_candidates"],
        hazard_axes=result["candidate_axes"],
        chosen_axes=result["chosen_axes"],
        realization_ids=realization_ids,
        selected_rows=result["selected_rows"],
    )
    print(
        f"[hazfill] chosen axes={result['chosen_axes']}\n"
        f"[hazfill] clusters={result['screen']['clusters']}\n"
        f"[hazfill] coverage={result['coverage']}\n"
        f"[hazfill] Staged final ensemble '{final_slug}' ({n} realizations) -> {out_dir}\n"
        f"[hazfill] Wrote hazard_image.npz ({len(result['candidate_axes'])} candidate axes) "
        f"for diagnostics."
    )


if __name__ == "__main__":
    main()
