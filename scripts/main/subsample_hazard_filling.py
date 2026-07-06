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
from src.ensembles import materialize_subset_from_master, staged_ensemble_dir


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
    from scengen.diagnostics import load_hazard_image, save_hazard_image
    from scengen.hazard_filling import select_from_candidate_image

    n = design.n_realizations
    seed = design.subset_seed or 0
    master_slug = design.kn_ensemble_slug()
    final_slug = design.hazard_filling_slug()

    # The candidate hazard image H is streamed once at master generation (bounded memory) and stored
    # as hazard_image.npz — so selection reads H directly and never loads the (possibly huge / chunked)
    # pool timeseries. The selected GLOBAL realizations are then materialized from the daily chunks.
    haz_path = staged_ensemble_dir(master_slug) / "hazard_image.npz"
    if not haz_path.exists():
        print(
            f"[hazfill] Master hazard image not staged: {haz_path}. Run Step 1 "
            f"(workflow/01) with NYCOPT_SCENARIO_DESIGN=hazard_filling first."
        )
        sys.exit(1)
    haz = load_hazard_image(haz_path)
    H, candidate_axes, realization_ids = haz["H"], haz["hazard_axes"], haz["realization_ids"]
    print(f"[hazfill] master='{master_slug}' pool={H.shape[0]} "
          f"({len(candidate_axes)} candidate axes); selecting n={n} (seed={seed}).")

    result = select_from_candidate_image(
        H, candidate_axes, n, seed=seed, selector_space=design.selector_space,
    )
    selected_global = [int(realization_ids[r]) for r in result["selected_rows"]]

    # Materialize the reduced ensemble from the master's daily chunks (reads only the selected
    # realizations per chunk); the optimizer resolves it directly via get_ensemble_spec.
    materialize_subset_from_master(
        master_slug, selected_global, final_slug,
        extra_meta={
            "design": design.name,
            "selector": design.selector,
            "selector_space": design.selector_space,
            "subset_seed": seed,
            "chosen_axes": result["chosen_axes"],
            "candidate_axes": list(candidate_axes),
            "redundancy_clusters": result["screen"]["clusters"],
            "coverage": result["coverage"],
        },
    )

    # Diagnostics hazard image beside the reduced ensemble (decoupled from pywrdrb).
    out_dir = staged_ensemble_dir(final_slug)
    save_hazard_image(
        out_dir / "hazard_image.npz",
        H=H, hazard_axes=candidate_axes, chosen_axes=result["chosen_axes"],
        realization_ids=realization_ids, selected_rows=result["selected_rows"],
    )
    print(
        f"[hazfill] chosen axes={result['chosen_axes']}\n"
        f"[hazfill] clusters={result['screen']['clusters']}\n"
        f"[hazfill] coverage={result['coverage']}\n"
        f"[hazfill] Staged final ensemble '{final_slug}' ({n} realizations) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
