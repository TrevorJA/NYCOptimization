"""select_hazard_filling.py - Workflow step 03: hazard-filling selection.

Stages the search ensemble of the active hazard-filling design by selecting N members
from the design's OWN candidate pool (staged by step 02). The selector is a
deterministic LHS anchor plan snapped to the nearest unused pool member in hazard
space; there is no discrepancy objective and no annealing, so L2-star discrepancy
stays an independent build-QC gate rather than the quantity the selector optimized.

The snap is intrinsic, not an approximation: hazard coordinates (drought deficit
volume, flood peak magnitude, ...) are EMERGENT properties of a realized flow
sequence, so a hazard-space design has nothing to generate *to* and must select from
a finite pool. Input-space designs, by contrast, generate at their design points
(step 02, ``lhs_theta``) and never reach this step.

All K ensemble draws are built in ONE job: the pool is fixed and only the anchor seed
varies (``ScenarioDesign.selector_seed(draw)``), so every draw reuses the single
``hazard_image.npz`` load and the selection itself is cheap. Draw-to-draw spread
therefore measures anchor-placement variance.

The heavy lifting (hazard metrics, selector, coverage diagnostics) lives in the
sibling ``scengen`` package; this script supplies the pywrdrb-side slug grammar and
materializes the selected realizations from the pool's daily chunks. Install the
sibling editable into this venv first::

    pip install -e ../NYCOptimization_scenario_generation

Run after step 02 (which stages the candidate pool + its hazard image)::

    NYCOPT_SCENARIO_DESIGN=hazard_filling_du python scripts/main/select_hazard_filling.py
"""

from __future__ import annotations

import sys

import config
from src.ensembles import materialize_subset, staged_ensemble_dir
from src.scenario_designs import ScenarioDesign


def _select_draw(design: ScenarioDesign, draw: int) -> None:
    """Select and stage one ensemble draw from that draw's candidate pool.

    Each draw has its OWN pool (step 02 regenerates it per draw), so a draw re-rolls
    both sources of construction randomness -- the pool and the LHS anchor plan. That
    is what makes hazard-filling's between-draw variance commensurable with
    ``fixed_probabilistic``'s, which re-rolls its whole sample each draw.

    Args:
        design: The active hazard-filling design.
        draw: Independent ensemble-draw index; keys the pool, the anchor seed, and
            the output slug.
    """
    from scengen.diagnostics import load_hazard_image, save_hazard_image
    from scengen.hazard_filling import select_from_candidate_image

    pool_slug = design.pool_slug(draw)
    haz_path = staged_ensemble_dir(pool_slug) / "hazard_image.npz"
    if not haz_path.exists():
        print(
            f"[hazfill] Candidate hazard image not staged: {haz_path}. Run workflow "
            f"step 02 with NYCOPT_SCENARIO_DESIGN={design.name} "
            f"NYCOPT_ENSEMBLE_DRAW={draw} first."
        )
        sys.exit(1)

    # H is streamed once at pool generation (bounded memory) and stored, so selection
    # reads it directly and never loads the pool's daily timeseries.
    haz = load_hazard_image(haz_path)
    H, candidate_axes, realization_ids = haz["H"], haz["hazard_axes"], haz["realization_ids"]

    n = design.n_realizations
    seed = design.selector_seed(draw)
    out_slug = design.search_ensemble_slug(draw)

    result = select_from_candidate_image(
        H, candidate_axes, n, seed=seed, selector_space=design.selector_space,
    )
    selected_global = [int(realization_ids[r]) for r in result["selected_rows"]]

    # Reads only the selected realizations from each of the pool's daily chunks, so
    # peak memory scales with N, not with the pool.
    materialize_subset(
        pool_slug, selected_global, out_slug,
        extra_meta={
            "design": design.name,
            "draw": draw,
            "source_pool": pool_slug,
            # Inherited from the pool this ensemble was selected from, so the reduced
            # search ensemble carries the same seed domain its realizations came from
            # and config.py's search-vs-test seed-domain guard can see it.
            "seed_domain": design.seed_domain,
            "selector": design.selector,
            "selector_space": design.selector_space,
            "selector_seed": seed,
            "chosen_axes": result["chosen_axes"],
            "candidate_axes": list(candidate_axes),
            "redundancy_clusters": result["screen"]["clusters"],
            "coverage": result["coverage"],
        },
    )

    out_dir = staged_ensemble_dir(out_slug)
    save_hazard_image(
        out_dir / "hazard_image.npz",
        H=H, hazard_axes=candidate_axes, chosen_axes=result["chosen_axes"],
        realization_ids=realization_ids, selected_rows=result["selected_rows"],
    )
    print(f"[hazfill] draw {draw}: pool='{pool_slug}' P={H.shape[0]} seed={seed} "
          f"axes={result['chosen_axes']} coverage={result['coverage']} -> "
          f"'{out_slug}' ({n} realizations)")


def main() -> None:
    """Stage every ensemble draw of the active hazard-filling design."""
    design = config.ACTIVE_SCENARIO_DESIGN
    if design.construction != "hazard_fill":
        print(
            f"[hazfill] Active design '{design.name}' is not a hazard-filling design "
            f"(construction={design.construction!r}); step 03 does not apply. Set "
            f"NYCOPT_SCENARIO_DESIGN to one of the hazard_filling_* designs."
        )
        sys.exit(1)

    print(f"[hazfill] design='{design.name}': selecting n={design.n_realizations} in "
          f"{design.selector_space} space for {design.n_ensemble_draws} draw(s).")
    for draw in range(design.n_ensemble_draws):
        _select_draw(design, draw)


if __name__ == "__main__":
    main()
