"""select_hazard_filling.py - Workflow step 03: hazard-filling selection.

Stages the search ensemble of the active hazard-filling design by selecting N members
from the draw's own candidate pool (staged by step 02): a deterministic LHS anchor
plan snapped to the nearest unused pool member in hazard space (``scengen``).
L2-star discrepancy is reported as an independent build-QC diagnostic.

All draws are staged in one job. Selection reads only the draw's stored
``hazard_image.npz``, never the pool timeseries; the selected realizations are then
materialized from the pool's daily chunks. Requires the sibling package::

    pip install -e ../NYCOptimization_scenario_generation

Run after step 02::

    NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary python scripts/main/select_hazard_filling.py
"""

from __future__ import annotations

import sys

import config
from src.ensembles import materialize_subset, staged_ensemble_dir
from src.scenario_designs import ScenarioDesign


def _select_draw(design: ScenarioDesign, draw: int) -> None:
    """Select and stage one ensemble draw from that draw's own candidate pool.

    Each draw re-rolls both its pool (step 02) and its LHS anchor plan, so draws
    are commensurable with ``monte_carlo`` draws.

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

    # The campaign selection axes (config.HAZARD_SELECTION_AXES) restrict the screen
    # and the snap; the image keeps all candidate axes for reporting.
    result = select_from_candidate_image(
        H, candidate_axes, n, seed=seed, selector_space=design.selector_space,
        selection_axes=config.HAZARD_SELECTION_AXES,
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
            # Campaign axis restriction applied before the screen.
            "selection_axes": list(config.HAZARD_SELECTION_AXES),
            "chosen_axes": result["chosen_axes"],
            "candidate_axes": list(candidate_axes),
            # Axis-screen QC: retained/dropped axes with reasons, Spearman matrix,
            # dedupe threshold.
            "axis_screen": result["screen"],
            "coverage": result["coverage"],
            # Robust bounds (p1/p99) actually used, plus per-axis clipped fractions.
            "normalization": result["normalization"],
        },
    )

    out_dir = staged_ensemble_dir(out_slug)
    save_hazard_image(
        out_dir / "hazard_image.npz",
        H=H, hazard_axes=candidate_axes, chosen_axes=result["chosen_axes"],
        realization_ids=realization_ids, selected_rows=result["selected_rows"],
        reference_start=haz["reference_start"],
    )
    cov = result["coverage"]["geometries"]
    print(f"[hazfill] draw {draw}: pool='{pool_slug}' P={H.shape[0]} seed={seed} "
          f"axes={result['chosen_axes']} -> '{out_slug}' ({n} realizations)")
    print("[hazfill]   L2* vs random null: " + "  ".join(
        f"{g}={c['selected_L2_star']:.4f} (null {c['null_mean']:.4f}+/-{c['null_std']:.4f}, "
        f"pctl {c['percentile']:.0f})" for g, c in cov.items()))


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
