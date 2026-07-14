"""generate_test_ensemble.py - Workflow step 12: build the held-out test ensemble E_test.

E_test is the measuring stick: every design's Pareto policies are re-simulated on it, and the
cross-design comparison happens there and nowhere else. It is NOT a scenario design — it never
enters search, is never subsampled, and is never a control (see ``src/etest.py``).

Construction, per variant in ``src.etest.E_TEST_VARIANTS``:

    LHS over the FULL (widened) range of the harmonic DU factors  ->  N_theta_test SOWs
      x R_test realizations per SOW (natural variability WITHIN each state of the world)
      x L_test years                                              ->  N_test realizations, chunked

Reuses the one generation path (``src.ensemble_generation.generate_forcing_ensemble``), with:

    theta_sampler="lhs"        space-filling; the i.i.d. requirement belongs to the SEARCH-side
                               candidate pools, which are subsampled. E_test is not.
    population="du_forced"     the DU box is the point of a test ensemble.
    realizations_per_profile   R_test > 1 -- what makes the SOW-unit robustness metric computable.
    compute_hazard_image=True  REQUIRED: scripts/main/scenario_discovery.py hard-fails without
                               E_test's hazard image.
    chunk_size > 0             E_test is the largest ensemble in the study; the chunked re-eval path
                               (src/chunk_reeval.py, workflow step 09) exists for exactly this.

Sizing lives in ``src/etest.py`` (provisional, env-overridable). No CLI value flags: ``--variant``
is an identifier.

    python3 -m scripts.main.generate_test_ensemble                  # the campaign E_test (kn)
    python3 -m scripts.main.generate_test_ensemble --variant hmm    # opt-in sensitivity

``kn`` (Kirsch-Nowak over the wide DU box) is THE test ensemble: the default, the only variant the
campaign requires, and the one ``NYCOPT_REEVAL_ENSEMBLE_PRESET`` should point at. ``hmm`` is a
registered, opt-in, unvalidated generator-structure sensitivity; nothing builds it automatically and
no workflow step depends on it.

E_test needs no ``PRESETS`` entry -- point ``NYCOPT_REEVAL_ENSEMBLE_PRESET`` at the printed slug.
Set ``NYCOPT_ENSEMBLE_FORCE=1`` to overwrite an already-staged slug.
"""

from __future__ import annotations

import argparse
import os

from config import (
    ENSEMBLE_FORCING_MEAN_ABS_CSV,
    ENSEMBLE_FORCING_MEAN_FRAC_CSV,
    ENSEMBLE_FORCING_STD_CSV,
    ENSEMBLE_FORCING_VARIANCE_AXIS,
    ENSEMBLE_MASTER_HAZARD_BLOCK,
    STAGED_ENSEMBLE_DIR,
)
from src.etest import (
    E_TEST_VARIANT,
    ETestVariant,
    assert_staged_etest_contract,
    get_etest_variant,
)

_FORCE = os.environ.get("NYCOPT_ENSEMBLE_FORCE", "").strip().lower() in (
    "1", "true", "yes", "on",
)


def etest_config(variant: ETestVariant):
    """Build the ``ForcingEnsembleConfig`` for one E_test variant.

    The single place the E_test construction contract is expressed as generator arguments. Exposed
    (rather than inlined in :func:`main`) so the contract is unit-testable without generating.

    Args:
        variant: The registered E_test variant.

    Returns:
        A ``scengen.forcing_ensemble.ForcingEnsembleConfig``.
    """
    from scengen.forcing_ensemble import ForcingEnsembleConfig

    return ForcingEnsembleConfig(
        root_seed=variant.seed,
        seed_domain=variant.seed_domain,
        generator=variant.generator,
        population="du_forced",
        theta_sampler="lhs",
        n_forcing_profiles=variant.n_theta,
        realizations_per_profile=variant.realizations_per_theta,
        realization_years=variant.realization_years,
        output_dir=STAGED_ENSEMBLE_DIR / variant.slug,
        mean_frac_csv=ENSEMBLE_FORCING_MEAN_FRAC_CSV,
        variance_axis=ENSEMBLE_FORCING_VARIANCE_AXIS,
        mean_abs_csv=ENSEMBLE_FORCING_MEAN_ABS_CSV if ENSEMBLE_FORCING_VARIANCE_AXIS else None,
        std_csv=ENSEMBLE_FORCING_STD_CSV if ENSEMBLE_FORCING_VARIANCE_AXIS else None,
        bound_pct=variant.bound_pct,
        margin=variant.margin,
        compute_hazard_image=True,
        store_daily=True,
        hazard_block_size=ENSEMBLE_MASTER_HAZARD_BLOCK,
        chunk_size=variant.chunk_size,
    )


def _parse_args() -> argparse.Namespace:
    """Parse the variant identifier (sizing lives in src/etest.py, not on the command line)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--variant", default=E_TEST_VARIANT,
        help="E_test variant identifier (default: NYCOPT_ETEST_VARIANT, else 'kn').",
    )
    return parser.parse_args()


def main() -> None:
    """Generate (or confirm) the staged E_test ensemble for the requested variant."""
    from src.ensemble_generation import generate_forcing_ensemble

    variant = get_etest_variant(_parse_args().variant)
    out = STAGED_ENSEMBLE_DIR / variant.slug

    if out.exists() and any(out.iterdir()) and not _FORCE:
        print(f"[etest] '{variant.slug}' already staged at {out}. "
              f"Set NYCOPT_ENSEMBLE_FORCE=1 to regenerate.")
    else:
        out.mkdir(parents=True, exist_ok=True)
        print(f"[etest] Building '{variant.slug}': generator={variant.generator}, "
              f"N_theta={variant.n_theta} x R={variant.realizations_per_theta} "
              f"= {variant.n_realizations} realizations, L={variant.realization_years}yr, "
              f"LHS box=pct{variant.bound_pct} margin={variant.margin}, seed={variant.seed} "
              f"({variant.seed_domain}), chunk={variant.chunk_size}.")
        generate_forcing_ensemble(etest_config(variant))

    assert_staged_etest_contract(variant.slug)
    print(f"[etest] Done. Set NYCOPT_REEVAL_ENSEMBLE_PRESET={variant.slug} for steps 05/08/09/11.")


if __name__ == "__main__":
    main()
