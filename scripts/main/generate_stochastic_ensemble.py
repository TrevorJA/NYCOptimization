"""generate_stochastic_ensemble.py - Workflow step 02: generate a design's realizations.

Every scenario design GENERATES its own realizations, from its OWN namespaced seed
stream (``ScenarioDesign.generation_seed``). Nothing is subsampled from a shared
"master" -- that concept is retired. Only the hazard-filling designs build a
*candidate pool*, and they subsample their OWN pool in workflow step 03.

Dispatch is on ``design.construction``; the five builders differ only in a handful
of generator flags, and that difference IS the per-design construction:

    construction    builder                 N_theta            R      theta   hazard   out slug
    --------------  ----------------------  -----------------  -----  ------  -------  ------------------------
    direct_iid      _build_direct_iid       n_realizations     1      iid     no       search_ensemble_slug(k)
    lhs_theta       _build_lhs_theta        n_theta_profiles   R      LHS     no       search_ensemble_slug(k)
    pool_resample   _build_resample_pool    pool_size          1      iid     no       pool_slug(k)
    hazard_fill     _build_candidate_pool   pool_size          1      iid     YES      pool_slug(k)
    stationary_kn   _build_scaling_kn       (direct Kirsch-Nowak, supplemental)        search_ensemble_slug()
    preset          (no-op: ``historic`` stages nothing)

``lhs_theta`` is the published ``input_stratified`` recipe: a Latin hypercube over
the generator's forcing parameters, with realizations GENERATED at each design
point. It is not a subsample, and it never snaps to a pool.

EVERY design stages one artifact PER DRAW, pools included: a draw is the design's
construction re-run from scratch with a fresh seed, and generating the pool IS part
of a pool-owning design's construction. Pinning a pool across draws would leave a
hazard-filling draw varying only its LHS anchor plan while a ``fixed_probabilistic``
draw re-rolls its whole sample -- the between-draw variances would not be
commensurable, and hazard-filling would look stable by construction rather than as a
finding. Step-02 cost therefore scales with K for every design. Within a draw the two
DU hazard designs share one pool slug, so the second is a no-op.

All configuration comes from ``config.py`` + the scenario-design registry -- no CLI
value flags. ``--draw`` / ``--all-draws`` are identifiers, not settings.

    sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/ffmp_obj7_hazfill_pilot.env \\
           --array=0-9 workflow/02_generate_ensemble.sh

Set ``NYCOPT_ENSEMBLE_FORCE=1`` to overwrite an already-staged slug.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from config import (
    ACTIVE_SCENARIO_DESIGN,
    ENSEMBLE_FORCING_BOUND_PCT,
    ENSEMBLE_FORCING_MARGIN,
    ENSEMBLE_FORCING_MEAN_ABS_CSV,
    ENSEMBLE_FORCING_MEAN_FRAC_CSV,
    ENSEMBLE_FORCING_STD_CSV,
    ENSEMBLE_FORCING_VARIANCE_AXIS,
    ENSEMBLE_MASTER_CHUNK_SIZE,
    ENSEMBLE_MASTER_HAZARD_BLOCK,
    ENSEMBLE_MASTER_STREAM_ONLY,
    STAGED_ENSEMBLE_DIR,
)
from src.ensemble_generation import generate_kirsch_nowak_ensemble
from src.scenario_designs import ScenarioDesign

# Overwrite an already-staged slug. Guards every construction kind.
_FORCE = os.environ.get("NYCOPT_ENSEMBLE_FORCE", "").strip().lower() in (
    "1", "true", "yes", "on",
)


def _already_staged(out: Path) -> bool:
    """Return True if ``out`` holds a staged ensemble and regeneration was not forced.

    This is the idempotency contract that lets the two DU hazard-filling designs
    share one candidate pool: whichever runs second finds the pool staged and
    returns without regenerating it.

    Args:
        out: Staged-ensemble directory for the slug being built.

    Returns:
        True when the slug is already staged and ``NYCOPT_ENSEMBLE_FORCE`` is unset.
    """
    if out.exists() and any(out.iterdir()) and not _FORCE:
        print(f"[gen] '{out.name}' already staged at {out}. "
              f"Set NYCOPT_ENSEMBLE_FORCE=1 to regenerate.")
        return True
    return False


def _generate_forcing(
    design: ScenarioDesign,
    *,
    slug: str,
    root_seed: int,
    n_forcing_profiles: int,
    realizations_per_profile: int,
    population: str,
    theta_sampler: str,
    compute_hazard_image: bool,
) -> None:
    """Run the shared forcing->realization generator for one staged slug.

    The single entry point behind four of the five builders. They differ only in the
    arguments passed here, so the generator stays one code path and the designs stay
    exactly controlled against each other.

    Args:
        design: The active scenario design (supplies L and provenance).
        slug: Staged-ensemble slug to write.
        root_seed: Namespaced generator seed from the design's seed domain.
        n_forcing_profiles: Number of theta profiles to draw (N_theta).
        realizations_per_profile: Realizations generated per profile (R).
        population: ``"stationary"`` (no climate perturbation) or ``"du_forced"``.
        theta_sampler: ``"iid"`` or ``"lhs"``. LHS only for ``input_stratified``.
        compute_hazard_image: Stream the hazard image while generating. True only
            for a hazard-filling candidate pool -- the SSI-6 fit and POT pass are
            pure waste otherwise.
    """
    from scengen.forcing_ensemble import ForcingEnsembleConfig, generate_forcing_ensemble

    out = STAGED_ENSEMBLE_DIR / slug
    if _already_staged(out):
        return
    out.mkdir(parents=True, exist_ok=True)

    cfg = ForcingEnsembleConfig(
        root_seed=root_seed,
        # Recorded in the staged _meta.json. config.py compares the SEARCH ensemble's
        # domain against the TEST ensemble's and HARD-ERRORS on a match, so E_test can
        # never be drawn from a search seed stream (selection bias, Bonham et al. 2024).
        # The guard is only live because the domain is written here.
        seed_domain=design.seed_domain,
        population=population,
        theta_sampler=theta_sampler,
        compute_hazard_image=compute_hazard_image,
        n_forcing_profiles=n_forcing_profiles,
        realizations_per_profile=realizations_per_profile,
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
    print(f"[gen] Building '{slug}' for design '{design.name}': "
          f"population={population}, theta={theta_sampler}, "
          f"N_theta={n_forcing_profiles} x R={realizations_per_profile} "
          f"= {n_forcing_profiles * realizations_per_profile}, "
          f"L={design.realization_years}yr, seed={root_seed}, "
          f"hazard_image={compute_hazard_image}, store_daily={cfg.store_daily}.")
    generate_forcing_ensemble(cfg)
    print(f"[gen] Done: {slug} -> {out}")


def _build_direct_iid(design: ScenarioDesign, draw: int) -> None:
    """Generate N i.i.d. realizations for one draw of ``fixed_probabilistic``.

    N independent theta draws x 1 realization each. Under the stationary population
    theta is vacuous, so this is N i.i.d. Kirsch-Nowak records -- the exact
    statistical control for ``hazard_filling_stationary``.

    Args:
        design: The active design (``construction == "direct_iid"``).
        draw: Independent ensemble-draw index; keys the output slug and the seed.
    """
    _generate_forcing(
        design,
        slug=design.search_ensemble_slug(draw),
        root_seed=design.generation_seed(draw),
        n_forcing_profiles=design.n_realizations,
        realizations_per_profile=1,
        population=design.population,
        theta_sampler="iid",
        compute_hazard_image=False,
    )


def _build_lhs_theta(design: ScenarioDesign, draw: int) -> None:
    """Generate one draw of ``input_stratified``: LHS over theta, realizations at each point.

    The published input-space recipe (Quinn et al. 2020; Bartholomew & Kwakkel 2020):
    a Latin hypercube over the harmonic forcing parameters, with R realizations
    GENERATED under each design point. Forcing parameters are a knob on the generator,
    so there is nothing to snap to and nothing to subsample.

    Args:
        design: The active design (``construction == "lhs_theta"``).
        draw: Independent ensemble-draw index; keys the output slug and the seed.
    """
    _generate_forcing(
        design,
        slug=design.search_ensemble_slug(draw),
        root_seed=design.generation_seed(draw),
        n_forcing_profiles=design.n_theta_profiles,
        realizations_per_profile=design.realizations_per_profile,
        population="du_forced",
        theta_sampler="lhs",
        compute_hazard_image=False,
    )


def _build_resample_pool(design: ScenarioDesign, draw: int) -> None:
    """Generate the stationary resampling pool for ``resampled_probabilistic``.

    The simulation layer redraws N indices from this pool at every function
    evaluation, so the pool itself is the staged artifact. The pool is i.i.d. (never
    LHS) -- a uniform random size-N subset of an i.i.d. pool has exactly the law of N
    fresh i.i.d. draws.

    Draw k gets a FRESH pool: the pool is this design's ensemble, so re-running its
    construction from scratch means re-generating it. Sharing one pool across draws
    would leave the per-evaluation resampling RNG as the only between-draw variation,
    which is within-run noise, not composition variance.

    Args:
        design: The active design (``construction == "pool_resample"``).
        draw: Ensemble-draw index.
    """
    _generate_forcing(
        design,
        slug=design.pool_slug(draw),
        root_seed=design.generation_seed(draw),
        n_forcing_profiles=design.pool_size,
        realizations_per_profile=1,
        population="stationary",
        theta_sampler="iid",
        compute_hazard_image=False,
    )


def _build_candidate_pool(design: ScenarioDesign, draw: int) -> None:
    """Generate a hazard-filling design's OWN candidate pool + its hazard image.

    The pool is sampled i.i.d. with one realization per theta, so its hazard image is
    the honest empirical hazard manifold rather than an artifact of a design imposed
    on theta. Step 03 selects N members from it by LHS anchors + nearest-neighbor snap.

    Draw k gets a FRESH pool, and that is load-bearing for the replication analysis.
    Generating the pool IS part of a hazard-filling design's construction, so a draw
    must re-roll it. If the pool were pinned across draws, a hazard-filling draw would
    vary only the LHS anchor plan while a ``fixed_probabilistic`` draw re-rolls its
    entire sample -- the two between-draw variances would not be commensurable, and
    hazard-filling would look more stable BY CONSTRUCTION rather than as a finding.

    Within a draw, ``hazard_filling_du`` and ``hazard_filling_absolute`` resolve to the
    SAME pool slug (they differ only in selector space), so it is generated once and the
    second design is a no-op via the ``_already_staged`` guard.

    Args:
        design: The active design (``construction == "hazard_fill"``).
        draw: Ensemble-draw index.
    """
    _generate_forcing(
        design,
        slug=design.pool_slug(draw),
        root_seed=design.generation_seed(draw),
        n_forcing_profiles=design.pool_size,
        realizations_per_profile=1,
        population=design.population,
        theta_sampler="iid",
        compute_hazard_image=True,
    )


def _build_scaling_kn(design: ScenarioDesign) -> None:
    """Generate the direct Kirsch-Nowak stand-in ensemble for the Anvil scaling runs.

    Supplemental only: times the trimmed-model ensemble-evaluation path, whose per-eval
    cost is set by the ensemble SHAPE (N x L), not by scenario content. Sized from the
    design (``NYCOPT_SCALING_KN_YEARS`` / ``NYCOPT_SCALING_KN_REALS``).

    Args:
        design: The active design (``construction == "stationary_kn"``).
    """
    slug = design.search_ensemble_slug()
    out = STAGED_ENSEMBLE_DIR / slug
    if _already_staged(out):
        return
    out.mkdir(parents=True, exist_ok=True)
    seed = design.generation_seed()
    print(f"[gen] Building '{slug}' for design '{design.name}': "
          f"years={design.realization_years}, reals={design.n_realizations}, seed={seed}.")
    generate_kirsch_nowak_ensemble(
        n_years=design.realization_years,
        n_realizations=design.n_realizations,
        seed=seed,
        output_dir=out,
    )
    print(f"[gen] Done: {slug} -> {out}")


_BUILDERS = {
    "direct_iid": _build_direct_iid,
    "lhs_theta": _build_lhs_theta,
    "pool_resample": _build_resample_pool,
    "hazard_fill": _build_candidate_pool,
}


def _parse_args() -> argparse.Namespace:
    """Parse the two draw identifiers (no settings; those live in the registries)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--draw", type=int, default=int(os.environ.get("NYCOPT_ENSEMBLE_DRAW", "0")),
        help="Independent ensemble-draw index (identifier). Defaults to "
             "NYCOPT_ENSEMBLE_DRAW, else 0.",
    )
    parser.add_argument(
        "--all-draws", action="store_true",
        help="Build every draw of the active design serially in this one process, "
             "instead of the single --draw.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the active scenario design's realizations for the requested draw(s)."""
    args = _parse_args()
    design = ACTIVE_SCENARIO_DESIGN

    if design.construction == "preset":
        print(f"[gen] Design '{design.name}' resolves to the static preset "
              f"'{design.ensemble_preset}'; step 02 is a no-op.")
        return

    if design.construction == "stationary_kn":
        _build_scaling_kn(design)
        return

    build = _BUILDERS.get(design.construction)
    if build is None:
        raise ValueError(
            f"Design '{design.name}' has construction '{design.construction}', which "
            f"step 02 cannot build. Known: {sorted(_BUILDERS) + ['stationary_kn', 'preset']}."
        )

    draws = range(design.n_ensemble_draws) if args.all_draws else [args.draw]
    for draw in draws:
        if not 0 <= draw < design.n_ensemble_draws:
            raise ValueError(
                f"draw {draw} out of range for design '{design.name}' "
                f"(K={design.n_ensemble_draws})."
            )
        build(design, draw)


if __name__ == "__main__":
    main()
