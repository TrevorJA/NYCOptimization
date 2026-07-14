"""
scenario_designs.py - Registry of scenario designs for the MOEA evaluation ensemble.

A *scenario design* is the construction recipe for the streamflow ensemble used
during many-objective search. Comparing scenario designs is the methodological
contribution of the study (see ``docs/notes/methods/experimental_design.md`` and
``docs/notes/methods/scenario_design_methods.md``). This module maps a design
name (e.g. ``"historic"``, ``"hazard_filling_du"``) to an immutable
``ScenarioDesign`` describing how its search ensemble is built. The design name
is the top level of the output tree: ``outputs/{design}/{moea_slug}/...``.

Each design is constructed by its OWN published recipe, from its OWN seed
stream. No design is subsampled from a shared pool -- no published study builds
its search ensemble that way, and doing so would misdescribe every method the
comparison claims to represent.

Two populations
---------------
A *population* is the law a design's realizations are drawn from. Population and
selection rule are independent choices; the comparison holds one fixed and
varies the other.

* ``stationary``  -- Kirsch-Nowak fitted to the historic record; no climate
  perturbation. The flow model of the prevailing water-supply optimization
  literature. Hazard variation comes from natural variability alone.
* ``du_forced``   -- forcing parameters theta sampled from the CMIP6 harmonic
  hypercube. Hazard variation comes from natural variability x forcing
  uncertainty.

Hazard-filling runs in BOTH, which is what makes every contrast exactly
controlled:

* ``fixed_probabilistic`` -> ``hazard_filling_stationary``: same generator, same
  population law, same N, same L. Only the SELECTION RULE differs.
* ``input_stratified`` -> ``hazard_filling_du``: same forcing space, same N,
  same L. Only the SELECTION SPACE differs (theta vs hazard). The central claim.
* ``hazard_filling_stationary`` -> ``hazard_filling_du``: what the DU forcing
  space adds.

A stationary-only pool would leave ``input_stratified`` with no input space to
stratify; a DU-only pool would leave hazard-filling with no exact
random-selection control.

Why only hazard-filling needs a pool
------------------------------------
Hazard coordinates are EMERGENT properties of a realized flow sequence -- no
generator can be asked to produce a realization at a prescribed drought
severity. Forcing parameters theta, by contrast, ARE a knob. So input-space
designs GENERATE TO their design points (LHS alone, nothing to snap to), while
hazard-space designs must SELECT FROM a finite candidate pool (LHS anchors +
nearest-neighbor snap). The snap is intrinsic to hazard-space design, not an
approximation of something better.

The i.i.d. condition (load-bearing)
-----------------------------------
Candidate pools are sampled i.i.d., never by LHS. A uniform random size-N subset
of an i.i.d. pool has exactly the joint law of N fresh i.i.d. draws -- which is
what makes ``fixed_probabilistic`` the EXACT statistical control for
``hazard_filling_stationary``. A random subset of an LHS design is NOT i.i.d.,
so an LHS-sampled pool would silently void the control. ``input_stratified`` is
the only design that uses LHS, and it uses it to GENERATE, never to build a
pool. Enforced by ``assert_iid_pools()`` (called at import) and by
``tests/test_design_registries.py``.

For the same reason a DU candidate pool must carry ``realizations_per_profile ==
1``: realizations sharing a theta are not independent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from scengen.seeds import design_seed

from src.ensembles import (
    EnsembleSpec,
    as_resampling_pool,
    get_ensemble_spec,
    kirsch_nowak_slug,
)

# Scenario length L (years) of each realization in the matched comparison -- the
# SINGLE project-wide source of truth. Re-exported as ``config.SCENARIO_YEARS``.
# L = 10 exceeds the 1960s DRB drought of record (~4-5 yr) plus onset and
# recovery, so a design-basis event fits inside a window, and it keeps N large
# enough to fill an m-dimensional hazard space at a fixed per-evaluation budget
# (methods §6). Changing L invalidates every staged L-conditional artifact.
SCENARIO_YEARS: int = int(os.environ.get("NYCOPT_SCENARIO_YEARS", "10"))

# Ensemble size N, common to every matched design (methods §6). N x L = 1000
# scenario-years per evaluation, at equal NFE, so per-evaluation cost, warm-up
# and wall-clock are identical across designs. A common (N, L) is REQUIRED: if L
# differed across designs the selection rule would be confounded with record
# length.
SEARCH_ENSEMBLE_N: int = int(os.environ.get("NYCOPT_SEARCH_N", "100"))

# Candidate-pool cardinality P for the hazard-filling designs, and the
# resampling-pool cardinality for ``resampled_probabilistic``. P >> N so that
# LHS anchors snap to near neighbours; the snap-distance distribution is a
# reported build diagnostic. PROVISIONAL test-scale defaults -- production
# targets 1e5-1e6 (methods §6, open parameters).
CANDIDATE_POOL_SIZE: int = int(os.environ.get("NYCOPT_CANDIDATE_POOL_N", "2000"))
RESAMPLE_POOL_SIZE: int = int(os.environ.get("NYCOPT_RESAMPLE_POOL_N", "1000"))

# input_stratified allocates N = N_theta x R. R = 1 is the literal published
# construction (Quinn et al. 2020; Bartholomew & Kwakkel 2020); R > 1 separates
# forcing uncertainty from natural variability within a forcing (Quinn et al.
# 2018), at the cost of input coverage at fixed N. The split is an open
# parameter to set before generation (methods §6).
INPUT_STRAT_N_THETA: int = int(os.environ.get("NYCOPT_INPUT_STRAT_N_THETA", "20"))
INPUT_STRAT_R: int = int(os.environ.get("NYCOPT_INPUT_STRAT_R", "5"))

# Independent ensemble draws K per design (the unit of analysis). Draws are now
# independent GENERATIONS, not re-indexings of shared data, so K must be fixed
# before workflow step 02 runs. Target 10 (methods §6).
N_ENSEMBLE_DRAWS: int = int(os.environ.get("NYCOPT_N_ENSEMBLE_DRAWS", "10"))

# Root seed for the whole campaign. Every generated artifact derives its seed as
# ``design_seed(SEED_ROOT, seed_domain, draw)``, so seed domains are disjoint by
# construction and one knob re-rolls everything.
SEED_ROOT: int = int(os.environ.get("NYCOPT_SEED_ROOT", "20260713"))

# Stationary-KN sizing for the Anvil scaling experiment's stand-in ensemble.
# Timing-only; scenario content does not affect per-eval cost.
_SCALING_KN_YEARS = int(os.environ.get("NYCOPT_SCALING_KN_YEARS", "20"))
_SCALING_KN_REALS = int(os.environ.get("NYCOPT_SCALING_KN_REALS", "20"))


###############################################################################
# ScenarioDesign
###############################################################################

@dataclass(frozen=True)
class ScenarioDesign:
    """Immutable specification of a scenario design for MOEA evaluation.

    Attributes:
        name: Key used to select this design (``NYCOPT_SCENARIO_DESIGN``) and as
            the top-level output directory.
        family: Taxonomy label tying the design to a literature family.
        description: One-line human-readable summary.
        population: The law realizations are drawn from -- ``"historic"``,
            ``"stationary"``, or ``"du_forced"``.
        construction: How the search ensemble is built. Dispatch key for
            :meth:`resolve_search_spec` and for the step-02 generation script.

            * ``"preset"``        -- a static ``src/ensembles.py`` preset.
            * ``"direct_iid"``    -- generate N x L realizations i.i.d.
            * ``"lhs_theta"``     -- LHS over the harmonic forcing parameters,
              generating R realizations per design point. LHS ALONE; no snap.
            * ``"pool_resample"`` -- generate an own pool; the simulation layer
              redraws N indices at every function evaluation.
            * ``"hazard_fill"``   -- generate an own candidate pool, then select
              N members by LHS anchors + nearest-neighbor snap in hazard space.
            * ``"stationary_kn"`` -- direct Kirsch-Nowak, sized from the design
              (supplemental scaling stand-in only).
        theta_sampler: ``"iid"`` or ``"lhs"``. MUST be ``"iid"`` for every design
            except ``lhs_theta`` -- see the module docstring. Vacuous for
            stationary designs (there is no theta to sample), but kept ``"iid"``
            so the invariant reads uniformly.
        resample_per_eval: ``True`` only for ``pool_resample``. A simulation-layer
            flag, not a dispatch key.
        ensemble_preset: Static preset name, for ``construction == "preset"``.
        n_realizations: Search-ensemble size N.
        realization_years: Realization length L, in years.
        realizations_per_profile: R -- realizations generated per forcing profile.
            Used by ``lhs_theta`` (N = n_theta_profiles x R). MUST be 1 for a DU
            candidate pool, or the pool is not i.i.d.
        n_theta_profiles: N_theta, for ``lhs_theta`` only.
        pool_size: P -- cardinality of the design's OWN pool. Only
            ``pool_resample`` and ``hazard_fill`` have one.
        n_ensemble_draws: K independent constructions (the unit of analysis).
        seed_domain: Namespace for this design's generator seed. Disjointness
            across domains is what keeps designs from sharing realizations now
            that each one generates rather than selecting indices from shared
            data.
        selector: For ``hazard_fill``: the selector name. The wired selector is
            LHS + nearest-neighbor (``"lhs_nn"``). Deterministic given its anchor
            seed; K draws vary the anchor plan and therefore measure
            anchor-placement variance. There is no simulated annealing.
        selector_space: For ``hazard_fill``: the space the anchors fill.
            ``"cdf"`` = empirical-CDF/rank space (the campaign designs);
            ``"abs"`` = absolute magnitude space (non-campaign sensitivity,
            deliberately over-representing hazard extremes).
        needs_hazard_image: Whether step 02 must stream a hazard image while
            generating this design's pool. True only for ``hazard_fill``; the
            SSI-6 fit and POT pass are pure waste otherwise.
        campaign: Whether this design is part of the manuscript comparison.
        notes: Precedent and open questions.
    """

    name: str
    family: str
    description: str
    population: str
    construction: str
    theta_sampler: str = "iid"
    resample_per_eval: bool = False
    ensemble_preset: str | None = None
    n_realizations: int | None = None
    realization_years: int | None = None
    realizations_per_profile: int = 1
    n_theta_profiles: int | None = None
    pool_size: int | None = None
    n_ensemble_draws: int = 1
    seed_domain: str | None = None
    selector: str = "lhs_nn"
    selector_space: str = "cdf"
    needs_hazard_image: bool = False
    campaign: bool = True
    notes: str = ""

    # -- slugs ---------------------------------------------------------------

    def pool_slug(self, draw: int = 0) -> str | None:
        """Return the slug of the design's OWN pool for ``draw``, or ``None``.

        Only ``pool_resample`` and ``hazard_fill`` own a pool. This is never
        called a "master" -- it belongs to the design.

        **The pool is re-drawn per draw, and that is load-bearing.** A draw is
        the design's construction re-run from scratch with a fresh seed, and
        generating the pool IS part of a hazard-filling design's construction.
        If the pool were pinned across draws, a hazard-filling draw would vary
        only the LHS anchor plan while a ``fixed_probabilistic`` draw re-rolls
        its entire sample -- so the two between-draw variances would not be
        commensurable, and hazard-filling would look more stable *by
        construction* rather than as a finding. The K draws must re-roll
        everything that is random about building the ensemble.

        The two DU hazard-filling designs share a pool *at the same draw* (same
        population, same axes, same N; they differ only in selector space), so
        it is generated once per draw.

        Args:
            draw: Independent ensemble-draw index.
        """
        if self.pool_size is None or self.realization_years is None:
            return None
        if self.construction == "pool_resample":
            stem = "respool"
        elif self.construction == "hazard_fill":
            stem = "statpool" if self.population == "stationary" else "dupool"
        else:
            return None
        return f"{stem}_{self.realization_years}yr_n{self.pool_size}_d{draw}"

    def search_ensemble_slug(self, draw: int = 0) -> str | None:
        """Return the staged directory the optimizer loads for ``draw``.

        One slug grammar for every design that stages a fixed ensemble. The draw
        index -- not the seed -- keys the slug: the draw is the human-facing
        replication unit, and the seed is recorded as provenance in
        ``_meta.json``.
        """
        if self.construction == "stationary_kn":
            return kirsch_nowak_slug(self.realization_years, self.n_realizations)
        if self.realization_years is None or self.n_realizations is None:
            return None
        stem = {
            "direct_iid": "fixprob",
            "lhs_theta": "inputstrat",
            "hazard_fill": (
                f"hazfill_{'stat' if self.population == 'stationary' else 'du'}"
                f"{'' if self.selector_space == 'cdf' else '_abs'}"
            ),
        }.get(self.construction)
        if stem is None:
            return None
        return f"{stem}_{self.realization_years}yr_n{self.n_realizations}_d{draw}"

    # -- seeds ---------------------------------------------------------------

    def generation_seed(self, draw: int = 0) -> int:
        """Seed for generating this design's realizations (or its pool)."""
        return design_seed(SEED_ROOT, self.seed_domain, draw)

    def selector_seed(self, draw: int = 0) -> int:
        """Seed for the hazard-filling LHS anchor plan of ``draw``.

        Deliberately NOT split by ``selector_space``: both hazard arms get the
        same anchor plan, so the only difference between the CDF and absolute
        designs is the normalization geometry the anchors snap into -- the
        cleanest possible paired comparison.
        """
        domain = (
            "hazard_select_stat"
            if self.population == "stationary"
            else "hazard_select_du"
        )
        return design_seed(SEED_ROOT, domain, draw)

    # -- resolution ----------------------------------------------------------

    def resolve_search_spec(self, draw: int = 0) -> EnsembleSpec:
        """Resolve this design's search ensemble to an ``EnsembleSpec``.

        A pure lookup. Every design's ensemble is constructed by workflow step 02
        (and step 03 for hazard-filling); nothing is generated here, so importing
        ``config`` performs no RNG draws and no bulk I/O.

        Args:
            draw: Index of the independent ensemble draw. A draw is the design's
                construction re-run from scratch with a fresh seed.

        Returns:
            The ``EnsembleSpec`` for the search ensemble. For ``pool_resample``
            this is the design's own pool marked ``resample_per_eval=True``; the
            simulation layer redraws ``n_realizations`` indices from it at every
            evaluation.

        Raises:
            ValueError: For ``draw != 0`` on a design with no fixed ensemble to
                replicate.
            NotImplementedError: When the design's staged data is missing, naming
                the exact workflow step that builds it.
        """
        if self.construction == "preset":
            self._reject_nonzero_draw(draw)
            return get_ensemble_spec(self.ensemble_preset)

        if self.construction == "stationary_kn":
            self._reject_nonzero_draw(draw)
            return self._staged_or_raise(self.search_ensemble_slug(), step="02")

        if self.construction == "pool_resample":
            # The search "ensemble" IS the pool -- the simulation layer redraws
            # n_realizations indices from it at every function evaluation. Draw k
            # is a fresh pool, so between-draw variance is real composition
            # variance rather than per-eval RNG noise.
            pool = self._staged_or_raise(self.pool_slug(draw), step="02")
            return as_resampling_pool(pool, self.n_realizations)

        step = "03" if self.construction == "hazard_fill" else "02"
        return self._staged_or_raise(self.search_ensemble_slug(draw), step=step)

    def _reject_nonzero_draw(self, draw: int) -> None:
        """Raise if ``draw != 0`` on a design with no ensemble to replicate."""
        if draw != 0:
            raise ValueError(
                f"Scenario design '{self.name}' has no fixed ensemble draw to "
                f"replicate (got draw={draw}). Replicate over MOEA seeds instead."
            )

    def _staged_or_raise(self, slug: str | None, *, step: str) -> EnsembleSpec:
        """Resolve a staged ensemble by slug, or explain how to build it."""
        if slug is None:
            raise NotImplementedError(
                f"Scenario design '{self.name}' is missing sizing "
                f"(n_realizations / realization_years / pool_size)."
            )
        try:
            return get_ensemble_spec(slug)
        except KeyError:
            raise NotImplementedError(
                f"Scenario design '{self.name}': ensemble '{slug}' is not staged "
                f"yet. Build it with workflow step {step} "
                f"(NYCOPT_SCENARIO_DESIGN={self.name}"
                + (
                    f", NYCOPT_ENSEMBLE_DRAW=k for k in 0..{self.n_ensemble_draws - 1}"
                    if self.n_ensemble_draws > 1
                    else ""
                )
                + ")."
            ) from None


###############################################################################
# Registry
###############################################################################

SCENARIO_DESIGNS: dict[str, ScenarioDesign] = {
    # ---------------- stationary population ----------------
    "historic": ScenarioDesign(
        name="historic",
        family="historical_record",
        description="The observed record, simulated as one continuous trace.",
        population="historic",
        construction="preset",
        ensemble_preset="historic_single",
        n_realizations=1,
        n_ensemble_draws=1,
        notes="Precedent: Giuliani et al. (2016); Herman et al. (2020). Reference "
              "for prevailing applied practice; cannot be size-matched, so it is "
              "reported rather than entered into the matched contrasts. K=1: "
              "composition variance is zero by construction.",
    ),
    "fixed_probabilistic": ScenarioDesign(
        name="fixed_probabilistic",
        family="fixed_probabilistic_ensemble",
        description="N x L realizations generated i.i.d. from the stationary "
                    "Kirsch-Nowak generator; frozen across the search.",
        population="stationary",
        construction="direct_iid",
        theta_sampler="iid",
        n_realizations=SEARCH_ENSEMBLE_N,
        realization_years=SCENARIO_YEARS,
        n_ensemble_draws=N_ENSEMBLE_DRAWS,
        seed_domain="fixed",
        notes="Precedent: Quinn et al. (2017); Zatarain Salazar et al. (2017). The "
              "reference against which designed selection is judged -- and, because "
              "a uniform random size-N subset of an i.i.d. pool has exactly the law "
              "of N i.i.d. draws, the EXACT statistical control for "
              "hazard_filling_stationary: the two differ only in the selection rule "
              "applied to the same population law.",
    ),
    "resampled_probabilistic": ScenarioDesign(
        name="resampled_probabilistic",
        family="resampled_probabilistic_ensemble",
        description="Own stationary pool; N realizations redrawn at every "
                    "function evaluation.",
        population="stationary",
        construction="pool_resample",
        theta_sampler="iid",
        resample_per_eval=True,
        n_realizations=SEARCH_ENSEMBLE_N,
        realization_years=SCENARIO_YEARS,
        pool_size=RESAMPLE_POOL_SIZE,
        n_ensemble_draws=N_ENSEMBLE_DRAWS,
        seed_domain="resample_pool",
        notes="Tests whether FREEZING the search ensemble causes overfitting. "
              "Primary precedent: Brodeur et al. (2020) (bagging / cross-validation "
              "in reservoir control-policy search). Trindade et al. (2017, 2019) and "
              "Gold et al. (2022, 2023) are cited ONLY for the principle that the "
              "search ensemble is re-randomized across evaluations -- NOT as the "
              "mechanism. DECLARED DEVIATION: Trindade evaluates ALL 1,000 "
              "realizations every evaluation and re-randomizes the flow<->DU-vector "
              "PAIRING; our theta is fused into the realization at generation, so "
              "there is no pairing to re-randomize. Ours is index resampling of N "
              "from a pre-staged pool. Requires a non-chunked (single-HDF5) pool.",
    ),
    "hazard_filling_stationary": ScenarioDesign(
        name="hazard_filling_stationary",
        family="hazard_filling_ensemble",
        description="Space-filling subsample, in hazard space, of its own "
                    "stationary candidate pool (proposed method).",
        population="stationary",
        construction="hazard_fill",
        theta_sampler="iid",
        n_realizations=SEARCH_ENSEMBLE_N,
        realization_years=SCENARIO_YEARS,
        pool_size=CANDIDATE_POOL_SIZE,
        n_ensemble_draws=N_ENSEMBLE_DRAWS,
        seed_domain="stat_pool",
        selector="lhs_nn",
        selector_space="cdf",
        needs_hazard_image=True,
        campaign=True,
        notes="The contribution, in the stationary population. Generalizes Zatarain "
              "Salazar et al. (2017) -- which subsamples a stationary Kirsch-Nowak "
              "pool by a realized-flow metric, in search -- from 1-D to m-D, and "
              "from probability-preserving to coverage. Controlled by "
              "fixed_probabilistic (same generator, same population law, same N, "
              "same L; only the selection rule differs). Hazard axes are SCREENED "
              "per pool (Olden & Poff redundancy screen, tail-balanced, target "
              "m=3-4). The selector is deterministic LHS + nearest-neighbor; the "
              "snap is intrinsic because hazard coordinates cannot be prescribed at "
              "generation. No simulated annealing.",
    ),

    # ---------------- DU-forced population ----------------
    "input_stratified": ScenarioDesign(
        name="input_stratified",
        family="input_stratified_ensemble",
        description="Latin hypercube over the harmonic forcing parameters; R "
                    "realizations GENERATED per design point.",
        population="du_forced",
        construction="lhs_theta",
        theta_sampler="lhs",
        n_realizations=INPUT_STRAT_N_THETA * INPUT_STRAT_R,
        realization_years=SCENARIO_YEARS,
        n_theta_profiles=INPUT_STRAT_N_THETA,
        realizations_per_profile=INPUT_STRAT_R,
        n_ensemble_draws=N_ENSEMBLE_DRAWS,
        seed_domain="input_strat",
        notes="Precedent: Quinn et al. (2020); Bartholomew & Kwakkel (2020); Eker & "
              "Kwakkel (2018); Watson & Kasprzyk (2017). The most common recent DMDU "
              "approach, and the foil for hazard_filling_du: the contrast isolates "
              "the central claim that uniform coverage in INPUT space need not give "
              "uniform coverage in HAZARD space, because distinct theta often yield "
              "hydrologically redundant realizations (Quinn et al. 2020; Guo et al. "
              "2018). LHS ALONE -- realizations are GENERATED at the design points, "
              "not selected from a pool, because theta is a knob on the generator. "
              "The LHS is over the INTRINSIC harmonic amplitudes [m, r1, r2] (+3 "
              "with the CV axis), NOT the derived 12-dim monthly change-factor "
              "vector, which is a deterministic function of them. R>1 separates "
              "forcing uncertainty from natural variability within a forcing (Quinn "
              "et al. 2018); R=1 maximizes input coverage at fixed N. The "
              "N_theta/R split is PROVISIONAL and set before generation.",
    ),
    "hazard_filling_du": ScenarioDesign(
        name="hazard_filling_du",
        family="hazard_filling_ensemble",
        description="Space-filling subsample, in hazard space, of its own "
                    "DU-forced candidate pool (proposed method).",
        population="du_forced",
        construction="hazard_fill",
        theta_sampler="iid",
        n_realizations=SEARCH_ENSEMBLE_N,
        realization_years=SCENARIO_YEARS,
        pool_size=CANDIDATE_POOL_SIZE,
        realizations_per_profile=1,
        n_ensemble_draws=N_ENSEMBLE_DRAWS,
        seed_domain="du_pool",
        selector="lhs_nn",
        selector_space="cdf",
        needs_hazard_image=True,
        campaign=True,
        notes="The contribution, in the DU-forced population, and the central "
              "novelty claim. Controlled by input_stratified (same forcing space, "
              "same N, same L; only the selection SPACE differs). Motivation: Cohen "
              "et al. (2021); Zaniolo et al. (2023). Machinery: Bonham et al. "
              "(2024). The pool's theta are sampled i.i.d. (NOT LHS) with "
              "realizations_per_profile=1, so the pool is an i.i.d. sample of the "
              "DU population and its hazard image is the honest empirical hazard "
              "manifold rather than an artifact of a design imposed on theta.",
    ),

    # ---------------- non-campaign ----------------
    "hazard_filling_absolute": ScenarioDesign(
        name="hazard_filling_absolute",
        family="hazard_filling_ensemble",
        description="Non-campaign sensitivity: hazard-filling in ABSOLUTE "
                    "magnitude space rather than empirical-CDF/rank space.",
        population="du_forced",
        construction="hazard_fill",
        theta_sampler="iid",
        n_realizations=SEARCH_ENSEMBLE_N,
        realization_years=SCENARIO_YEARS,
        pool_size=CANDIDATE_POOL_SIZE,
        realizations_per_profile=1,
        n_ensemble_draws=1,
        seed_domain="du_pool",
        selector="lhs_nn",
        selector_space="abs",
        needs_hazard_image=True,
        campaign=False,
        notes="NOT part of the manuscript comparison. Shares the DU candidate pool, "
              "axes, N and anchor plan with hazard_filling_du; differs ONLY in the "
              "selection space (absolute magnitude rather than empirical-CDF/rank), "
              "so it deliberately over-represents rare-but-severe corners relative "
              "to their frequency in the pool. Retained as a sensitivity on how much "
              "of any hazard-filling effect is attributable to tail enrichment "
              "specifically. Cf. Hilbers et al. (2019) on deliberate tail "
              "over-representation for an optimization.",
    ),
    "scaling_stationary": ScenarioDesign(
        name="scaling_stationary",
        family="fixed_probabilistic_ensemble",
        description="Stationary Kirsch-Nowak ensemble sized for the Anvil "
                    "parallel-scaling experiment (timing stand-in).",
        population="stationary",
        construction="stationary_kn",
        theta_sampler="iid",
        n_realizations=_SCALING_KN_REALS,
        realization_years=_SCALING_KN_YEARS,
        n_ensemble_draws=1,
        seed_domain="fixed",
        campaign=False,
        notes="Supplemental-only, for workflow/supplemental/anvil_scaling_*. Times "
              "the trimmed-model ensemble-evaluation path on a directly generated "
              "kn_{Y}yr_n{N} ensemble; scenario content does not affect per-eval "
              "cost. Sizes via NYCOPT_SCALING_KN_YEARS / NYCOPT_SCALING_KN_REALS. "
              "Not part of the manuscript design comparison.",
    ),
}


###############################################################################
# Invariants
###############################################################################

def assert_iid_pools() -> None:
    """Assert the two conditions the cross-design control depends on.

    1. Every design except ``lhs_theta`` samples theta i.i.d. A random subset of
       an LHS design is not i.i.d., so an LHS-sampled pool would silently void
       the distributional-equivalence control that makes ``fixed_probabilistic``
       the exact reference for ``hazard_filling_stationary``. Nothing else in the
       pipeline would fail if this were broken -- hence the assertion.
    2. A DU candidate pool carries ``realizations_per_profile == 1``.
       Realizations sharing a theta are not independent, so R > 1 would break the
       same argument from the other side.

    **E_test is deliberately EXEMPT from both, and must stay exempt.** It iterates
    ``SCENARIO_DESIGNS`` only, and the held-out test ensemble (``src/etest.py``) is an
    ``EnsembleSpec``, never a ``ScenarioDesign`` -- so it is out of scope here by
    construction. That is not an oversight to be "fixed". Both conditions above exist
    for ONE reason: a design's candidate pool is later SUBSAMPLED, and the control
    argument (a uniform random size-N subset of an i.i.d. pool has exactly the law of
    N i.i.d. draws) holds only for an i.i.d. pool with one realization per theta.
    E_test is never subsampled and is never a control -- it is the measuring stick --
    so neither condition applies to it. It is sampled by LHS over the full DU range,
    with R >> 1 realizations per theta, and that is CORRECT: space-filling is what a
    measuring stick wants, and the within-theta replication is what makes the SOW-unit
    robustness metric (Herman 2014; Trindade 2017; Gold 2022) computable at all.
    Applying this assertion to E_test would forbid exactly the construction the
    literature standardizes on.

    Raises:
        AssertionError: If either condition is violated.
    """
    for design in SCENARIO_DESIGNS.values():
        if design.construction == "lhs_theta":
            assert design.theta_sampler == "lhs", (
                f"'{design.name}' is an input-stratified design and must sample "
                f"theta by LHS, got {design.theta_sampler!r}."
            )
            continue
        assert design.theta_sampler == "iid", (
            f"'{design.name}' must sample theta i.i.d., got "
            f"{design.theta_sampler!r}. A random subset of an LHS design is not "
            f"i.i.d., which would void the control that makes "
            f"fixed_probabilistic the exact reference for the hazard-filling "
            f"designs. See scenario_design_methods.md §3.2."
        )
        if design.construction == "hazard_fill" and design.population == "du_forced":
            assert design.realizations_per_profile == 1, (
                f"'{design.name}' is a DU candidate pool and must carry "
                f"realizations_per_profile=1; realizations sharing a theta are "
                f"not independent."
            )


def assert_seed_domains_disjoint(max_draws: int = 64) -> None:
    """Assert no two (design, draw) pairs collide on a generator seed.

    Before per-design generation this did not matter -- designs merely selected
    indices from shared data. Now that every design *generates*, two designs
    sharing a seed would produce correlated realizations, reintroducing exactly
    the confound the architecture removes.

    Args:
        max_draws: Number of draws to check per seed domain.

    Raises:
        AssertionError: If two distinct (domain, draw) pairs map to one seed.
    """
    seen: dict[int, tuple[str, int]] = {}
    domains = {d.seed_domain for d in SCENARIO_DESIGNS.values() if d.seed_domain}
    domains |= {"hazard_select_stat", "hazard_select_du", "etest:kn", "etest:hmm"}
    for domain in sorted(domains):
        for draw in range(max_draws):
            seed = design_seed(SEED_ROOT, domain, draw)
            assert seed not in seen, (
                f"Seed collision: ({domain}, {draw}) and {seen[seed]} both map to "
                f"seed {seed}."
            )
            seen[seed] = (domain, draw)


assert_iid_pools()


###############################################################################
# Resolver + helpers
###############################################################################

def get_scenario_design(name: str) -> ScenarioDesign:
    """Resolve a scenario-design name to its ``ScenarioDesign``.

    Args:
        name: A key of ``SCENARIO_DESIGNS``.

    Returns:
        The matching ``ScenarioDesign``.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return SCENARIO_DESIGNS[name]
    except KeyError:
        raise KeyError(
            f"Unknown scenario design '{name}'. "
            f"Available designs: {list_scenario_designs()}."
        ) from None


def list_scenario_designs() -> list[str]:
    """Return the registered scenario-design names in sorted order."""
    return sorted(SCENARIO_DESIGNS)


def campaign_designs() -> list[str]:
    """Return the names of the designs in the manuscript comparison."""
    return sorted(n for n, d in SCENARIO_DESIGNS.items() if d.campaign)
