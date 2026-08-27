"""
scenario_designs.py - Registry of scenario designs for the MOEA search ensemble.

A scenario design is the construction recipe for the streamflow ensemble used
during search (``docs/notes/methods/scenario_design_methods.md``). This module
maps a design name (e.g. ``"historic"``, ``"hazard_filling_stationary"``) to an
immutable ``ScenarioDesign``; the design name is the top level of the output
tree, ``outputs/{design}/{moea_slug}/...``.

The campaign is the three designs flagged ``campaign=True``: ``historic``,
``fixed_probabilistic`` and ``hazard_filling_stationary`` (``campaign_designs()``).
The other registered designs are wired but non-campaign. Every design generates
its own realizations from its own seed domain; only the hazard-filling designs
select from a candidate pool, because hazard coordinates are emergent properties
of a realized sequence and cannot be prescribed at generation.

Candidate pools are sampled i.i.d., never by LHS: a uniform random size-N subset
of an i.i.d. pool has exactly the law of N fresh i.i.d. draws, which is what
makes ``fixed_probabilistic`` the exact control for ``hazard_filling_stationary``.
A DU candidate pool must also carry ``realizations_per_profile == 1``. Both
conditions are enforced by ``assert_iid_pools()`` at import.
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

# Scenario length L (years) of each realization; the single source of truth,
# re-exported as ``config.SCENARIO_YEARS``. Changing L invalidates every staged
# L-conditional artifact.
SCENARIO_YEARS: int = int(os.environ.get("NYCOPT_SCENARIO_YEARS", "10"))

# Ensemble size N common to every matched design (campaign_design.md;
# ensemble_size_diagnostics.md). A common (N, L) keeps per-evaluation cost
# identical across designs. Changing N invalidates every staged search
# ensemble and the step-05 baselines.
SEARCH_ENSEMBLE_N: int = int(os.environ.get("NYCOPT_SEARCH_N", "300"))

# Candidate-pool cardinality P for the hazard-filling designs and the
# resampling-pool cardinality for ``resampled_probabilistic``. The default is
# the laptop scale; production P = 1e6 via NYCOPT_CANDIDATE_POOL_N at
# generation time.
CANDIDATE_POOL_SIZE: int = int(os.environ.get("NYCOPT_CANDIDATE_POOL_N", "2000"))
RESAMPLE_POOL_SIZE: int = int(os.environ.get("NYCOPT_RESAMPLE_POOL_N", "1000"))

# input_stratified allocates N = N_theta x R; set before generation.
INPUT_STRAT_N_THETA: int = int(os.environ.get("NYCOPT_INPUT_STRAT_N_THETA", "20"))
INPUT_STRAT_R: int = int(os.environ.get("NYCOPT_INPUT_STRAT_R", "5"))

# Independent ensemble draws staged per design (independent generations, fixed
# before step 02 runs). Draw 0 is the searched ensemble; draws 1-2 serve the
# SI draw-sensitivity re-evaluation of the final Pareto sets and are never
# searched.
N_ENSEMBLE_DRAWS: int = int(os.environ.get("NYCOPT_N_ENSEMBLE_DRAWS", "3"))

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
        n_ensemble_draws: Independent constructions staged for the design.
            The campaign searches draw 0 only; the others serve the SI
            draw-sensitivity re-evaluation (see ``N_ENSEMBLE_DRAWS``).
        seed_domain: Namespace for this design's generator seed; disjoint
            domains keep designs from sharing realizations.
        selector: For ``hazard_fill``: the selector name. The wired selector is
            LHS + nearest-neighbor (``"lhs_nn"``), deterministic given its
            anchor seed; draws vary the anchor plan.
        selector_space: For ``hazard_fill``: the space the anchors fill.
            ``"abs"`` = absolute, range-scaled magnitude space -- the CAMPAIGN
            selector, which deliberately over-represents the severe hazard
            corners relative to their pool frequency; ``"cdf"`` =
            empirical-CDF/rank space, the non-campaign sensitivity, which
            preserves the pool marginals and distorts only the joint dependence
            among axes.
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
        """Return the slug of the design's own pool for ``draw``, or ``None``.

        Only ``pool_resample`` and ``hazard_fill`` own a pool. The pool is
        regenerated per draw so a hazard-filling draw re-rolls everything that
        is random about its construction and its between-draw variance is
        commensurable with ``fixed_probabilistic``. Hazard-filling designs of
        the same population share the pool at the same draw.

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

        Deliberately not split by ``selector_space``: the CDF and absolute
        designs share one anchor plan and differ only in the normalization
        geometry the anchors snap into.
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
        campaign=False,
        notes="NON-CAMPAIGN (retained for future work). Tests whether FREEZING "
              "the search ensemble causes overfitting. "
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
        selector_space="abs",
        needs_hazard_image=True,
        campaign=True,
        notes="THE CAMPAIGN CONTRIBUTION, in the stationary population. Generalizes "
              "Zatarain Salazar et al. (2017) -- which subsamples a stationary "
              "Kirsch-Nowak pool by a realized-flow metric, in search -- from 1-D to "
              "m-D, and from probability-preserving to coverage. Selection is in "
              "ABSOLUTE, robust range-scaled hazard space (selector_space='abs'; "
              "per-axis p1/p99 bounds — see scengen.subsample.ROBUST_LO_PCT), which "
              "deliberately over-represents the severe (rare) hazard corners relative "
              "to their pool frequency -- the deliberate distribution shift the study "
              "tests. Controlled by fixed_probabilistic (same generator, same "
              "population law, same N, same L; only the selection rule differs). "
              "Hazard axes are SCREENED per pool (degenerate drop + near-duplicate "
              "dedupe at |rho_S| >= 0.95; all other non-degenerate descriptors "
              "retained). The selector is deterministic LHS + "
              "nearest-neighbor; the snap is intrinsic because hazard coordinates "
              "cannot be prescribed at generation. The rank-space "
              "(empirical-CDF) variant is the non-campaign "
              "hazard_filling_stationary_cdf sensitivity.",
    ),
    "hazard_filling_stationary_cdf": ScenarioDesign(
        name="hazard_filling_stationary_cdf",
        family="hazard_filling_ensemble",
        description="Non-campaign sensitivity: stationary hazard-filling in "
                    "empirical-CDF/rank space rather than absolute magnitude space.",
        population="stationary",
        construction="hazard_fill",
        theta_sampler="iid",
        n_realizations=SEARCH_ENSEMBLE_N,
        realization_years=SCENARIO_YEARS,
        pool_size=CANDIDATE_POOL_SIZE,
        n_ensemble_draws=1,
        seed_domain="stat_pool",
        selector="lhs_nn",
        selector_space="cdf",
        needs_hazard_image=True,
        campaign=False,
        notes="NOT part of the manuscript campaign. Shares the stationary candidate "
              "pool, axes and anchor plan with hazard_filling_stationary; differs "
              "ONLY in the selection space (empirical-CDF/rank rather than absolute "
              "magnitude), so it preserves the pool marginals and distorts only the "
              "joint dependence among axes. Retained as a sensitivity that isolates "
              "how much of any hazard-filling effect is attributable to absolute-space "
              "tail over-representation specifically.",
    ),

    # ---------------- DU-forced designs (non-campaign; retained/future work) ------
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
        campaign=False,
        notes="NON-CAMPAIGN (retained for future work). Precedent: Quinn et al. "
              "(2020); Bartholomew & Kwakkel (2020); Eker & Kwakkel (2018); Watson & "
              "Kasprzyk (2017). The most common recent DMDU approach, and the foil "
              "for hazard_filling_du: the contrast isolates "
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
              "N_theta/R split is set before generation.",
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
        campaign=False,
        notes="NON-CAMPAIGN (retained for future work). Hazard-filling in the "
              "DU-forced population. Controlled by input_stratified (same forcing "
              "space, "
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

    1. Every design except ``lhs_theta`` samples theta i.i.d.; a random subset
       of an LHS design is not i.i.d., which would void the control that makes
       ``fixed_probabilistic`` the exact reference for the hazard-filling
       designs. Nothing else in the pipeline would fail if this were broken.
    2. A DU candidate pool carries ``realizations_per_profile == 1``, since
       realizations sharing a theta are not independent.

    E_test (``src/etest.py``) is an ``EnsembleSpec``, never a ``ScenarioDesign``,
    and is out of scope by construction: it is never a control, so LHS with
    R > 1 is correct there.

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

    Two designs sharing a seed would produce correlated realizations.

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
