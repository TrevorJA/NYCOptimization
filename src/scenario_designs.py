"""
scenario_designs.py - Registry of scenario designs for the MOEA evaluation ensemble.

A *scenario design* is the construction recipe for the streamflow ensemble used
during many-objective search. Comparing scenario designs is the methodological
contribution of the study (see ``docs/notes/methods/experimental_design.md``).
This module is the single source of truth that maps a *single-string design
name* (e.g. ``"historic"``, ``"hazard_filling"``) to an immutable
``ScenarioDesign`` describing how its search ensemble is built.

The scenario design is one of two orthogonal axes that specify an optimization
run; the other is the MOEA algorithm configuration (see ``src/moea_config.py``).
The design name is the **top level** of the output tree:
``outputs/{scenario_design_name}/{moea_slug}/...``.

This module supersedes the former stub ``config.ENSEMBLE_PRESETS``. It bridges to
the lower-level ``src/ensembles.py`` ``EnsembleSpec`` machinery: a design knows
how to resolve its search ensemble into one or more ``EnsembleSpec`` instances.

Status: ``historic`` (single-trace preset), the two fixed probabilistic designs
``fixed_probabilistic_short`` / ``fixed_probabilistic_long``, and
``resampled_probabilistic`` are wired. The fixed designs resolve to a directly
generated Kirsch-Nowak ensemble via the ``kn_{Y}yr_n{N}`` slug grammar (Step-1
generation stages exactly the ensemble the design names; no subsampling-from-a-
master step is required for the stand-up). Their sizes/lengths here are
*provisional small test values* for standing up the generation->optimization
workflow; the final sizes (and the eventual subsample-from-master construction)
remain open decisions (#2, #5 in ``experimental_design.md``). The remaining
three designs (``resampled_probabilistic``, ``input_stratified``,
``hazard_filling``) still raise a clear ``NotImplementedError`` from
``resolve_search_spec`` until their machinery (per-eval re-index hook; CMIP6
forcing space; hazard metrics + subsample selectors) lands.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ensembles import (
    EnsembleSpec,
    as_resampling_pool,
    get_ensemble_spec,
    kirsch_nowak_slug,
)


###############################################################################
# ScenarioDesign
###############################################################################

@dataclass(frozen=True)
class ScenarioDesign:
    """Immutable specification of a scenario design for MOEA evaluation.

    Attributes
    ----------
    name
        Single-string key used to select this design (via
        ``NYCOPT_SCENARIO_DESIGN``) and as the top-level output directory.
    family
        Taxonomy label tying the design to a literature family (see the
        "Scenario designs compared" table in ``experimental_design.md``).
    description
        One-line human-readable summary.
    resample_per_eval
        ``True`` only for the resampled-probabilistic design, whose search
        ensemble is redrawn at every function evaluation. ``False`` for the
        fixed designs.
    selection
        Subsampling method used to draw the search ensemble from the master
        ensemble: ``"none"`` (historic), ``"random"`` (probabilistic),
        ``"lhs_input"`` (input-stratified), ``"hazard_fill"`` (hazard-filling).
        ``None`` until decided.
    ensemble_preset
        Name of a static ``src/ensembles.py`` preset this design maps to, when
        applicable (e.g. ``historic`` -> ``"historic_single"``). ``None`` for
        designs whose ensemble is constructed dynamically.
    n_realizations
        Number of realizations in the search ensemble. ``None`` until decided.
    realization_years
        Length (in years) of each realization. ``None`` until decided.
    n_ensemble_draws
        Replication structure: number of independent ensemble constructions
        (each optimized across several seeds). ``None`` until decided.
    master_pool_size
        For pool-based designs (``resample_per_eval`` and ``hazard_fill``): size
        of the master pool that Step-1 stages and from which ``n_realizations``
        are drawn (per-eval for resampled; once, offline, for hazard-filling).
        ``None`` for the fixed designs that stage their search ensemble directly.
    subset_seed
        For ``hazard_fill`` only: the selector seed identifying the precomputed
        subset manifest written by the scengen hazard-filling subsample step.
    notes
        Free-form notes (open questions, literature pointers).
    """

    name: str
    family: str
    description: str
    resample_per_eval: bool = False
    selection: str | None = None
    ensemble_preset: str | None = None
    n_realizations: int | None = None
    realization_years: int | None = None
    n_ensemble_draws: int | None = None
    master_pool_size: int | None = None
    subset_seed: int | None = None
    notes: str = ""

    def kn_staged_dims(self) -> tuple[int, int] | None:
        """Return ``(n_years, n_realizations)`` of the KN ensemble Step-1 must stage.

        This is the single source of truth shared by :meth:`resolve_search_spec`
        and the Step-1 generation script, so the generated ensemble and the
        resolved search spec always name the same staged ``kn_{Y}yr_n{N}``
        directory. Returns ``None`` for designs that do not stage a direct KN
        ensemble (historic, preset-backed, or not-yet-wired designs).

        - Fixed probabilistic (direct-KN): stage exactly the search ensemble
          ``(realization_years, n_realizations)``.
        - Resampled probabilistic / hazard-filling: stage the *master pool*
          ``(realization_years, master_pool_size)``; the search ensemble is a
          subset of that pool (per-evaluation random for resampled; a fixed
          space-filling subset for hazard-filling).
        """
        if self.ensemble_preset is not None:
            return None
        if self.selection not in ("random", "hazard_fill"):
            return None
        if self.realization_years is None or self.n_realizations is None:
            return None
        uses_master_pool = self.resample_per_eval or self.selection == "hazard_fill"
        if uses_master_pool:
            if self.master_pool_size is None:
                return None
            return (self.realization_years, self.master_pool_size)
        return (self.realization_years, self.n_realizations)

    def kn_ensemble_slug(self) -> str | None:
        """Return the ``kn_{Y}yr_n{N}`` slug this design stages, or ``None``."""
        dims = self.kn_staged_dims()
        return None if dims is None else kirsch_nowak_slug(dims[0], dims[1])

    def resolve_search_spec(self, draw: int = 0) -> EnsembleSpec:
        """Resolve this design's search ensemble to an ``EnsembleSpec``.

        Args:
            draw: Index of the independent ensemble draw (replication). Only the
                static-preset and direct-KN paths are wired today, and only for
                ``draw == 0``; multi-draw replication awaits the
                subsample-from-master construction.

        Returns:
            The ``EnsembleSpec`` describing the search ensemble. For the
            resampled design this is the master-pool spec marked
            ``resample_per_eval=True``; the simulation layer redraws a subset of
            ``n_realizations`` indices from the pool at each evaluation.

        Raises:
            NotImplementedError: For designs whose construction is not yet wired
                (``input_stratified``, ``hazard_filling``), or for ``draw != 0``
                on a direct-KN design.
        """
        if self.ensemble_preset is not None:
            return get_ensemble_spec(self.ensemble_preset)
        slug = self.kn_ensemble_slug()
        if slug is not None:
            if draw != 0:
                raise NotImplementedError(
                    f"Scenario design '{self.name}': multi-draw replication "
                    f"(draw={draw}) is not wired yet. The stand-up path stages a "
                    f"single ensemble; independent draws await the "
                    f"subsample-from-master construction (experimental_design.md "
                    f"#5)."
                )
            spec = get_ensemble_spec(slug)
            if self.selection == "hazard_fill":
                # Fixed space-filling subset of the master pool, precomputed
                # offline by the scengen hazard-filling subsample step and read
                # from the staged subset manifest. The original ensemble (the
                # master pool slug above) and the hazard metrics used to select
                # the subset are both expected to change later.
                from src.ensembles import load_hazard_filling_spec
                hf = load_hazard_filling_spec(
                    slug, self.n_realizations, self.subset_seed,
                )
                if hf is None:
                    raise NotImplementedError(
                        f"hazard_filling subset not computed yet for master pool "
                        f"'{slug}'. Generate the pool (workflow 01), then run the "
                        f"scengen hazard-filling subsample "
                        f"(scripts/main/subsample_hazard_filling.py) to write "
                        f"hazard_filling_n{self.n_realizations}_"
                        f"seed{self.subset_seed}.json under the staged pool dir."
                    )
                return hf
            if self.resample_per_eval:
                # Master pool: the simulation layer draws `n_realizations`
                # indices from it at every evaluation (Trindade et al. 2017).
                return as_resampling_pool(spec, self.n_realizations)
            return spec
        raise NotImplementedError(
            f"Scenario design '{self.name}' ({self.family}) has no ensemble "
            f"construction wired yet. Its sizes/lengths/draws/selection are "
            f"open decisions (see experimental_design.md). Runnable today: "
            f"'historic', 'fixed_probabilistic_short', "
            f"'fixed_probabilistic_long', 'resampled_probabilistic'."
        )


###############################################################################
# Registry
###############################################################################
# The six scenario designs of the manuscript comparison. Construction
# parameters for the five non-historic designs are deliberately ``None`` until
# the ensemble-design discussion (experimental_design.md open decisions #2/#4/#5)
# finalizes them. Add or revise entries here as those decisions land.

SCENARIO_DESIGNS: dict[str, ScenarioDesign] = {
    "historic": ScenarioDesign(
        name="historic",
        family="historical_record",
        description="The observed record, simulated as one continuous trace.",
        resample_per_eval=False,
        selection="none",
        ensemble_preset="historic_single",
        n_realizations=1,
        notes="Reference for prevailing practice; cannot be size-matched.",
    ),
    "fixed_probabilistic_short": ScenarioDesign(
        name="fixed_probabilistic_short",
        family="fixed_probabilistic_ensemble",
        description="Many short synthetic sequences, drawn once at random from "
                    "the master ensemble (here generated directly).",
        resample_per_eval=False,
        selection="random",
        n_realizations=10,
        realization_years=5,
        notes="Baseline structured design (sample average approximation). "
              "PROVISIONAL small test sizes (10 x 5yr = 50 scenario-years) for "
              "the workflow stand-up; matched in scenario-years to "
              "fixed_probabilistic_long. Final sizes + subsample-from-master TBD.",
    ),
    "fixed_probabilistic_long": ScenarioDesign(
        name="fixed_probabilistic_long",
        family="fixed_probabilistic_ensemble_long",
        description="A few multi-decadal synthetic records at equal total "
                    "simulated years.",
        resample_per_eval=False,
        selection="random",
        n_realizations=2,
        realization_years=25,
        notes="Tests many-short vs few-long at equal simulated years. "
              "PROVISIONAL small test sizes (2 x 25yr = 50 scenario-years), "
              "matched to fixed_probabilistic_short. Final sizes TBD.",
    ),
    "resampled_probabilistic": ScenarioDesign(
        name="resampled_probabilistic",
        family="resampled_probabilistic_ensemble",
        description="Scenarios redrawn at random from the master ensemble at "
                    "every function evaluation (per Trindade et al. 2017).",
        resample_per_eval=True,
        selection="random",
        n_realizations=10,
        realization_years=5,
        master_pool_size=50,
        notes="No fixed ensemble to replicate; seeds only. Comparisons rely "
              "entirely on the held-out re-evaluation. PROVISIONAL small test "
              "sizes: draw 10 of a 50-realization master pool, 5yr each "
              "(per-eval draw matches fixed_probabilistic_short). The simulation "
              "layer redraws indices each evaluation. Final sizes TBD.",
    ),
    "input_stratified": ScenarioDesign(
        name="input_stratified",
        family="input_stratified_ensemble",
        description="Latin hypercube sample over generator parameters, one "
                    "realization per parameter set.",
        resample_per_eval=False,
        selection="lhs_input",
        notes="Most common recent DMDU approach; contrast with hazard_filling "
              "isolates the central claim. Sizes/draws TBD.",
    ),
    "hazard_filling": ScenarioDesign(
        name="hazard_filling",
        family="hazard_filling_ensemble",
        description="Space-filling subsample of the master ensemble in "
                    "hazard-metric space (proposed method).",
        resample_per_eval=False,
        selection="hazard_fill",
        n_realizations=10,
        realization_years=5,
        master_pool_size=200,
        subset_seed=0,
        notes="The proposed contribution. INITIAL DRAFT: subsample 10 of a "
              "200-realization stationary Kirsch-Nowak pool of 5yr records "
              "(same generator as fixed_probabilistic_short) using the scengen "
              "hazard-filling selector with MOEA-FIND's 'primary' SSI metrics. "
              "Both the master pool and the hazard metrics are expected to "
              "change. The subset is precomputed offline; resolve reads the "
              "subset manifest staged under the pool dir.",
    ),
}


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
