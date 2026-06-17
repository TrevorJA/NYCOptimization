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

Status: only ``historic`` is fully wired (it maps to the legacy
``historic_single`` preset). The other five designs are registered with their
taxonomy metadata, but their ensemble-construction parameters (sizes, lengths,
draws, selection method) are intentionally left ``None`` until the
ensemble-design discussion finalizes them (open decisions #2, #4, #5 in
``experimental_design.md``). ``resolve_search_spec`` raises a clear
``NotImplementedError`` for those until they are wired.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ensembles import EnsembleSpec, get_ensemble_spec


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
    notes: str = ""

    def resolve_search_spec(self, draw: int = 0) -> EnsembleSpec:
        """Resolve this design's search ensemble to an ``EnsembleSpec``.

        Args:
            draw: Index of the independent ensemble draw (replication). Only
                meaningful once ``n_ensemble_draws`` is wired; ignored by the
                static ``ensemble_preset`` path.

        Returns:
            The ``EnsembleSpec`` describing the search ensemble.

        Raises:
            NotImplementedError: For designs whose construction parameters are
                not yet decided (everything except ``historic``).
        """
        if self.ensemble_preset is not None:
            return get_ensemble_spec(self.ensemble_preset)
        raise NotImplementedError(
            f"Scenario design '{self.name}' ({self.family}) has no ensemble "
            f"construction wired yet. Its sizes/lengths/draws/selection are "
            f"open decisions (see experimental_design.md). Only 'historic' is "
            f"runnable today."
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
    "fixed_probabilistic": ScenarioDesign(
        name="fixed_probabilistic",
        family="fixed_probabilistic_ensemble",
        description="Scenarios drawn once at random from the master ensemble "
                    "of many short sequences.",
        resample_per_eval=False,
        selection="random",
        notes="Baseline structured design (sample average approximation). "
              "Sizes/lengths/draws TBD.",
    ),
    "fixed_probabilistic_long": ScenarioDesign(
        name="fixed_probabilistic_long",
        family="fixed_probabilistic_ensemble_long",
        description="A few multi-decadal synthetic records at equal total "
                    "simulated years.",
        resample_per_eval=False,
        selection="random",
        notes="Tests many-short vs few-long at equal simulated years. "
              "Sizes/lengths TBD.",
    ),
    "resampled_probabilistic": ScenarioDesign(
        name="resampled_probabilistic",
        family="resampled_probabilistic_ensemble",
        description="Scenarios redrawn at random from the master ensemble at "
                    "every function evaluation (per Trindade et al. 2017).",
        resample_per_eval=True,
        selection="random",
        notes="No fixed ensemble to replicate; seeds only. Comparisons rely "
              "entirely on the held-out re-evaluation. Size TBD.",
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
        notes="The proposed contribution. Hazard-metric axes (open decision "
              "#4), sizes, and draws TBD.",
    ),
    # Dev-only: a tiny runnable ensemble design (bridges to the existing
    # wcu_kirsch_n5 preset) so the ensemble-aware path is testable end-to-end
    # before the method designs are wired. Mirrors the 'smoke' MOEA config.
    # NOT a manuscript design.
    "smoke_ensemble": ScenarioDesign(
        name="smoke_ensemble",
        family="dev_smoke",
        description="Dev/smoke ensemble (N=5 Kirsch-Nowak) for plumbing tests.",
        resample_per_eval=False,
        selection="random",
        ensemble_preset="wcu_kirsch_n5",
        n_realizations=5,
        realization_years=20,
        notes="Plumbing exercise only — not a method choice.",
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
