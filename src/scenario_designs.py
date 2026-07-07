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

Status: all registered designs are code-wired. ``historic`` (single-trace
preset), the two fixed probabilistic designs ``fixed_probabilistic_short`` /
``fixed_probabilistic_long``, and ``resampled_probabilistic`` resolve to a
directly generated Kirsch-Nowak ensemble via the ``kn_{Y}yr_n{N}`` slug grammar
(step-02 generation stages exactly the ensemble the design names; no
subsampling-from-a-master step is required for the stand-up). Their
sizes/lengths here are *provisional small test values* for standing up the
generation->optimization workflow; the final sizes (and the eventual
subsample-from-master construction) remain open decisions (#2, #5 in
``experimental_design.md``). The forcing-master designs (``input_stratified``,
``hazard_filling``, ``hazard_filling_absolute``) raise ``NotImplementedError``
from ``resolve_search_spec`` only when their staged data (forcing master /
reduced subsample, workflow steps 02-03) is missing — the gate is staged-data
availability, not unimplemented code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.ensembles import (
    EnsembleSpec,
    as_resampling_pool,
    get_ensemble_spec,
    kirsch_nowak_slug,
    staged_ensemble_dir,
)

# Forcing-master sizing shared by the two designs that subsample a CMIP6-forced master
# (hazard_filling, input_stratified). Provisional small test values for the workflow stand-up;
# override via env for a tiny smoke run so the master slug, the Step-1 driver, and the subset
# size stay mutually consistent (methods §3.2). No CLI value flags.
_MASTER_N_FORCING = int(os.environ.get("NYCOPT_MASTER_N_FORCING", "200"))
_MASTER_REALS_PER_PROFILE = int(os.environ.get("NYCOPT_MASTER_REALS_PER_PROFILE", "1"))
_MASTER_YEARS = int(os.environ.get("NYCOPT_MASTER_YEARS", "5"))
_MASTER_SUBSET_N = int(os.environ.get("NYCOPT_MASTER_SUBSET_N", "64"))


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
        For ``hazard_fill`` only: the selector (LHS) seed; part of the final
        ensemble slug so different selector seeds stage to distinct directories.
    selector
        For ``hazard_fill`` only: the subsample selector name (provenance/record;
        the wired selector is LHS + nearest-neighbor, ``"lhs_nn"``).
    selector_space
        For ``hazard_fill`` only: the space the LHS+NN selector fills.
        ``"cdf"`` = empirical-CDF/rank space (faithful, marginally representative;
        the primary design); ``"abs"`` = absolute magnitude space (distorted,
        deliberately over-represents hazard extremes in the search ensemble; the
        supplementary uniform-hazard-magnitude design). The uniform-coverage
        rationale is borrowed from the bottom-up/scenario-neutral tradition, but
        applying it to the in-loop search ensemble (distorting the search
        measure) is the deliberate departure -- NOT decision scaling, which
        samples for post-hoc evaluation and re-imposes likelihoods. Part of the
        final ensemble slug so the two do not collide.
    master_kind
        How the master (pool) this design draws from is built: ``"stationary"``
        (direct Kirsch-Nowak, ``kn_{Y}yr_n{N}``) or ``"forcing"`` (the shared
        CMIP6-forced master ``master_{L}yr_n{N_M}``; methods §3.2). Forcing
        designs (``hazard_filling``, ``input_stratified``) share one master.
    n_forcing_profiles
        For ``master_kind == "forcing"``: number of CMIP6 forcing profiles
        (N_Theta). Master cardinality = ``n_forcing_profiles *
        realizations_per_profile``.
    realizations_per_profile
        For ``master_kind == "forcing"``: realizations per forcing profile
        (default 1, so each realization carries a distinct theta — the §4.5
        input-stratified view).
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
    selector: str = "lhs_nn"
    selector_space: str = "cdf"
    master_kind: str = "stationary"
    n_forcing_profiles: int | None = None
    realizations_per_profile: int = 1
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
        if self.master_kind == "forcing":
            return None  # forcing designs stage a CMIP6 master, not a stationary kn ensemble
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

    def master_n_realizations(self) -> int | None:
        """Cardinality N_M of the forcing master, or ``None`` for non-forcing designs."""
        if self.master_kind != "forcing" or self.n_forcing_profiles is None:
            return None
        return self.n_forcing_profiles * self.realizations_per_profile

    def master_slug(self) -> str | None:
        """Return the shared forcing-master slug ``master_{L}yr_n{N_M}``, or ``None``.

        Both forcing designs (``hazard_filling``, ``input_stratified``) resolve to the *same* slug
        when their ``(realization_years, n_forcing_profiles, realizations_per_profile)`` match, so
        Step 1 stages one design-independent master (methods §3.2).
        """
        n_m = self.master_n_realizations()
        if n_m is None or self.realization_years is None:
            return None
        return f"master_{self.realization_years}yr_n{n_m}"

    def kn_ensemble_slug(self) -> str | None:
        """Return the pool slug Step 1 stages for this design, or ``None``.

        For a ``forcing`` design this is the shared CMIP6 master (:meth:`master_slug`); for
        ``hazard_fill`` on a stationary master it is the ``kn_{Y}yr_n{N}`` master pool that the
        subsample step reads — not the design's search ensemble (that is the reduced ensemble named
        by :meth:`hazard_filling_slug`).
        """
        if self.master_kind == "forcing":
            return self.master_slug()
        dims = self.kn_staged_dims()
        return None if dims is None else kirsch_nowak_slug(dims[0], dims[1])

    def hazard_filling_slug(self) -> str | None:
        """Return the final reduced-ensemble slug for a ``hazard_fill`` design.

        This is the standalone ensemble the scengen subsample step stages and the
        optimizer consumes (``hazfill_{L}yr_n{N}_s{seed}``). It is the single
        source of truth shared by the subsample script (output dir) and
        :meth:`resolve_search_spec` (resolution). Returns ``None`` for non-
        hazard-filling designs or before sizes are decided.
        """
        if self.selection != "hazard_fill":
            return None
        if self.realization_years is None or self.n_realizations is None:
            return None
        seed = self.subset_seed or 0
        space_tag = "" if self.selector_space == "cdf" else f"_{self.selector_space}"
        return (
            f"hazfill{space_tag}_{self.realization_years}yr_"
            f"n{self.n_realizations}_s{seed}"
        )

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
            NotImplementedError: For ``draw != 0`` on a direct-KN design, or when a design's
                master/reduced ensemble is not staged yet (with instructions to build it).
        """
        if self.ensemble_preset is not None:
            return get_ensemble_spec(self.ensemble_preset)
        if self.selection == "lhs_input":
            return self._resolve_input_stratified(draw)
        if self.selection == "hazard_fill":
            # The search ensemble is the FINAL reduced ensemble that the scengen
            # subsample step stages (its own HDF5 + _meta.json), not the master
            # pool. The optimizer loads it directly by slug — no manifest, no
            # realization-index override.
            final_slug = self.hazard_filling_slug()
            try:
                return get_ensemble_spec(final_slug)
            except KeyError:
                raise NotImplementedError(
                    f"hazard_filling ensemble '{final_slug}' is not staged yet. "
                    f"Generate the master pool ('{self.kn_ensemble_slug()}', "
                    f"workflow step 02 with NYCOPT_SCENARIO_DESIGN=hazard_filling), then "
                    f"run workflow step 03 (scripts/main/subsample_hazard_filling.py) to stage the "
                    f"reduced ensemble."
                ) from None
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
            f"'fixed_probabilistic_long', 'resampled_probabilistic', "
            f"'input_stratified', 'hazard_filling'."
        )

    def _resolve_input_stratified(self, draw: int) -> EnsembleSpec:
        """Resolve the input-stratified search ensemble (methods §4.5).

        Space-fills the generator parameter space theta by running the same LHS+nearest-neighbor
        selector as ``hazard_filling`` but over the staged master's per-realization forcing
        coordinates (``forcing_profiles.npz``) instead of its hazard image, then materializes the
        selected realizations from the master's daily chunks into a small reduced ensemble (resolved
        by slug). Contrasting this with ``hazard_filling`` isolates the claim that uniform coverage in
        *input* space need not produce uniform coverage in *hazard* space. Idempotent: reuses the
        staged reduced ensemble on later resolves.
        """
        import numpy as np
        from scengen.subsample import input_stratified_subsample
        from src.ensembles import materialize_subset_from_master

        master = self.master_slug()
        if master is None or self.n_realizations is None:
            raise NotImplementedError(
                f"Scenario design '{self.name}' is missing forcing-master sizing "
                f"(master_kind/n_forcing_profiles/realization_years/n_realizations)."
            )
        seed = (self.subset_seed or 0) + draw
        out_slug = f"inputstrat_{self.realization_years}yr_n{self.n_realizations}_s{seed}"
        try:
            return get_ensemble_spec(out_slug)  # already materialized
        except KeyError:
            pass

        npz_path = staged_ensemble_dir(master) / "forcing_profiles.npz"
        if not npz_path.exists():
            raise NotImplementedError(
                f"input_stratified master '{master}' is not staged (no forcing_profiles.npz). "
                f"Generate it with workflow step 02 (NYCOPT_SCENARIO_DESIGN={self.name})."
            )
        with np.load(npz_path) as data:
            theta = data["mean_factor_a"]
            if "cv_factor_v" in data.files:
                theta = np.hstack([theta, data["cv_factor_v"]])
        idx = [int(i) for i in input_stratified_subsample(theta, self.n_realizations, seed=seed)]
        try:
            materialize_subset_from_master(master, idx, out_slug, extra_meta={
                "design": self.name, "selection": "lhs_input", "subset_seed": seed})
        except KeyError as exc:  # master daily chunks absent
            raise NotImplementedError(
                f"input_stratified could not materialize from master '{master}': {exc}"
            ) from None
        return get_ensemble_spec(out_slug)


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
        master_kind="forcing",
        n_forcing_profiles=_MASTER_N_FORCING,
        realizations_per_profile=_MASTER_REALS_PER_PROFILE,
        realization_years=_MASTER_YEARS,
        n_realizations=_MASTER_SUBSET_N,
        subset_seed=0,
        notes="Most common recent DMDU approach; contrast with hazard_filling "
              "isolates the central claim (uniform coverage in INPUT space need not "
              "give uniform coverage in HAZARD space). Space-fills the shared forcing "
              "master's theta (forcing_profiles.npz) with the same LHS+NN selector as "
              "hazard_filling, then overrides the master's realization indices. "
              "PROVISIONAL sizes (env-overridable); draws TBD.",
    ),
    "hazard_filling": ScenarioDesign(
        name="hazard_filling",
        family="hazard_filling_ensemble",
        description="Space-filling subsample of the master ensemble in "
                    "hazard-metric space (proposed method).",
        resample_per_eval=False,
        selection="hazard_fill",
        master_kind="forcing",
        n_forcing_profiles=_MASTER_N_FORCING,
        realizations_per_profile=_MASTER_REALS_PER_PROFILE,
        n_realizations=_MASTER_SUBSET_N,
        realization_years=_MASTER_YEARS,
        subset_seed=0,
        selector="lhs_nn",
        notes="The proposed contribution. Subsample N of the shared CMIP6-forced "
              "master (master_{L}yr_n{N_M}, also used by input_stratified) using the "
              "scengen LHS+nearest-neighbor "
              "selector (methods 4.6). Hazard axes are SCREENED per pool from an "
              "8-candidate wet+dry event-descriptor pool (SSI-6 controlling-event "
              "drought magnitude/duration/intensity/onset/recovery + POT flood "
              "peak/duration/rise), Olden & Poff redundancy screen with a "
              "tail-balanced final set (2 dry + 2 wet, target m=4 -> N>=~64 to "
              "fill). Both the master pool and candidate axes may change. The "
              "subsample step stages a standalone reduced ensemble "
              "(hazfill_5yr_n64_s0 + _meta.json) the optimizer loads by slug.",
    ),
    "hazard_filling_absolute": ScenarioDesign(
        name="hazard_filling_absolute",
        family="hazard_filling_ensemble",
        description="SUPPLEMENTARY distorted counterpart to hazard_filling: "
                    "space-filling subsample uniform in absolute hazard-magnitude "
                    "space (over-represents hazard extremes in the search "
                    "ensemble).",
        resample_per_eval=False,
        selection="hazard_fill",
        master_kind="forcing",
        n_forcing_profiles=_MASTER_N_FORCING,
        realizations_per_profile=_MASTER_REALS_PER_PROFILE,
        n_realizations=_MASTER_SUBSET_N,
        realization_years=_MASTER_YEARS,
        subset_seed=0,
        selector="lhs_nn",
        selector_space="abs",
        notes="Supplementary arm (run with fewer seeds/draws than the main "
              "designs). Same master pool, axes, and N as hazard_filling; differs "
              "ONLY in the selection space: LHS+nearest-neighbor in absolute "
              "magnitude space rather than empirical-CDF/rank space. This "
              "deliberately distorts the search measure toward uniform coverage "
              "of hazard MAGNITUDE (rare-but-severe scenarios over-represented "
              "relative to frequency), in contrast to hazard_filling, which is "
              "marginally representative. The uniform-coverage rationale borrows "
              "from the bottom-up/scenario-neutral tradition but is NOT decision "
              "scaling (that samples for post-hoc evaluation and re-imposes "
              "likelihoods; it never distorts the search). Objectives are held "
              "fixed across both arms; the held-out re-evaluation is the common "
              "comparison basis. Stages hazfill_abs_5yr_n64_s0.",
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
