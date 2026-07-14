"""
etest.py - The held-out test ensemble E_test: registry, sizing, and staging contract.

E_test is the MEASURING STICK of the study. Every scenario design's Pareto policies are re-simulated
on it, and the cross-design comparison is made there and nowhere else. It is an ``EnsembleSpec``,
never a ``ScenarioDesign``: it never enters search, it is never subsampled, and it is never a
control.

Construction (and why it is not what a search ensemble is)
---------------------------------------------------------
* **LHS over the FULL range of the deeply-uncertain factors** — the intrinsic harmonic forcing
  coordinates ``[m, r1, r2]`` (+3 CV axes when the variance axis is on). The envelope is
  deliberately WIDER than the search-side forcing box (:data:`E_TEST_BOUND_PCT` /
  :data:`E_TEST_MARGIN` vs ``config.ENSEMBLE_FORCING_BOUND_PCT`` / ``..._MARGIN``), so E_test is
  not a subset of the space any design searched in. A measuring stick contained in the search box
  can only ever reward interpolation.

  LHS is CORRECT here, and the i.i.d. requirement of the search side does NOT apply. That
  requirement exists solely because a uniform random subset of an i.i.d. pool is distributionally
  identical to i.i.d. draws, which is what makes ``fixed_probabilistic`` the exact control for
  ``hazard_filling_stationary`` (see ``src/scenario_designs.py``). E_test is never subsampled and is
  never a control, so nothing about it requires i.i.d. sampling — and space-filling is precisely
  what a measuring stick wants.

* **Many realizations per LHS point** (``R_test >> 1``), so natural variability is sampled *within*
  each deeply-uncertain state of the world (SOW). This is what makes the SOW-unit robustness metric
  (``src.robustness.satisficing_multivariate_sow``) computable: the R realizations sharing a theta
  are one SOW, collapsed in stage 1 before the Starr domain criterion is applied across SOWs. It is
  the Triangle-lineage standard (Herman et al. 2014; Trindade et al. 2017; Gold et al. 2022, 2023).

* ``N_test = N_theta_test x R_test``, and E_test is BY FAR the largest ensemble in the study
  (``N_test >> N = SEARCH_ENSEMBLE_N``). ``L_test >= L``.

* **One default construction; a second, opt-in.** ``kn`` (Kirsch-Nowak over the wide DU box) is THE
  test ensemble: it is the default variant, the only one the campaign requires, and what
  ``NYCOPT_REEVAL_ENSEMBLE_PRESET`` should point at. ``hmm`` swaps in the multi-site
  Gaussian-mixture HMM (Gold et al. 2024) on annual flows, which models interannual wet/dry
  sequencing as a fitted Markov process rather than inheriting it from the historical record as
  Kirsch does. It is an OPTIONAL, UNVALIDATED generator-structure sensitivity — registered and
  selectable (``--variant hmm``), built by nothing automatically, required by no workflow step.

Precedent for the construction: Trindade et al. (2017) (10,000 LHS DU samples x 1,000 flow
realizations); Gold et al. (2022) (1e6 SOWs); Kasprzyk et al. (2013) and Bartholomew & Kwakkel
(2020) (10,000 LHS SOWs); Quinn et al. (2020) (LHS over generator parameters).

SIZING IS PROVISIONAL
---------------------
``N_theta_test``, ``R_test`` and ``L_test`` are being set against the SU budget. They live HERE, as
single-source-of-truth env-overridable constants with clearly-flagged provisional defaults, and are
hardcoded nowhere else.

E_test needs no ``src.ensembles.PRESETS`` entry: ``_spec_from_staged_dir`` resolves any staged slug
carrying a ``_meta.json``. Point ``NYCOPT_REEVAL_ENSEMBLE_PRESET`` at the slug.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from scengen.seeds import design_seed

from src.scenario_designs import SCENARIO_YEARS, SEED_ROOT

###############################################################################
# Sizing (PROVISIONAL - being set against the SU budget)
###############################################################################

#: Number of LHS design points (deeply-uncertain states of the world) in E_test.
#: PROVISIONAL. Precision of the SOW-unit robustness metric is governed by THIS number, not by
#: N_test — see ``src.robustness.satisficing_multivariate_sow``.
E_TEST_N_THETA: int = int(os.environ.get("NYCOPT_ETEST_N_THETA", "200"))

#: Realizations generated per LHS point (R_test). MUST be > 1: with R = 1 there is no within-SOW
#: sample, so the SOW unit collapses onto the realization unit and the two metrics coincide.
#: PROVISIONAL.
E_TEST_R: int = int(os.environ.get("NYCOPT_ETEST_R", "10"))

#: Realization length L_test (years). Must be >= the search-side L (``SCENARIO_YEARS``), enforced
#: below. PROVISIONAL.
E_TEST_YEARS: int = int(os.environ.get("NYCOPT_ETEST_YEARS", str(SCENARIO_YEARS)))

#: Realizations per staged daily chunk. E_test is the largest ensemble in the study, so it is
#: chunked by default; the chunked re-eval path (``src.chunk_reeval``) exists for exactly this.
#: Must be a multiple of ``E_TEST_R`` so chunks align to forcing profiles (SOWs are never split).
E_TEST_CHUNK_SIZE: int = int(os.environ.get("NYCOPT_ETEST_CHUNK_SIZE", "500"))

#: Percentiles bounding each harmonic forcing parameter over the CMIP6 anchors. The search box uses
#: (5, 95) (the empirical 90% range, trimming outlier GCM runs); E_test uses the FULL empirical
#: range, widened by ``E_TEST_MARGIN``, so it strictly contains the search box.
E_TEST_BOUND_PCT: tuple[float, float] = (
    float(os.environ.get("NYCOPT_ETEST_BOUND_LO", "0.0")),
    float(os.environ.get("NYCOPT_ETEST_BOUND_HI", "100.0")),
)

#: Fractional widening of each harmonic-parameter range beyond the full CMIP6 span. PROVISIONAL.
E_TEST_MARGIN: float = float(os.environ.get("NYCOPT_ETEST_MARGIN", "0.25"))

#: The default E_test construction. ``"kn"`` is THE test ensemble of the campaign; every other
#: registered variant is an opt-in sensitivity that nothing builds automatically.
E_TEST_DEFAULT_VARIANT: str = "kn"

#: Selected E_test variant (which construction the generation step means by "E_test").
E_TEST_VARIANT: str = (
    os.environ.get("NYCOPT_ETEST_VARIANT", E_TEST_DEFAULT_VARIANT).strip()
    or E_TEST_DEFAULT_VARIANT
)


###############################################################################
# Variant registry
###############################################################################

@dataclass(frozen=True)
class ETestVariant:
    """One construction of the held-out test ensemble.

    Attributes:
        name: Registry key and ``--variant`` identifier.
        generator: Flow-generator family — ``"kn"`` (Kirsch monthly + Nowak) or ``"hmm"``
            (multi-site Gaussian-mixture HMM on annual flows + Nowak annual->monthly->daily).
        n_theta: N_theta_test — LHS design points (SOWs).
        realizations_per_theta: R_test — realizations per SOW. Must be > 1.
        realization_years: L_test.
        bound_pct: Percentiles bounding each harmonic parameter over the CMIP6 anchors.
        margin: Fractional widening of that box.
        chunk_size: Realizations per staged daily chunk (0 = single directory).
        seed_domain: Namespaced seed domain; must be one of the reserved ``etest:*`` domains, which
            are disjoint from every search-side domain by construction (``scengen.seeds``).
        campaign: Whether the manuscript campaign requires this variant. Exactly one variant is the
            campaign's E_test; the rest are opt-in sensitivities that nothing builds automatically.
        description: One-line human-readable summary.
    """

    name: str
    generator: str
    n_theta: int
    realizations_per_theta: int
    realization_years: int
    bound_pct: tuple[float, float]
    margin: float
    chunk_size: int
    seed_domain: str
    campaign: bool = False
    description: str = ""

    @property
    def n_realizations(self) -> int:
        """N_test = N_theta_test x R_test."""
        return self.n_theta * self.realizations_per_theta

    @property
    def slug(self) -> str:
        """Staged-ensemble slug, e.g. ``etest_kn_10yr_n2000``."""
        return f"etest_{self.generator}_{self.realization_years}yr_n{self.n_realizations}"

    @property
    def seed(self) -> int:
        """Generator root seed, namespaced to this variant's reserved seed domain."""
        return design_seed(SEED_ROOT, self.seed_domain, 0)


E_TEST_VARIANTS: dict[str, ETestVariant] = {
    # THE test ensemble. The default, the only one the campaign requires, and what
    # NYCOPT_REEVAL_ENSEMBLE_PRESET points at.
    "kn": ETestVariant(
        name="kn",
        generator="kn",
        n_theta=E_TEST_N_THETA,
        realizations_per_theta=E_TEST_R,
        realization_years=E_TEST_YEARS,
        bound_pct=E_TEST_BOUND_PCT,
        margin=E_TEST_MARGIN,
        chunk_size=E_TEST_CHUNK_SIZE,
        seed_domain="etest:kn",
        campaign=True,
        description="Kirsch-Nowak over the wide DU forcing box; LHS theta x R realizations.",
    ),
    # OPT-IN generator-structure sensitivity. NOT part of the campaign, NOT built by any workflow
    # step by default, and NOT validated as a hydrology product for this basin: the HMM is fitted on
    # 76 annual multi-site totals, and on the DRB record the fitted transition matrix is close to
    # memoryless, so it should not be described as adding interannual persistence until that is
    # checked. It also imposes the forcing differently (a monthly delta-change on the disaggregated
    # totals, since state-conditional HMM emissions have no analogue of the Kirsch eqs-10-11
    # log-moment adjustment), so it varies generator AND forcing mechanism together. Build it only
    # to ask "does the comparison survive a different flow generator", and say so when reporting.
    "hmm": ETestVariant(
        name="hmm",
        generator="hmm",
        n_theta=E_TEST_N_THETA,
        realizations_per_theta=E_TEST_R,
        realization_years=E_TEST_YEARS,
        bound_pct=E_TEST_BOUND_PCT,
        margin=E_TEST_MARGIN,
        chunk_size=E_TEST_CHUNK_SIZE,
        seed_domain="etest:hmm",
        campaign=False,
        description="OPT-IN sensitivity (not campaign): multi-site Gaussian-mixture HMM on annual "
                    "flows (Gold et al. 2024), same wide DU box, same LHS theta x R design.",
    ),
}


def campaign_etest_variant() -> ETestVariant:
    """Return the one variant the campaign's E_test is built from.

    Raises:
        AssertionError: If the registry does not declare exactly one campaign variant.
    """
    campaign = [v for v in E_TEST_VARIANTS.values() if v.campaign]
    assert len(campaign) == 1, (
        f"exactly one E_test variant must be the campaign's measuring stick; got "
        f"{[v.name for v in campaign]}"
    )
    return campaign[0]


def get_etest_variant(name: str) -> ETestVariant:
    """Resolve a variant name to its :class:`ETestVariant`.

    Args:
        name: A key of :data:`E_TEST_VARIANTS`.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return E_TEST_VARIANTS[name]
    except KeyError:
        raise KeyError(
            f"Unknown E_test variant '{name}'. Available: {list_etest_variants()}."
        ) from None


def list_etest_variants() -> list[str]:
    """Return the registered E_test variant names in sorted order."""
    return sorted(E_TEST_VARIANTS)


###############################################################################
# Invariants
###############################################################################

def assert_etest_contract() -> None:
    """Assert the properties E_test's role as the measuring stick depends on.

    Raises:
        AssertionError: If any variant violates the construction contract.
    """
    from src.scenario_designs import SCENARIO_DESIGNS

    campaign_etest_variant()  # exactly one campaign variant
    assert E_TEST_DEFAULT_VARIANT in E_TEST_VARIANTS, (
        f"default E_test variant '{E_TEST_DEFAULT_VARIANT}' is not registered."
    )
    assert E_TEST_VARIANTS[E_TEST_DEFAULT_VARIANT].campaign, (
        f"the default E_test variant '{E_TEST_DEFAULT_VARIANT}' must be the campaign one; every "
        f"other variant is an opt-in sensitivity that nothing builds automatically."
    )

    search_domains = {d.seed_domain for d in SCENARIO_DESIGNS.values() if d.seed_domain}
    for v in E_TEST_VARIANTS.values():
        assert v.realizations_per_theta > 1, (
            f"E_test variant '{v.name}' has R_test={v.realizations_per_theta}. R_test must be > 1: "
            f"with one realization per theta there is no within-SOW sample, the SOW unit collapses "
            f"onto the realization unit, and sat_multivariate_sow is not a distinct quantity."
        )
        assert v.realization_years >= SCENARIO_YEARS, (
            f"E_test variant '{v.name}' has L_test={v.realization_years} < the search L="
            f"{SCENARIO_YEARS}. Policies must be tested on records at least as long as they were "
            f"searched on."
        )
        assert v.seed_domain.startswith("etest:"), (
            f"E_test variant '{v.name}' must use a reserved 'etest:*' seed domain, got "
            f"'{v.seed_domain}'."
        )
        assert v.seed_domain not in search_domains, (
            f"E_test variant '{v.name}' shares seed domain '{v.seed_domain}' with a scenario "
            f"design. The held-out ensemble would not be held out (Bonham et al. 2024)."
        )
        if v.chunk_size:
            assert v.chunk_size % v.realizations_per_theta == 0, (
                f"E_test variant '{v.name}': chunk_size ({v.chunk_size}) must be a multiple of "
                f"R_test ({v.realizations_per_theta}) so a SOW is never split across chunks."
            )


assert_etest_contract()


def assert_staged_etest_contract(slug: str) -> dict:
    """Verify a STAGED E_test directory records the construction it claims.

    The registry can be right and the staged artifact still wrong (a stale directory generated
    before a sizing change, or one built by the search-side step 02, which samples theta i.i.d. with
    R = 1). Both failures are silent at re-eval time and would quietly void the SOW-unit metric, so
    the generator calls this immediately after staging.

    Args:
        slug: Staged-ensemble slug.

    Returns:
        The parsed ``_meta.json``.

    Raises:
        FileNotFoundError: If the slug is not staged.
        AssertionError: If the staged meta is not an LHS x R>1 forcing design.
    """
    from src.ensembles import staged_ensemble_dir

    meta_path = Path(staged_ensemble_dir(slug)) / "_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"E_test '{slug}' is not staged ({meta_path} missing).")
    meta = json.loads(meta_path.read_text())

    assert meta.get("theta_sampler") == "lhs", (
        f"Staged E_test '{slug}' records theta_sampler={meta.get('theta_sampler')!r}, expected "
        f"'lhs'. E_test is a space-filling LHS over the full DU box; an i.i.d. draw is the "
        f"search-side pool construction."
    )
    assert int(meta.get("realizations_per_profile", 0)) > 1, (
        f"Staged E_test '{slug}' records realizations_per_profile="
        f"{meta.get('realizations_per_profile')!r}, expected > 1. Without replication within a "
        f"theta there is no SOW to collapse, and the SOW-unit robustness metric is undefined."
    )
    assert str(meta.get("seed_domain", "")).startswith("etest:"), (
        f"Staged E_test '{slug}' records seed_domain={meta.get('seed_domain')!r}; E_test must be "
        f"generated from a reserved 'etest:*' seed stream, or the held-out re-evaluation is not "
        f"held out."
    )
    return meta
