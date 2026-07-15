"""
test_design_registries.py - Tests for the two run-axis registries.

Covers the scenario-design registry (src/scenario_designs.py), the MOEA-config
registry (src/moea_config.py), their resolvers, and the two-axis output-path
helper (config.run_output_dir). These are pure-Python and fast (no simulation).

The scenario-design tests here are not incidental coverage: two of them
(``test_pools_are_iid_sampled``, ``test_du_candidate_pool_has_one_realization_per_theta``)
guard the statistical control the whole cross-design comparison rests on, and
nothing else in the pipeline would fail if that control were broken.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scengen.seeds import design_seed

from src.scenario_designs import (
    SCENARIO_DESIGNS,
    SCENARIO_YEARS,
    SEARCH_ENSEMBLE_N,
    SEED_ROOT,
    ScenarioDesign,
    assert_iid_pools,
    assert_seed_domains_disjoint,
    campaign_designs,
    get_scenario_design,
    list_scenario_designs,
)
from src.moea_config import (
    MOEA_CONFIGS,
    MOEAConfig,
    get_moea_config,
    list_moea_configs,
)


# ---------------------------------------------------------------------------
# Scenario designs
# ---------------------------------------------------------------------------

# The manuscript campaign is three designs, all stationary-population. Deep
# uncertainty enters only via E_test (src/etest.py), not as a search design.
CAMPAIGN_DESIGNS = {
    "historic",
    "fixed_probabilistic",
    "hazard_filling_stationary",
}
# Retained, fully wired, but not in the main campaign: the DU-forced designs, the
# resampled design, the rank-space stationary hazard variant, and the scaling
# stand-in. Campaign membership is the config-level switch (the `campaign` flag);
# no construction code is deleted.
NON_CAMPAIGN_DESIGNS = {
    "resampled_probabilistic",
    "input_stratified",
    "hazard_filling_du",
    "hazard_filling_stationary_cdf",
    "hazard_filling_absolute",
    "scaling_stationary",
}

# Resolve without reading any staged data: `historic` from a static preset, and
# the supplemental scaling stand-in from the `kn_{Y}yr_n{N}` slug grammar.
WIRED_DESIGNS = {"historic", "scaling_stationary"}

# Every other design GENERATES its own realizations in workflow step 02 (and
# selects in step 03 for hazard-filling), so resolution raises a clear
# NotImplementedError until its data is staged.
STAGING_REQUIRED_DESIGNS = (
    CAMPAIGN_DESIGNS | NON_CAMPAIGN_DESIGNS
) - WIRED_DESIGNS


def _stage(root: Path, slug: str, n: int, years: int, **extra) -> None:
    """Stage a fake ensemble directory (``_meta.json`` only) resolvable by slug."""
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    meta = {"slug": slug, "n_realizations": n, "realization_years": years}
    meta.update(extra)
    (d / "_meta.json").write_text(json.dumps(meta))


def test_all_designs_present_and_resolvable():
    assert set(SCENARIO_DESIGNS) == CAMPAIGN_DESIGNS | NON_CAMPAIGN_DESIGNS
    for name in SCENARIO_DESIGNS:
        d = get_scenario_design(name)
        assert isinstance(d, ScenarioDesign)
        assert d.name == name


def test_campaign_designs_are_the_three():
    assert set(campaign_designs()) == CAMPAIGN_DESIGNS


def test_list_scenario_designs_sorted():
    assert list_scenario_designs() == sorted(SCENARIO_DESIGNS)


def test_unknown_design_raises():
    with pytest.raises(KeyError):
        get_scenario_design("does_not_exist")


def test_historic_resolves_to_single_trace():
    spec = get_scenario_design("historic").resolve_search_spec()
    assert spec.is_ensemble is False
    assert spec.n_realizations == 1


# -- the two invariants the control depends on ------------------------------

def test_pools_are_iid_sampled():
    """Only ``input_stratified`` may sample theta by LHS.

    The cross-design control is distributional equivalence: a uniform random
    size-N subset of an i.i.d. pool has exactly the joint law of N fresh i.i.d.
    draws, which is what makes ``fixed_probabilistic`` the EXACT statistical
    control for ``hazard_filling_stationary`` -- they differ only in the
    selection rule applied to the same population law.

    A random subset of an LHS design is NOT i.i.d. So if anyone "improves" a
    candidate pool to LHS sampling because it fills the space better, the
    control is silently void and NOTHING else in the pipeline would fail. Hence
    this test.
    """
    assert_iid_pools()  # the import-time assertion, exercised explicitly
    for name, d in SCENARIO_DESIGNS.items():
        expected = "lhs" if d.construction == "lhs_theta" else "iid"
        assert d.theta_sampler == expected, (
            f"'{name}' has theta_sampler={d.theta_sampler!r}, expected {expected!r}"
        )


def test_du_candidate_pool_has_one_realization_per_theta():
    """A DU candidate pool must draw an independent theta per realization.

    R > 1 would make realizations sharing a theta dependent, breaking the i.i.d.
    property of the pool from the other side.
    """
    for name, d in SCENARIO_DESIGNS.items():
        if d.construction == "hazard_fill" and d.population == "du_forced":
            assert d.realizations_per_profile == 1, f"'{name}' has R != 1"


def test_seed_domains_do_not_collide():
    """No two (design, draw) pairs -- nor any design and E_test -- share a seed.

    Now that every design GENERATES rather than selecting indices from shared
    data, two designs on the same seed would produce correlated realizations,
    reintroducing exactly the confound the per-design architecture removes.
    """
    assert_seed_domains_disjoint(max_draws=64)


def test_every_design_has_a_seed_domain():
    for name, d in SCENARIO_DESIGNS.items():
        if d.construction == "preset":
            continue
        assert d.seed_domain, f"'{name}' has no seed_domain"


def test_design_seed_is_deterministic_and_draw_dependent():
    d = get_scenario_design("fixed_probabilistic")
    assert d.generation_seed(3) == d.generation_seed(3)
    assert d.generation_seed(3) != d.generation_seed(4)
    assert d.generation_seed(0) == design_seed(SEED_ROOT, "fixed", 0)


# -- populations and construction -------------------------------------------

def test_campaign_search_population_is_stationary():
    """Campaign designs search only the stationary population.

    Deep uncertainty enters solely through the held-out test ensemble
    (src/etest.py), which makes the re-evaluation a generalization test. The
    du_forced population is used only by retained non-campaign designs and E_test.
    """
    pops = {d.population for d in SCENARIO_DESIGNS.values() if d.campaign}
    assert pops == {"historic", "stationary"}


def test_matched_designs_share_one_n_and_l():
    """The selection comparison requires a common (N, L).

    If L differed across designs the selection rule would be confounded with
    record length; if N differed, per-evaluation cost would differ and equal-NFE
    would no longer coincide with equal-scenario-years.
    """
    matched = CAMPAIGN_DESIGNS - {"historic"}
    for name in matched:
        d = get_scenario_design(name)
        assert d.n_realizations == SEARCH_ENSEMBLE_N, f"'{name}' N != {SEARCH_ENSEMBLE_N}"
        assert d.realization_years == SCENARIO_YEARS, f"'{name}' L != {SCENARIO_YEARS}"


def test_input_stratified_allocates_n_as_n_theta_times_r():
    d = get_scenario_design("input_stratified")
    assert d.n_theta_profiles * d.realizations_per_profile == d.n_realizations


def test_only_hazard_filling_and_resample_own_a_pool():
    """Hazard-filling is the only *selecting* design, because hazard coordinates
    are emergent and cannot be prescribed at generation."""
    with_pool = {n for n, d in SCENARIO_DESIGNS.items() if d.pool_slug(0) is not None}
    assert with_pool == {
        "resampled_probabilistic",
        "hazard_filling_stationary",
        "hazard_filling_stationary_cdf",
        "hazard_filling_du",
        "hazard_filling_absolute",
    }


def test_du_hazard_designs_share_one_candidate_pool_per_draw():
    """The CDF and absolute arms differ ONLY in selector space, so within a draw
    the pool is generated once and both select from it."""
    du = get_scenario_design("hazard_filling_du")
    absolute = get_scenario_design("hazard_filling_absolute")
    assert du.pool_slug(0) == absolute.pool_slug(0)
    assert du.selector_space == "cdf"
    assert absolute.selector_space == "abs"
    # ...and they get the SAME anchor plan, so the only difference is the
    # normalization geometry the anchors snap into.
    assert du.selector_seed(0) == absolute.selector_seed(0)


def test_pools_are_redrawn_per_draw():
    """A draw re-rolls EVERYTHING random about building the ensemble, pool included.

    If the pool were pinned across draws, a hazard-filling draw would vary only its
    LHS anchor plan while a ``fixed_probabilistic`` draw re-rolls its entire sample.
    The two between-draw variances would then not be commensurable, and hazard-filling
    would look more stable BY CONSTRUCTION rather than as a finding.
    """
    for name in ("hazard_filling_du", "hazard_filling_stationary", "resampled_probabilistic"):
        d = get_scenario_design(name)
        assert d.pool_slug(0) != d.pool_slug(1), f"'{name}' pins its pool across draws"
        assert d.generation_seed(0) != d.generation_seed(1)


def test_stationary_and_du_pools_are_distinct():
    stat = get_scenario_design("hazard_filling_stationary")
    du = get_scenario_design("hazard_filling_du")
    assert stat.pool_slug(0) != du.pool_slug(0)
    assert stat.generation_seed(0) != du.generation_seed(0)
    assert stat.selector_seed(0) != du.selector_seed(0)


def test_no_simulated_annealing_selector():
    """The selector is deterministic LHS + nearest-neighbor snap. K draws vary
    the anchor plan, not an annealer."""
    for d in SCENARIO_DESIGNS.values():
        if d.construction == "hazard_fill":
            assert d.selector == "lhs_nn"


def test_resample_flag_only_on_resampled():
    resampling = {n for n, d in SCENARIO_DESIGNS.items() if d.resample_per_eval}
    assert resampling == {"resampled_probabilistic"}


def test_hazard_image_only_for_hazard_filling():
    """Streaming the hazard image costs an SSI-6 fit + POT pass per realization;
    it is pure waste for designs that never subsample."""
    needs = {n for n, d in SCENARIO_DESIGNS.items() if d.needs_hazard_image}
    assert needs == {
        "hazard_filling_stationary",
        "hazard_filling_stationary_cdf",
        "hazard_filling_du",
        "hazard_filling_absolute",
    }


# -- slugs and resolution ---------------------------------------------------

def test_slugs_key_on_draw_not_seed():
    d = get_scenario_design("fixed_probabilistic")
    assert d.search_ensemble_slug(0).endswith("_d0")
    assert d.search_ensemble_slug(2).endswith("_d2")
    assert d.search_ensemble_slug(0) != d.search_ensemble_slug(2)


def test_hazard_filling_slugs_distinguish_population_and_space():
    # Campaign stationary design is now ABSOLUTE space -> "hazfill_stat_abs_";
    # the rank-space sensitivity keeps the plain "hazfill_stat_" stem.
    assert get_scenario_design("hazard_filling_stationary").search_ensemble_slug(0).startswith("hazfill_stat_abs_")
    assert get_scenario_design("hazard_filling_stationary_cdf").search_ensemble_slug(0).startswith("hazfill_stat_")
    assert not get_scenario_design("hazard_filling_stationary_cdf").search_ensemble_slug(0).startswith("hazfill_stat_abs_")
    assert get_scenario_design("hazard_filling_du").search_ensemble_slug(0).startswith("hazfill_du_")
    assert get_scenario_design("hazard_filling_absolute").search_ensemble_slug(0).startswith("hazfill_du_abs_")


def test_stationary_hazard_designs_share_one_candidate_pool_and_anchor_plan():
    """The absolute (campaign) and rank-space (sensitivity) stationary hazard
    designs differ ONLY in selector space, so within a draw they share the pool
    and the anchor plan -- the cleanest possible paired sensitivity."""
    abs_design = get_scenario_design("hazard_filling_stationary")
    cdf_design = get_scenario_design("hazard_filling_stationary_cdf")
    assert abs_design.pool_slug(0) == cdf_design.pool_slug(0)
    assert abs_design.selector_seed(0) == cdf_design.selector_seed(0)
    assert abs_design.selector_space == "abs"
    assert cdf_design.selector_space == "cdf"


@pytest.mark.parametrize("name", sorted(STAGING_REQUIRED_DESIGNS))
def test_staging_required_designs_raise_when_not_staged(name, tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    with pytest.raises(NotImplementedError):
        get_scenario_design(name).resolve_search_spec()


@pytest.mark.parametrize("name", ["fixed_probabilistic", "input_stratified"])
def test_generated_designs_resolve_own_staged_slug(name, tmp_path, monkeypatch):
    """Generated designs resolve their OWN staged ensemble -- no pool, no subset."""
    import config
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    d = get_scenario_design(name)
    slug = d.search_ensemble_slug(0)
    _stage(tmp_path, slug, d.n_realizations, d.realization_years)

    spec = d.resolve_search_spec(draw=0)
    assert spec.is_ensemble is True
    assert spec.inflow_type == slug
    assert spec.n_realizations == d.n_realizations
    assert spec.realization_years == d.realization_years
    assert d.pool_slug(0) is None  # it generates; it does not select


def test_hazard_filling_resolves_reduced_ensemble_per_draw(tmp_path, monkeypatch):
    """Step 03 stages one reduced ensemble per draw; resolution reads only its meta."""
    import config
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    d = get_scenario_design("hazard_filling_du")

    slug1 = d.search_ensemble_slug(draw=1)
    _stage(tmp_path, slug1, d.n_realizations, d.realization_years)

    assert d.resolve_search_spec(draw=1).inflow_type == slug1
    with pytest.raises(NotImplementedError):
        d.resolve_search_spec(draw=2)  # that draw is not staged


def test_resampled_probabilistic_resolves_to_its_own_pool(tmp_path, monkeypatch):
    """The pool belongs to the design -- it is not a shared master."""
    import config
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    d = get_scenario_design("resampled_probabilistic")
    pool = d.pool_slug(0)
    assert pool.startswith("respool_")
    _stage(tmp_path, pool, d.pool_size, d.realization_years)

    spec = d.resolve_search_spec()
    assert spec.resample_per_eval is True
    assert spec.resample_size == d.n_realizations
    assert spec.n_realizations == d.pool_size  # the POOL, drawn from per eval


def test_draw_replication_structurally_rejected():
    """historic has no ensemble to redraw."""
    with pytest.raises(ValueError):
        get_scenario_design("historic").resolve_search_spec(draw=1)


def test_resolution_is_a_pure_lookup(tmp_path, monkeypatch):
    """Resolving must not generate anything.

    Construction lives in workflow step 02/03. If resolution generated, importing
    config would do RNG draws and bulk I/O on every process -- including every
    Borg worker.
    """
    import config
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    d = get_scenario_design("fixed_probabilistic")
    with pytest.raises(NotImplementedError):
        d.resolve_search_spec()
    assert not list(tmp_path.iterdir()), "resolution wrote to the staging dir"


# ---------------------------------------------------------------------------
# MOEA configs
# ---------------------------------------------------------------------------

def test_moea_configs_present_and_resolvable():
    assert {"smoke", "production"} <= set(MOEA_CONFIGS)
    for name in MOEA_CONFIGS:
        c = get_moea_config(name)
        assert isinstance(c, MOEAConfig)
        assert c.name == name


def test_list_moea_configs_sorted():
    assert list_moea_configs() == sorted(MOEA_CONFIGS)


def test_unknown_moea_config_raises():
    with pytest.raises(KeyError):
        get_moea_config("does_not_exist")


def test_smoke_is_fully_specified():
    c = get_moea_config("smoke")
    assert c.n_islands is not None
    assert c.max_evaluations is not None
    assert c.runtime_frequency is not None
    assert c.total_ntasks_mpi == 1 + c.n_islands * (c.n_workers_per_island + 1)


def test_production_numbers_are_tbd():
    c = get_moea_config("production")
    assert c.max_evaluations is None
    assert c.n_islands is None
    assert c.total_ntasks_mpi is None  # unset workers/islands -> None


def test_max_time_seconds_conversion():
    assert MOEAConfig(name="x", max_time_hours=2).max_time_seconds == 7200
    assert MOEAConfig(name="x").max_time_seconds is None


# ---------------------------------------------------------------------------
# Output-path helper (two-axis)
# ---------------------------------------------------------------------------

def test_run_output_dir_layout(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    p = config.run_output_dir("hazard_filling_du", "ffmp_obj7_sal", "sets")
    assert p == tmp_path / "hazard_filling_du" / "ffmp_obj7_sal" / "sets"
    assert p.is_dir()


def test_figure_dir_for_stable_vs_exploratory(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(config, "FIG_EXPLORATORY_DIR", tmp_path / "_exploratory")
    stable = config.figure_dir_for("historic", "ffmp_obj7", "pareto")
    assert stable == tmp_path / "historic" / "ffmp_obj7" / "pareto"
    expl = config.figure_dir_for("historic", "ffmp_obj7", "made_up_kind")
    assert expl == tmp_path / "_exploratory" / "historic" / "ffmp_obj7" / "made_up_kind"
