"""
test_design_registries.py - Tests for the two run-axis registries.

Covers the scenario-design registry (src/scenario_designs.py), the MOEA-config
registry (src/moea_config.py), their resolvers, and the two-axis output-path
helper (config.run_output_dir). These are pure-Python and fast (no simulation).
"""

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.scenario_designs import (
    SCENARIO_DESIGNS,
    ScenarioDesign,
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

EXPECTED_DESIGNS = {
    "historic", "fixed_probabilistic_short", "fixed_probabilistic_long",
    "resampled_probabilistic", "input_stratified", "hazard_filling",
    "hazard_filling_absolute",
}

# Designs whose search ensemble is wired and resolves to an EnsembleSpec
# directly (no offline staging step needed to resolve — the kn slug parses
# from its name, or it is a static preset).
WIRED_DESIGNS = {
    "historic",
    "fixed_probabilistic_short", "fixed_probabilistic_long",
    "resampled_probabilistic",
}

# Wired, but resolution requires an offline Step-1/Step-2 staging step first (the forcing master
# and/or the reduced ensemble). Raises a clear NotImplementedError until then.
#   - hazard_filling(_absolute): the scengen subsample step stages the reduced ensemble.
#   - input_stratified: needs the shared forcing master (forcing_profiles.npz) staged.
STAGING_REQUIRED_DESIGNS = {"hazard_filling", "hazard_filling_absolute", "input_stratified"}

# The two designs that subsample the shared CMIP6-forced master (methods §3.2).
FORCING_MASTER_DESIGNS = {"hazard_filling", "hazard_filling_absolute", "input_stratified"}


def test_all_designs_present_and_resolvable():
    assert EXPECTED_DESIGNS <= set(SCENARIO_DESIGNS)
    for name in SCENARIO_DESIGNS:
        d = get_scenario_design(name)
        assert isinstance(d, ScenarioDesign)
        assert d.name == name


def test_list_scenario_designs_sorted():
    assert list_scenario_designs() == sorted(SCENARIO_DESIGNS)


def test_unknown_design_raises():
    with pytest.raises(KeyError):
        get_scenario_design("does_not_exist")


def test_historic_resolves_to_single_trace():
    spec = get_scenario_design("historic").resolve_search_spec()
    assert spec.preset_name == "historic_single"
    assert spec.is_ensemble is False


@pytest.mark.parametrize(
    "name,n_real,n_years",
    [("fixed_probabilistic_short", 10, 5), ("fixed_probabilistic_long", 2, 25)],
)
def test_fixed_probabilistic_designs_resolve_to_kn_ensemble(name, n_real, n_years):
    """The fixed probabilistic designs resolve to a directly generated KN ensemble."""
    d = get_scenario_design(name)
    spec = d.resolve_search_spec()
    assert spec.is_ensemble is True
    assert spec.n_realizations == n_real
    assert spec.realization_years == n_years
    assert spec.inflow_type == f"kn_{n_years}yr_n{n_real}"
    # Generation and resolution name the same staged slug.
    assert d.kn_ensemble_slug() == spec.inflow_type


def test_fixed_probabilistic_short_and_long_equal_scenario_years():
    short = get_scenario_design("fixed_probabilistic_short")
    long = get_scenario_design("fixed_probabilistic_long")
    assert short.n_realizations * short.realization_years == \
        long.n_realizations * long.realization_years


def test_fixed_probabilistic_multidraw_not_wired():
    with pytest.raises(NotImplementedError):
        get_scenario_design("fixed_probabilistic_short").resolve_search_spec(draw=1)


def test_resampled_probabilistic_resolves_to_master_pool():
    """Resampled design resolves to the master-pool spec marked for per-eval resampling."""
    d = get_scenario_design("resampled_probabilistic")
    spec = d.resolve_search_spec()
    # Staged ensemble is the master POOL (master_pool_size), not the draw size.
    assert d.kn_ensemble_slug() == "kn_5yr_n50"
    assert spec.inflow_type == "kn_5yr_n50"
    assert spec.n_realizations == 50            # full pool
    assert spec.resample_per_eval is True
    assert spec.resample_size == 10             # drawn per evaluation


def test_resampled_per_eval_draw_is_a_distinct_subset():
    """The per-eval hook draws resample_size indices from the pool, varying per eval."""
    from src.simulation import _resampled_eval_spec
    pool_spec = get_scenario_design("resampled_probabilistic").resolve_search_spec()
    s1 = _resampled_eval_spec(pool_spec, eval_count=1)
    s2 = _resampled_eval_spec(pool_spec, eval_count=2)
    # Correct size, valid subset of the pool, no resampling flag carried into use.
    assert s1.n_realizations == 10
    assert set(s1.realization_indices) <= set(pool_spec.realization_indices)
    assert len(set(s1.realization_indices)) == 10  # without replacement
    # Different evaluations draw different subsets; same eval is reproducible.
    assert s1.realization_indices != s2.realization_indices
    s1_again = _resampled_eval_spec(pool_spec, eval_count=1)
    assert s1.realization_indices == s1_again.realization_indices


def test_every_design_is_wired():
    """No design is left with an un-decided construction (all raise only pending staging)."""
    assert EXPECTED_DESIGNS - WIRED_DESIGNS - STAGING_REQUIRED_DESIGNS == set()


@pytest.mark.parametrize("name", sorted(STAGING_REQUIRED_DESIGNS))
def test_staging_required_designs_raise_when_not_staged(name, tmp_path, monkeypatch):
    """Designs that need an offline staged master/ensemble raise a clear error until it exists."""
    import config
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)  # empty staging
    with pytest.raises(NotImplementedError):
        get_scenario_design(name).resolve_search_spec()


@pytest.mark.parametrize(
    "name,dims",
    [
        ("fixed_probabilistic_short", (5, 10)),    # stages the search ensemble
        ("fixed_probabilistic_long", (25, 2)),
        ("resampled_probabilistic", (5, 50)),      # stages the master pool
    ],
)
def test_kn_staged_dims(name, dims):
    assert get_scenario_design(name).kn_staged_dims() == dims


@pytest.mark.parametrize("name", sorted(FORCING_MASTER_DESIGNS))
def test_forcing_designs_share_one_master(name):
    """Forcing designs stage no stationary kn pool; their pool slug IS the shared forcing master."""
    d = get_scenario_design(name)
    assert d.master_kind == "forcing"
    assert d.kn_staged_dims() is None
    assert d.master_slug() is not None
    assert d.kn_ensemble_slug() == d.master_slug()
    # All forcing designs resolve to the same design-independent master (methods §3.2).
    masters = {get_scenario_design(n).master_slug() for n in FORCING_MASTER_DESIGNS}
    assert len(masters) == 1


def test_hazard_filling_resolves_from_staged_ensemble(tmp_path, monkeypatch):
    """When the final reduced ensemble is staged, hazard_filling resolves to it by slug.

    The scengen subsample step stages a standalone ensemble (HDF5s + _meta.json)
    under ``hazfill_{L}yr_n{N}_s{seed}``; resolution reads only the _meta.json,
    with no manifest and no realization-index override.
    """
    import json
    import config
    from src import ensembles

    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    d = get_scenario_design("hazard_filling")
    final_slug = d.hazard_filling_slug()
    out_dir = ensembles.staged_ensemble_dir(final_slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_meta.json").write_text(
        json.dumps({
            "slug": final_slug,
            "n_realizations": d.n_realizations,
            "realization_years": d.realization_years,
        })
    )

    spec = d.resolve_search_spec()
    assert spec.is_ensemble is True
    assert spec.inflow_type == final_slug            # loads the staged reduced HDF5
    assert spec.realization_indices == tuple(range(d.n_realizations))
    assert spec.n_realizations == d.n_realizations
    assert spec.realization_years == d.realization_years


def test_resample_flag_only_on_resampled():
    resampling = {n for n, d in SCENARIO_DESIGNS.items() if d.resample_per_eval}
    assert resampling == {"resampled_probabilistic"}


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
    p = config.run_output_dir("hazard_filling", "ffmp_obj7_sal", "sets")
    assert p == tmp_path / "hazard_filling" / "ffmp_obj7_sal" / "sets"
    assert p.is_dir()


def test_figure_dir_for_stable_vs_exploratory(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(config, "FIG_EXPLORATORY_DIR", tmp_path / "_exploratory")
    stable = config.figure_dir_for("historic", "ffmp_obj7", "pareto")
    assert stable == tmp_path / "historic" / "ffmp_obj7" / "pareto"
    expl = config.figure_dir_for("historic", "ffmp_obj7", "made_up_kind")
    assert expl == tmp_path / "_exploratory" / "historic" / "ffmp_obj7" / "made_up_kind"
