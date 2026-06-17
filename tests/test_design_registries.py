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
    "historic", "fixed_probabilistic", "fixed_probabilistic_long",
    "resampled_probabilistic", "input_stratified", "hazard_filling",
    "smoke_ensemble",
}


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


def test_smoke_ensemble_resolves_to_ensemble():
    spec = get_scenario_design("smoke_ensemble").resolve_search_spec()
    assert spec.is_ensemble is True
    assert spec.n_realizations == 5


@pytest.mark.parametrize("name", sorted(EXPECTED_DESIGNS - {"historic", "smoke_ensemble"}))
def test_method_designs_not_wired_yet(name):
    """The five manuscript designs raise until their construction is decided."""
    with pytest.raises(NotImplementedError):
        get_scenario_design(name).resolve_search_spec()


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
