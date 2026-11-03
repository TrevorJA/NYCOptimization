"""tests/test_demand_source.py - NYC demand is always the time-dynamic series.

Guards the 2026-11-03 decision: NYC demand is Pywr-DRB's extrapolated diversion
series everywhere (per realization for ensembles, the historic-flow series for
the single trace) and a constant 800 MGD NYC demand is never used -- not in
search, re-evaluation, baselines or diagnostics. NJ demand is the constant
100 MGD Decree quantity. No env file may select a demand source.
"""

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
import src.simulation as sim  # noqa: E402
from src.ensembles import STAGED_ENSEMBLE_FILES  # noqa: E402


def test_nyc_demand_sources_are_time_dynamic():
    assert config.nyc_demand_source(True) == "custom"
    assert config.nyc_demand_source(False) == "historical"
    for is_ensemble in (True, False):
        config.assert_time_dynamic_nyc_demand(config.nyc_demand_source(is_ensemble))


def test_constant_nyc_demand_is_refused():
    with pytest.raises(ValueError, match="constant 800 MGD"):
        config.assert_time_dynamic_nyc_demand("constant_max")
    with pytest.raises(ValueError):
        config.assert_time_dynamic_nyc_demand("anything_else")


def test_model_builder_demand_options():
    """Every model build carries a time-dynamic NYC source and the constant NJ one."""
    for is_ensemble, expected in ((True, "custom"), (False, "historical")):
        opts = sim._demand_options(is_ensemble)
        assert opts["nyc_nj_demand_source"] == expected
        assert opts["nj_demand_source"] == "constant_max"


def test_nj_demand_is_the_decree_constant():
    assert config.NJ_DEMAND_SOURCE == "constant_max"


def test_no_env_file_selects_a_demand_source():
    envs = sorted((PROJECT_DIR / "workflow" / "envs").glob("*.env"))
    assert envs, "no env files found"
    offenders = [p.name for p in envs
                 if "DEMAND_SOURCE" in p.read_text(encoding="utf-8")]
    assert offenders == [], f"env files must not select a demand source: {offenders}"


def test_demand_series_is_part_of_the_staging_contract():
    assert STAGED_ENSEMBLE_FILES["diversion_nyc_extrapolated_mgd.hdf5"] == "04"
