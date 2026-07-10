"""
test_design_registries.py - Tests for the two run-axis registries.

Covers the scenario-design registry (src/scenario_designs.py), the MOEA-config
registry (src/moea_config.py), their resolvers, and the two-axis output-path
helper (config.run_output_dir). These are pure-Python and fast (no simulation).
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.scenario_designs import (
    LONG_RECORD_YEARS,
    SCENARIO_DESIGNS,
    SCENARIO_YEARS,
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

# Designs whose search ensemble resolves to an EnsembleSpec without any offline
# staging step (static preset only — every ensemble design now draws from a
# staged forcing master, methods §2.1).
WIRED_DESIGNS = {"historic"}

# Wired, but resolution requires the offline Step-02 (forcing master) and, for
# hazard_filling*, Step-03 (reduced subsample) staging first. Raises a clear
# NotImplementedError until then.
STAGING_REQUIRED_DESIGNS = {
    "fixed_probabilistic_short", "fixed_probabilistic_long",
    "resampled_probabilistic",
    "hazard_filling", "hazard_filling_absolute", "input_stratified",
}

# Every manuscript ensemble design is forcing-master-backed.
FORCING_MASTER_DESIGNS = STAGING_REQUIRED_DESIGNS

# The short-window designs that must share ONE design-independent master
# (master_{L}yr_n{N_M}) so cross-design differences are attributable to
# selection alone. fixed_probabilistic_long uses its own long master
# (master_{L'}yr_n{N'}) of the same construction.
SHARED_MASTER_DESIGNS = FORCING_MASTER_DESIGNS - {"fixed_probabilistic_long"}


def _stage_fake_forcing_master(root: Path, slug: str, n: int, years: int,
                               with_daily: bool = True) -> None:
    """Stage a tiny fake single-dir forcing master (meta + optional daily HDF5s)."""
    import numpy as np
    import pandas as pd

    out = root / slug
    out.mkdir(parents=True, exist_ok=True)
    if with_daily:
        from synhydro.core.ensemble import Ensemble
        dates = pd.date_range("2000-01-01", periods=8, freq="D")
        data = {
            k: pd.DataFrame(
                {"siteA": np.full(8, float(k)), "siteB": np.full(8, 100.0 + k)},
                index=dates,
            )
            for k in range(n)
        }
        for fname in ("gage_flow_mgd.hdf5", "catchment_inflow_mgd.hdf5"):
            Ensemble(data).to_hdf5(str(out / fname))
    (out / "_meta.json").write_text(json.dumps({
        "slug": slug, "n_realizations": n, "realization_years": years,
        "source_kind": "synhydro_kn",
    }))


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


def test_scenario_length_is_single_sourced():
    """SCENARIO_YEARS is the one scenario-length constant: every
    short-window design uses it; the long design keeps its own L'; config and
    supplemental_config consume the same value."""
    for name in sorted(SHARED_MASTER_DESIGNS):
        assert get_scenario_design(name).realization_years == SCENARIO_YEARS
    assert get_scenario_design("fixed_probabilistic_long").realization_years \
        == LONG_RECORD_YEARS
    import config
    assert config.SCENARIO_YEARS == SCENARIO_YEARS
    import supplemental_config as scfg
    if not scfg.ENS_SMOKE:
        assert scfg.ENS_REALIZATION_YEARS == SCENARIO_YEARS


def test_fixed_probabilistic_short_and_long_equal_scenario_years():
    short = get_scenario_design("fixed_probabilistic_short")
    long = get_scenario_design("fixed_probabilistic_long")
    assert short.n_realizations * short.realization_years == \
        long.n_realizations * long.realization_years


@pytest.mark.parametrize(
    "name,staged_n",
    [("fixed_probabilistic_short", 24), ("fixed_probabilistic_long", 4)],
)
def test_fixed_probabilistic_materializes_random_subset(name, staged_n, tmp_path,
                                                        monkeypatch):
    """Fixed probabilistic designs draw a random index subset from the staged
    forcing master and materialize it as a standalone reduced ensemble."""
    import config
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    d = get_scenario_design(name)
    _stage_fake_forcing_master(tmp_path, d.master_slug(), n=staged_n,
                               years=d.realization_years)

    spec = d.resolve_search_spec(draw=0)
    assert spec.is_ensemble is True
    assert spec.n_realizations == d.n_realizations
    assert spec.realization_years == d.realization_years
    assert spec.inflow_type == \
        f"rand_{d.realization_years}yr_n{d.n_realizations}_s{d.subset_seed or 0}"
    # Provenance: the reduced ensemble records the master and the drawn indices.
    meta = json.loads((tmp_path / spec.inflow_type / "_meta.json").read_text())
    assert meta["source_master"] == d.master_slug()
    gids = meta["global_realization_ids"]
    assert len(set(gids)) == d.n_realizations          # without replacement
    assert all(0 <= g < staged_n for g in gids)
    # Idempotent: a second resolve reuses the staged reduced ensemble.
    assert d.resolve_search_spec(draw=0).inflow_type == spec.inflow_type


def test_multidraw_selects_distinct_deterministic_subsets(tmp_path, monkeypatch):
    """Draw k stages its own reduced ensemble (seed = subset_seed + k) with a
    different deterministic index subset."""
    import config
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    d = get_scenario_design("fixed_probabilistic_short")
    _stage_fake_forcing_master(tmp_path, d.master_slug(), n=24,
                               years=d.realization_years)

    spec0 = d.resolve_search_spec(draw=0)
    spec1 = d.resolve_search_spec(draw=1)
    assert spec0.inflow_type != spec1.inflow_type
    assert spec0.inflow_type.endswith("_s0") and spec1.inflow_type.endswith("_s1")
    gids = {
        s.inflow_type: json.loads(
            (tmp_path / s.inflow_type / "_meta.json").read_text()
        )["global_realization_ids"]
        for s in (spec0, spec1)
    }
    assert gids[spec0.inflow_type] != gids[spec1.inflow_type]
    # Deterministic: re-resolving a draw reproduces the same subset.
    assert json.loads(
        (tmp_path / d.resolve_search_spec(draw=1).inflow_type / "_meta.json").read_text()
    )["global_realization_ids"] == gids[spec1.inflow_type]


def test_resampled_probabilistic_resolves_to_master_pool(tmp_path, monkeypatch):
    """Resampled design's per-eval POOL is the shared forcing master itself,
    marked for per-evaluation resampling."""
    import config
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    d = get_scenario_design("resampled_probabilistic")
    _stage_fake_forcing_master(tmp_path, d.master_slug(), n=24,
                               years=d.realization_years, with_daily=False)

    spec = d.resolve_search_spec()
    assert spec.inflow_type == d.master_slug()   # pool IS the shared master
    assert spec.n_realizations == 24             # full staged pool
    assert spec.resample_per_eval is True
    assert spec.resample_size == d.n_realizations  # drawn per evaluation


def test_resampled_per_eval_draw_is_a_distinct_subset(tmp_path, monkeypatch):
    """The per-eval hook draws resample_size indices from the pool, varying per eval."""
    import config
    from src.simulation import _resampled_eval_spec

    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    d = get_scenario_design("resampled_probabilistic")
    _stage_fake_forcing_master(tmp_path, d.master_slug(), n=24,
                               years=d.realization_years, with_daily=False)
    pool_spec = d.resolve_search_spec()
    s1 = _resampled_eval_spec(pool_spec, eval_count=1)
    s2 = _resampled_eval_spec(pool_spec, eval_count=2)
    # Correct size, valid subset of the pool, no resampling flag carried into use.
    assert s1.n_realizations == d.n_realizations
    assert set(s1.realization_indices) <= set(pool_spec.realization_indices)
    assert len(set(s1.realization_indices)) == d.n_realizations  # w/o replacement
    # Different evaluations draw different subsets; same eval is reproducible.
    assert s1.realization_indices != s2.realization_indices
    s1_again = _resampled_eval_spec(pool_spec, eval_count=1)
    assert s1.realization_indices == s1_again.realization_indices


@pytest.mark.parametrize("name", ["historic", "resampled_probabilistic"])
def test_draw_replication_structurally_rejected(name):
    """Designs with no fixed ensemble to redraw accept only draw=0."""
    with pytest.raises(ValueError, match="draw"):
        get_scenario_design(name).resolve_search_spec(draw=1)


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


def test_kn_staged_dims_only_for_stationary_standin():
    """Only the supplemental scaling design stages a direct stationary KN
    ensemble; every manuscript design stages a forcing master (or none)."""
    assert get_scenario_design("scaling_stationary").kn_staged_dims() is not None
    for name in sorted(EXPECTED_DESIGNS):
        assert get_scenario_design(name).kn_staged_dims() is None


@pytest.mark.parametrize("name", sorted(FORCING_MASTER_DESIGNS))
def test_forcing_designs_share_one_master(name):
    """Forcing designs stage no stationary kn pool; their pool slug IS a forcing master."""
    d = get_scenario_design(name)
    assert d.master_kind == "forcing"
    assert d.kn_staged_dims() is None
    assert d.master_slug() is not None
    assert d.kn_ensemble_slug() == d.master_slug()
    # All short-window designs — including the probabilistic controls — resolve
    # to the SAME design-independent master (methods §2.1/§3.2, review A1).
    shared = {get_scenario_design(n).master_slug() for n in SHARED_MASTER_DESIGNS}
    assert len(shared) == 1
    # The long design uses its own long master of the same construction.
    long_master = get_scenario_design("fixed_probabilistic_long").master_slug()
    assert long_master not in shared
    assert f"{LONG_RECORD_YEARS}yr" in long_master


def test_hazard_filling_resolves_from_staged_ensemble(tmp_path, monkeypatch):
    """When the final reduced ensemble is staged, hazard_filling resolves to it by slug.

    The scengen subsample step stages a standalone ensemble (HDF5s + _meta.json)
    under ``hazfill_{L}yr_n{N}_s{seed}``; resolution reads only the _meta.json,
    with no manifest and no realization-index override.
    """
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


def test_hazard_filling_draw_resolves_per_draw_slug(tmp_path, monkeypatch):
    """Draw k of hazard_filling loads the reduced ensemble staged with
    selector seed subset_seed + k."""
    import config
    from src import ensembles

    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    d = get_scenario_design("hazard_filling")
    slug1 = d.hazard_filling_slug(draw=1)
    assert slug1.endswith("_s1")
    out_dir = ensembles.staged_ensemble_dir(slug1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_meta.json").write_text(json.dumps({
        "slug": slug1,
        "n_realizations": d.n_realizations,
        "realization_years": d.realization_years,
    }))

    assert d.resolve_search_spec(draw=1).inflow_type == slug1
    with pytest.raises(NotImplementedError):
        d.resolve_search_spec(draw=2)  # that draw is not staged


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
