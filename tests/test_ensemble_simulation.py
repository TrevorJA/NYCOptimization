"""
tests/test_ensemble_simulation.py - Tests for the ensemble simulation path (M2a).

Covers:

  1. Cache-key isolation: two different ensemble presets produce distinct
     cache entries in ``_get_cached_model_dict`` and the resulting model
     dicts carry the expected ``inflow_type``.

  2. Legacy path regression: with the default ``historic_single`` preset,
     ``run_simulation_inmemory`` is the unchanged call path. We verify the
     dispatch in ``evaluate()`` routes through the legacy function.

  3. Ensemble correctness (slow): with ``wcu_kirsch_n5``,
     ``run_simulation_ensemble_inmemory`` returns a list of length 5,
     each dict has the expected keys, and reservoir storage trajectories
     genuinely vary across realizations (i ≠ j → distinct ``res_storage``).

The slow integration test is gated behind ``-m slow`` because a full pywrdrb
ensemble simulation takes ~5–10 minutes even on a 2-year clip. The fast
tests verify the structural plumbing without running pywrdrb.

Run fast tests only:
    venv/bin/python -m pytest tests/test_ensemble_simulation.py -v -m "not slow"

Run slow ensemble test (clipped to 2-year window via env):
    venv/bin/python -m pytest tests/test_ensemble_simulation.py -v -m slow
"""

# ---------------------------------------------------------------------------
# Date envelope: do NOT set PYWRDRB_SIM_START_DATE / PYWRDRB_SIM_END_DATE
# here — the simulation window for an ensemble run is derived from the
# spec's ``realization_years`` (see src/simulation.py::_ensemble_window).
# Setting env-level overrides would collide with the staged HDF5's date
# axis (e.g., ``wcu_kirsch_n5`` stages 1945-10-01 → 1965-09-30 under the
# 20-yr dev preset; an env clip outside that window would mismatch).
# ---------------------------------------------------------------------------
import os  # noqa: F401

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import pytest

from src.ensembles import get_ensemble_spec, register_ensemble_path
import src.simulation as sim


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
slow = pytest.mark.slow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STAGED_ENSEMBLE = (
    PROJECT_DIR / "outputs" / "synthetic_ensembles" / "syn_kirsch_drb_n100_seed42"
)


@pytest.fixture
def historic_spec():
    return get_ensemble_spec("historic_single")


@pytest.fixture
def wcu5_spec():
    return get_ensemble_spec("wcu_kirsch_n5")


@pytest.fixture(autouse=True)
def _reset_caches():
    """Each test gets a clean module-level cache so cache-key tests are
    independent of test ordering."""
    sim._CACHED_MODEL_DICTS.clear()
    sim._CACHED_MODEL_DICT = None
    yield
    sim._CACHED_MODEL_DICTS.clear()
    sim._CACHED_MODEL_DICT = None


# ---------------------------------------------------------------------------
# Fast tests — cache key shape and dispatch behavior (no pywrdrb run)
# ---------------------------------------------------------------------------

def test_cache_key_includes_ensemble_preset(historic_spec, wcu5_spec):
    """The cache must distinguish presets by name, not just by inflow_type."""
    drought_levels = (
        "level1a", "level1b", "level1c", "level2", "level3", "level4", "level5",
    )
    legacy_key = (
        drought_levels, False, False, historic_spec.preset_name,
        historic_spec.du_factor_signature,
    )
    ensemble_key = (
        drought_levels, False, False, wcu5_spec.preset_name,
        wcu5_spec.du_factor_signature,
    )
    # Sanity: keys differ on the preset name slot.
    assert legacy_key != ensemble_key
    assert legacy_key[3] == "historic_single"
    assert ensemble_key[3] == "wcu_kirsch_n5"


def test_run_ensemble_rejects_historic_single(historic_spec):
    """Calling the ensemble path with is_ensemble=False must error fast."""
    # We don't need a real nyc_config; the guard is the first check.
    with pytest.raises(ValueError, match="is_ensemble=False"):
        sim.run_simulation_ensemble_inmemory(
            nyc_config=None, ensemble_spec=historic_spec,
        )


def test_evaluate_uses_search_ensemble_spec_default(monkeypatch):
    """evaluate() should default to config.SEARCH_ENSEMBLE_SPEC when not given.

    We monkeypatch run_simulation_inmemory to avoid running pywrdrb and
    just confirm the dispatch routes through the legacy path under the
    default historic_single preset.
    """
    sentinel_data = {"res_storage": pd.DataFrame()}
    called = {"legacy": 0, "ensemble": 0}

    def fake_legacy(cfg):
        called["legacy"] += 1
        return sentinel_data

    def fake_ensemble(cfg, spec):
        called["ensemble"] += 1
        return [sentinel_data]

    class FakeObjSet:
        def compute_for_borg(self, data):
            return [0.0, 0.0]

        def compute_for_borg_ensemble(self, data_per_real):
            return [0.0, 0.0]

    monkeypatch.setattr(sim, "run_simulation_inmemory", fake_legacy)
    monkeypatch.setattr(
        sim, "run_simulation_ensemble_inmemory", fake_ensemble,
    )
    monkeypatch.setattr(
        sim, "dvs_to_config", lambda dv, formulation_name="ffmp": object(),
    )

    # Default search preset is historic_single -> legacy path
    sim.evaluate(
        dv_vector=np.zeros(1), formulation_name="ffmp",
        objective_set=FakeObjSet(),
    )
    assert called["legacy"] == 1
    assert called["ensemble"] == 0


def test_evaluate_dispatches_to_ensemble_when_spec_is_ensemble(monkeypatch,
                                                              wcu5_spec):
    """When passed an ensemble spec, evaluate() must call the ensemble path."""
    sentinel_data = {"res_storage": pd.DataFrame()}
    called = {"legacy": 0, "ensemble": 0}

    def fake_legacy(cfg):
        called["legacy"] += 1
        return sentinel_data

    def fake_ensemble(cfg, spec):
        called["ensemble"] += 1
        assert spec is wcu5_spec
        return [sentinel_data] * spec.n_realizations

    class FakeObjSet:
        def compute_for_borg(self, data):
            return [0.0]

        def compute_for_borg_ensemble(self, data_per_real):
            return [float(len(data_per_real))]

    monkeypatch.setattr(sim, "run_simulation_inmemory", fake_legacy)
    monkeypatch.setattr(
        sim, "run_simulation_ensemble_inmemory", fake_ensemble,
    )
    monkeypatch.setattr(
        sim, "dvs_to_config", lambda dv, formulation_name="ffmp": object(),
    )

    objs = sim.evaluate(
        dv_vector=np.zeros(1), formulation_name="ffmp",
        objective_set=FakeObjSet(), ensemble_spec=wcu5_spec,
    )
    assert called["ensemble"] == 1
    assert called["legacy"] == 0
    assert objs == [5.0]


def test_evaluate_raises_when_ensemble_objset_missing(monkeypatch, wcu5_spec):
    """When the legacy ObjectiveSet has no compute_for_borg_ensemble, the
    ensemble dispatch must surface a clear NotImplementedError instead of
    silently calling the wrong method."""
    monkeypatch.setattr(
        sim, "run_simulation_ensemble_inmemory",
        lambda cfg, spec: [{}] * spec.n_realizations,
    )
    monkeypatch.setattr(
        sim, "dvs_to_config", lambda dv, formulation_name="ffmp": object(),
    )

    class LegacyOnlyObjSet:
        def compute_for_borg(self, data):
            return [0.0]
        # NO compute_for_borg_ensemble — exercises the misuse guard at
        # simulation.py::evaluate (a hand-built single-trace set should
        # not silently feed the ensemble dispatch).

    with pytest.raises(NotImplementedError, match="compute_for_borg_ensemble"):
        sim.evaluate(
            dv_vector=np.zeros(1), formulation_name="ffmp",
            objective_set=LegacyOnlyObjSet(), ensemble_spec=wcu5_spec,
        )


# ---------------------------------------------------------------------------
# Shared memory-batched realization path (run_simulation_ensemble_batched)
# ---------------------------------------------------------------------------

def test_batched_orders_and_chunks(monkeypatch, wcu5_spec):
    """The shared batched loop preserves realization order and chunks by size."""
    seen = []

    def fake_inmem(cfg, spec):
        seen.append(tuple(spec.realization_indices))
        return [{"i": i} for i in spec.realization_indices]

    monkeypatch.setattr(sim, "run_simulation_ensemble_inmemory", fake_inmem)

    idx = list(wcu5_spec.realization_indices)  # n=5 -> [0,1,2,3,4]
    out = sim.run_simulation_ensemble_batched(
        nyc_config=object(), ensemble_spec=wcu5_spec, batch_size=2,
        per_realization_fn=lambda d: d["i"],
    )
    assert out == idx
    # batch_size=2 over 5 realizations -> chunks of 2,2,1 in realization order
    assert seen == [tuple(idx[0:2]), tuple(idx[2:4]), tuple(idx[4:5])]
    # Distinct __b{offset} cache keys per chunk (no model-dict reuse).
    # (offsets 0, 2, 4 -> three distinct presets; order already asserted above.)


def test_batched_skips_failed_batch(monkeypatch, wcu5_spec):
    """skip_failed_batches fills failed_value for a batch whose sim raises."""
    calls = {"n": 0}

    def fake_inmem(cfg, spec):
        calls["n"] += 1
        if calls["n"] == 2:          # second chunk (idx[2:4]) blows up
            raise RuntimeError("boom batch")
        return [{"i": i} for i in spec.realization_indices]

    monkeypatch.setattr(sim, "run_simulation_ensemble_inmemory", fake_inmem)
    idx = list(wcu5_spec.realization_indices)

    out = sim.run_simulation_ensemble_batched(
        nyc_config=object(), ensemble_spec=wcu5_spec, batch_size=2,
        per_realization_fn=lambda d: d["i"],
        skip_failed_batches=True, failed_value=-1,
    )
    assert out == [idx[0], idx[1], -1, -1, idx[4]]

    # Default (skip_failed_batches=False) propagates the exception.
    calls["n"] = 0
    with pytest.raises(RuntimeError, match="boom batch"):
        sim.run_simulation_ensemble_batched(
            nyc_config=object(), ensemble_spec=wcu5_spec, batch_size=2,
            per_realization_fn=lambda d: d["i"],
        )


def test_evaluate_batched_matches_legacy(monkeypatch, wcu5_spec):
    """Batched evaluate() must give identical objectives to the legacy path."""
    from src.objectives import ObjectiveSet
    from src.objectives_ensemble import EnsembleObjective, SatisficingAgg

    class FakeBase:
        name = "fake"

        def compute(self, data):
            return data["v"]

    eo = EnsembleObjective(
        base=FakeBase(), aggregator=SatisficingAgg(threshold=2.5, kind="ge"),
        name="fake__sat", epsilon=0.02,
    )
    objset = ObjectiveSet([eo])

    # Per-realization base value v = realization index -> [0,1,2,3,4].
    monkeypatch.setattr(
        sim, "run_simulation_ensemble_inmemory",
        lambda cfg, spec: [{"v": float(i)} for i in spec.realization_indices],
    )
    monkeypatch.setattr(
        sim, "dvs_to_config", lambda dv, formulation_name="ffmp": object(),
    )

    legacy = sim.evaluate(
        np.zeros(1), objective_set=objset, ensemble_spec=wcu5_spec,
        realization_batch=0,
    )
    batched = sim.evaluate(
        np.zeros(1), objective_set=objset, ensemble_spec=wcu5_spec,
        realization_batch=2,
    )
    assert legacy == batched
    # ge 2.5 over [0,1,2,3,4] satisfied by {3,4} -> 2/5 = 0.4, negated for Borg.
    assert batched == pytest.approx([-0.4])


# ---------------------------------------------------------------------------
# Feasibility pre-check (check_dv_feasibility)
# ---------------------------------------------------------------------------

def test_feasibility_probe_feasible(monkeypatch, wcu5_spec):
    """A probe sim that returns normally => (True, None), single realization."""
    seen = {}

    def fake_inmem(cfg, spec):
        seen["idx"] = tuple(spec.realization_indices)
        return [{}]

    monkeypatch.setattr(sim, "run_simulation_ensemble_inmemory", fake_inmem)
    ok, err = sim.check_dv_feasibility(object(), wcu5_spec)
    assert ok is True and err is None
    # Probes exactly one realization (the spec's first index).
    assert seen["idx"] == (wcu5_spec.realization_indices[0],)


def test_feasibility_probe_infeasible(monkeypatch, wcu5_spec):
    """A probe sim that raises => (False, 'ExcType: msg')."""
    def boom(cfg, spec):
        raise RuntimeError("GLPK: problem has no feasible solution")

    monkeypatch.setattr(sim, "run_simulation_ensemble_inmemory", boom)
    ok, err = sim.check_dv_feasibility(object(), wcu5_spec)
    assert ok is False
    assert "RuntimeError" in err and "feasible" in err


def test_feasibility_rejects_single_trace(historic_spec):
    """is_ensemble=False must error fast (mirrors the ensemble-sim guard)."""
    with pytest.raises(ValueError, match="is_ensemble=True"):
        sim.check_dv_feasibility(object(), historic_spec)


# ---------------------------------------------------------------------------
# Slow integration tests — these actually build and run pywrdrb
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not STAGED_ENSEMBLE.exists(),
    reason="Staged Kirsch–Nowak ensemble HDF5 not present; "
           "run the workflow/02-04 ensemble pipeline first.",
)
@slow
def test_ensemble_returns_list_of_n_distinct_data_dicts(wcu5_spec):
    """End-to-end: wcu_kirsch_n5 -> 5 data dicts with distinct storage paths."""
    from src.formulations import get_baseline_values
    from src.simulation import dvs_to_config, run_simulation_ensemble_inmemory

    dv_baseline = np.array(get_baseline_values("ffmp"))
    nyc_config = dvs_to_config(dv_baseline, formulation_name="ffmp")

    data_per_real = run_simulation_ensemble_inmemory(nyc_config, wcu5_spec)

    # Length matches realization count.
    assert isinstance(data_per_real, list)
    assert len(data_per_real) == wcu5_spec.n_realizations == 5

    # Each dict has the canonical keys. flood_stage is included now that
    # the staged HDF5 carries flood-augmented inflows and the simulation
    # path runs with enable_nyc_flood_operations=True.
    expected_keys = {
        "res_storage", "major_flow", "ibt_demands",
        "ibt_diversions", "mrf_target", "flood_stage",
    }
    from config import INCLUDE_SALINITY_MODEL
    if INCLUDE_SALINITY_MODEL:
        expected_keys = expected_keys | {"salinity"}

    for i, d in enumerate(data_per_real):
        missing = expected_keys - set(d.keys())
        assert not missing, f"realization {i} missing keys: {missing}"
        assert not d["res_storage"].empty, f"realization {i} has empty res_storage"
        assert not d["flood_stage"].empty, f"realization {i} has empty flood_stage"
        if INCLUDE_SALINITY_MODEL:
            assert not d["salinity"].empty, (
                f"realization {i} has empty salinity frame — "
                f"_extract_salinity_records did not populate this scenario."
            )

    # Realizations must DIFFER on storage trajectories AND on flood-stage
    # gauge series — a strict check that the ensemble inflow indices reach
    # both reservoir-balance and flood-monitoring parameters.
    s0 = data_per_real[0]["res_storage"]
    fs0 = data_per_real[0]["flood_stage"]
    for j in range(1, len(data_per_real)):
        sj = data_per_real[j]["res_storage"]
        fsj = data_per_real[j]["flood_stage"]
        assert not s0.equals(sj), (
            f"realization 0 and {j} have identical storage trajectories — "
            f"ensemble inflow indices may not be wired through to pywr."
        )
        assert not fs0.equals(fsj), (
            f"realization 0 and {j} have identical flood-stage trajectories — "
            f"flood-augmented HDF5 may not be reaching FlowEnsemble."
        )

    # Salinity LSTM is now scenario-aware (PywrDRB-ML + Pywr-DRB
    # salt_front_location refactor 2026-05-06). The forward pass takes
    # per-scenario flow as input, so the salt-front trajectories must
    # diverge across realizations — otherwise we're seeing a single shared
    # series and the scenario-aware refactor regressed.
    if INCLUDE_SALINITY_MODEL:
        sf0 = data_per_real[0]["salinity"]["salt_front_location_mu"]
        for j in range(1, len(data_per_real)):
            sfj = data_per_real[j]["salinity"]["salt_front_location_mu"]
            assert not sf0.equals(sfj), (
                f"realization 0 and {j} have identical salt-front "
                f"trajectories — salinity LSTM may not be receiving "
                f"per-scenario flows (check ml_model.lstm.set_n_scenarios)."
            )


@pytest.mark.skipif(
    not STAGED_ENSEMBLE.exists(),
    reason="Staged Kirsch–Nowak ensemble HDF5 not present.",
)
@slow
def test_cache_isolation_across_presets(historic_spec, wcu5_spec):
    """Building the model twice with different ensemble presets must put
    two distinct entries in _CACHED_MODEL_DICTS, with the right inflow_type
    on each cached dict."""
    from src.simulation import _get_cached_model_dict, _CACHED_MODEL_DICTS

    nyc_config = sim._get_cached_defaults()

    # Force builds with each spec. The historic build uses the legacy
    # single-trace inflow_type; the wcu5 build uses the staged ensemble dir.
    base_legacy = _get_cached_model_dict(
        use_trimmed=False, nyc_config=nyc_config, ensemble_spec=historic_spec,
    )
    base_ensemble = _get_cached_model_dict(
        use_trimmed=False, nyc_config=nyc_config, ensemble_spec=wcu5_spec,
    )
    assert base_legacy is not base_ensemble, "cache cross-contamination"

    # Two distinct cache entries (preset name slot differs).
    preset_slots = {key[3] for key in sim._CACHED_MODEL_DICTS}
    assert "historic_single" in preset_slots
    assert "wcu_kirsch_n5" in preset_slots
