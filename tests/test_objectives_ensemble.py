"""
tests/test_objectives_ensemble.py - Unit tests for the v1 ensemble objective
framework in src.objectives_ensemble.

Covers:
  1. SatisficingAgg ge / le / NaN handling.
  2. EnsembleObjective wraps a base Objective and applies the aggregator
     once per realization, including the salt-front objective on a synthetic
     two-realization data_per_real list.
  3. build_ensemble_objective_set resolves the 7 v1 names to maximise-direction
     EnsembleObjectives that ObjectiveSet accepts.
  4. ObjectiveSet.compute_for_borg_ensemble returns a 7-element vector of
     negated maximised satisficing rates, all in [-1, 0].
  5. NYCOPT_SAT_THRESHOLDS env override patches the registry without code
     changes.

Run:
    venv/bin/python -m pytest tests/test_objectives_ensemble.py -v
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.objectives import OBJECTIVES, ObjectiveSet
import src.objectives_ensemble as obj_ens
from src.objectives_ensemble import (
    EnsembleObjective,
    ENSEMBLE_OBJECTIVES,
    SatisficingAgg,
    build_ensemble_objective_set,
)


# ---------------------------------------------------------------------------
# SatisficingAgg
# ---------------------------------------------------------------------------

def test_satisficing_ge_basic():
    agg = SatisficingAgg(threshold=0.95, kind="ge")
    assert agg([0.99, 0.98, 0.92, 0.96]) == pytest.approx(0.75)


def test_satisficing_le_basic():
    agg = SatisficingAgg(threshold=10.0, kind="le")
    # 5, 9 satisfy (<= 10); 12, 11 fail.
    assert agg([5.0, 9.0, 12.0, 11.0]) == pytest.approx(0.5)


def test_satisficing_nan_counts_as_unsatisfied():
    agg = SatisficingAgg(threshold=0.95, kind="ge")
    # NaN ⇒ unsatisfied, even with a "ge" check that has no upper bound.
    assert agg([0.99, float("nan"), 0.97]) == pytest.approx(2.0 / 3.0)


def test_satisficing_all_nan_returns_zero():
    agg = SatisficingAgg(threshold=10.0, kind="le")
    assert agg([float("nan"), float("nan")]) == 0.0


def test_satisficing_empty_returns_zero():
    agg = SatisficingAgg(threshold=10.0, kind="le")
    assert agg([]) == 0.0


def test_satisficing_rejects_bad_kind():
    with pytest.raises(ValueError, match="kind"):
        SatisficingAgg(threshold=0.0, kind="eq")


# ---------------------------------------------------------------------------
# EnsembleObjective
# ---------------------------------------------------------------------------

def _make_salinity_data(peak_rm: float, n_days: int = 400) -> dict:
    """Build a minimal data dict with a 'salinity' frame whose post-warmup max
    salt-front RM equals `peak_rm`. Other base objectives don't read this
    dict, so they aren't relevant here."""
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
    series = pd.Series(np.full(n_days, peak_rm - 5.0), index=idx)
    # Place the peak well into the post-warmup window (default WARMUP_DAYS=365).
    series.iloc[380] = peak_rm
    return {
        "salinity": pd.DataFrame(
            {"salt_front_location_mu": series, "salt_front_location_sd": series},
            index=idx,
        ),
    }


def test_ensemble_objective_wraps_base_per_realization():
    salt_le = SatisficingAgg(threshold=92.47, kind="le")
    ens = EnsembleObjective(
        base=OBJECTIVES["salt_front_max_rm"],
        aggregator=salt_le,
        name="salt_front_max_rm__test",
        epsilon=0.02,
    )
    data_per_real = [
        _make_salinity_data(peak_rm=90.0),   # satisfies (<=92.47)
        _make_salinity_data(peak_rm=95.0),   # fails
    ]
    raw = ens.compute(data_per_real)
    assert raw == pytest.approx(0.5)
    assert ens.compute_for_borg(data_per_real) == pytest.approx(-0.5)
    assert ens.direction == "maximize"
    assert ens.sign == 1


def test_ensemble_objective_calls_base_once_per_realization():
    """Wrapping should invoke the base's compute exactly once per realization,
    in order, and pass each per-realization data dict through unchanged."""
    calls = []

    class Spy:
        name = "spy"
        direction = "minimize"

        def compute(self, d):
            calls.append(d)
            return 91.0  # always satisficing

    ens = EnsembleObjective(
        base=Spy(),
        aggregator=SatisficingAgg(threshold=92.47, kind="le"),
        name="x", epsilon=0.02,
    )
    data_per_real = [{"a": 1}, {"b": 2}, {"c": 3}]
    raw = ens.compute(data_per_real)
    assert raw == pytest.approx(1.0)
    assert calls == data_per_real


# ---------------------------------------------------------------------------
# build_ensemble_objective_set + ObjectiveSet.compute_for_borg_ensemble
# ---------------------------------------------------------------------------

V1_ENSEMBLE_NAMES = [
    "nyc_reliability_weekly_decree__sat95",
    "nyc_max_deficit_weekly_decree__sat10pp",
    "montague_reliability_weekly_decree__sat85",
    "montague_max_deficit_weekly_decree__sat25pp",
    "salt_front_max_rm__sat92rm",
    "flood_days_downstream_action_anygauge__sat30d",
    "storage_min_combined_pct__sat25",
]


def test_registry_has_seven_objectives():
    assert set(ENSEMBLE_OBJECTIVES) == set(V1_ENSEMBLE_NAMES)


def test_build_ensemble_set_returns_seven_maximize():
    obj_set = build_ensemble_objective_set(V1_ENSEMBLE_NAMES)
    assert isinstance(obj_set, ObjectiveSet)
    assert obj_set.n_objs == 7
    assert obj_set.names == V1_ENSEMBLE_NAMES
    assert all(d == 1 for d in obj_set.directions)  # all maximize


def test_build_ensemble_set_rejects_legacy_name():
    with pytest.raises(KeyError, match="Unknown ensemble objective"):
        build_ensemble_objective_set(["salt_front_max_rm"])  # legacy name


def test_compute_for_borg_ensemble_salt_front_only():
    """End-to-end: feed only the salt-front ensemble objective a 2-realization
    data_per_real and confirm the Borg vector is the negated 0.5."""
    obj_set = build_ensemble_objective_set(["salt_front_max_rm__sat92rm"])
    data_per_real = [
        _make_salinity_data(peak_rm=90.0),
        _make_salinity_data(peak_rm=95.0),
    ]
    borg = obj_set.compute_for_borg_ensemble(data_per_real)
    assert borg == pytest.approx([-0.5])


def test_borg_vector_is_in_negated_unit_interval(monkeypatch):
    """Every element of compute_for_borg_ensemble must be in [-1, 0]."""
    # Patch each base Objective.compute to return a finite value so the
    # non-salinity metrics don't NaN-out on a synthetic data dict that
    # doesn't carry their input keys (ibt_demands, major_flow, etc.).
    monkey_results = {
        "salt_front_max_rm":                       91.0,   # le 92.47 ⇒ satisfies
        "nyc_reliability_weekly_decree":           0.99,
        "nyc_max_deficit_weekly_decree":           5.0,
        "montague_reliability_weekly_decree":      0.90,
        "montague_max_deficit_weekly_decree":      15.0,
        "flood_days_downstream_action_anygauge":   10.0,
        "storage_min_combined_pct":                40.0,
    }
    for base_name, val in monkey_results.items():
        monkeypatch.setattr(
            OBJECTIVES[base_name], "compute",
            lambda _d, v=val: v,
        )

    obj_set = build_ensemble_objective_set(V1_ENSEMBLE_NAMES)
    data_per_real = [_make_salinity_data(peak_rm=90.0) for _ in range(3)]
    borg = obj_set.compute_for_borg_ensemble(data_per_real)
    assert len(borg) == 7
    for v in borg:
        assert -1.0 <= v <= 0.0


# ---------------------------------------------------------------------------
# Threshold env override
# ---------------------------------------------------------------------------

def test_env_threshold_override(monkeypatch):
    monkeypatch.setenv(
        "NYCOPT_SAT_THRESHOLDS",
        json.dumps({"salt_front_max_rm__sat92rm": 80.0}),
    )
    importlib.reload(obj_ens)
    try:
        agg = obj_ens.ENSEMBLE_OBJECTIVES["salt_front_max_rm__sat92rm"].aggregator
        assert isinstance(agg, obj_ens.SatisficingAgg)
        assert agg.threshold == pytest.approx(80.0)
    finally:
        monkeypatch.delenv("NYCOPT_SAT_THRESHOLDS", raising=False)
        importlib.reload(obj_ens)


def test_env_threshold_override_rejects_unknown(monkeypatch):
    monkeypatch.setenv(
        "NYCOPT_SAT_THRESHOLDS",
        json.dumps({"not_a_real_objective": 0.5}),
    )
    with pytest.raises(KeyError, match="not_a_real_objective"):
        importlib.reload(obj_ens)
    monkeypatch.delenv("NYCOPT_SAT_THRESHOLDS", raising=False)
    importlib.reload(obj_ens)
