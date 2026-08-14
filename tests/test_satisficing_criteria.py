"""
tests/test_satisficing_criteria.py - Subset criterion-set semantics.

The criterion sets are Quinn-2017-style subsets: each thresholds only its
member axes and leaves every other axis non-binding. What must hold:

  1. ``thresholds(adopted, kinds)`` returns a FULL vector -- member axes at
     their placement, all others at the non-binding infinity of their kind --
     so ``robustness._satisfaction_cube``'s missing-threshold guard stays
     armed (no axis is ever silently absent).
  2. The joint Starr criterion under a subset vector counts ONLY the member
     axes' conjunction.
  3. A reference set passes the adopted snapshot through unchanged.
  4. Unknown member axes are a hard error.
  5. ``score_criteria`` reproduces ``sat_multivariate_sow`` exactly for the
     reference set, and its shortfall columns measure the McPhail
     satisficing-regret ``max(0, c - f)``.

Run:
    venv/bin/python -m pytest tests/test_satisficing_criteria.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import src.robustness as rob  # noqa: E402
from src.satisficing_criteria import (  # noqa: E402
    ALL_SETS, CRITERION_SETS, NAMED_SETS, CriterionSet, criterion_by_key,
    focal_criterion, nonbinding_threshold,
)

OBJ_NAMES = ["rel", "deficit", "storage"]
ADOPTED = {"rel": 0.8, "deficit": 10.0, "storage": 20.0}
KINDS = {"rel": "ge", "deficit": "le", "storage": "ge"}


def _cube() -> rob.RawCube:
    """2 solutions x 4 SOWs x 3 objectives with known pass/fail structure."""
    cube = np.array([
        # solution 0: rel passes everywhere; deficit fails SOW 3;
        #             storage fails SOWs 2-3.
        [[0.9, 5.0, 25.0], [0.9, 8.0, 30.0], [0.9, 9.0, 10.0], [0.9, 12.0, 5.0]],
        # solution 1: rel fails everywhere.
        [[0.5, 1.0, 50.0], [0.5, 1.0, 50.0], [0.5, 1.0, 50.0], [0.5, 1.0, 50.0]],
    ], dtype=float)
    return rob.RawCube(
        cube=cube, solution_ids=[0, 1], sow_labels=[0, 1, 2, 3],
        obj_names=OBJ_NAMES, thresholds=dict(ADOPTED), kinds=dict(KINDS),
        directions={"rel": "maximize", "deficit": "minimize",
                    "storage": "maximize"},
        is_ensemble=True, realizations_per_sow=1, meta={},
    )


def test_nonbinding_threshold_signs():
    assert nonbinding_threshold("ge") == -math.inf
    assert nonbinding_threshold("le") == math.inf
    with pytest.raises(ValueError):
        nonbinding_threshold("eq")


def test_subset_thresholds_fill_every_axis():
    cset = CriterionSet("t", "T", "test", criteria={"rel": 0.8})
    thr = cset.thresholds(ADOPTED, KINDS)
    assert set(thr) == set(ADOPTED), "the vector must cover every adopted axis"
    assert thr["rel"] == 0.8
    assert thr["deficit"] == math.inf      # "le" axis, non-binding
    assert thr["storage"] == -math.inf     # "ge" axis, non-binding


def test_reference_set_passes_adopted_through():
    ref = CriterionSet("r", "R", "ref", reference=True)
    assert ref.thresholds(ADOPTED, KINDS) == ADOPTED


def test_unknown_member_axis_is_a_hard_error():
    cset = CriterionSet("t", "T", "test", criteria={"nope": 1.0})
    with pytest.raises(KeyError):
        cset.thresholds(ADOPTED, KINDS)


def test_joint_starr_counts_only_member_axes():
    raw = _cube()
    rel_only = CriterionSet("t", "T", "test", criteria={"rel": 0.8})
    thr = rel_only.thresholds(raw.thresholds, raw.kinds)
    # The guard stays armed: every axis has a threshold, none is skipped.
    joint = rob.satisficing_multivariate_sow(raw, thr)
    assert joint[0] == pytest.approx(1.0)   # rel passes in all 4 SOWs
    assert joint[1] == pytest.approx(0.0)   # rel fails everywhere

    two = CriterionSet("t2", "T2", "test",
                       criteria={"rel": 0.8, "storage": 20.0})
    joint2 = rob.satisficing_multivariate_sow(
        raw, two.thresholds(raw.thresholds, raw.kinds))
    assert joint2[0] == pytest.approx(0.5)  # storage fails SOWs 2-3


def test_registry_shape_and_focal_env(monkeypatch):
    keys = [c.key for c in CRITERION_SETS]
    assert len(keys) == len(set(keys)), "criterion keys must be unique"
    for c in NAMED_SETS:
        assert 1 <= len(c.axes) <= 3, f"{c.key}: subsets hold 1-3 axes"
        assert not c.reference
    assert ALL_SETS[-1].reference, "the reference set displays last"
    assert criterion_by_key("reference_all8").reference

    monkeypatch.delenv("NYCOPT_FOCAL_CRITERION", raising=False)
    assert focal_criterion().key == "compromise"
    monkeypatch.setenv("NYCOPT_FOCAL_CRITERION", "flood")
    assert focal_criterion().key == "flood"
    with pytest.raises(KeyError):
        monkeypatch.setenv("NYCOPT_FOCAL_CRITERION", "not_a_set")
        focal_criterion()


def test_score_criteria_reference_matches_primary_and_shortfalls():
    raw = _cube()
    sets = (
        CriterionSet("rel_only", "Rel", "test", criteria={"rel": 0.8}),
        CriterionSet("ref", "Ref", "test", reference=True),
    )
    scorecard, higher = rob.score_criteria(raw, baseline=None, sets=sets)

    primary = rob.satisficing_multivariate_sow(raw)
    assert np.allclose(scorecard["sat_set__ref"], primary), \
        "the reference set must reproduce sat_multivariate_sow exactly"
    assert higher["sat_set__ref"] is True

    # Shortfall = max(0, c - f) for "ge": solution 1 misses rel 0.8 by 0.3 in
    # every SOW; solution 0 never misses it.
    col = "shortfall_mean__rel_only__rel"
    assert col in scorecard
    assert scorecard[col][0] == pytest.approx(0.0)
    assert scorecard[col][1] == pytest.approx(0.3)
    assert higher[col] is False


def test_criterion_ranking_stability_shape():
    raw = _cube()
    sets = (
        CriterionSet("a", "A", "test", criteria={"rel": 0.8}),
        CriterionSet("b", "B", "test", criteria={"storage": 20.0}),
    )
    scorecard, _ = rob.score_criteria(raw, baseline=None, sets=sets)
    tau = rob.criterion_ranking_stability(scorecard)
    assert list(tau.index) == ["sat_set__a", "sat_set__b"]
    assert tau.iloc[0, 0] == pytest.approx(1.0)


def test_single_trace_cube_nans_the_criteria_scorecard():
    raw = _cube()
    single = rob.RawCube(
        cube=raw.cube[:, :1, :], solution_ids=raw.solution_ids,
        sow_labels=[0], obj_names=raw.obj_names, thresholds=raw.thresholds,
        kinds=raw.kinds, directions=raw.directions, is_ensemble=False,
        realizations_per_sow=1, meta={},
    )
    scorecard, _ = rob.score_criteria(single, baseline=None, sets=(
        CriterionSet("a", "A", "test", criteria={"rel": 0.8}),))
    assert scorecard.isna().all().all()
