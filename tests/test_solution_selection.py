"""Unit tests for src/solution_selection.py — dominance, scaling, selectors."""

import numpy as np
import pytest

from src.pareto_filter import to_natural
from src.solution_selection import (
    Selection,
    best_single,
    compromise,
    compromise_scores,
    dominance_mask,
    dominates,
    n_objectives_beaten,
    nondominated_mask,
    normalized_dv,
    orient_maximize,
    pairwise_distances,
    scale_objectives,
    select_by_rules,
    select_diverse,
)

#: One maximize objective and one minimize objective — the mixed-sense case
#: every dominance test must survive.
MIXED = [1, -1]


###############################################################################
# Dominance
###############################################################################

def test_dominates_strict_weak_and_self():
    """Weak dominance: no-worse everywhere plus strictly better somewhere."""
    # Both objectives maximize; a beats b on both -> dominates.
    assert dominates([2.0, 2.0], [1.0, 1.0], [1, 1])
    # Tie on one axis, strictly better on the other -> STILL dominates. This is
    # the case strong dominance would wrongly reject.
    assert dominates([2.0, 1.0], [1.0, 1.0], [1, 1])
    # Equal on every axis -> no strict improvement -> no dominance.
    assert not dominates([1.0, 1.0], [1.0, 1.0], [1, 1])
    # A vector never dominates itself.
    v = [0.3, 7.5]
    assert not dominates(v, v, MIXED)
    # Better on one, worse on the other -> mutually nondominated.
    assert not dominates([2.0, 0.0], [1.0, 1.0], [1, 1])
    assert not dominates([1.0, 1.0], [2.0, 0.0], [1, 1])
    # "Better on average" is NOT dominance: 10 + 0 beats 1 + 1 on the mean.
    assert not dominates([10.0, 0.0], [1.0, 1.0], [1, 1])


def test_dominates_respects_mixed_senses():
    """A lower value wins on a minimize axis and loses on a maximize axis."""
    # obj0 maximize, obj1 minimize. a: higher reliability AND lower deficit.
    assert dominates([0.9, 10.0], [0.8, 20.0], MIXED)
    # Same numbers read in the wrong orientation would flip the verdict, so
    # the all-maximize reading must NOT dominate.
    assert not dominates([0.9, 10.0], [0.8, 20.0], [1, 1])
    # Tie on the maximize axis, better on the minimize axis -> dominates.
    assert dominates([0.8, 10.0], [0.8, 20.0], MIXED)


def test_dominates_tolerance_is_symmetric_slack():
    """tol widens both the no-worse and the strictly-better comparisons."""
    a, b = [1.0, 1.0], [1.0 + 1e-9, 1.0]
    # Exact comparison: a is worse on obj0 by 1e-9, so no dominance.
    assert not dominates(a, b, [1, 1])
    # With slack larger than the gap, a is "no worse" — but it also has no
    # strict win left, so it still does not dominate.
    assert not dominates(a, b, [1, 1], tol=1e-6)
    # A genuine win beyond the slack does dominate.
    assert dominates([1.0, 2.0], b, [1, 1], tol=1e-6)


def test_dominance_mask_and_beaten_count_agree_on_a_hand_case():
    """Row-wise dominance vs a reference, plus the near-miss count."""
    ref = np.array([0.5, 20.0])                 # max obj0, min obj1
    front = np.array([
        [0.6, 10.0],   # beats both              -> dominates, 2 beaten
        [0.5, 10.0],   # ties obj0, beats obj1   -> dominates, 1 beaten
        [0.6, 20.0],   # beats obj0, ties obj1   -> dominates, 1 beaten
        [0.5, 20.0],   # identical               -> no dominance, 0 beaten
        [0.6, 30.0],   # trade-off               -> no dominance, 1 beaten
        [0.4, 30.0],   # worse on both           -> no dominance, 0 beaten
    ])
    mask = dominance_mask(front, ref, MIXED)
    assert mask.tolist() == [True, True, True, False, False, False]
    assert n_objectives_beaten(front, ref, MIXED).tolist() == [2, 1, 1, 0, 1, 0]
    # Row-by-row agreement with the scalar predicate.
    assert [dominates(r, ref, MIXED) for r in front] == mask.tolist()


def test_nondominated_mask_keeps_duplicates_and_drops_dominated():
    """Duplicate rows are mutually nondominated and both survive."""
    front = np.array([
        [0.9, 30.0],   # best obj0
        [0.5, 10.0],   # best obj1
        [0.7, 20.0],   # interior, nondominated
        [0.6, 25.0],   # dominated by row 2
        [0.9, 30.0],   # exact duplicate of row 0
    ])
    assert nondominated_mask(front, MIXED).tolist() == [
        True, True, True, False, True]


def test_nondominated_mask_matches_brute_force_on_random_stored_objectives():
    """to_natural + our dominance agrees with an O(n^2) reference on the
    all-minimized storage orientation."""
    rng = np.random.default_rng(20260805)
    directions = [1, -1, 1, -1, 1]
    stored = rng.normal(size=(60, 5)).round(2)   # rounding forces real ties
    natural = to_natural(stored, directions)

    # Brute force, written against the STORED orientation where every column
    # is minimized: a dominates b iff a <= b on all and a < b on at least one.
    n = stored.shape[0]
    expected = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (np.all(stored[j] <= stored[i])
                    and np.any(stored[j] < stored[i])):
                dominated = True
                break
        expected.append(not dominated)

    assert nondominated_mask(natural, directions).tolist() == expected

    # And the pairwise predicate agrees, vector by vector, on the same data.
    ref = natural[0]
    brute = [(np.all(stored[j] <= stored[0]) and np.any(stored[j] < stored[0]))
             for j in range(n)]
    assert dominance_mask(natural, ref, directions).tolist() == brute


###############################################################################
# Scaling
###############################################################################

def test_scale_objectives_orients_one_as_best():
    """1.0 marks the best value on both maximize and minimize axes."""
    front = np.array([[0.5, 20.0], [1.0, 10.0], [0.75, 15.0]])
    s = scale_objectives(front, MIXED)
    # obj0 maximize: 1.0 is best; obj1 minimize: 10.0 is best.
    assert s[1].tolist() == [1.0, 1.0]
    assert s[0].tolist() == [0.0, 0.0]
    assert s[2] == pytest.approx([0.5, 0.5])


def test_scale_objectives_handles_a_constant_objective():
    """A zero-range axis becomes all 1.0 instead of dividing by zero."""
    front = np.array([[0.5, 3.0], [0.9, 3.0], [0.7, 3.0]])
    s = scale_objectives(front, MIXED)
    assert np.all(np.isfinite(s))
    assert s[:, 1].tolist() == [1.0, 1.0, 1.0]
    assert s[:, 0].tolist() == [0.0, 1.0, pytest.approx(0.5)]
    # A constant axis must not move the decision: it contributes no regret and
    # never binds, so the compromise winner is decided by obj0 alone.
    assert compromise(front, MIXED, method="distance_to_ideal") == 1
    assert compromise(front, MIXED, method="maximin") == 1
    # An explicit ideal == nadir is the same degenerate case.
    s2 = scale_objectives(front, MIXED, ideal=[0.9, 3.0], nadir=[0.9, 3.0])
    assert np.all(s2 == 1.0)


def test_scale_objectives_accepts_an_external_common_frame():
    """Explicit ideal/nadir let two runs be scored on one frame."""
    front = np.array([[0.6, 15.0], [0.8, 12.0]])
    s = scale_objectives(front, MIXED, ideal=[1.0, 10.0], nadir=[0.5, 20.0])
    assert s[0] == pytest.approx([0.2, 0.5])
    assert s[1] == pytest.approx([0.6, 0.8])
    # Outside the frame: clipped by default, raw when asked.
    outside = np.array([[1.5, 5.0]])
    assert scale_objectives(outside, MIXED, ideal=[1.0, 10.0],
                            nadir=[0.5, 20.0])[0].tolist() == [1.0, 1.0]
    raw = scale_objectives(outside, MIXED, ideal=[1.0, 10.0], nadir=[0.5, 20.0],
                           clip=False)[0]
    assert raw[0] > 1.0 and raw[1] > 1.0


def test_orient_maximize_flips_only_minimize_columns():
    arr = np.array([[0.5, 20.0]])
    assert orient_maximize(arr, MIXED).tolist() == [[0.5, -20.0]]


###############################################################################
# Compromise rules
###############################################################################

def _hand_case():
    """Three solutions whose scaled scores are exactly (0,1), (1,0), (.6,.6).

    obj0 maximizes over [0, 10]; obj1 minimizes over [0, 10].
    """
    return np.array([
        [0.0, 0.0],    # scaled (0.0, 1.0) — perfect on obj1, worst on obj0
        [10.0, 10.0],  # scaled (1.0, 0.0) — perfect on obj0, worst on obj1
        [6.0, 4.0],    # scaled (0.6, 0.6) — balanced
    ])


def test_mean_scaled_picks_the_best_average():
    """Means are 0.5, 0.5, 0.6 -> the balanced solution wins."""
    front = _hand_case()
    scores = compromise_scores(front, MIXED, method="mean_scaled")
    assert scores == pytest.approx([0.5, 0.5, 0.6])
    assert compromise(front, MIXED, method="mean_scaled") == 2


def test_distance_to_ideal_p1_p2_and_chebyshev():
    """Hand-computed Lp regrets from the ideal point (1, 1)."""
    front = _hand_case()
    # Regrets: (1.0, 0.0), (0.0, 1.0), (0.4, 0.4); weights are 0.5 each.
    p1 = compromise_scores(front, MIXED, method="distance_to_ideal", p=1)
    assert p1 == pytest.approx([-0.5, -0.5, -0.4])
    p2 = compromise_scores(front, MIXED, method="distance_to_ideal", p=2)
    assert p2 == pytest.approx([-np.sqrt(0.5), -np.sqrt(0.5),
                               -np.sqrt(0.5 * 0.32)])
    cheb = compromise_scores(front, MIXED, method="distance_to_ideal",
                             p=float("inf"))
    assert cheb == pytest.approx([-0.5, -0.5, -0.2])
    for p in (1, 2, float("inf")):
        assert compromise(front, MIXED, method="distance_to_ideal", p=p) == 2


def test_maximin_maximizes_the_worst_axis():
    """Worst scaled values are 0.0, 0.0, 0.6 -> the balanced solution wins."""
    front = _hand_case()
    scores = compromise_scores(front, MIXED, method="maximin")
    # Uniform weights (0.5 each) divide, so scores are 2x the raw minimum.
    assert scores == pytest.approx([0.0, 0.0, 1.2])
    assert compromise(front, MIXED, method="maximin") == 2


def test_maximin_and_mean_scaled_can_disagree():
    """The average-best solution is not the worst-case-best solution."""
    # scaled: A = (1.0, 0.0), B = (0.5, 0.5). Means 0.5 vs 0.5 (tie -> lowest
    # index), minima 0.0 vs 0.5 (B wins outright).
    front = np.array([[10.0, 10.0], [5.0, 5.0], [0.0, 0.0]])
    assert compromise(front, MIXED, method="mean_scaled") == 0
    assert compromise(front, MIXED, method="maximin") == 1


def test_weights_shift_the_compromise_toward_the_weighted_objective():
    """Heavier weight on obj0 pulls the choice to the obj0-strong solution."""
    front = np.array([[8.0, 8.0], [4.0, 2.0], [0.0, 0.0]])
    # Scaled: (1.0, 0.0), (0.5, 0.75), (0.0, 1.0).
    assert compromise(front, MIXED, method="mean_scaled") == 1
    assert compromise(front, MIXED, method="mean_scaled",
                      weights=[9.0, 1.0]) == 0
    # Weights are normalized internally, so scale is irrelevant.
    a = compromise_scores(front, MIXED, method="mean_scaled", weights=[3.0, 1.0])
    b = compromise_scores(front, MIXED, method="mean_scaled",
                          weights=[0.75, 0.25])
    assert a == pytest.approx(b)


def test_compromise_rejects_unknown_method_and_empty_front():
    front = _hand_case()
    with pytest.raises(ValueError, match="unknown method"):
        compromise(front, MIXED, method="best_vibes")
    with pytest.raises(ValueError, match="empty front"):
        compromise(np.empty((0, 2)), MIXED)
    with pytest.raises(ValueError, match="empty front"):
        best_single(np.empty((0, 2)), MIXED, 0)


###############################################################################
# Selectors
###############################################################################

def test_best_single_uses_direction_and_breaks_ties_by_lowest_index():
    front = np.array([[0.9, 20.0], [0.9, 10.0], [0.4, 10.0]])
    # obj0 maximize: rows 0 and 1 tie at 0.9 -> lowest index wins.
    assert best_single(front, MIXED, 0) == 0
    # obj1 minimize: rows 1 and 2 tie at 10.0 -> lowest index wins.
    assert best_single(front, MIXED, 1) == 1


def test_select_by_rules_falls_through_to_keep_choices_distinct():
    """Two rules whose top pick collides resolve to different solutions."""
    front = np.array([[0.9, 10.0], [0.9, 30.0], [0.5, 40.0]])
    rules = [
        ("NYC priority", "best_nyc", front[:, 0]),          # ties rows 0, 1
        ("Flow priority", "best_flow", -front[:, 1]),       # prefers row 0
    ]
    picks = select_by_rules(rules)
    assert [s.index for s in picks] == [0, 1]
    assert picks[0] == Selection("NYC priority", "best_nyc", 0)
    # Without the distinctness pass, both rules land on row 0.
    assert [s.index for s in select_by_rules(rules, distinct=False)] == [0, 0]
    # Deterministic across repeats.
    assert [s.index for s in select_by_rules(rules)] == [0, 1]


def test_select_by_rules_handles_non_finite_scores():
    """NaN/-inf scores rank last rather than winning or crashing."""
    scores = np.array([np.nan, 0.2, np.inf, -np.inf])
    picks = select_by_rules([("robust", "max_sat", scores)])
    assert picks[0].index == 2
    # An all-NaN rule still returns something rather than raising.
    assert select_by_rules([("x", "x", np.full(4, np.nan))])[0].index == 0


def test_select_diverse_spreads_across_the_front_and_is_deterministic():
    """Farthest-point picks the extremes, not three neighbours."""
    # A clustered front: three near-identical solutions plus two extremes.
    front = np.array([
        [5.00, 5.0],
        [5.01, 5.0],
        [4.99, 5.0],
        [10.0, 10.0],
        [0.0, 0.0],
    ])
    picks = select_diverse(front, MIXED, 3)
    assert len(picks) == 3
    assert len(set(picks)) == 3
    # Both extremes must be represented; the cluster contributes at most one.
    assert 3 in picks and 4 in picks
    assert sum(p in (0, 1, 2) for p in picks) == 1
    assert select_diverse(front, MIXED, 3) == picks       # deterministic


def test_select_diverse_honours_seeds_candidates_and_duplicate_guard():
    front = np.array([[0.0, 0.0], [10.0, 10.0], [5.0, 5.0], [5.0, 5.0]])
    # Seeds come first, in the order given.
    assert select_diverse(front, MIXED, 3, seed_indices=[1, 0])[:2] == [1, 0]
    # Candidates restrict what may be added after the seeds.
    assert select_diverse(front, MIXED, 3, seed_indices=[0],
                          candidates=[2]) == [0, 2]
    # Exact duplicates are refused, so fewer than n may come back.
    assert select_diverse(front, MIXED, 4, seed_indices=[2]) == [2, 0, 1]
    # Asking for more than exists never raises.
    assert len(select_diverse(front, MIXED, 99)) <= 4
    assert select_diverse(np.empty((0, 2)), MIXED, 3) == []


def test_select_diverse_can_spread_in_an_external_space():
    """`space` lets the caller disperse over decision variables instead."""
    front = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])   # identical objs
    dv_space = np.array([[0.0], [0.5], [1.0]])
    picks = select_diverse(front, MIXED, 2, seed_indices=[0], space=dv_space)
    assert picks == [0, 2]


###############################################################################
# Decision-variable helpers
###############################################################################

def test_normalized_dv_and_pairwise_distances():
    dv = np.array([[0.0, 5.0], [10.0, 5.0]])
    bounds = (np.array([0.0, 5.0]), np.array([10.0, 5.0]))
    nd = normalized_dv(dv, bounds)
    # Second variable has a zero-width bound -> 0.0, never a divide-by-zero.
    assert nd.tolist() == [[0.0, 0.0], [1.0, 0.0]]
    d = pairwise_distances(nd)
    assert d.shape == (2, 2)
    assert d[0, 0] == 0.0
    assert d[0, 1] == pytest.approx(1.0)
    assert np.allclose(d, d.T)
