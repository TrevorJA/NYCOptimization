"""solution_selection.py - Pick traceable representatives out of a Pareto front.

Pure array math: no plotting, no side effects, and no I/O beyond the one
convenience loader :func:`load_natural_front`. Every selector consumes
objectives in **NATURAL** orientation (the human-facing units produced by
:func:`src.pareto_filter.to_natural`, NOT the all-minimized values stored in a
``.set``) together with the formulation's ``directions`` vector, and returns
**integer row indices into the front**. Those indices are the ``.set`` row
numbers — the same ids the re-evaluation cube uses as ``solution_id`` — so a
selection made here stays referenceable in every downstream artifact.

Orientation
-----------
``directions[k]`` is ``+1`` when objective ``k`` is maximized and ``-1`` when it
is minimized. :func:`orient_maximize` re-signs a natural array so LARGER IS
BETTER on every axis; all comparisons below happen in that orientation, which
is what makes one dominance rule serve mixed-sense objective sets.

Determinism
-----------
Every selector breaks ties by the LOWEST row index. Rankings use a stable sort,
so two solutions with identical scores always resolve the same way across runs
and machines. This matters here: on the historic trace the weekly reliabilities
live on a 1/76 lattice, so exact ties are common and real, not float noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Compromise rules accepted by :func:`compromise_scores` / :func:`compromise`.
COMPROMISE_METHODS = ("mean_scaled", "distance_to_ideal", "maximin")


###############################################################################
# Orientation and scaling
###############################################################################

def orient_maximize(natural_obj: np.ndarray, directions) -> np.ndarray:
    """Re-sign natural objectives so that larger is better on every axis.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` (or a single ``(n_objs,)`` row)
            of objectives in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).

    Returns:
        A float copy with minimize columns negated. Values are no longer in
        natural units and are only meaningful for comparison.
    """
    arr = np.asarray(natural_obj, dtype=float)
    return arr * np.asarray(directions, dtype=float)


def scale_objectives(natural_obj: np.ndarray, directions,
                     ideal=None, nadir=None, clip: bool = True) -> np.ndarray:
    """Min-max scale each objective to [0, 1] oriented so **1 = best**.

    The scaling frame defaults to the front itself (per-objective best and
    worst observed), which makes the result self-referential: adding or
    removing solutions changes the scores. Pass ``ideal``/``nadir`` explicitly
    to score several runs on one common frame.

    Degenerate axes: an objective with zero range (every solution identical,
    or ``ideal == nadir``) carries no information for discriminating
    solutions. Rather than divide by zero it is defined as **all 1.0** — every
    solution sits at that axis's ideal. That is the choice that keeps
    :func:`compromise` honest: a constant axis then contributes exactly zero
    regret to ``distance_to_ideal`` and never becomes the binding minimum in
    ``maximin``, so it drops out of the decision instead of shifting it.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` objectives in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        ideal: Optional ``(n_objs,)`` best-case values in NATURAL units.
            Defaults to the per-objective best over ``natural_obj``.
        nadir: Optional ``(n_objs,)`` worst-case values in NATURAL units.
            Defaults to the per-objective worst over ``natural_obj``.
        clip: Clamp results into [0, 1]. Only bites when an explicit frame is
            narrower than the data (a solution outside the given ideal/nadir);
            set False to let such solutions score outside the unit interval.

    Returns:
        ``(n_solutions, n_objs)`` array where 1.0 is best on every axis.
    """
    z = orient_maximize(natural_obj, directions)
    z = np.atleast_2d(z)
    if z.shape[0] == 0:
        return z.copy()
    best = (z.max(axis=0) if ideal is None
            else orient_maximize(np.asarray(ideal, dtype=float), directions))
    worst = (z.min(axis=0) if nadir is None
             else orient_maximize(np.asarray(nadir, dtype=float), directions))
    span = best - worst
    degenerate = span <= 0.0
    safe = np.where(degenerate, 1.0, span)
    scaled = (z - worst) / safe
    scaled[:, degenerate] = 1.0
    if clip:
        scaled = np.clip(scaled, 0.0, 1.0)
    return scaled


def _normalized_weights(weights, n_objs: int) -> np.ndarray:
    """Per-objective weights normalized to sum 1 (uniform when None)."""
    if weights is None:
        return np.full(n_objs, 1.0 / n_objs)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n_objs,):
        raise ValueError(f"weights must have shape ({n_objs},), got {w.shape}")
    if np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be non-negative with at least one positive")
    return w / w.sum()


###############################################################################
# Dominance
###############################################################################

def dominates(a, b, directions, tol: float = 0.0) -> bool:
    """Does objective vector ``a`` Pareto-dominate ``b``?

    The standard (weak) definition, applied after both vectors are oriented so
    larger is better: ``a`` dominates ``b`` iff ``a_i >= b_i`` for **all** i
    **and** ``a_j > b_j`` for **at least one** j. This is deliberately NOT
    "strictly better on every objective" (that is strong dominance and
    undercounts whenever objectives tie) and NOT "better on average".

    A vector never dominates itself: equality on every axis fails the
    at-least-one-strictly-better clause.

    Args:
        a: ``(n_objs,)`` candidate, natural units.
        b: ``(n_objs,)`` reference, natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        tol: Comparison slack in oriented units. ``a_i`` counts as no-worse
            when ``a_i >= b_i - tol`` and as strictly better when
            ``a_j > b_j + tol``. Default 0.0 — exact comparison. Objectives on
            the historic trace sit on a 1/76 weekly lattice, so exact ties are
            genuine outcomes rather than float noise and inflating ``tol``
            would erase real distinctions; raise it only for values carried
            through a lossy round trip.

    Returns:
        True iff ``a`` dominates ``b``.
    """
    za = orient_maximize(np.asarray(a, dtype=float).ravel(), directions)
    zb = orient_maximize(np.asarray(b, dtype=float).ravel(), directions)
    return bool(np.all(za >= zb - tol) and np.any(za > zb + tol))


def dominance_mask(natural_obj: np.ndarray, reference_vector, directions,
                   tol: float = 0.0) -> np.ndarray:
    """Which front members dominate a single reference vector.

    Vectorized :func:`dominates` over the rows of ``natural_obj``; see that
    docstring for the definition and the meaning of ``tol``.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        reference_vector: ``(n_objs,)`` comparison point in natural units (e.g.
            the FFMP baseline).
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        tol: Comparison slack in oriented units.

    Returns:
        Boolean array of length ``n_solutions``.
    """
    z = np.atleast_2d(orient_maximize(natural_obj, directions))
    if z.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    r = orient_maximize(np.asarray(reference_vector, dtype=float).ravel(),
                        directions)
    return (z >= r - tol).all(axis=1) & (z > r + tol).any(axis=1)


def n_objectives_beaten(natural_obj: np.ndarray, reference_vector, directions,
                        tol: float = 0.0) -> np.ndarray:
    """Per-solution count of objectives strictly better than a reference.

    The companion diagnostic to :func:`dominance_mask`: a solution can beat the
    reference on seven of eight objectives and still not dominate it. Reporting
    the count distribution alongside the dominance count shows how close the
    near-misses are.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        reference_vector: ``(n_objs,)`` comparison point in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        tol: Strictly-better slack in oriented units.

    Returns:
        Integer array of length ``n_solutions``, values in ``0..n_objs``.
    """
    z = np.atleast_2d(orient_maximize(natural_obj, directions))
    if z.shape[0] == 0:
        return np.zeros(0, dtype=int)
    r = orient_maximize(np.asarray(reference_vector, dtype=float).ravel(),
                        directions)
    return (z > r + tol).sum(axis=1).astype(int)


def nondominated_mask(natural_obj: np.ndarray, directions,
                      tol: float = 0.0) -> np.ndarray:
    """Boolean mask of the Pareto-nondominated rows of a set.

    A row is nondominated iff no other row dominates it under
    :func:`dominates`. Duplicated rows are all kept: identical vectors do not
    dominate each other.

    Runs an O(n^2) scan with an early skip for already-dominated rows, which is
    ample for archive-sized fronts (thousands of members).

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        tol: Comparison slack in oriented units.

    Returns:
        Boolean array of length ``n_solutions`` (True = nondominated).
    """
    z = np.atleast_2d(orient_maximize(natural_obj, directions))
    n = z.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        beats_i = (z >= z[i] - tol).all(axis=1) & (z > z[i] + tol).any(axis=1)
        if beats_i.any():
            keep[i] = False
    return keep


###############################################################################
# Single-objective and compromise selectors
###############################################################################

def best_single(natural_obj: np.ndarray, directions, k: int) -> int:
    """Row index of the front's best solution on objective ``k`` alone.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        k: Objective column index.

    Returns:
        Row index of the maximum (``directions[k] == 1``) or minimum
        (``directions[k] == -1``) natural value. Ties resolve to the LOWEST row
        index, so the answer is stable across runs.

    Raises:
        ValueError: If the front is empty.
    """
    z = np.atleast_2d(orient_maximize(natural_obj, directions))
    if z.shape[0] == 0:
        raise ValueError("cannot select from an empty front")
    return int(np.argmax(z[:, k]))


def compromise_scores(natural_obj: np.ndarray, directions,
                      method: str = "mean_scaled", weights=None,
                      p: float = 2.0, ideal=None, nadir=None) -> np.ndarray:
    """Per-solution compromise score, higher = more preferred.

    All three rules operate on :func:`scale_objectives` output (1 = best) and
    share one weight convention: **a larger weight makes an objective more
    important**, i.e. harder to trade away.

    ``"mean_scaled"``
        Weighted mean scaled score. Rewards good average attainment and will
        accept a poor axis if the rest are strong.
    ``"distance_to_ideal"``
        Compromise programming (Zeleny 1973): score is the negated weighted
        Lp distance from the ideal point ``(1, ..., 1)``,
        ``-(sum_i w_i * (1 - s_i)^p)^(1/p)``. ``p=1`` is the weighted
        Manhattan regret, ``p=2`` the Euclidean compromise, and ``p=inf``
        collapses to Chebyshev / minimax regret ``-max_i w_i * (1 - s_i)``.
    ``"maximin"``
        Maximize the worst weighted scaled objective, ``min_i s_i / w_i``.
        Weights divide, so a larger weight pushes that axis toward being the
        binding minimum and therefore protects it; uniform weights reduce the
        rule to plain ``min_i s_i``.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        method: One of :data:`COMPROMISE_METHODS`.
        weights: Optional ``(n_objs,)`` non-negative importances, normalized
            internally to sum 1. None = uniform.
        p: Lp order for ``"distance_to_ideal"``; 1, 2, or ``float("inf")``.
        ideal: Optional explicit scaling ideal, natural units (see
            :func:`scale_objectives`).
        nadir: Optional explicit scaling nadir, natural units.

    Returns:
        ``(n_solutions,)`` scores; larger is more preferred for every method,
        so a single ``argmax`` selects under any of them.

    Raises:
        ValueError: On an unknown ``method``.
    """
    scaled = scale_objectives(natural_obj, directions, ideal=ideal, nadir=nadir)
    if scaled.shape[0] == 0:
        return np.zeros(0, dtype=float)
    w = _normalized_weights(weights, scaled.shape[1])

    if method == "mean_scaled":
        return scaled @ w
    if method == "distance_to_ideal":
        regret = 1.0 - scaled
        if np.isinf(p):
            return -np.max(w * regret, axis=1)
        if p <= 0:
            raise ValueError(f"p must be positive or inf, got {p}")
        return -np.power(np.sum(w * np.power(regret, p), axis=1), 1.0 / p)
    if method == "maximin":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(w > 0, scaled / np.where(w > 0, w, 1.0), np.inf)
        return np.min(ratio, axis=1)
    raise ValueError(f"unknown method '{method}'; expected one of "
                     f"{COMPROMISE_METHODS}")


def compromise(natural_obj: np.ndarray, directions,
               method: str = "mean_scaled", weights=None, p: float = 2.0,
               ideal=None, nadir=None) -> int:
    """Row index of the best compromise solution; see :func:`compromise_scores`.

    Ties resolve to the LOWEST row index.

    Raises:
        ValueError: If the front is empty, or on an unknown ``method``.
    """
    scores = compromise_scores(natural_obj, directions, method=method,
                               weights=weights, p=p, ideal=ideal, nadir=nadir)
    if scores.size == 0:
        raise ValueError("cannot select from an empty front")
    return int(np.argmax(scores))


###############################################################################
# Diverse and rule-driven selection
###############################################################################

def select_diverse(natural_obj: np.ndarray, directions, n: int, *,
                   seed_indices=(), candidates=None, space=None,
                   ideal=None, nadir=None,
                   min_separation: float = 0.0) -> list[int]:
    """Greedily pick ``n`` maximally-spread representatives from the front.

    Farthest-point (maximin dispersion) sampling: start from ``seed_indices``
    -- or, when none are given, from the ``mean_scaled`` compromise solution so
    the result is deterministic and centred -- then repeatedly add the
    candidate whose nearest already-chosen neighbour is farthest away. This is
    what makes N representatives operationally DISTINCT instead of N variations
    on the same corner of the front, which is the failure mode of taking the
    top-N of any single ranking.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        n: Number of representatives wanted (including any seeds).
        seed_indices: Row indices to start from, kept in the given order.
        candidates: Optional row indices eligible for selection (default: all
            rows). Seeds are always kept even if outside this list.
        space: Optional ``(n_solutions, d)`` array to measure distance in --
            e.g. bound-normalized decision variables, to spread in DV space
            rather than objective space. Default: the scaled objectives.
        ideal: Optional explicit scaling ideal, natural units (ignored when
            ``space`` is given).
        nadir: Optional explicit scaling nadir, natural units.
        min_separation: Stop early once the best remaining candidate sits
            within this Euclidean distance of an already-chosen solution. The
            default 0.0 still refuses exact duplicates, so the returned list
            may be shorter than ``n``.

    Returns:
        Row indices, seeds first, then greedy additions. Length
        ``<= min(n, len(candidates) + len(seeds))``.
    """
    pts = (scale_objectives(natural_obj, directions, ideal=ideal, nadir=nadir)
           if space is None else np.atleast_2d(np.asarray(space, dtype=float)))
    n_rows = pts.shape[0]
    if n_rows == 0 or n <= 0:
        return []

    chosen = [int(i) for i in seed_indices][:n]
    if not chosen:
        chosen = [compromise(natural_obj, directions, method="mean_scaled",
                             ideal=ideal, nadir=nadir)]
    pool = (np.arange(n_rows) if candidates is None
            else np.asarray(candidates, dtype=int))
    pool = np.array([i for i in pool if i not in set(chosen)], dtype=int)

    while len(chosen) < n and pool.size:
        d = np.linalg.norm(pts[pool][:, None, :] - pts[chosen][None, :, :],
                           axis=2).min(axis=1)
        best = int(np.argmax(d))          # stable: first max wins
        if d[best] <= min_separation:
            break
        chosen.append(int(pool[best]))
        pool = np.delete(pool, best)
    return chosen


@dataclass(frozen=True)
class Selection:
    """One chosen solution and the rule that chose it.

    Attributes:
        label: Human-facing name for the figure legend / CSV.
        rule: Machine-facing identifier of the selection rule.
        index: Row index into the reference set (== re-eval ``solution_id``).
    """

    label: str
    rule: str
    index: int


def select_by_rules(rules, *, distinct: bool = True) -> list[Selection]:
    """Resolve a list of scored rules into distinct solution choices.

    Each rule contributes a full ranking rather than a single winner, so when
    two rules would land on the same solution the later rule can fall through
    to its own next-best candidate instead of being dropped or silently
    duplicated.

    Args:
        rules: Sequence of ``(label, rule, scores)`` where ``scores`` is a
            ``(n_solutions,)`` array with larger = more preferred (e.g. from
            :func:`compromise_scores`, or an external robustness score).
            ``NaN`` (an unscored solution) ranks last; ``+/-inf`` sort
            normally.
        distinct: Skip solutions already taken by an earlier rule. When a rule
            has no untaken candidate left it keeps its own top choice, so the
            returned list always has one entry per rule.

    Returns:
        One :class:`Selection` per rule, in the given order.
    """
    taken: set[int] = set()
    out: list[Selection] = []
    for label, rule, scores in rules:
        s = np.asarray(scores, dtype=float)
        if s.size == 0:
            continue
        order = np.argsort(-np.where(np.isnan(s), -np.inf, s), kind="stable")
        pick = int(order[0])
        if distinct:
            for cand in order:
                if int(cand) not in taken:
                    pick = int(cand)
                    break
        taken.add(pick)
        out.append(Selection(label=label, rule=rule, index=pick))
    return out


def normalized_dv(dv: np.ndarray, bounds) -> np.ndarray:
    """Decision variables rescaled to [0, 1] by their search bounds.

    Args:
        dv: ``(n_solutions, n_vars)`` decision-variable array.
        bounds: ``(lower, upper)`` arrays as returned by
            ``src.formulations.get_bounds``.

    Returns:
        ``(n_solutions, n_vars)`` array; a zero-width bound maps to 0.0.
    """
    lo, hi = (np.asarray(b, dtype=float) for b in bounds)
    span = hi - lo
    safe = np.where(span == 0.0, 1.0, span)
    out = (np.atleast_2d(np.asarray(dv, dtype=float)) - lo) / safe
    out[:, span == 0.0] = 0.0
    return out


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Symmetric Euclidean distance matrix over the rows of ``points``.

    Used to confirm that a selection is operationally distinct: near-zero
    off-diagonal entries over bound-normalized decision variables mean two
    "different" representatives encode nearly the same policy.
    """
    a = np.atleast_2d(np.asarray(points, dtype=float))
    return np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)


###############################################################################
# Convenience loader
###############################################################################

def load_natural_front(set_file, formulation: str = "ffmp") -> tuple:
    """Load a ``.set``/``.ref`` and return its objectives in natural units.

    Thin wrapper over ``src.load.reference_set.load_reference_set`` that also
    applies the formulation's objective count as a fail-loud column guard and
    un-negates maximize objectives, so callers of this module never touch the
    all-minimized storage orientation.

    Args:
        set_file: Path to the reference set.
        formulation: Formulation name (fixes ``n_vars``).

    Returns:
        ``(dv, natural_obj, obj_names, directions)``.
    """
    from src.formulations import (get_n_objs, get_n_vars, get_obj_directions,
                                  get_obj_names)
    from src.load.reference_set import load_reference_set
    from src.pareto_filter import to_natural

    dv, obj = load_reference_set(Path(set_file), get_n_vars(formulation),
                                 n_objs=get_n_objs())
    directions = get_obj_directions()
    return dv, to_natural(obj, directions), get_obj_names(), directions
