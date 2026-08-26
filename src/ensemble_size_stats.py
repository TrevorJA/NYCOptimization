"""ensemble_size_stats.py - Pure statistics for the ensemble-size diagnostics.

Every function here is arithmetic on a persisted per-realization annual-unit
library (``(n_policy, n_realization, n_objective, n_unit)``) or on objective
vectors composed from it; nothing simulates. Used by
``scripts/supplemental/ensemble_size_figures.py`` and covered by
``tests/test_ensemble_size_diagnostics.py``. Design and pre-registered
criteria: ``docs/notes/methods/ensemble_size_diagnostics.md`` §4.

Conventions
-----------
* The unit of independence is the REALIZATION: replicates, bootstraps, and
  subsamples index the realization axis; unit-years are pooled, never
  resampled, except in the deliberately naive unit-level bootstrap that the
  effective-sample-size ratio compares against.
* "Natural" values follow each objective's own orientation (reliabilities
  maximized, deficits minimized). "Borg form" negates maximized objectives so
  every objective is minimized; epsilon-dominance is evaluated in Borg form
  with the Borg box convention (``floor(f / eps)``).
* A dominance relation between policies ``a`` and ``b`` is coded ``+1`` (a
  epsilon-dominates b), ``-1`` (b epsilon-dominates a), ``0`` (incomparable,
  including the same box).
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from src.sensitivity_common import apply_operator_rows


###############################################################################
# Composition: member set -> objective vector
###############################################################################

def compose_objectives(units: np.ndarray, rows: Sequence[int],
                       operators: Sequence) -> np.ndarray:
    """Objective values of every policy on the ensemble formed by ``rows``.

    Stage (ii) of the two-layer scheme applied offline: each policy's
    unit-years over the member realizations are pooled and collapsed with the
    objective's own unit operator, exactly as the search driver does on the
    live ensemble.

    Args:
        units: Library tensor ``(n_policy, n_real, n_obj, n_unit)`` in natural
            units (all-NaN slab = failed realization; NaN unit-years are the
            operators' own failure sentinels).
        rows: Realization indices (into axis 1) of the ensemble members.
        operators: One unit operator per objective, in axis-2 order.

    Returns:
        Array ``(n_policy, n_obj)`` of natural-unit objective values.
    """
    rows = np.asarray(rows, dtype=int)
    n_policy, _, n_obj, _ = units.shape
    out = np.empty((n_policy, n_obj), dtype=float)
    for k, op in enumerate(operators):
        pooled = units[:, rows, k, :].reshape(n_policy, -1)
        out[:, k] = apply_operator_rows(op, pooled)
    return out


def borg_form(values: np.ndarray, directions: Sequence[str]) -> np.ndarray:
    """Negate maximized objectives so every column is minimized."""
    signs = np.array([-1.0 if d == "maximize" else 1.0 for d in directions])
    return np.asarray(values, dtype=float) * signs


def better_sign(directions: Sequence[str]) -> np.ndarray:
    """``+1`` where larger natural values are better, ``-1`` where smaller are."""
    return np.array([1.0 if d == "maximize" else -1.0 for d in directions])


###############################################################################
# Replicate construction
###############################################################################

def disjoint_prefix_blocks(p_ref: int, n: int) -> list[np.ndarray]:
    """Contiguous disjoint blocks ``[r*n, (r+1)*n)`` of an i.i.d. reference prefix.

    A prefix of an i.i.d. pool is an exact i.i.d. sample, so any partition of it
    into size-``n`` blocks yields independent size-``n`` i.i.d. samples.
    """
    if n <= 0 or p_ref < n:
        raise ValueError(f"cannot cut size-{n} blocks from a {p_ref}-row reference")
    return [np.arange(r * n, (r + 1) * n) for r in range(p_ref // n)]


def supplemented_replicates(p_ref: int, n: int, min_replicates: int,
                            seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Disjoint blocks, topped up to ``min_replicates`` with random subsets.

    Args:
        p_ref: Reference size.
        n: Replicate size.
        min_replicates: Target count; random overlapping subsets are appended
            when the disjoint count falls short.
        seed: RNG seed for the appended subsets.

    Returns:
        ``(replicates, is_random)`` — the index arrays and a boolean flag per
        replicate marking the supplemented (overlapping) ones.
    """
    reps = disjoint_prefix_blocks(p_ref, n)
    flags = [False] * len(reps)
    rng = np.random.default_rng([seed, n])
    while len(reps) < min_replicates:
        reps.append(np.sort(rng.choice(p_ref, size=n, replace=False)))
        flags.append(True)
    return reps, np.asarray(flags, dtype=bool)


###############################################################################
# Precision statistics
###############################################################################

def level_se(values: np.ndarray) -> np.ndarray:
    """SD across replicates (ddof=1) of ``(R, n_policy, n_obj)`` values."""
    v = np.asarray(values, dtype=float)
    if v.shape[0] < 2:
        return np.full(v.shape[1:], np.nan)
    return np.std(v, axis=0, ddof=1)


def pair_index(n_policy: int) -> list[tuple[int, int]]:
    """All unordered policy pairs ``(a, b)`` with ``a < b``."""
    return list(combinations(range(n_policy), 2))


def paired_differences(values: np.ndarray) -> np.ndarray:
    """Per-replicate differences ``J_a - J_b`` for every pair: ``(R, n_pair, n_obj)``."""
    v = np.asarray(values, dtype=float)
    pairs = pair_index(v.shape[1])
    return np.stack([v[:, a, :] - v[:, b, :] for a, b in pairs], axis=1)


def paired_se(values: np.ndarray) -> np.ndarray:
    """SD across replicates of the paired difference, per pair: ``(n_pair, n_obj)``."""
    d = paired_differences(values)
    if d.shape[0] < 2:
        return np.full(d.shape[1:], np.nan)
    return np.std(d, axis=0, ddof=1)


def summarize_over_pairs(per_pair: np.ndarray) -> dict[str, np.ndarray]:
    """Max, P90, and median over the pair axis of a ``(n_pair, n_obj)`` array."""
    return {
        "max": np.nanmax(per_pair, axis=0),
        "p90": np.nanpercentile(per_pair, 90, axis=0),
        "median": np.nanmedian(per_pair, axis=0),
    }


###############################################################################
# Epsilon-dominance relations and flip rate
###############################################################################

def epsilon_relations(values: np.ndarray, epsilons: Sequence[float],
                      directions: Sequence[str]) -> np.ndarray:
    """Pairwise epsilon-dominance codes for one composed objective matrix.

    Args:
        values: ``(n_policy, n_obj)`` natural-unit objective values.
        epsilons: Positive per-objective precisions.
        directions: Per-objective ``"maximize"`` / ``"minimize"``.

    Returns:
        ``(n_pair,)`` int array over :func:`pair_index` order: ``+1`` if the
        first policy's epsilon box weakly dominates the second's with at least
        one strict coordinate, ``-1`` for the reverse, else ``0``.
    """
    F = borg_form(values, directions)
    eps = np.asarray(epsilons, dtype=float)
    if np.any(eps <= 0):
        raise ValueError(f"epsilons must be positive, got {eps}")
    boxes = np.floor(F / eps)
    codes = []
    for a, b in pair_index(F.shape[0]):
        if not (np.all(np.isfinite(boxes[a])) and np.all(np.isfinite(boxes[b]))):
            codes.append(0)
            continue
        a_le = np.all(boxes[a] <= boxes[b])
        b_le = np.all(boxes[b] <= boxes[a])
        if a_le and np.any(boxes[a] < boxes[b]):
            codes.append(1)
        elif b_le and np.any(boxes[b] < boxes[a]):
            codes.append(-1)
        else:
            codes.append(0)
    return np.asarray(codes, dtype=int)


def majority_relation(codes: np.ndarray) -> np.ndarray:
    """Per-pair modal code over replicates of ``(R, n_pair)`` codes.

    Ties resolve toward ``0`` (incomparable), then ``+1``, then ``-1`` — the
    conservative reading when replicates disagree.
    """
    c = np.asarray(codes, dtype=int)
    out = np.zeros(c.shape[1], dtype=int)
    for j in range(c.shape[1]):
        counts = {v: int(np.sum(c[:, j] == v)) for v in (0, 1, -1)}
        out[j] = max(counts, key=lambda v: (counts[v], {0: 2, 1: 1, -1: 0}[v]))
    return out


def flip_rate(codes: np.ndarray, reference: np.ndarray) -> float:
    """Mean over replicates of the fraction of pairs whose code differs from ``reference``."""
    c = np.asarray(codes, dtype=int)
    ref = np.asarray(reference, dtype=int)
    return float(np.mean(c != ref[None, :]))


###############################################################################
# Bias / construction shift
###############################################################################

def optimism(values: np.ndarray, reference: np.ndarray,
             directions: Sequence[str]) -> np.ndarray:
    """Signed ``(R, n_policy, n_obj)`` gaps, positive when the replicate looks better.

    ``sign * (J_N - J_ref)`` with ``sign = +1`` for maximized objectives. For
    fixed policies this is estimator bias (order-statistic operators), not
    selection optimism; the mean over replicates is the reported quantity.
    """
    s = better_sign(directions)
    return (np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)[None]) * s


###############################################################################
# Bootstraps (realization-level vs naive unit-level) and n_eff
###############################################################################

def bootstrap_sd(units_po: np.ndarray, op, b: int, rng: np.random.Generator,
                 level: str = "realization") -> float:
    """Bootstrap SD of one policy x objective statistic on one member set.

    Args:
        units_po: ``(R, U)`` unit-years of the member realizations.
        op: The objective's unit operator.
        b: Bootstrap draws.
        rng: Generator.
        level: ``"realization"`` resamples whole realizations with replacement
            (the honest block bootstrap); ``"unit"`` resamples unit-years
            i.i.d. from the pooled sample (the naive comparator).

    Returns:
        SD (ddof=1) over the ``b`` bootstrap statistics.
    """
    u = np.asarray(units_po, dtype=float)
    n_real, n_unit = u.shape
    if level == "realization":
        idx = rng.integers(0, n_real, size=(b, n_real))
        pools = u[idx].reshape(b, n_real * n_unit)
    elif level == "unit":
        flat = u.reshape(-1)
        idx = rng.integers(0, flat.size, size=(b, flat.size))
        pools = flat[idx]
    else:
        raise ValueError(f"unknown bootstrap level {level!r}")
    stats = apply_operator_rows(op, pools)
    return float(np.std(stats, ddof=1))


def n_eff_ratio(sd_unit: float, sd_realization: float) -> float:
    """``n_eff / (N (L-1))`` = ``(SD_unit / SD_realization)^2``.

    For a mean-type statistic ``Var = sigma^2 / n_eff`` under dependence and
    ``sigma^2 / (N (L-1))`` under the i.i.d.-unit fiction, so the squared SD
    ratio is the fraction of the pooled unit count that acts independently.
    """
    if not np.isfinite(sd_realization) or sd_realization <= 0:
        return np.nan
    return float((sd_unit / sd_realization) ** 2)


###############################################################################
# Closed forms
###############################################################################

def p_at_least_one_beyond(q: float, n: int) -> float:
    """P(an i.i.d. sample of ``n`` holds >= 1 member beyond pool quantile ``q``) = 1 - q^n."""
    return float(1.0 - q ** n)
