"""sensitivity_common.py - Shared helpers for the objective-sensitivity experiments.

The supplemental experiments live in ``scripts/supplemental/``:

* the **historic** (single-trace) random-DV diagnostic
  (``objective_sensitivity_{run,figures}.py``),
* the **epsilon-calibration** experiment
  (``epsilon_calibration_{run,figures}.py``),
* the **satisfaction-factor** sweep
  (``satisfaction_factor_{run,figures}.py``), and
* the **framing-convention** cube analysis
  (``framing_convention_analysis.py``).

They share six kinds of logic, factored here so no script copies another
(per the no-duplication / refactor-all-callers project convention):

1. **MPI plumbing** — ``get_mpi_context``, ``assign_rank_slots``, and the
   filesystem-barrier primitives (``prepare_partial_dir`` / ``mark_rank_done``
   / ``await_all_done``). The barrier avoids ``comm.bcast``/``comm.gather``,
   which are flaky on the cluster's OpenMPI build; ranks coordinate through
   ``.done`` marker files instead. The combine step itself stays in each script
   because the shard payload differs (flat CSV rows vs. a 3-D HDF5 matrix).
2. **DV sampling** — ``sample_lhs_dvs`` (Latin-hypercube within ``get_bounds``)
   and ``sample_feasible_dvs`` (uniform on the feasible region via rejection
   against the formal Borg constraints).
3. **Objective-set resolution** — ``resolve_objective_names``.
4. **Rank-correlation diagnostics** — ``kendall_tau_b`` and
   ``spearman_and_flagged`` (the Olden & Poff redundancy screen).
5. **Unit-operator vectorization** — ``apply_operator_rows`` (row-wise annual-
   unit operators for the bootstrap noise estimate).
6. **Epsilon-dominance utilities** — ``epsilon_nondominated`` (Borg-convention
   ε-box archive filter) and ``ceil_to_clean_step`` (epsilon rounding).

To preserve the import-order contract (``supplemental_config`` must set the
``NYCOPT_*`` env knobs before ``config`` is imported), this module imports
``config`` / ``src.formulations`` / ``src.objectives`` **lazily inside
functions**, never at module load.
"""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc


###############################################################################
# MPI plumbing
###############################################################################

def get_mpi_context():
    """Return ``(comm, rank, size)``; falls back to ``(None, 0, 1)`` serially.

    The fallback covers both a missing ``mpi4py`` and an importable ``mpi4py``
    whose MPI runtime fails to initialize (e.g. a laptop with no MPI installed).
    """
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        return None, 0, 1


def assign_rank_slots(n_items: int, rank: int, size: int) -> list:
    """Return the contiguous ``np.array_split`` slice of item indices for ``rank``.

    Args:
        n_items: Total number of work items (e.g. DV vectors).
        rank: This process's MPI rank.
        size: Total number of ranks.

    Returns:
        The list of integer item indices this rank owns.
    """
    return list(np.array_split(np.arange(n_items, dtype=int), size)[rank])


def prepare_partial_dir(partial_dir: Path, rank: int, *, wait_s: float = 60.0) -> None:
    """Create the per-run shard directory on rank 0; workers wait for it.

    Args:
        partial_dir: Directory that will hold per-rank shards + ``.done`` markers.
        rank: This process's MPI rank.
        wait_s: Maximum seconds a worker waits for rank 0 to create the dir.
    """
    if rank == 0:
        partial_dir.mkdir(parents=True, exist_ok=True)
        return
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if partial_dir.exists():
            return
        time.sleep(0.5)


def mark_rank_done(partial_dir: Path, rank: int) -> None:
    """Touch this rank's ``rank_{rank:03d}.done`` completion marker."""
    (partial_dir / f"rank_{rank:03d}.done").touch()


def await_all_done(partial_dir: Path, size: int, *, deadline_s: float = 1800.0,
                   poll_s: float = 2.0) -> bool:
    """Block (rank 0) until every rank's ``.done`` marker appears.

    Args:
        partial_dir: Shard directory holding the ``.done`` markers.
        size: Number of ranks expected.
        deadline_s: Hard cap on the wait, in seconds.
        poll_s: Polling interval, in seconds.

    Returns:
        ``True`` if all markers appeared before the deadline, else ``False``.
    """
    expected = {f"rank_{r:03d}.done" for r in range(size)}
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        seen = {p.name for p in partial_dir.glob("rank_*.done")}
        if seen >= expected:
            return True
        time.sleep(poll_s)
    return False


###############################################################################
# DV sampling
###############################################################################

def sample_lhs_dvs(formulation: str, seed: int, n_samples: int) -> np.ndarray:
    """Latin-hypercube DV sample scaled to the formulation bounds.

    Args:
        formulation: Formulation name whose bounds define the sampling box.
        seed: RNG seed for the LHS engine (reproducibility).
        n_samples: Number of DV vectors.

    Returns:
        Array of shape ``(n_samples, n_vars)`` within ``get_bounds(formulation)``.
    """
    from src.formulations import get_bounds

    lows, highs = get_bounds(formulation)
    sampler = qmc.LatinHypercube(d=len(lows), seed=seed)
    unit = sampler.random(n=n_samples)
    return qmc.scale(unit, lows, highs)


def sample_feasible_dvs(formulation: str, seed: int, n_samples: int, *,
                        chunk: int = 8192,
                        max_draws: int = 5_000_000) -> tuple:
    """Uniform sample of the constraint-feasible DV region, via rejection.

    Draws i.i.d. uniform vectors within ``get_bounds(formulation)`` and keeps
    only those with zero violation on both formal Borg constraints
    (``src.simulation.compute_constraint_violations`` — pure DV arithmetic, no
    simulation). Rejection from i.i.d. uniform draws yields an *exactly*
    uniform distribution on the feasible region, which is the population the
    Borg archive lives in (constraint-dominance keeps infeasible vectors out
    of the archive). Random 36-DV vectors are ~1% feasible (the flood-ordering
    constraint dominates), so expect ~100x oversampling; the constraint check
    costs microseconds per vector after the one-time defaults load.

    A plain LHS is deliberately NOT used here: conditioning an LHS on
    feasibility destroys its stratification guarantee while leaving an
    ill-characterized measure, whereas uniform + rejection has a clean one.

    Args:
        formulation: Formulation name whose bounds define the sampling box.
        seed: RNG seed (reproducibility; ranks regenerating with the same seed
            obtain the identical sample).
        n_samples: Number of feasible DV vectors to return.
        chunk: Uniform draws generated per rejection round.
        max_draws: Hard cap on total draws before giving up.

    Returns:
        ``(dvs, info)`` where ``dvs`` has shape ``(n_samples, n_vars)`` and
        ``info`` is ``{"n_draws": int, "acceptance_rate": float}`` (QC — a
        collapsing acceptance rate signals a constraint or bounds change).

    Raises:
        RuntimeError: If ``max_draws`` draws yield fewer than ``n_samples``
            feasible vectors.
    """
    from src.formulations import get_bounds
    from src.simulation import compute_constraint_violations

    lows, highs = get_bounds(formulation)
    rng = np.random.default_rng(seed)
    accepted: list = []
    n_draws = 0
    while len(accepted) < n_samples:
        if n_draws >= max_draws:
            raise RuntimeError(
                f"sample_feasible_dvs: {n_draws} draws yielded only "
                f"{len(accepted)}/{n_samples} feasible vectors "
                f"(acceptance {len(accepted) / max(n_draws, 1):.2e}); "
                "check the constraint functions / bounds."
            )
        block = rng.uniform(lows, highs, size=(chunk, len(lows)))
        for dv in block:
            n_draws += 1
            cons = compute_constraint_violations(dv, formulation)
            if all(c == 0.0 for c in cons):
                accepted.append(np.asarray(dv, dtype=float))
                if len(accepted) == n_samples:
                    break
    info = {"n_draws": int(n_draws),
            "acceptance_rate": float(n_samples) / float(n_draws)}
    return np.asarray(accepted, dtype=float), info


###############################################################################
# Objective-set resolution
###############################################################################

def resolve_objective_names(mode) -> list:
    """Resolve an objective-set selector to an ordered list of registry names.

    Args:
        mode: Either ``"full_registry"`` (every objective in
            ``src.objectives.OBJECTIVES``), ``"active"``
            (``config.ACTIVE_OBJECTIVES``), or an explicit list of registry
            names used verbatim.

    Returns:
        Ordered objective names.

    Raises:
        ValueError: If ``mode`` is an unrecognized string.
    """
    if isinstance(mode, str):
        from src.objectives import OBJECTIVES

        if mode == "full_registry":
            return list(OBJECTIVES.keys())
        if mode == "active":
            from config import ACTIVE_OBJECTIVES

            return list(ACTIVE_OBJECTIVES)
        raise ValueError(
            f"Unknown objective-set mode '{mode}'. "
            "Use 'full_registry', 'active', or a list of registry names."
        )
    return list(mode)


###############################################################################
# Rank-correlation diagnostics
###############################################################################

def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall rank correlation τ_b between two equal-length score vectors.

    Non-finite pairs are dropped. Returns ``nan`` if fewer than two valid pairs
    remain or either vector is constant after filtering. Used to compare
    DV rankings induced by different ensemble sizes / aggregation operators
    (Herman et al. 2015; McPhail et al. 2020).

    Args:
        x: First score vector (one value per DV).
        y: Second score vector (one value per DV), aligned to ``x``.

    Returns:
        τ_b in ``[-1, 1]``, or ``nan`` when undefined.
    """
    from scipy.stats import kendalltau

    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    tau, _ = kendalltau(a, b)
    return float(tau)


#: Stability rank for the redundancy keep-recommendation: a HIGHER score is the
#: more stable / stakeholder-interpretable member to retain when a pair is
#: collinear (favoring frequency/satisficing and tail/percentile forms over
#: single-worst-case extremes; Bonham et al. 2024; Quinn et al. 2017).
_STABILITY_KEYWORDS: list = [
    ("reliability", 5),
    ("cvar90", 4),
    ("_p5_", 4),
    ("_minor", 3),
    ("_action", 2),
    ("_major", 1),
    ("_max", 0),
    ("_min", 0),
    ("salt_front", 0),
]


def stability_score(name: str) -> int:
    """Heuristic 'keep me' score for redundancy pruning; higher = more stable."""
    from config import ACTIVE_OBJECTIVES

    score = 10 if name in ACTIVE_OBJECTIVES else 0  # recommended set wins first
    for kw, pts in _STABILITY_KEYWORDS:
        if kw in name:
            score += pts
            break
    return score


def spearman_and_flagged(samples: pd.DataFrame, obj_names: list,
                         threshold: float):
    """Spearman matrix (pairwise-complete) and the flagged collinear pairs.

    Objectives with fewer than 3 valid samples or zero variance are excluded
    from the matrix (they cannot yield a meaningful rank correlation). From each
    ``|rho| > threshold`` pair the more stable member (``stability_score``) is
    recommended for retention (Olden & Poff 2003; Bonham et al. 2024).

    Args:
        samples: Rows = samples, columns include the objective values.
        obj_names: Objective columns to screen, in display order.
        threshold: ``|Spearman rho|`` above which a pair is flagged collinear.

    Returns:
        ``(spearman_df, flagged_df, excluded_names)``.
    """
    usable = [n for n in obj_names
              if n in samples.columns
              and samples[n].notna().sum() >= 3
              and samples[n].nunique(dropna=True) > 1]
    excluded = [n for n in obj_names if n not in usable]

    spearman = samples[usable].corr(method="spearman")  # pairwise-complete

    flagged = []
    for a, b in combinations(usable, 2):
        rho = spearman.loc[a, b]
        if pd.notna(rho) and abs(rho) > threshold:
            keep = a if stability_score(a) >= stability_score(b) else b
            drop = b if keep == a else a
            flagged.append({"obj_a": a, "obj_b": b, "rho": float(rho),
                            "keep": keep, "consider_dropping": drop})
    flagged_df = pd.DataFrame(
        flagged, columns=["obj_a", "obj_b", "rho", "keep", "consider_dropping"]
    ).sort_values("rho", key=lambda s: s.abs(), ascending=False)
    return spearman, flagged_df, excluded


###############################################################################
# Unit-operator vectorization (epsilon-calibration bootstrap)
###############################################################################

def apply_operator_rows(op, arr: np.ndarray) -> np.ndarray:
    """Apply an annual-unit operator to each row of a 2-D pooled-unit array.

    Vectorized re-implementations of the three §2 unit operators
    (``FailureFrequencyOp`` / ``PooledPercentileOp`` / ``PooledMeanOp``),
    reproducing their non-finite sentinel semantics exactly; any other callable
    falls back to a row loop. Used by the epsilon-calibration bootstrap, where
    the operator is evaluated on thousands of realization-resampled unit pools
    per objective.

    Args:
        op: A unit operator (callable ``units -> float``).
        arr: Array of shape ``(n_rows, n_pooled_units)``; each row is one
            pooled unit-year sample.

    Returns:
        Array of shape ``(n_rows,)`` with ``float(op(row))`` per row.
    """
    from src.objectives_ensemble import (
        FailureFrequencyOp,
        PooledMeanOp,
        PooledPercentileOp,
    )

    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {a.shape}")
    if isinstance(op, FailureFrequencyOp):
        ok = np.isfinite(a) & (a < op.k)
        return ok.sum(axis=1) / float(a.shape[1])
    if isinstance(op, PooledPercentileOp):
        filled = np.where(np.isfinite(a), a, op.worst_value)
        return np.percentile(filled, op.q, axis=1)
    if isinstance(op, PooledMeanOp):
        filled = np.where(np.isfinite(a), a, op.worst_value)
        return filled.mean(axis=1)
    return np.array([float(op(row)) for row in a], dtype=float)


###############################################################################
# Epsilon-dominance utilities (epsilon-calibration analysis)
###############################################################################

def epsilon_nondominated(objs: np.ndarray, epsilons: Sequence[float]) -> np.ndarray:
    """Indices of the ε-box nondominated subset (Borg archive convention).

    All objectives MINIMIZED (Borg sign convention — negate maximize values
    before calling). Each solution maps to the ε-box ``floor(f / ε)``; a box
    dominates another when it is weakly smaller in every coordinate and
    strictly smaller in at least one; within a box the solution closest
    (Euclidean, in ε units) to the box's lower corner represents it
    (Laumanns et al. 2002; Hadka & Reed 2013). Rows with any non-finite
    objective are excluded — they cannot enter a Borg archive.

    Args:
        objs: Array of shape ``(n_solutions, n_objectives)``, minimized.
        epsilons: Positive per-objective epsilon values.

    Returns:
        Sorted integer indices (into ``objs`` rows) of the archive members.

    Raises:
        ValueError: On a non-positive epsilon or a shape mismatch.
    """
    F = np.asarray(objs, dtype=float)
    eps = np.asarray(epsilons, dtype=float)
    if F.ndim != 2 or F.shape[1] != eps.size:
        raise ValueError(f"shape mismatch: objs {F.shape} vs {eps.size} epsilons")
    if not np.all(eps > 0.0):
        raise ValueError(f"epsilons must be positive, got {eps}")

    valid = np.flatnonzero(np.isfinite(F).all(axis=1))
    if valid.size == 0:
        return np.array([], dtype=int)
    scaled = F[valid] / eps
    boxes = np.floor(scaled)
    corner_dist = ((scaled - boxes) ** 2).sum(axis=1)

    # One representative per occupied box: closest to the box lower corner
    # (stable tie-break on original order).
    rep_for_box: dict = {}
    for local, box in enumerate(map(tuple, boxes)):
        best = rep_for_box.get(box)
        if best is None or corner_dist[local] < corner_dist[best]:
            rep_for_box[box] = local

    box_arr = np.array(list(rep_for_box.keys()), dtype=float)
    reps = np.array(list(rep_for_box.values()), dtype=int)
    keep = np.ones(len(reps), dtype=bool)
    for i in range(len(reps)):
        if not keep[i]:
            continue
        dominated = (
            (box_arr <= box_arr[i]).all(axis=1)
            & (box_arr < box_arr[i]).any(axis=1)
        )
        if dominated.any():
            keep[i] = False
    return np.sort(valid[reps[keep]])


def ceil_to_clean_step(x: float,
                       mantissas: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 5.0)
                       ) -> float:
    """Smallest 'clean' value >= ``x``: a mantissa from ``mantissas`` x 10^k.

    Used to round a raw epsilon requirement (max of signal scale, noise floor,
    and metric granularity) UP to an interpretable step in native units, so
    the published epsilon never under-resolves any of its floors. Returns
    ``nan`` for non-finite or non-positive input (a degenerate objective —
    flag upstream rather than fabricate an epsilon).

    Args:
        x: Raw positive requirement.
        mantissas: Ascending clean mantissas in ``[1, 10)``.

    Returns:
        The clean step, e.g. ``0.0117 -> 0.015``, ``0.9 -> 1.0``, ``1.34 -> 1.5``.
    """
    if not np.isfinite(x) or x <= 0.0:
        return float("nan")
    exp = np.floor(np.log10(x))
    rel_tol = 1.0 + 1e-12
    for m in sorted(mantissas):
        candidate = m * 10.0 ** exp
        if candidate * rel_tol >= x:
            return float(candidate)
    return float(10.0 ** (exp + 1))
