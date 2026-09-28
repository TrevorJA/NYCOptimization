"""transfer_stats.py - Arithmetic for the transfer-evaluation matrix.

Every function here is arithmetic over arrays that are already in memory:
nothing simulates, nothing reads a file, nothing imports ``config`` or
``supplemental_config``. That keeps the module importable from a test without
any environment set up, and keeps the definitions of the three readouts in one
reviewable place.

No new metric is defined. The dominance readout is a thin aggregation over
:mod:`src.solution_selection`, which already implements weak Pareto dominance
against a reference vector and is verified against brute force in
``tests/test_solution_selection.py``. The merged-reference-set readout is
attribution bookkeeping over an epsilon-nondominated archive computed by
:func:`src.sensitivity_common.epsilon_nondominated`.

Conventions
-----------
Orientation is the single most dangerous detail in this instrument, because the
two conventions appear within one function call of each other:

- **Natural units** - objectives as a hydrologist reads them (reliability
  0.71, deficit 35.0 %). The scenario-matched baseline CSVs written by
  ``scripts/main/run_baseline.py`` are in natural units, and
  :mod:`src.solution_selection` expects natural units plus a ``directions``
  vector.
- **Borg orientation** - every objective minimized, so a maximize objective is
  stored negated. ``.set`` files (both dialects) and
  ``src.sensitivity_common.epsilon_nondominated`` use this.

``directions`` is the per-objective ``+1`` (maximize) / ``-1`` (minimize)
vector from ``src.formulations.get_obj_directions``. Converting between the two
conventions is elementwise multiplication by ``directions``; this module does
it explicitly at every boundary rather than relying on the caller.

A unit of attribution is a **solution**, identified by its 36-value decision
vector. Matching on decision variables rather than objective vectors is exact,
is unique across independent searches, and sidesteps the case of two distinct
policies landing on the same objective vector.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "to_borg",
    "to_natural",
    "dominance_summary",
    "epsilon_dominance_mask",
    "dv_key_array",
    "attribute_members",
    "contribution_table",
    "eps_fraction_table",
    "enrichment_table",
    "leave_one_out_enrichment",
    "occupied_boxes",
]


###############################################################################
# Orientation
###############################################################################

def to_borg(natural_obj: np.ndarray, directions) -> np.ndarray:
    """Natural units -> Borg orientation (all objectives minimized).

    Borg minimizes everything, so a **maximize** objective is stored negated
    and a minimize objective is stored unchanged - matching
    ``AnnualUnitObjective.compute_for_borg``. The multiplier is therefore
    ``-directions``, NOT ``directions``.

    Note this is the exact opposite of
    ``src.solution_selection.orient_maximize``, which multiplies by
    ``+directions`` to make larger mean better. The two conventions differ by a
    global sign and both appear in this instrument, so every conversion goes
    through these two functions rather than an inline multiplication.

    Args:
        natural_obj: ``(n, n_objs)`` or ``(n_objs,)`` in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).

    Returns:
        A float copy with maximize columns negated.
    """
    return np.asarray(natural_obj, dtype=float) * -np.asarray(directions, dtype=float)


def to_natural(borg_obj: np.ndarray, directions) -> np.ndarray:
    """Borg orientation -> natural units.

    The inverse of :func:`to_borg`; the transform is its own inverse because
    ``directions`` is elementwise +/-1, but both names exist so call sites read
    in the direction they mean.
    """
    return np.asarray(borg_obj, dtype=float) * -np.asarray(directions, dtype=float)


###############################################################################
# Readout 1: dominance over the FFMP baseline
###############################################################################

def dominance_summary(natural_obj: np.ndarray, baseline_natural, directions,
                      epsilons, tol: float = 0.0) -> dict:
    """Summarize how a solution set compares to one baseline vector.

    Four numbers, because strict dominance on eight objectives is a stringent
    test that can legitimately be zero everywhere (the campaign's incumbent
    already has a joint satisficing score of zero, with Montague reliability
    binding), and a table of zeros would say nothing about how close the set
    came.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        baseline_natural: ``(n_objs,)`` baseline in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        epsilons: Per-objective epsilon values, used for the
            epsilon-relaxed dominance count (a solution counts as no-worse when
            it is within one epsilon box of the baseline on that axis).
        tol: Exact-comparison slack passed through to
            :func:`src.solution_selection.dominance_mask`. Default 0.0.

    Returns:
        Dict with ``n_solutions``, ``n_dominating``, ``frac_dominating``,
        ``n_eps_dominating``, ``frac_eps_dominating``, ``n_not_dominated``,
        ``frac_not_dominated``, ``mean_objectives_beaten`` and
        ``objectives_beaten_hist`` (length ``n_objs + 1``).

    Raises:
        ValueError: On a shape mismatch between the set, the baseline, the
            directions and the epsilons.
    """
    from src.solution_selection import dominance_mask, n_objectives_beaten

    obj = np.atleast_2d(np.asarray(natural_obj, dtype=float))
    base = np.asarray(baseline_natural, dtype=float).ravel()
    dirs = np.asarray(directions, dtype=float).ravel()
    eps = np.asarray(epsilons, dtype=float).ravel()
    n_objs = obj.shape[1]
    if base.shape[0] != n_objs or dirs.shape[0] != n_objs or eps.shape[0] != n_objs:
        raise ValueError(
            f"shape mismatch: set has {n_objs} objectives but baseline has "
            f"{base.shape[0]}, directions {dirs.shape[0]}, epsilons {eps.shape[0]}")

    mask = dominance_mask(obj, base, dirs, tol=tol)
    beaten = n_objectives_beaten(obj, base, dirs, tol=tol)

    # "Not dominated by the baseline" is the converse test: does the single
    # baseline vector dominate this solution? Reported because a set can fail
    # to dominate the incumbent while still not being beaten by it.
    z = obj * dirs
    r = base * dirs
    dominated_by_base = (r >= z - tol).all(axis=1) & (r > z + tol).any(axis=1)

    eps_mask = epsilon_dominance_mask(obj, base, dirs, eps)

    n = obj.shape[0]
    return {
        "n_solutions": int(n),
        "n_dominating": int(mask.sum()),
        "frac_dominating": float(mask.mean()) if n else float("nan"),
        "n_eps_dominating": int(eps_mask.sum()),
        "frac_eps_dominating": float(eps_mask.mean()) if n else float("nan"),
        "n_not_dominated": int((~dominated_by_base).sum()),
        "frac_not_dominated": float((~dominated_by_base).mean()) if n else float("nan"),
        "mean_objectives_beaten": float(beaten.mean()) if n else float("nan"),
        "objectives_beaten_hist": np.bincount(beaten, minlength=n_objs + 1).tolist(),
    }


def epsilon_dominance_mask(natural_obj: np.ndarray, reference_natural,
                           directions, epsilons) -> np.ndarray:
    """Which solutions dominate a reference vector at epsilon resolution.

    Weak dominance applied to epsilon-box indices rather than raw values, which
    is the resolution the search itself worked at: a difference smaller than
    ``epsilon`` was never a difference the optimizer could see, so counting it
    as dominance overstates the result.

    Args:
        natural_obj: ``(n_solutions, n_objs)`` in natural units.
        reference_natural: ``(n_objs,)`` in natural units.
        directions: Per-objective direction ints (+1 maximize, -1 minimize).
        epsilons: Per-objective epsilon values, all strictly positive.

    Returns:
        Boolean array of length ``n_solutions``.

    Raises:
        ValueError: If any epsilon is not strictly positive.
    """
    eps = np.asarray(epsilons, dtype=float).ravel()
    if np.any(~np.isfinite(eps)) or np.any(eps <= 0):
        raise ValueError(f"epsilons must be finite and positive, got {eps.tolist()}")
    obj = np.atleast_2d(np.asarray(natural_obj, dtype=float))
    if obj.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    ref = np.asarray(reference_natural, dtype=float).ravel()
    dirs = np.asarray(directions, dtype=float).ravel()

    # Work in Borg orientation so "better" is uniformly "smaller box index".
    zb = np.floor(to_borg(obj, dirs) / eps)
    rb = np.floor(to_borg(ref, dirs) / eps)
    return (zb <= rb).all(axis=1) & (zb < rb).any(axis=1)


###############################################################################
# Readout 2: merged reference set attribution
###############################################################################

def dv_key_array(dvs: np.ndarray) -> np.ndarray:
    """Hashable per-row keys for a decision-variable matrix.

    Uses the exact float bit pattern of each row, so two rows match only when
    every decision variable is bit-identical. That is the right strictness
    here: rows are copied verbatim out of the same ``.set`` text, never
    recomputed, so a genuine match is exact and a near-match is a different
    policy.

    Args:
        dvs: ``(n_solutions, n_vars)`` decision variables.

    Returns:
        ``(n_solutions,)`` array of bytes objects.
    """
    arr = np.ascontiguousarray(np.asarray(dvs, dtype=float))
    if arr.ndim != 2:
        raise ValueError(f"dvs must be 2-D, got shape {arr.shape}")
    return np.array([row.tobytes() for row in arr], dtype=object)


def attribute_members(member_keys, source_keys: dict) -> tuple[dict, dict, int]:
    """Attribute merged-set members to the source sets that contain them.

    Args:
        member_keys: Sequence of decision-vector keys, one per merged-set
            member (from :func:`dv_key_array`).
        source_keys: Mapping of source name -> array of that source's keys.

    Returns:
        ``(credited, disjoint, n_shared)``:

        - ``credited`` maps source -> count of merged-set members that source
          contains. A member present in two sources is credited to **both**, so
          these counts can sum to more than the merged-set size.
        - ``disjoint`` maps source -> count under a deterministic tie-break
          (the first source in sorted ``source_keys`` order that contains the
          member), so these counts sum to exactly the number of attributed
          members.
        - ``n_shared`` is how many members were claimed by more than one
          source, i.e. how much the two attributions can differ.

        Reporting both is deliberate: one of them has to make an arbitrary
        choice, and the reader should be able to see how much that choice
        moved the answer rather than inherit it silently.
    """
    names = sorted(source_keys)
    lookup = {name: set(np.asarray(source_keys[name], dtype=object).tolist())
              for name in names}
    credited = {name: 0 for name in names}
    disjoint = {name: 0 for name in names}
    n_shared = 0

    for key in member_keys:
        owners = [name for name in names if key in lookup[name]]
        if not owners:
            continue
        for name in owners:
            credited[name] += 1
        disjoint[owners[0]] += 1
        if len(owners) > 1:
            n_shared += 1
    return credited, disjoint, n_shared


def contribution_table(credited: dict, disjoint: dict, source_sizes: dict,
                       merged_size: int) -> list:
    """Per-source contribution rows for one target ensemble.

    Two shares, because the three source sets differ in size by a factor of
    three (335 / 991 / 784) and the two questions have different answers:

    - ``composition_share`` = contributed / merged-set size. What the merged
      reference set is made of. A larger archive contributes more members
      partly because it has more members to contribute.
    - ``contribution_rate`` = contributed / source-set size. How likely one
      solution from that search is to survive the merge. Size-invariant, and
      the fairer cross-design comparison.

    Args:
        credited: Source -> credited count (may double-count shared members).
        disjoint: Source -> tie-broken count.
        source_sizes: Source -> number of solutions in that source set.
        merged_size: Number of members in the merged reference set.

    Returns:
        List of dicts, one per source, sorted by source name.
    """
    rows = []
    for name in sorted(credited):
        n_src = int(source_sizes.get(name, 0))
        n_cred = int(credited[name])
        rows.append({
            "source": name,
            "source_set_size": n_src,
            "merged_set_size": int(merged_size),
            "n_contributed": n_cred,
            "n_contributed_disjoint": int(disjoint.get(name, 0)),
            "composition_share": n_cred / merged_size if merged_size else float("nan"),
            "composition_share_disjoint": (
                disjoint.get(name, 0) / merged_size if merged_size else float("nan")),
            "contribution_rate": n_cred / n_src if n_src else float("nan"),
        })
    return rows


###############################################################################
# Path consistency
###############################################################################

def eps_fraction_table(driver_borg: np.ndarray, stored_borg: np.ndarray,
                       epsilons, obj_names) -> list:
    """Per-objective agreement between two evaluation paths, in epsilon units.

    The instrument reads its diagonal cells from stored ``.set`` objective
    columns (the search code path) and its off-diagonal cells from the driver.
    Readout 2 puts both into one nondominated sort, so the question that
    matters is not "are they equal" - pywrdrb's LP carries accepted run-to-run
    jitter - but "is the difference small enough that no solution changes
    epsilon box". Expressing the difference as a fraction of each objective's
    epsilon answers exactly that, in the study's own currency.

    Args:
        driver_borg: ``(n, n_objs)`` objectives from this driver, Borg orientation.
        stored_borg: ``(n, n_objs)`` matching rows from the ``.set``, Borg orientation.
        epsilons: Per-objective epsilon values.
        obj_names: Objective names, in column order.

    Returns:
        List of dicts, one per objective, with ``max_abs_diff``,
        ``median_abs_diff``, ``max_diff_over_eps``, ``median_diff_over_eps``
        and ``n_compared``.

    Raises:
        ValueError: On a shape mismatch between the two matrices.
    """
    a = np.atleast_2d(np.asarray(driver_borg, dtype=float))
    b = np.atleast_2d(np.asarray(stored_borg, dtype=float))
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: driver {a.shape} vs stored {b.shape}")
    eps = np.asarray(epsilons, dtype=float).ravel()
    names = list(obj_names)
    if a.shape[1] != eps.shape[0] or a.shape[1] != len(names):
        raise ValueError(
            f"{a.shape[1]} objective columns but {eps.shape[0]} epsilons and "
            f"{len(names)} names")

    rows = []
    d = np.abs(a - b)
    for k, name in enumerate(names):
        col = d[:, k]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            rows.append({"objective": name, "n_compared": 0,
                         "max_abs_diff": float("nan"),
                         "median_abs_diff": float("nan"),
                         "eps": float(eps[k]),
                         "max_diff_over_eps": float("nan"),
                         "median_diff_over_eps": float("nan")})
            continue
        rows.append({
            "objective": name,
            "n_compared": int(finite.size),
            "max_abs_diff": float(finite.max()),
            "median_abs_diff": float(np.median(finite)),
            "eps": float(eps[k]),
            "max_diff_over_eps": float(finite.max() / eps[k]),
            "median_diff_over_eps": float(np.median(finite) / eps[k]),
        })
    return rows


###############################################################################
# Readout 2 diagnostics: is a contribution share a design effect, or arithmetic?
###############################################################################

def enrichment_table(member_sources, pool_sizes: dict) -> list:
    """Contribution enrichment relative to a set's share of the pooled input.

    A composition share is not interpretable on its own. If every solution were
    equally likely to survive the merge, each source would contribute in
    proportion to how many solutions it put into the pool, so a set holding
    47 % of the pool would supply 47 % of the merged set with no advantage
    whatsoever. Enrichment divides the observed share by that expectation:

    ``enrichment = (contributed / merged size) / (source size / pool size)``

    Under the null of no design effect it is 1.0 regardless of set sizes, which
    is what makes it comparable across the three unequal archives
    (335 / 991 / 784). It is the same normalisation an over-representation
    analysis uses, and it is reported alongside the raw share rather than
    instead of it.

    Args:
        member_sources: Sequence of source names, one per merged-set member.
        pool_sizes: Source name -> number of solutions that source contributed
            to the pooled input.

    Returns:
        List of dicts with ``source``, ``pool_share``, ``merged_share``,
        ``enrichment``, ``n_contributed`` and ``merged_set_size``, sorted by
        source name. Enrichment is NaN for a source absent from the pool.
    """
    members = list(member_sources)
    total_pool = sum(int(v) for v in pool_sizes.values())
    n_merged = len(members)
    rows = []
    for name in sorted(pool_sizes):
        n_src = int(pool_sizes[name])
        contributed = sum(1 for m in members if m == name)
        pool_share = n_src / total_pool if total_pool else float("nan")
        merged_share = contributed / n_merged if n_merged else float("nan")
        rows.append({
            "source": name,
            "pool_share": pool_share,
            "merged_share": merged_share,
            "enrichment": (merged_share / pool_share) if pool_share else float("nan"),
            "n_contributed": contributed,
            "merged_set_size": int(n_merged),
        })
    return rows


def occupied_boxes(borg_obj: np.ndarray, epsilons) -> int:
    """Number of distinct epsilon boxes a set occupies.

    The epsilon archive retains at most one member per box, so a set spread
    over many boxes can survive the thinning better than an equally good set
    whose members crowd into few boxes. Reporting box occupancy per solution
    separates that mechanism from a genuine dominance advantage.
    """
    eps = np.asarray(epsilons, dtype=float).ravel()
    arr = np.atleast_2d(np.asarray(borg_obj, dtype=float))
    finite = arr[np.isfinite(arr).all(axis=1)]
    if finite.size == 0:
        return 0
    return int(np.unique(np.floor(finite / eps).astype(np.int64), axis=0).shape[0])


def leave_one_out_enrichment(archive_fn, n_objs: int, obj_names) -> list:
    """Enrichment with each objective dropped in turn.

    An eight-objective nondominated archive can be driven by a single axis: a
    set that is clearly better on one objective is nondominated regardless of
    the other seven. Recomputing the archive on each seven-objective subset
    localises the effect. A source whose enrichment collapses when objective
    ``k`` is removed owed its contribution to objective ``k``.

    Args:
        archive_fn: Callable taking a list of retained objective indices and
            returning ``{source: enrichment}`` for the archive built on them.
        n_objs: Number of objectives.
        obj_names: Objective names, in column order.

    Returns:
        List of dicts, one per (dropped objective) plus one baseline row with
        ``dropped = "none"``.
    """
    names = list(obj_names)
    rows = [{"dropped": "none", **archive_fn(list(range(n_objs)))}]
    for k in range(n_objs):
        keep = [j for j in range(n_objs) if j != k]
        rows.append({"dropped": names[k], **archive_fn(keep)})
    return rows
