"""reeval_core.py - shared helpers for re-evaluating Pareto-optimal policies on
a COMMON (held-out) streamflow ensemble.

Both ``src.reevaluate`` (multiprocessing) and ``src.reevaluate_mpi`` (MPI) use
these so the two execution paths score solutions identically.

Why this exists
---------------
Comparing the objective scores an MOEA produced for *different* search
ensembles is invalid — each arm's scores are computed on its own ensemble. The
only sound cross-design comparison re-evaluates every arm's Pareto **policies**
(decision-variable vectors) on ONE common ensemble and compares those scores.

Flexibility
-----------
The common ensemble is whatever ``config.REEVAL_ENSEMBLE_SPEC`` resolves to,
selected by the ``NYCOPT_REEVAL_ENSEMBLE_PRESET`` env var. It can be ANY
registered preset, a ``kn_{Y}yr_n{N}`` slug, or any staged ensemble directory
carrying a ``_meta.json`` (see ``src.ensembles.get_ensemble_spec``). Swap the
re-eval ensemble by changing that one env var — nothing else. Outputs are
written under a per-ensemble subdir (``reeval/{tag}/``) so re-evals on different
common ensembles never clobber each other.

The objective *set* is resolved against the re-eval ensemble (ensemble
satisficing objectives when it ``is_ensemble``, else the single-trace set),
using the same objective names (``config.ACTIVE_OBJECTIVES``) for every arm —
so scores are directly comparable across arms.
"""
from __future__ import annotations

import re

# Lazily-built, process-local cache of (objective_set, ensemble_spec, is_ensemble)
# for the default (config-driven) re-eval target. Avoids rebuilding per solution
# and sidesteps pickling the objective set into multiprocessing workers (each
# spawned worker re-imports config and rebuilds from inherited env vars).
_REEVAL_CACHE = None


def resolve_reeval(objectives=None, reeval_spec=None):
    """Resolve the (objective_set, ensemble_spec, is_ensemble) for re-eval.

    With no args, reads ``config.REEVAL_ENSEMBLE_SPEC`` and
    ``config.ACTIVE_OBJECTIVES`` and caches the result. Pass explicit
    ``objectives`` / ``reeval_spec`` to override (not cached).
    """
    global _REEVAL_CACHE
    if _REEVAL_CACHE is not None and objectives is None and reeval_spec is None:
        return _REEVAL_CACHE

    from config import REEVAL_ENSEMBLE_SPEC, ACTIVE_OBJECTIVES
    spec = reeval_spec if reeval_spec is not None else REEVAL_ENSEMBLE_SPEC
    names = objectives if objectives is not None else ACTIVE_OBJECTIVES

    is_ensemble = bool(spec is not None and spec.is_ensemble)
    if is_ensemble:
        from src.objectives_ensemble import build_ensemble_objective_set
        obj_set = build_ensemble_objective_set(names)
    else:
        from src.objectives import build_objective_set
        obj_set = build_objective_set(names)

    result = (obj_set, spec, is_ensemble)
    if objectives is None and reeval_spec is None:
        _REEVAL_CACHE = result
    return result


def reeval_tag(spec) -> str:
    """Filesystem-safe label for the re-eval ensemble (its preset name)."""
    name = getattr(spec, "preset_name", None) or "historic_single"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def reeval_output_dir(scenario: str, slug: str, spec, seed=None):
    """``outputs/{scenario}/{slug}/reeval/{reeval_tag}[/seed_NN]`` (created)."""
    from config import run_output_dir
    d = run_output_dir(scenario, slug, "reeval") / reeval_tag(spec)
    if seed is not None:
        d = d / f"seed_{seed:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reeval_obj_names() -> list:
    """Objective names for the re-eval objective set (CSV columns)."""
    obj_set, _, _ = resolve_reeval()
    return list(obj_set.names)


def evaluate_solution(solution_id: int, dv_vector, formulation: str,
                      out_path=None):
    """Re-evaluate one policy on the common re-eval ensemble.

    Ensemble re-eval reuses the search-path ``evaluate()`` (in-memory ensemble
    simulation + the identical objective aggregation), so re-eval and search
    compute objectives byte-for-byte the same way — only the ensemble differs.
    Returns objectives in **natural units** (not Borg-minimized): satisficing
    fractions in [0, 1] for the ensemble path, raw metric values otherwise.

    Single-trace re-eval keeps the legacy per-solution HDF5 path. Ensemble
    re-eval is in-memory (no giant per-solution HDF5s) and ignores ``out_path``.

    Returns:
        (solution_id, natural_objectives | None, error_message | None)
    """
    try:
        obj_set, spec, is_ensemble = resolve_reeval()
        if is_ensemble:
            from src.simulation import evaluate
            borg = evaluate(dv_vector, formulation_name=formulation,
                            objective_set=obj_set, ensemble_spec=spec)
            # Borg minimizes; undo the sign flip to recover natural values.
            dirs = obj_set.directions
            natural = [(-v if d == 1 else v) for v, d in zip(borg, dirs)]
        else:
            from src.simulation import dvs_to_config, run_simulation_to_disk
            cfg = dvs_to_config(dv_vector, formulation)
            data = run_simulation_to_disk(cfg, out_path)
            natural = list(obj_set.compute(data))
        return solution_id, [float(x) for x in natural], None
    except Exception as e:
        return solution_id, None, f"{type(e).__name__}: {e}"


def evaluate_solution_raw(solution_id: int, dv_vector, formulation: str,
                          out_path=None):
    """Re-evaluate one policy and return its raw per-realization base matrix.

    Companion to :func:`evaluate_solution` for the decoupled robustness path:
    instead of collapsing the ensemble to satisficing fractions, return the full
    ``(n_realizations, n_base_objs)`` matrix of base-objective values in NATURAL
    units so robustness metrics are scored offline (see ``src.robustness``).
    Reuses the search-path :func:`src.simulation.evaluate_raw`, so re-eval base
    metrics match search byte-for-byte; only the ensemble differs.

    ``out_path`` is accepted for signature parity with ``evaluate_solution`` and
    is unused (raw re-eval is in-memory).

    Returns:
        ``(solution_id, base_matrix | None, base_names | None, error | None)``.
        ``base_matrix`` is ``(n_realizations, n_base_objs)``; for a single-trace
        re-eval spec it is ``(1, n_obj)``.
    """
    try:
        obj_set, spec, _ = resolve_reeval()
        from src.simulation import evaluate_raw
        base_matrix, base_names = evaluate_raw(
            dv_vector, formulation_name=formulation,
            objective_set=obj_set, ensemble_spec=spec,
        )
        return solution_id, base_matrix, base_names, None
    except Exception as e:
        return solution_id, None, None, f"{type(e).__name__}: {e}"


def satisficing_from_raw(base_matrix, base_names=None) -> list:
    """Reproduce :func:`evaluate_solution`'s natural satisficing fractions.

    Aggregates each column of the raw base matrix with its ensemble objective's
    aggregator (natural, un-negated), so the legacy ``objectives_summary.csv``
    can be derived from the persisted matrix instead of a second simulation —
    guaranteeing the two are consistent. For a single-trace re-eval set the matrix
    row IS the natural objective vector, so it is returned directly.

    Args:
        base_matrix: ``(n_realizations, n_base_objs)`` natural-unit array.
        base_names: Optional column names for validation against the resolved set.

    Returns:
        List of natural objective values aligned to :func:`reeval_obj_names`.
    """
    import numpy as np

    obj_set, _, is_ensemble = resolve_reeval()
    arr = np.asarray(base_matrix, dtype=float)
    if not is_ensemble:
        return [float(x) for x in arr[0, :]]
    ens_objs = list(obj_set)
    if base_names is not None:
        expected = [o.base.name for o in ens_objs]
        if list(base_names) != expected:
            raise ValueError(
                f"base_names {list(base_names)} do not match resolved re-eval "
                f"objective set {expected}"
            )
    return [float(o.aggregator(arr[:, k])) for k, o in enumerate(ens_objs)]


def reeval_raw_meta(formulation: str, n_solutions: int, seed=None) -> dict:
    """Self-describing metadata sidecar for the persisted raw matrix.

    Snapshots everything the offline scorer needs to compute robustness WITHOUT
    re-importing the live objective registry or honoring a changed
    ``NYCOPT_SAT_THRESHOLDS`` at scoring time (the moving-measuring-stick guard,
    McPhail et al. 2020). Carries per-objective thresholds/kinds/directions, the
    base-objective column order, the realization indices each matrix row maps to,
    and the run provenance ``(scenario_design, slug, seed)`` so re-evals are
    poolable across designs for cross-design comparison.
    """
    obj_set, spec, is_ensemble = resolve_reeval()

    if is_ensemble:
        ens_objs = list(obj_set)
        base_names = [o.base.name for o in ens_objs]
        thresholds = {o.base.name: getattr(o.aggregator, "threshold", None)
                      for o in ens_objs}
        kinds = {o.base.name: getattr(o.aggregator, "kind", None)
                 for o in ens_objs}
        directions = {o.base.name: o.base.direction for o in ens_objs}
        realization_indices = [int(i) for i in spec.realization_indices]
    else:
        base_names = list(obj_set.names)
        thresholds, kinds = {}, {}
        directions = {o.name: o.direction for o in obj_set}
        realization_indices = [0]

    from config import active_scenario_name, derive_slug
    return {
        "scenario_design": active_scenario_name(),
        "slug": derive_slug(formulation),
        "formulation": formulation,
        "seed": seed,
        "reeval_tag": reeval_tag(spec),
        "is_ensemble": bool(is_ensemble),
        "n_solutions": int(n_solutions),
        "n_realizations": len(realization_indices),
        "base_names": base_names,
        "ensemble_obj_names": list(obj_set.names),
        "realization_indices": realization_indices,
        "thresholds": thresholds,
        "kinds": kinds,
        "directions": directions,
    }


def persist_reeval_raw(reeval_dir, raw_results, formulation, n_solutions,
                       seed=None):
    """Write the raw per-realization matrix + self-describing meta, derive summary.

    The single persistence path shared by the multiprocessing and MPI drivers.
    Writes ``reeval_raw.parquet`` (long format; ``reeval_raw.csv.gz`` fallback if
    ``pyarrow`` is unavailable) and ``reeval_raw_meta.json``, then derives
    ``objectives_summary.csv`` from the SAME matrix via :func:`satisficing_from_raw`
    (no second simulation, so summary and matrix are guaranteed consistent).

    Args:
        reeval_dir: Output directory (already created).
        raw_results: Iterable of ``(solution_id, base_matrix | None,
            base_names | None, error | None)`` from :func:`evaluate_solution_raw`.
        formulation: Formulation name (for meta provenance).
        n_solutions: Total solutions attempted (for meta).
        seed: Optional seed (for meta provenance).

    Returns:
        ``(summary_csv_path, raw_path, meta_path)``.
    """
    import json

    import numpy as np
    import pandas as pd

    raw_results = list(raw_results)
    meta = reeval_raw_meta(formulation, n_solutions, seed)
    realization_indices = meta["realization_indices"]
    base_names = meta["base_names"]

    # Long-format raw matrix; failed solutions contribute no rows. Rows carry the
    # actual realization index (not batch position) so they join to the ensemble's
    # hazard coordinates.
    sids, rids, objs, vals = [], [], [], []
    for sid, mat, names, _err in raw_results:
        if mat is None:
            continue
        arr = np.asarray(mat, dtype=float)
        cols = list(names) if names is not None else base_names
        for j in range(arr.shape[0]):
            rid = realization_indices[j] if j < len(realization_indices) else j
            for k, name in enumerate(cols):
                sids.append(int(sid))
                rids.append(int(rid))
                objs.append(name)
                vals.append(float(arr[j, k]))
    long_df = pd.DataFrame(
        {"solution_id": sids, "realization_id": rids,
         "objective": objs, "value": vals}
    )

    raw_path = reeval_dir / "reeval_raw.parquet"
    try:
        long_df.to_parquet(raw_path, index=False)
    except Exception:  # noqa: BLE001 - pyarrow/fastparquet missing -> csv.gz
        raw_path = reeval_dir / "reeval_raw.csv.gz"
        long_df.to_csv(raw_path, index=False, compression="gzip")

    meta_path = reeval_dir / "reeval_raw_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    # Derive objectives_summary.csv from the matrix (one simulation source).
    obj_names = reeval_obj_names()
    by_sid = {sid: (mat, names) for sid, mat, names, _e in raw_results}
    index = sorted(by_sid)
    rows = []
    for sid in index:
        mat, names = by_sid[sid]
        rows.append([np.nan] * len(obj_names) if mat is None
                    else satisficing_from_raw(mat, names))
    summary_df = pd.DataFrame(
        rows, columns=obj_names, index=pd.Index(index, name="solution_id"),
    )
    summary_csv = reeval_dir / "objectives_summary.csv"
    summary_df.to_csv(summary_csv)
    return summary_csv, raw_path, meta_path
