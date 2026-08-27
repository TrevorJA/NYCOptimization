"""reeval_core.py - shared helpers for re-evaluating Pareto policies on the
common held-out test ensemble.

Used by ``src.reevaluate`` (multiprocessing), ``src.reevaluate_mpi`` (MPI),
and ``src.chunk_reeval`` (chunked MPI) so every driver scores solutions
identically. Re-evaluation computes the SAME annual-unit objectives the search
optimized (``src.objectives_ensemble``), recomputed per SOW: each SOW's R
realizations contribute their stage-(i) unit-years to one pooled sample and
the §2 unit operator collapses it. The persisted artifact is the per-SOW
objective matrix, which ``src.robustness`` scores offline.

The common ensemble is ``config.REEVAL_ENSEMBLE_SPEC``, selected by
``NYCOPT_REEVAL_ENSEMBLE_PRESET`` (any registered preset, ``kn_{Y}yr_n{N}``
slug, or staged directory with a ``_meta.json``). Outputs are written under
``reeval/{tag}/`` per ensemble.
"""
from __future__ import annotations

import re

import numpy as np

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

    The objective set is ALWAYS the annual-unit set, including for a
    single-trace spec (the N = 1 case over that trace's unit-years).
    """
    global _REEVAL_CACHE
    if _REEVAL_CACHE is not None and objectives is None and reeval_spec is None:
        return _REEVAL_CACHE

    from config import REEVAL_ENSEMBLE_SPEC, ACTIVE_OBJECTIVES
    spec = reeval_spec if reeval_spec is not None else REEVAL_ENSEMBLE_SPEC
    names = objectives if objectives is not None else ACTIVE_OBJECTIVES

    from src.objectives_ensemble import build_ensemble_objective_set
    obj_set = build_ensemble_objective_set(names)

    is_ensemble = bool(spec is not None and spec.is_ensemble)
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
    """Column names for ``objectives_summary.csv``, naming what the cells CONTAIN.

    For an ensemble re-eval the cells are the MEAN over SOWs of the per-SOW
    annual-unit objective values (the Laplace / risk-neutral summary of the
    persisted per-SOW matrix), so the columns are ``sowmean__{objective}``.
    For a single-trace re-eval the row IS the natural annual-unit objective
    vector of that trace, so the objective names are used unchanged.
    """
    obj_set, _, is_ensemble = resolve_reeval()
    if not is_ensemble:
        return list(obj_set.names)
    return [f"sowmean__{o.name}" for o in obj_set]


def sow_objective_matrix(units: np.ndarray, obj_set, sow_ids) -> tuple:
    """Pool each SOW's unit-years through the §2 unit operators.

    Stage (i) already happened in the simulation worker
    (``src.simulation.evaluate_annual_units``); this is the re-evaluation's
    stage (ii): for each SOW, the unit-years of its R realizations are pooled
    into ONE sample per objective and collapsed with that objective's own unit
    operator — the search aggregation recomputed per deeply-uncertain state.

    Failed realizations (all-NaN slabs, e.g. a crashed simulation batch) are
    EXCLUDED from their SOW's pool so a failed run cannot masquerade as bad
    performance; a SOW with no surviving realization is NaN. Non-finite
    unit-years inside a surviving realization keep the search-side convention
    (the unit operator counts them as failure-years / worst-sentinels). The
    returned survivor counts make partial failures VISIBLE: a SOW scored on
    fewer than its R realizations pools fewer unit-years, so its percentile
    operators are a different order statistic, and nothing else records that.

    Args:
        units: ``(R, M, U)`` stage-(i) tensor in realization order.
        obj_set: The annual-unit ObjectiveSet (column order = objective order).
        sow_ids: Length-R SOW id of each realization row.

    Returns:
        ``(J, sow_labels, survivors)`` — ``J`` float array ``(n_sow, M)`` of
        per-SOW objective values in natural units; ``sow_labels`` the ascending
        SOW ids its rows are keyed by; ``survivors`` int array ``(n_sow,)`` of
        surviving (non-all-NaN) realizations pooled into each row.

    Raises:
        ValueError: If the tensor carries zero unit-years — ``np.all`` over an
            empty axis would mark every realization dead and silently NaN
            every SOW.
    """
    ens_objs = list(obj_set)
    units = np.asarray(units, dtype=float)
    if units.ndim != 3 or units.shape[1] != len(ens_objs):
        raise ValueError(
            f"units tensor has shape {units.shape}; expected "
            f"(R, {len(ens_objs)}, U)"
        )
    if units.shape[2] == 0:
        raise ValueError(
            "units tensor carries zero unit-years per realization; the "
            "simulation window yielded no complete FFMP-year units."
        )
    sow_ids = [int(s) for s in sow_ids]
    if len(sow_ids) != units.shape[0]:
        raise ValueError(
            f"{len(sow_ids)} sow_ids for {units.shape[0]} realizations"
        )

    groups: dict[int, list[int]] = {}
    for r, s in enumerate(sow_ids):
        groups.setdefault(s, []).append(r)
    labels = sorted(groups)

    alive = ~np.all(~np.isfinite(units), axis=(1, 2))  # (R,) simulation survived
    J = np.full((len(labels), len(ens_objs)), np.nan, dtype=float)
    survivors = np.zeros(len(labels), dtype=int)
    for g, label in enumerate(labels):
        rows = [r for r in groups[label] if alive[r]]
        survivors[g] = len(rows)
        if not rows:
            continue
        for k, obj in enumerate(ens_objs):
            J[g, k] = obj.unit_operator(units[rows, k, :].ravel())
    return J, labels, survivors


def evaluate_solution_raw(solution_id: int, dv_vector, formulation: str):
    """Re-evaluate one policy and return its per-SOW objective matrix.

    The re-eval work unit: simulate the policy on the common ensemble, reduce
    each realization to its stage-(i) annual metrics
    (``AnnualUnitObjective.annual_units``, the search path's own reduction),
    and pool each SOW's unit-years through the §2 unit operators.

    Returns:
        ``(solution_id, sow_matrix | None, obj_names | None,
        survivors | None, error | None)``. ``sow_matrix`` is
        ``(n_sow, n_objs)`` in natural units, rows keyed by ascending SOW id;
        ``survivors`` is the aligned per-SOW surviving-realization count from
        :func:`sow_objective_matrix`. For a single-trace re-eval spec the
        matrix is ``(1, n_objs)`` (the trace's own annual-unit objective
        vector).
    """
    try:
        obj_set, spec, is_ensemble = resolve_reeval()
        from src.simulation import evaluate_annual_units
        units, obj_names = evaluate_annual_units(
            dv_vector, formulation_name=formulation,
            objective_set=obj_set, ensemble_spec=spec,
        )
        if is_ensemble:
            sow_ids, _n_sow, _rps = sow_grouping(spec, spec.realization_indices)
            if sow_ids is None:
                raise ValueError(
                    "the re-eval ensemble carries no SOW grouping (no forcing "
                    "profiles), so the per-SOW objective unit is undefined for "
                    "it. Robustness re-evaluation requires a DU-forced ensemble."
                )
            sow_matrix, _labels, survivors = sow_objective_matrix(units, obj_set, sow_ids)
        else:
            sow_matrix, _labels, survivors = sow_objective_matrix(units, obj_set, [0])
        return solution_id, sow_matrix, obj_names, survivors, None
    except Exception as e:
        return solution_id, None, None, None, f"{type(e).__name__}: {e}"


def sow_grouping(spec, realization_indices) -> tuple[list | None, int | None, int | None]:
    """Recover which deeply-uncertain state of the world (SOW) each realization belongs to.

    A SOW is one forcing profile theta. With ``realizations_per_profile = R``,
    realization ``k`` belongs to profile ``p = k // R``. Read from the staged
    ensemble's ``forcing_profiles.npz``, falling back to ``_meta.json``. An
    ensemble with no forcing profiles (stationary, or the historic trace) has
    no SOW structure and returns ``(None, None, None)``; a grouping is never
    fabricated.

    Args:
        spec: The re-eval ``EnsembleSpec``.
        realization_indices: The realization ids to group.

    Returns:
        ``(sow_ids, n_sow, realizations_per_sow)``; ``sow_ids`` is aligned 1:1 with
        ``realization_indices``. All three are ``None`` when the ensemble has no
        forcing profiles.
    """
    import json

    if spec is None or not spec.is_ensemble:
        return None, None, None

    from src.ensembles import staged_ensemble_dir

    staged = staged_ensemble_dir(spec.inflow_type)
    r_per_sow = None

    npz = staged / "forcing_profiles.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=True) as z:
            if "realizations_per_profile" in z:
                r_per_sow = int(z["realizations_per_profile"])

    if r_per_sow is None:
        meta_path = staged / "_meta.json"
        if not meta_path.exists():
            return None, None, None
        meta = json.loads(meta_path.read_text())
        # No forcing profiles -> no SOW structure. `population` is authoritative;
        # `theta_sampler` is None for a stationary ensemble.
        if meta.get("population") != "du_forced" or meta.get("theta_sampler") is None:
            return None, None, None
        if meta.get("realizations_per_profile") is None:
            return None, None, None
        r_per_sow = int(meta["realizations_per_profile"])

    if r_per_sow < 1:
        return None, None, None

    sow_ids = [int(i) // r_per_sow for i in realization_indices]
    return sow_ids, len(set(sow_ids)), r_per_sow


def reeval_raw_meta(formulation: str, n_solutions: int, seed=None) -> dict:
    """Self-describing metadata sidecar for the persisted per-SOW matrix.

    Snapshots everything the offline scorer needs WITHOUT re-importing the
    live objective registry or honoring a changed ``NYCOPT_SAT_THRESHOLDS``:
    per-objective thresholds/kinds/directions, unit-operator provenance, the
    objective column order, the SOW labels each matrix row maps to, and the
    run provenance ``(scenario_design, slug, seed)``.
    """
    obj_set, spec, is_ensemble = resolve_reeval()
    ens_objs = list(obj_set)

    obj_names = [o.name for o in ens_objs]
    thresholds = {o.name: o.sat_threshold for o in ens_objs}
    kinds = {o.name: o.sat_kind for o in ens_objs}
    directions = {o.name: o.direction for o in ens_objs}
    unit_operators = {o.name: _op_descriptor(o.unit_operator) for o in ens_objs}

    if is_ensemble:
        realization_indices = [int(i) for i in spec.realization_indices]
        sow_ids, n_sow, realizations_per_sow = sow_grouping(
            spec, realization_indices)
        sow_labels = sorted(set(sow_ids)) if sow_ids is not None else None
    else:
        realization_indices = [0]
        sow_ids, n_sow, realizations_per_sow = None, None, None
        sow_labels = [0]

    from config import active_scenario_name, derive_slug
    return {
        "scenario_design": active_scenario_name(),
        "slug": derive_slug(formulation),
        "formulation": formulation,
        "seed": seed,
        "reeval_tag": reeval_tag(spec),
        "is_ensemble": bool(is_ensemble),
        "substrate": "sow_annual_unit",
        "n_solutions": int(n_solutions),
        "n_realizations": len(realization_indices),
        "obj_names": obj_names,
        # SOW labels the matrix rows are keyed by (ascending). None when the
        # ensemble carries no forcing profiles -- the per-SOW unit is then
        # undefined and the scorer reports robustness N/A.
        "sow_labels": sow_labels,
        "n_sow": n_sow,
        "realizations_per_sow": realizations_per_sow,
        "thresholds": thresholds,
        "kinds": kinds,
        "directions": directions,
        "unit_operators": unit_operators,
    }


def _op_descriptor(op) -> dict:
    """JSON-able provenance for a §2 unit operator (which statistic was pooled)."""
    from src.objectives_ensemble import (
        FailureFrequencyOp, PooledMeanOp, PooledPercentileOp,
    )
    if isinstance(op, FailureFrequencyOp):
        return {"type": "failure_frequency", "k": op.k}
    if isinstance(op, PooledPercentileOp):
        return {"type": "pooled_percentile", "q": op.q, "worst_value": op.worst_value}
    if isinstance(op, PooledMeanOp):
        return {"type": "pooled_mean", "worst_value": op.worst_value}
    return {"type": type(op).__name__}


def persist_reeval_raw(reeval_dir, raw_results, formulation, n_solutions,
                       seed=None):
    """Write the per-SOW objective matrix + self-describing meta, derive summary.

    The single persistence path shared by the multiprocessing, MPI, and chunked
    drivers. Writes ``reeval_raw.parquet`` (long format keyed by ``sow_id``;
    ``reeval_raw.csv.gz`` fallback if ``pyarrow`` is unavailable) and
    ``reeval_raw_meta.json``, then derives ``objectives_summary.csv`` from the
    SAME matrix (mean over SOWs; no second simulation, so summary and matrix
    are guaranteed consistent).

    Args:
        reeval_dir: Output directory (already created).
        raw_results: Iterable of ``(solution_id, sow_matrix | None,
            obj_names | None, survivors | None, error | None)`` from
            :func:`evaluate_solution_raw`.
        formulation: Formulation name (for meta provenance).
        n_solutions: Total solutions attempted (for meta).
        seed: Optional seed (for meta provenance).

    Returns:
        ``(summary_csv_path, raw_path, meta_path)``.
    """
    import json
    import warnings

    import pandas as pd

    raw_results = list(raw_results)
    meta = reeval_raw_meta(formulation, n_solutions, seed)
    # Record every attempted solution id (including fully-failed ones that
    # contribute no rows) so the offline scorer reconstructs the full solution
    # axis; otherwise an all-failed solution vanishes from the scorecard.
    meta["solution_ids"] = sorted(int(sid) for sid, *_ in raw_results)
    obj_names = list(meta["obj_names"])
    sow_labels = meta["sow_labels"] or [0]

    # Long-format per-SOW matrix; failed solutions contribute no rows. Rows
    # carry the SOW label (not matrix position) so they join to the ensemble's
    # hazard coordinates. Built vectorized per solution (np.repeat/np.tile);
    # row order: row-major over (sow, objective).
    sl = np.asarray(sow_labels, dtype=int)
    frames = []
    for sid, mat, names, survivors, _err in raw_results:
        if mat is None:
            continue
        arr = np.asarray(mat, dtype=float)
        g_i, m_i = arr.shape
        cols = list(names) if names is not None else obj_names
        if g_i != sl.shape[0]:
            raise ValueError(
                f"solution {sid}: sow matrix has {g_i} rows but the ensemble "
                f"has {sl.shape[0]} SOWs"
            )
        # Per-(solution, SOW) surviving-realization count: a SOW scored on
        # fewer than R realizations pooled fewer unit-years (its percentile
        # is a different order statistic), and this column is the only record
        # of that. NaN when the caller supplied none.
        surv = (np.asarray(survivors, dtype=float) if survivors is not None
                else np.full(g_i, np.nan))
        if surv.shape[0] != g_i:
            raise ValueError(
                f"solution {sid}: {surv.shape[0]} survivor counts for "
                f"{g_i} SOWs"
            )
        frames.append(pd.DataFrame({
            "solution_id": np.full(g_i * m_i, int(sid), dtype=int),
            "sow_id": np.repeat(sl, m_i),
            "objective": np.tile(np.asarray(cols, dtype=object), g_i),
            "value": arr.reshape(-1),
            "n_survivors": np.repeat(surv, m_i),
        }))
    long_df = (
        pd.concat(frames, ignore_index=True) if frames
        else pd.DataFrame({
            "solution_id": np.array([], dtype=int),
            "sow_id": np.array([], dtype=int),
            "objective": np.array([], dtype=object),
            "value": np.array([], dtype=float),
            "n_survivors": np.array([], dtype=float),
        })
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

    # Derive objectives_summary.csv from the matrix (one simulation source):
    # the mean over SOWs of the per-SOW objective values (single-trace rows
    # pass through unchanged, mean over one row being an identity).
    summary_cols = reeval_obj_names()
    by_sid = {sid: mat for sid, mat, _names, _surv, _e in raw_results}
    index = sorted(by_sid)
    rows = []
    for sid in index:
        mat = by_sid[sid]
        if mat is None:
            rows.append([np.nan] * len(summary_cols))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column
            rows.append(list(np.nanmean(np.asarray(mat, dtype=float), axis=0)))
    summary_df = pd.DataFrame(
        rows, columns=summary_cols, index=pd.Index(index, name="solution_id"),
    )
    summary_csv = reeval_dir / "objectives_summary.csv"
    summary_df.to_csv(summary_csv)
    return summary_csv, raw_path, meta_path
