"""chunk_reeval.py - Simulate a large chunked test ensemble and score it, metrics-only.

Re-evaluates a set of policies (decision-variable vectors) against **every chunk** of a chunked
test ensemble (``src.ensemble_generation.generate_forcing_ensemble`` with ``chunk_size > 0``),
computing objectives/robustness from **in-memory reduced metrics** — full simulation-output
timeseries are never persisted. Memory is bounded three ways: (1) each chunk is a small standalone
ensemble; (2) ``run_simulation_ensemble_batched`` (``SEARCH_REALIZATION_BATCH``) batches realizations
within a chunk, freeing timeseries per batch; (3) work is distributed across MPI ranks.

Design (reusing the re-evaluation stack):
- Work units are ``(solution, chunk)`` pairs. Ranks coordinate through marker/claim files, not
  flaky MPI collectives.
- Each unit reduces to the chunk's per-SOW annual-unit objective rows: the chunk's realizations
  are reduced to their stage-(i) annual metrics via :func:`src.simulation.evaluate_annual_units`
  (recorder -> ``/dev/null``) and each SOW's unit-years are pooled through the §2 unit operators
  (:func:`src.reeval_core.sow_objective_matrix`). A chunk holds WHOLE SOWs by construction
  (``chunk_size`` is a multiple of ``realizations_per_profile``; asserted in ``src.etest``), so
  rows are keyed by the ensemble's **global** SOW ids and persistence layout, scheduling, and
  merge placement provably cannot change the merged cube (``tests/test_chunk_reeval.py`` asserts
  equality across all of them).
- **Incremental mode** (``NYCOPT_CHUNK_INCREMENTAL=1``, default): each completed unit is flushed
  atomically to ``partial/units/chunk{j:03d}/sol{sid:05d}.parquet``; a failed unit leaves a
  ``.failed`` sidecar (its rows stay NaN, exactly like the one-shot no-rows semantics). Restarting
  the same submission resumes — done units are skipped regardless of rank count. A wall guard
  (``NYCOPT_CHUNK_STOP_EPOCH``/``NYCOPT_CHUNK_UNIT_SECONDS``) stops cleanly before the limit.
  Scheduling (``NYCOPT_CHUNK_SCHEDULE``): ``claim`` (default) = dynamic pull via O_CREAT|O_EXCL
  claim files over a chunk-major list (each node's ranks share 1-2 chunks' HDF5 working set);
  ``interleave`` = static strided; ``contiguous`` = static s-major ``np.array_split``.
- **One-shot mode** (``NYCOPT_CHUNK_INCREMENTAL=0``): each rank accumulates its rows and writes one
  long-format parquet partial at the end (the original path, kept as the reference).
- Merge: rank 0 reassembles per-solution ``(n_sow, M)`` matrices and reuses
  :func:`src.reeval_core.persist_reeval_raw` (so ``reeval_raw.parquet`` / ``reeval_raw_meta.json`` /
  ``objectives_summary.csv`` are byte-compatible with the normal re-eval path) then
  :func:`src.robustness.run`. With ``NYCOPT_CHUNK_MERGE=off`` the simulate job skips the merge
  (and the ``await_all_done`` barrier entirely); run it afterwards via
  ``workflow/09b_merge_test_chunks.sh`` -> :func:`merge_test_chunks`, which is resumable and
  refuses on missing units unless ``NYCOPT_CHUNK_MERGE_ALLOW_PARTIAL=1``.

The re-eval ensemble (``config.REEVAL_ENSEMBLE_SPEC``, via ``NYCOPT_REEVAL_ENSEMBLE_PRESET``) must be
the test-ensemble slug: its ``realization_indices == range(N_M)`` are exactly the global ids, so the reused
persistence keys every row to its true global realization.
"""

from __future__ import annotations

import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.sensitivity_common import (
    assign_rank_slots,
    await_all_done,
    get_mpi_context,
    mark_rank_done,
    prepare_partial_dir,
)


def _print_unit_line(sid: int, chunk_idx: int, t0: float) -> None:
    """Per-unit progress/telemetry: wall time and the rank's peak RSS so far.

    ``ru_maxrss`` is the process high-water mark (kB on Linux), so the printed
    value is cumulative-peak, not per-unit — it sizes ranks-per-node.
    """
    from src.sensitivity_common import get_mpi_context

    _, rank, _ = get_mpi_context()
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[unit] rank={rank} sol={sid} chunk={chunk_idx} "
          f"elapsed_s={time.perf_counter() - t0:.1f} rss_gb={rss_gb:.2f}",
          flush=True)


def _evaluate_unit(
    sol_idx: int,
    chunk_idx: int,
    dvs: np.ndarray,
    solution_ids: list[int],
    formulation: str,
    obj_set,
    chunks: list[tuple[object, list[int]]],
    realization_batch: int | None,
    r_per_sow: int,
) -> tuple[pd.DataFrame | None, str | None]:
    """Simulate one (solution, chunk) unit; return (long rows, None) or (None, error).

    The row construction is the single source of truth for unit payloads — both
    the one-shot per-rank accumulation and the incremental per-unit flush go
    through it, so the two persistence layouts carry identical rows. Rows are
    the chunk's per-SOW annual-unit objective values, keyed by GLOBAL SOW id
    (``global_realization_id // r_per_sow``; a chunk holds whole SOWs).
    """
    from src.reeval_core import sow_objective_matrix
    from src.simulation import evaluate_annual_units

    chunk_spec, global_ids = chunks[chunk_idx]
    sid = solution_ids[sol_idx]
    t0 = time.perf_counter()
    try:
        units, obj_names = evaluate_annual_units(
            dvs[sol_idx], formulation_name=formulation,
            objective_set=obj_set, ensemble_spec=chunk_spec,
            realization_batch=realization_batch,
        )
        sow_ids = [int(g) // int(r_per_sow)
                   for g in global_ids[:units.shape[0]]]
        sow_matrix, sow_labels, survivors = sow_objective_matrix(
            units, obj_set, sow_ids)
    except Exception as exc:  # noqa: BLE001 - a failed unit contributes no rows
        err = f"{type(exc).__name__}: {exc}"
        print(f"[chunk-reeval] solution {sid} x chunk {chunk_idx} failed: {err}")
        return None, err
    _print_unit_line(sid, chunk_idx, t0)
    # Vectorized long rows (row-major over SOW, objective), keyed by the
    # ensemble's global SOW ids. n_survivors records how many realizations
    # each SOW's pool actually held (a crashed batch shrinks the pool, and
    # this column is the only record of it).
    arr = np.asarray(sow_matrix, dtype=float)
    g_i, m_i = arr.shape
    sow_row = np.asarray(sow_labels, dtype=int)
    return pd.DataFrame({
        "solution_id": np.full(g_i * m_i, int(sid), dtype=int),
        "sow_id": np.repeat(sow_row, m_i),
        "objective": np.tile(np.asarray(obj_names, dtype=object), g_i),
        "value": arr.reshape(-1),
        "n_survivors": np.repeat(np.asarray(survivors, dtype=float), m_i),
    }), None


def _empty_long_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "solution_id": np.array([], dtype=int),
        "sow_id": np.array([], dtype=int),
        "objective": np.array([], dtype=object),
        "value": np.array([], dtype=float),
        "n_survivors": np.array([], dtype=float),
    })


def _rank_long_rows(
    work_slots: list[tuple[int, int]],
    dvs: np.ndarray,
    solution_ids: list[int],
    formulation: str,
    obj_set,
    chunks: list[tuple[object, list[int]]],
    realization_batch: int | None,
    r_per_sow: int,
) -> pd.DataFrame:
    """One-shot driver: simulate this rank's units, returning accumulated long rows."""
    frames = []
    for sol_idx, chunk_idx in work_slots:
        df, _err = _evaluate_unit(sol_idx, chunk_idx, dvs, solution_ids,
                                  formulation, obj_set, chunks,
                                  realization_batch, r_per_sow)
        if df is not None:
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return _empty_long_frame()


###############################################################################
# Incremental per-unit persistence (+ resume, wall guard, scheduling)
###############################################################################

def _unit_stem(units_dir: Path, chunk_idx: int, sid: int) -> Path:
    """Canonical extension-less path for one unit's artifacts."""
    return units_dir / f"chunk{chunk_idx:03d}" / f"sol{int(sid):05d}"


def _flush_unit(df: pd.DataFrame, stem: Path) -> None:
    """Atomically write one unit's rows (tmp + os.replace; parquet, csv.gz fallback).

    A killed rank can never leave a half-written unit that a resume would trust.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    tmp = stem.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, stem.with_suffix(".parquet"))
    except Exception:  # noqa: BLE001 - pyarrow/fastparquet missing (mirrors _write_partial)
        tmp = stem.with_suffix(".csv.gz.tmp")
        df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, stem.with_suffix(".csv.gz"))


def _read_unit(stem: Path) -> pd.DataFrame | None:
    """Read one unit's rows, or None if the unit has no result file."""
    if stem.with_suffix(".parquet").exists():
        return pd.read_parquet(stem.with_suffix(".parquet"))
    if stem.with_suffix(".csv.gz").exists():
        return pd.read_csv(stem.with_suffix(".csv.gz"))
    return None


def _completed_units(units_dir: Path, *, retry_failed: bool) -> set[tuple[int, int]]:
    """(sid, chunk_idx) pairs that need no work: done, or failed and not retried."""
    done: set[tuple[int, int]] = set()
    if not units_dir.exists():
        return done
    for p in units_dir.glob("chunk*/sol*"):
        name = p.name
        try:
            chunk_idx = int(p.parent.name[len("chunk"):])
            sid = int(name.split(".", 1)[0][len("sol"):])
        except ValueError:
            continue
        if name.endswith(".parquet") or name.endswith(".csv.gz"):
            done.add((sid, chunk_idx))
        elif name.endswith(".failed") and not retry_failed:
            done.add((sid, chunk_idx))
    return done


def _try_claim(claims_dir: Path, chunk_idx: int, sid: int) -> bool:
    """Atomically claim a unit for this rank (O_CREAT|O_EXCL; False if taken)."""
    path = claims_dir / f"chunk{chunk_idx:03d}_sol{int(sid):05d}.claim"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def _out_of_wall_time(unit_seconds: float, stop_epoch: float) -> bool:
    """True when the next unit could not finish before the job's stop epoch."""
    if not stop_epoch or not unit_seconds:
        return False
    return time.time() + 1.25 * unit_seconds >= stop_epoch


def _run_units_incremental(
    work: list[tuple[int, int]],
    dvs: np.ndarray,
    solution_ids: list[int],
    formulation: str,
    obj_set,
    chunks: list[tuple[object, list[int]]],
    realization_batch: int | None,
    r_per_sow: int,
    partial_dir: Path,
    rank: int,
    size: int,
) -> None:
    """Evaluate units with per-unit atomic flush, resume skip, and scheduling."""
    from config import (CHUNK_RETRY_FAILED, CHUNK_SCHEDULE, CHUNK_STOP_EPOCH,
                        CHUNK_UNIT_SECONDS)

    units_dir = partial_dir / "units"
    done = _completed_units(units_dir, retry_failed=bool(CHUNK_RETRY_FAILED))

    schedule = CHUNK_SCHEDULE
    claims_dir = None
    if schedule == "contiguous":
        my_units = [work[i] for i in assign_rank_slots(len(work), rank, size)]
    elif schedule == "interleave":
        my_units = work[rank::size]
    elif schedule == "claim":
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        claims_dir = partial_dir / f"claims_{job_id}"
        claims_dir.mkdir(parents=True, exist_ok=True)
        # Rank-dependent start offset minimizes claim collisions; every rank
        # still scans the whole list, so no unit is orphaned by a dead rank
        # that never claimed it (claims are job-scoped: a killed job's claims
        # die with its claims_{job_id} dir being ignored by the next job).
        offset = (rank * len(work)) // max(size, 1)
        my_units = work[offset:] + work[:offset]
    else:
        raise ValueError(
            f"Unknown NYCOPT_CHUNK_SCHEDULE='{schedule}' "
            f"(expected 'claim', 'interleave', or 'contiguous')."
        )

    n_run = n_skip = 0
    for sol_idx, chunk_idx in my_units:
        sid = solution_ids[sol_idx]
        if (int(sid), chunk_idx) in done:
            n_skip += 1
            continue
        if claims_dir is not None and not _try_claim(claims_dir, chunk_idx, sid):
            continue
        if _out_of_wall_time(CHUNK_UNIT_SECONDS, CHUNK_STOP_EPOCH):
            print(f"[chunk-reeval] rank {rank}: wall guard stop after {n_run} "
                  f"unit(s); resubmit the same job to resume.", flush=True)
            break
        df, err = _evaluate_unit(sol_idx, chunk_idx, dvs, solution_ids,
                                 formulation, obj_set, chunks,
                                 realization_batch, r_per_sow)
        stem = _unit_stem(units_dir, chunk_idx, sid)
        if df is None:
            stem.parent.mkdir(parents=True, exist_ok=True)
            stem.with_suffix(".failed").write_text(str(err) + "\n")
        else:
            _flush_unit(df, stem)
            failed = stem.with_suffix(".failed")
            if failed.exists():
                failed.unlink()
        n_run += 1
    print(f"[chunk-reeval] rank {rank}: {n_run} unit(s) evaluated, "
          f"{n_skip} already done.", flush=True)


###############################################################################
# Persistence + merge
###############################################################################

def _write_partial(df: pd.DataFrame, stem: Path) -> None:
    """Write a rank's long-format chunk as parquet, falling back to csv.gz (mirrors persist_reeval_raw)."""
    try:
        df.to_parquet(stem.with_suffix(".parquet"), index=False)
    except Exception:  # noqa: BLE001 - pyarrow/fastparquet missing
        df.to_csv(stem.with_suffix(".csv.gz"), index=False, compression="gzip")


def _read_partials(partial_dir: Path) -> list[pd.DataFrame]:
    """Read all rank chunks (parquet or csv.gz)."""
    parts = []
    for p in sorted(partial_dir.glob("rank_*.parquet")):
        parts.append(pd.read_parquet(p))
    for p in sorted(partial_dir.glob("rank_*.csv.gz")):
        parts.append(pd.read_csv(p))
    return parts


def _persist_and_score(
    reeval_dir: Path, raw_results: list, formulation: str, n_solutions: int, seed,
) -> Path:
    """Persist the raw cube via the re-eval path and run the robustness scorecards."""
    from src.reeval_core import persist_reeval_raw
    from src import robustness

    persist_reeval_raw(reeval_dir, raw_results, formulation, n_solutions, seed)

    # Pass the status-quo re-eval matrix if step 05 staged one under this same
    # reeval tag. Without it the incumbent-relative regret family warns and is
    # silently dropped.
    from config import REEVALUATION_SETTINGS

    baseline_dir = reeval_dir / "baseline"
    has_baseline = any(
        (baseline_dir / f).exists()
        for f in ("reeval_raw.parquet", "reeval_raw.csv.gz")
    )
    robustness.run(
        reeval_dir,
        baseline_dir=baseline_dir if has_baseline else None,
        metrics=tuple(REEVALUATION_SETTINGS["robustness_metrics"]),
    )
    return reeval_dir


def _merge_and_persist(
    partial_dir: Path, reeval_dir: Path, solution_ids: list[int], n_sow: int,
    formulation: str, obj_set, seed,
) -> Path:
    """One-shot merge: concatenate rank partials, reassemble per-solution matrices."""
    parts = _read_partials(partial_dir)
    long_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["solution_id", "sow_id", "objective", "value", "n_survivors"]
    )
    obj_names = [o.name for o in obj_set]

    # Reassemble each solution's (n_sow, M) matrix in global-SOW order (NaN for
    # failed/absent cells); persist_reeval_raw maps row g -> sow_labels[g] ==
    # global SOW id g. The per-SOW survivor counts travel the same way (rows
    # repeat the count per objective; "first" recovers the scalar).
    raw_results = []
    for sid in solution_ids:
        sub = long_df[long_df["solution_id"] == sid]
        if sub.empty:
            raw_results.append((sid, None, None, None, "no rows"))
            continue
        piv = (sub.pivot_table(index="sow_id", columns="objective", values="value")
               .reindex(index=range(n_sow), columns=obj_names))
        surv = (sub.groupby("sow_id")["n_survivors"].first()
                .reindex(range(n_sow)).to_numpy(dtype=float))
        raw_results.append((sid, piv.to_numpy(dtype=float), obj_names, surv, None))

    return _persist_and_score(reeval_dir, raw_results, formulation,
                              len(solution_ids), seed)


def merge_units(
    reeval_dir: Path,
    solution_ids: list[int],
    n_sow: int,
    formulation: str,
    obj_set,
    chunks: list[tuple[object, list[int]]],
    seed,
    *,
    allow_partial: bool = False,
) -> Path:
    """Merge per-unit files into the re-eval cube (no pivots, no full-table scans).

    Direct placement by (global SOW id, objective) reproduces the one-shot
    ``pivot_table`` + ``reindex`` semantics exactly for the unique-cell case;
    absent units (failed or missing) leave NaN rows, the one-shot no-rows
    behavior. Stateless over the unit files -> trivially resumable.

    Args:
        reeval_dir: Re-eval output dir holding ``partial/units``.
        solution_ids: Solution ids in ``dvs`` row order.
        n_sow: Global SOW count (matrix row space).
        formulation: Formulation name (persist metadata).
        obj_set: Resolved objective set (column names + order).
        chunks: ``pool_chunk_specs`` output (chunk count for the completeness gate).
        seed: Provenance seed for persist metadata.
        allow_partial: Merge despite missing units (their rows stay NaN).

    Raises:
        FileNotFoundError: If units are missing and ``allow_partial`` is False.
        ValueError: If a unit file carries an objective not in ``obj_set``.
    """
    units_dir = Path(reeval_dir) / "partial" / "units"
    obj_names = [o.name for o in obj_set]
    col_pos = {name: k for k, name in enumerate(obj_names)}

    missing: list[tuple[int, int]] = []
    n_failed = 0
    raw_results = []
    for sid in solution_ids:
        mat = np.full((n_sow, len(obj_names)), np.nan)
        surv = np.full(n_sow, np.nan)
        any_rows = False
        for j in range(len(chunks)):
            stem = _unit_stem(units_dir, j, sid)
            df = _read_unit(stem)
            if df is None:
                if stem.with_suffix(".failed").exists():
                    n_failed += 1  # evaluated and failed -> NaN rows, not "missing"
                else:
                    missing.append((int(sid), j))
                continue
            obj_idx = df["objective"].map(col_pos)
            if obj_idx.isna().any():
                bad = sorted(set(df["objective"][obj_idx.isna()]))
                raise ValueError(
                    f"unit sol{sid}/chunk{j} carries objectives not in the "
                    f"active set: {bad}"
                )
            sow_rows = df["sow_id"].to_numpy(dtype=int)
            mat[sow_rows,
                obj_idx.to_numpy(dtype=int)] = df["value"].to_numpy(dtype=float)
            # Rows repeat the per-SOW count once per objective; repeated
            # assignment writes the same scalar (chunks hold disjoint SOWs).
            surv[sow_rows] = df["n_survivors"].to_numpy(dtype=float)
            any_rows = True
        if any_rows:
            raw_results.append((sid, mat, obj_names, surv, None))
        else:
            raw_results.append((sid, None, None, None, "no rows"))

    if missing and not allow_partial:
        preview = ", ".join(f"sol{s}/chunk{j}" for s, j in missing[:20])
        raise FileNotFoundError(
            f"merge_units: {len(missing)} unit(s) missing (e.g. {preview}"
            f"{', ...' if len(missing) > 20 else ''}). Resubmit the simulate "
            f"job to complete them, or set NYCOPT_CHUNK_MERGE_ALLOW_PARTIAL=1 "
            f"to merge with NaN rows."
        )
    if missing:
        print(f"[chunk-reeval] merge: {len(missing)} missing unit(s) merged "
              f"as NaN rows (allow_partial).", flush=True)
    if n_failed:
        print(f"[chunk-reeval] merge: {n_failed} failed unit(s) contribute "
              f"NaN rows.", flush=True)

    return _persist_and_score(Path(reeval_dir), raw_results, formulation,
                              len(solution_ids), seed)


###############################################################################
# Entry points
###############################################################################

def _resolve_campaign(formulation: str, seed, reeval_dir: Path | None):
    """Shared campaign resolution for the simulate and merge entry points.

    Both must reconstruct identical (obj_set, chunks, reeval_dir) from the same
    env so a standalone merge scores exactly the campaign the simulate jobs ran.
    """
    from config import (REEVAL_ENSEMBLE_SPEC, active_scenario_name, derive_slug)
    from src.ensembles import pool_chunk_specs
    from src.reeval_core import reeval_output_dir, resolve_reeval, sow_grouping

    if REEVAL_ENSEMBLE_SPEC is None or not REEVAL_ENSEMBLE_SPEC.is_ensemble:
        raise ValueError(
            "chunk re-eval requires NYCOPT_REEVAL_ENSEMBLE_PRESET to resolve to the chunked "
            "test ensemble (an is_ensemble spec whose realization_indices span its global index "
            "space)."
        )
    obj_set, test_spec, _ = resolve_reeval()
    chunks = pool_chunk_specs(test_spec.inflow_type)
    sow_ids, n_sow, r_per_sow = sow_grouping(
        test_spec, test_spec.realization_indices)
    if sow_ids is None:
        raise ValueError(
            "chunk re-eval requires a DU-forced test ensemble with forcing "
            "profiles; this spec carries no SOW grouping, so the per-SOW "
            "objective unit is undefined for it."
        )
    if reeval_dir is None:
        reeval_dir = reeval_output_dir(active_scenario_name(), derive_slug(formulation),
                                       test_spec, seed)
    return obj_set, test_spec, chunks, n_sow, r_per_sow, Path(reeval_dir)


def simulate_test_chunks(
    formulation: str, dvs: np.ndarray, solution_ids: list[int] | None = None,
    *, seed=None, realization_batch: int | None = None, reeval_dir: Path | None = None,
) -> Path | None:
    """Re-evaluate ``dvs`` against every chunk of the test ensemble and write robustness artifacts.

    Args:
        formulation: Formulation name (DV grammar).
        dvs: ``(n_solutions, n_vars)`` decision-variable matrix.
        solution_ids: Ids aligned to ``dvs`` rows (default ``range(n_solutions)``).
        seed: Optional provenance seed (output subdir + meta).
        realization_batch: Realizations per within-chunk simulation batch (default
            ``config.SEARCH_REALIZATION_BATCH``).
        reeval_dir: Output dir (default ``reeval_output_dir`` under the test ensemble's re-eval tag).

    Returns:
        The re-eval output directory on rank 0 (``None`` when the merge is
        deferred via ``NYCOPT_CHUNK_MERGE=off``); ``None`` on worker ranks.
    """
    from config import (CHUNK_DONE_DEADLINE_S, CHUNK_INCREMENTAL, CHUNK_MERGE,
                        SEARCH_REALIZATION_BATCH)

    obj_set, test_spec, chunks, n_sow, r_per_sow, reeval_dir = _resolve_campaign(
        formulation, seed, reeval_dir)
    n_realizations = test_spec.n_realizations

    dvs = np.atleast_2d(np.asarray(dvs, dtype=float))
    n_solutions = dvs.shape[0]
    if solution_ids is None:
        solution_ids = list(range(n_solutions))
    if realization_batch is None:
        realization_batch = SEARCH_REALIZATION_BATCH

    comm, rank, size = get_mpi_context()
    partial_dir = Path(reeval_dir) / "partial"
    prepare_partial_dir(partial_dir, rank)
    incremental = bool(CHUNK_INCREMENTAL)
    if rank == 0 and incremental:
        # A resumed submission must not satisfy await_all_done with the
        # previous job's markers.
        for stale in partial_dir.glob("rank_*.done"):
            stale.unlink()

    if incremental:
        # Chunk-major: at any moment the active ranks work within a few
        # consecutive chunks, so a node's ranks re-read the same chunk HDF5
        # set through the page cache instead of touching all chunks at once.
        work = [(s, j) for j in range(len(chunks)) for s in range(n_solutions)]
    else:
        # Static s-major ordering (the reference path).
        work = [(s, j) for s in range(n_solutions) for j in range(len(chunks))]

    if rank == 0:
        print(f"[chunk-reeval] {n_solutions} solutions x {len(chunks)} chunks = {len(work)} units "
              f"across {size} rank(s); N_M={n_realizations}, batch={realization_batch}, "
              f"incremental={int(incremental)}, merge={CHUNK_MERGE}.")

    if incremental:
        _run_units_incremental(work, dvs, solution_ids, formulation, obj_set,
                               chunks, realization_batch, r_per_sow,
                               partial_dir, rank, size)
    else:
        my_slots = [work[i] for i in assign_rank_slots(len(work), rank, size)]
        rows = _rank_long_rows(my_slots, dvs, solution_ids, formulation, obj_set,
                               chunks, realization_batch, r_per_sow)
        _write_partial(rows, partial_dir / f"rank_{rank:03d}")
    mark_rank_done(partial_dir, rank)

    if rank != 0:
        return None
    if CHUNK_MERGE == "off":
        print("[chunk-reeval] merge deferred (NYCOPT_CHUNK_MERGE=off); run "
              "workflow/09b_merge_test_chunks.sh when all units are complete.",
              flush=True)
        return None
    if not await_all_done(partial_dir, size, deadline_s=float(CHUNK_DONE_DEADLINE_S)):
        raise TimeoutError("chunk re-eval: not all ranks reported done before the deadline.")
    if incremental:
        from config import CHUNK_MERGE_ALLOW_PARTIAL

        return merge_units(reeval_dir, solution_ids, n_sow, formulation,
                           obj_set, chunks, seed,
                           allow_partial=bool(CHUNK_MERGE_ALLOW_PARTIAL))
    return _merge_and_persist(partial_dir, Path(reeval_dir), solution_ids, n_sow,
                              formulation, obj_set, seed)


def merge_test_chunks(
    formulation: str, solution_ids: list[int],
    *, seed=None, reeval_dir: Path | None = None,
) -> Path:
    """Standalone merge of a completed (or partial) incremental simulate run.

    Resolves the campaign from the same env the simulate jobs used, so the ids
    and objective set match exactly. Stateless over the unit files: safe to
    re-run, overwrites its own outputs.
    """
    from config import CHUNK_MERGE_ALLOW_PARTIAL

    obj_set, test_spec, chunks, n_sow, _r_per_sow, reeval_dir = _resolve_campaign(
        formulation, seed, reeval_dir)
    return merge_units(reeval_dir, list(solution_ids), n_sow,
                       formulation, obj_set, chunks, seed,
                       allow_partial=bool(CHUNK_MERGE_ALLOW_PARTIAL))
