"""chunk_reeval.py - Simulate a large chunked test ensemble and score it, metrics-only.

Re-evaluates policies against every chunk of a chunked test ensemble
(``src.ensemble_generation.generate_forcing_ensemble`` with ``chunk_size > 0``)
from in-memory reduced metrics; simulation timeseries are never persisted.

- Work units are ``(solution, chunk, block)`` triples, distributed across MPI
  ranks. A block is one within-chunk simulation batch
  (``NYCOPT_SEARCH_REALIZATION_BATCH`` realizations, widened to whole SOWs:
  :func:`unit_block_size`), so a policy contributes ``chunks x blocks`` units
  and one policy alone can occupy that many ranks. With no batch set the block
  is the whole chunk.
- Each unit reduces to its block's per-SOW annual-unit objective rows via
  :func:`src.simulation.evaluate_annual_units` and
  :func:`src.reeval_core.sow_objective_matrix`, keyed by the ensemble's GLOBAL
  SOW ids (a chunk and a block hold whole SOWs), so unit granularity,
  persistence layout, scheduling, and merge placement cannot change the merged
  cube (``tests/test_chunk_reeval.py``).
- Incremental mode writes each unit atomically to
  ``partial/units/chunk{j:03d}/sol{sid:05d}_b{k:03d}.parquet`` (``.failed``
  sidecar when the block's simulation raises) and resumes on resubmission;
  ``partial/units/layout.json`` records the block layout, which a resumed pass
  must match and the merge reads. One-shot mode writes one partial per rank.
- Merge (rank 0, or ``workflow/11_merge_test_chunks.sh`` ->
  :func:`merge_test_chunks`) reassembles per-solution ``(n_sow, M)`` matrices
  and reuses :func:`src.reeval_core.persist_reeval_raw` then
  :func:`src.robustness.run`.

Knobs (``NYCOPT_CHUNK_*``) are documented once in config.py. The re-eval
ensemble (``NYCOPT_REEVAL_ENSEMBLE_PRESET``) must be the test-ensemble slug so
``realization_indices`` are the global ids.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import replace

try:
    import resource  # POSIX only; the RSS line degrades to NaN elsewhere
except ImportError:  # pragma: no cover - Windows
    resource = None
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


def _print_unit_line(sid: int, chunk_idx: int, block_idx: int, t0: float) -> None:
    """Per-unit progress/telemetry: wall time and the rank's peak RSS so far.

    ``ru_maxrss`` is the process high-water mark (kB on Linux), so the printed
    value is cumulative-peak, not per-unit — it sizes ranks-per-node.
    """
    from src.sensitivity_common import get_mpi_context

    _, rank, _ = get_mpi_context()
    rss_gb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
              if resource is not None else float("nan"))
    print(f"[unit] rank={rank} sol={sid} chunk={chunk_idx} block={block_idx} "
          f"elapsed_s={time.perf_counter() - t0:.1f} rss_gb={rss_gb:.2f}",
          flush=True)


###############################################################################
# Unit layout: (solution, chunk, block)
###############################################################################

def unit_block_size(realization_batch: int | None, r_per_sow: int) -> int:
    """Realizations per work unit; 0 means the whole chunk is one unit.

    A unit is one within-chunk simulation batch. Each unit pools its own SOWs,
    so it must hold whole SOWs: a batch that is not a multiple of ``r_per_sow``
    is widened to their least common multiple (the unit then simulates several
    sequential batches). No batch (``<= 0`` or ``None``) keeps the chunk whole.
    """
    if not realization_batch or int(realization_batch) <= 0:
        return 0
    return math.lcm(int(realization_batch), int(r_per_sow))


def unit_layout(chunks: list[tuple[object, list[int]]],
                realization_batch: int | None, r_per_sow: int) -> list[list[tuple[int, int]]]:
    """Per chunk, the ``[(r0, r1), ...]`` local realization ranges of its blocks.

    Blocks are contiguous, start at multiples of :func:`unit_block_size`, and
    tile the chunk; the last block of a chunk may be shorter.

    Raises:
        ValueError: If a chunk does not hold whole SOWs.
    """
    block = unit_block_size(realization_batch, r_per_sow)
    layout: list[list[tuple[int, int]]] = []
    for j, (_spec, global_ids) in enumerate(chunks):
        n = len(global_ids)
        if n % int(r_per_sow):
            raise ValueError(
                f"chunk {j} holds {n} realizations, not a multiple of the "
                f"{r_per_sow} realizations per SOW; a unit pools whole SOWs."
            )
        step = block if 0 < block < n else n
        layout.append([(r0, min(r0 + step, n)) for r0 in range(0, n, step)])
    return layout


def _layout_manifest(layout: list[list[tuple[int, int]]],
                     realization_batch: int | None, r_per_sow: int) -> dict:
    """The record of a unit layout that resume and merge must agree with."""
    return {
        "unit": "solution_chunk_block",
        "realization_batch": int(realization_batch or 0),
        "realizations_per_sow": int(r_per_sow),
        "blocks": [[[int(r0), int(r1)] for r0, r1 in chunk] for chunk in layout],
    }


def _read_layout(units_dir: Path) -> list[list[tuple[int, int]]] | None:
    """The block layout recorded in ``units_dir`` (None when no pass wrote one)."""
    path = units_dir / "layout.json"
    if not path.exists():
        return None
    blocks = json.loads(path.read_text())["blocks"]
    return [[(int(r0), int(r1)) for r0, r1 in chunk] for chunk in blocks]


def _check_or_write_layout(units_dir: Path, manifest: dict, rank: int) -> None:
    """Refuse to mix unit layouts in one partial dir; rank 0 records a new one.

    Unit files are named by block index, so a pass run with another realization
    batch would write units that overlap or miss the recorded blocks.
    """
    path = units_dir / "layout.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        if recorded.get("blocks") != manifest["blocks"]:
            raise RuntimeError(
                f"{path} records a unit layout written with realization batch "
                f"{recorded.get('realization_batch')}, but this pass uses "
                f"{manifest['realization_batch']}. Resubmit with the same "
                f"NYCOPT_SEARCH_REALIZATION_BATCH, or remove the partial "
                f"directory to start over."
            )
        return
    if rank == 0:
        units_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest) + "\n")
        os.replace(tmp, path)


def _evaluate_unit(
    sol_idx: int,
    chunk_idx: int,
    block_idx: int,
    dvs: np.ndarray,
    solution_ids: list[int],
    formulation: str,
    obj_set,
    chunks: list[tuple[object, list[int]]],
    layout: list[list[tuple[int, int]]],
    realization_batch: int | None,
    r_per_sow: int,
) -> tuple[pd.DataFrame | None, str | None]:
    """Simulate one (solution, chunk, block) unit; return (long rows, None) or (None, error).

    The row construction is the single source of truth for unit payloads — both
    the one-shot per-rank accumulation and the incremental per-unit flush go
    through it, so the two persistence layouts carry identical rows. Rows are
    the block's per-SOW annual-unit objective values, keyed by GLOBAL SOW id
    (``global_realization_id // r_per_sow``; a block holds whole SOWs).

    The block is simulated as the chunk spec restricted to its realizations,
    the same restriction :func:`src.simulation.run_simulation_ensemble_batched`
    applies to a batch, so a realization's values do not depend on the unit
    granularity.
    """
    from src.reeval_core import sow_objective_matrix
    from src.simulation import evaluate_annual_units

    chunk_spec, global_ids = chunks[chunk_idx]
    r0, r1 = layout[chunk_idx][block_idx]
    indices = list(chunk_spec.realization_indices)
    if (r0, r1) == (0, len(indices)):
        unit_spec = chunk_spec          # the whole chunk: the spec as staged
    else:
        unit_spec = replace(
            chunk_spec,
            preset_name=f"{chunk_spec.preset_name}__u{r0}",
            realization_indices=tuple(indices[r0:r1]),
        )
    sid = solution_ids[sol_idx]
    t0 = time.perf_counter()
    try:
        units, obj_names = evaluate_annual_units(
            dvs[sol_idx], formulation_name=formulation,
            objective_set=obj_set, ensemble_spec=unit_spec,
            realization_batch=realization_batch,
        )
        sow_ids = [int(g) // int(r_per_sow)
                   for g in global_ids[r0:r0 + units.shape[0]]]
        sow_matrix, sow_labels, survivors = sow_objective_matrix(
            units, obj_set, sow_ids)
    except Exception as exc:  # noqa: BLE001 - a failed unit contributes no rows
        err = f"{type(exc).__name__}: {exc}"
        print(f"[chunk-reeval] solution {sid} x chunk {chunk_idx} x block "
              f"{block_idx} failed: {err}")
        return None, err
    _print_unit_line(sid, chunk_idx, block_idx, t0)
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
    work_slots: list[tuple[int, int, int]],
    dvs: np.ndarray,
    solution_ids: list[int],
    formulation: str,
    obj_set,
    chunks: list[tuple[object, list[int]]],
    layout: list[list[tuple[int, int]]],
    realization_batch: int | None,
    r_per_sow: int,
) -> pd.DataFrame:
    """One-shot driver: simulate this rank's units, returning accumulated long rows."""
    frames = []
    for sol_idx, chunk_idx, block_idx in work_slots:
        df, _err = _evaluate_unit(sol_idx, chunk_idx, block_idx, dvs, solution_ids,
                                  formulation, obj_set, chunks, layout,
                                  realization_batch, r_per_sow)
        if df is not None:
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return _empty_long_frame()


###############################################################################
# Incremental per-unit persistence (+ resume, wall guard, scheduling)
###############################################################################

def _unit_stem(units_dir: Path, chunk_idx: int, sid: int, block_idx: int) -> Path:
    """Canonical extension-less path for one unit's artifacts."""
    return units_dir / f"chunk{chunk_idx:03d}" / f"sol{int(sid):05d}_b{int(block_idx):03d}"


#: Unit artifact names: ``sol{sid}_b{block}`` + result or failure suffix. A name
#: without the block index is a (solution, chunk) unit of the earlier layout.
_UNIT_FILE = re.compile(r"^sol(\d+)_b(\d+)\.(parquet|csv\.gz|failed)$")
_LEGACY_UNIT_FILE = re.compile(r"^sol(\d+)\.(parquet|csv\.gz|failed)$")


def _refuse_legacy_units(units_dir: Path) -> None:
    """Fail loudly on unit files written before units carried a block index."""
    legacy = sorted(p for p in units_dir.glob("chunk*/sol*")
                    if _LEGACY_UNIT_FILE.match(p.name))
    if legacy:
        raise RuntimeError(
            f"{units_dir} holds {len(legacy)} (solution, chunk) unit file(s) of "
            f"the earlier layout (e.g. {legacy[0].relative_to(units_dir)}). "
            f"Units are now (solution, chunk, block); remove the partial "
            f"directory and resubmit."
        )


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


def _completed_units(units_dir: Path, *, retry_failed: bool) -> set[tuple[int, int, int]]:
    """(sid, chunk_idx, block_idx) triples that need no work: done, or failed and not retried."""
    done: set[tuple[int, int, int]] = set()
    if not units_dir.exists():
        return done
    _refuse_legacy_units(units_dir)
    for p in units_dir.glob("chunk*/sol*"):
        m = _UNIT_FILE.match(p.name)
        if m is None:
            continue
        try:
            chunk_idx = int(p.parent.name[len("chunk"):])
        except ValueError:
            continue
        sid, block_idx, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
        if suffix != "failed" or not retry_failed:
            done.add((sid, chunk_idx, block_idx))
    return done


def _try_claim(claims_dir: Path, chunk_idx: int, sid: int, block_idx: int) -> bool:
    """Atomically claim a unit for this rank (O_CREAT|O_EXCL; False if taken)."""
    path = claims_dir / f"chunk{chunk_idx:03d}_sol{int(sid):05d}_b{int(block_idx):03d}.claim"
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
    work: list[tuple[int, int, int]],
    dvs: np.ndarray,
    solution_ids: list[int],
    formulation: str,
    obj_set,
    chunks: list[tuple[object, list[int]]],
    layout: list[list[tuple[int, int]]],
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
    _check_or_write_layout(
        units_dir, _layout_manifest(layout, realization_batch, r_per_sow), rank)
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
    for sol_idx, chunk_idx, block_idx in my_units:
        sid = solution_ids[sol_idx]
        if (int(sid), chunk_idx, block_idx) in done:
            n_skip += 1
            continue
        if claims_dir is not None and not _try_claim(claims_dir, chunk_idx, sid, block_idx):
            continue
        if _out_of_wall_time(CHUNK_UNIT_SECONDS, CHUNK_STOP_EPOCH):
            print(f"[chunk-reeval] rank {rank}: wall guard stop after {n_run} "
                  f"unit(s); resubmit the same job to resume.", flush=True)
            break
        df, err = _evaluate_unit(sol_idx, chunk_idx, block_idx, dvs, solution_ids,
                                 formulation, obj_set, chunks, layout,
                                 realization_batch, r_per_sow)
        stem = _unit_stem(units_dir, chunk_idx, sid, block_idx)
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

    # Pass the status-quo re-eval matrix if a baseline-policy run (step 10 with
    # NYCOPT_CHUNK_POLICIES=baseline, simulated once per E_test tag) exists for
    # this tag. Without it the incumbent-relative regret family warns and is
    # silently dropped.
    from config import REEVALUATION_SETTINGS
    from src.reeval_core import find_incumbent_cube

    robustness.run(
        reeval_dir,
        baseline_dir=find_incumbent_cube(reeval_dir),
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
    behavior. Stateless over the unit files -> trivially resumable. The block
    layout comes from ``partial/units/layout.json``, written by the simulate
    pass, so the merge does not depend on the realization batch in its own env.

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
        FileNotFoundError: If no simulate pass recorded a unit layout, or units
            are missing and ``allow_partial`` is False.
        RuntimeError: If the partial dir holds unit files of the earlier
            (solution, chunk) layout.
        ValueError: If the recorded layout does not match the chunk count, or a
            unit file carries an objective not in ``obj_set``.
    """
    units_dir = Path(reeval_dir) / "partial" / "units"
    obj_names = [o.name for o in obj_set]
    col_pos = {name: k for k, name in enumerate(obj_names)}

    if units_dir.exists():
        _refuse_legacy_units(units_dir)
    layout = _read_layout(units_dir)
    if layout is None:
        raise FileNotFoundError(
            f"merge_units: {units_dir / 'layout.json'} is missing, so no "
            f"incremental simulate pass has written units here. Run the "
            f"simulate job first."
        )
    if len(layout) != len(chunks):
        raise ValueError(
            f"merge_units: the recorded unit layout covers {len(layout)} "
            f"chunk(s) but the test ensemble has {len(chunks)}."
        )

    missing: list[tuple[int, int, int]] = []
    n_failed = 0
    raw_results = []
    for sid in solution_ids:
        mat = np.full((n_sow, len(obj_names)), np.nan)
        surv = np.full(n_sow, np.nan)
        any_rows = False
        for j, blocks in enumerate(layout):
            for k in range(len(blocks)):
                stem = _unit_stem(units_dir, j, sid, k)
                df = _read_unit(stem)
                if df is None:
                    if stem.with_suffix(".failed").exists():
                        n_failed += 1  # evaluated and failed -> NaN rows, not "missing"
                    else:
                        missing.append((int(sid), j, k))
                    continue
                obj_idx = df["objective"].map(col_pos)
                if obj_idx.isna().any():
                    bad = sorted(set(df["objective"][obj_idx.isna()]))
                    raise ValueError(
                        f"unit sol{sid}/chunk{j}/block{k} carries objectives "
                        f"not in the active set: {bad}"
                    )
                sow_rows = df["sow_id"].to_numpy(dtype=int)
                mat[sow_rows,
                    obj_idx.to_numpy(dtype=int)] = df["value"].to_numpy(dtype=float)
                # Rows repeat the per-SOW count once per objective; repeated
                # assignment writes the same scalar (blocks hold disjoint SOWs).
                surv[sow_rows] = df["n_survivors"].to_numpy(dtype=float)
                any_rows = True
        if any_rows:
            raw_results.append((sid, mat, obj_names, surv, None))
        else:
            raw_results.append((sid, None, None, None, "no rows"))

    if missing and not allow_partial:
        preview = ", ".join(f"sol{s}/chunk{j}/block{k}" for s, j, k in missing[:20])
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

def _resolve_campaign(formulation: str, seed, reeval_dir: Path | None,
                      *, baseline_policy: bool = False):
    """Shared campaign resolution for the simulate and merge entry points.

    Both must reconstruct identical (obj_set, chunks, reeval_dir) from the same
    env so a standalone merge scores exactly the campaign the simulate jobs ran.
    A baseline-policy run (``NYCOPT_CHUNK_POLICIES=baseline``) lands in
    ``reeval/{tag}/baseline/``, the incumbent cube the regret family of every
    later policy-set run under the same tag reads (:func:`_persist_and_score`).
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
        if baseline_policy:
            from src.reeval_core import incumbent_cube_dir
            reeval_dir = incumbent_cube_dir(reeval_dir)   # shared across designs
            reeval_dir.mkdir(parents=True, exist_ok=True)
    return obj_set, test_spec, chunks, n_sow, r_per_sow, Path(reeval_dir)


def simulate_test_chunks(
    formulation: str, dvs: np.ndarray, solution_ids: list[int] | None = None,
    *, seed=None, realization_batch: int | None = None, reeval_dir: Path | None = None,
    baseline_policy: bool = False,
) -> Path | None:
    """Re-evaluate ``dvs`` against every chunk of the test ensemble and write robustness artifacts.

    Args:
        formulation: Formulation name (DV grammar).
        dvs: ``(n_solutions, n_vars)`` decision-variable matrix.
        solution_ids: Ids aligned to ``dvs`` rows (default ``range(n_solutions)``).
        seed: Optional provenance seed (output subdir + meta).
        realization_batch: Realizations per within-chunk simulation batch (default
            ``config.SEARCH_REALIZATION_BATCH``); also the unit granularity
            (:func:`unit_block_size`).
        reeval_dir: Output dir (default ``reeval_output_dir`` under the test ensemble's re-eval tag).
        baseline_policy: ``dvs`` is the current FFMP policy alone; the output
            dir is the design-independent ``outputs/baseline/reeval/{tag}/``
            (``reeval_core.incumbent_cube_dir``, the incumbent cube every
            design's regret family reads).

    Returns:
        The re-eval output directory on rank 0 (``None`` when the merge is
        deferred via ``NYCOPT_CHUNK_MERGE=off``); ``None`` on worker ranks.
    """
    from config import (CHUNK_DONE_DEADLINE_S, CHUNK_INCREMENTAL, CHUNK_MERGE,
                        SEARCH_REALIZATION_BATCH)

    obj_set, test_spec, chunks, n_sow, r_per_sow, reeval_dir = _resolve_campaign(
        formulation, seed, reeval_dir, baseline_policy=baseline_policy)
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

    layout = unit_layout(chunks, realization_batch, r_per_sow)
    n_blocks = sum(len(blocks) for blocks in layout)
    if incremental:
        # Chunk-major, then block: at any moment the active ranks work within a
        # few consecutive chunks, so a node's ranks re-read the same chunk HDF5
        # set through the page cache instead of touching all chunks at once,
        # and a rank's consecutive solutions reuse one block's cached model.
        work = [(s, j, k) for j, blocks in enumerate(layout)
                for k in range(len(blocks)) for s in range(n_solutions)]
    else:
        # Static s-major ordering (the reference path).
        work = [(s, j, k) for s in range(n_solutions)
                for j, blocks in enumerate(layout) for k in range(len(blocks))]

    if rank == 0:
        print(f"[chunk-reeval] {n_solutions} solutions x {n_blocks} blocks "
              f"({len(chunks)} chunks) = {len(work)} units "
              f"across {size} rank(s); N_M={n_realizations}, batch={realization_batch}, "
              f"incremental={int(incremental)}, merge={CHUNK_MERGE}.")

    if incremental:
        _run_units_incremental(work, dvs, solution_ids, formulation, obj_set,
                               chunks, layout, realization_batch, r_per_sow,
                               partial_dir, rank, size)
    else:
        my_slots = [work[i] for i in assign_rank_slots(len(work), rank, size)]
        rows = _rank_long_rows(my_slots, dvs, solution_ids, formulation, obj_set,
                               chunks, layout, realization_batch, r_per_sow)
        _write_partial(rows, partial_dir / f"rank_{rank:03d}")
    mark_rank_done(partial_dir, rank)

    if rank != 0:
        return None
    if CHUNK_MERGE == "off":
        print("[chunk-reeval] merge deferred (NYCOPT_CHUNK_MERGE=off); run "
              "workflow/11_merge_test_chunks.sh when all units are complete.",
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
    *, seed=None, reeval_dir: Path | None = None, baseline_policy: bool = False,
) -> Path:
    """Standalone merge of a completed (or partial) incremental simulate run.

    Resolves the campaign from the same env the simulate jobs used, so the ids,
    objective set and output leaf (``baseline/`` for a baseline-policy run)
    match exactly. Stateless over the unit files: safe to re-run, overwrites
    its own outputs.
    """
    from config import CHUNK_MERGE_ALLOW_PARTIAL

    obj_set, test_spec, chunks, n_sow, _r_per_sow, reeval_dir = _resolve_campaign(
        formulation, seed, reeval_dir, baseline_policy=baseline_policy)
    return merge_units(reeval_dir, list(solution_ids), n_sow,
                       formulation, obj_set, chunks, seed,
                       allow_partial=bool(CHUNK_MERGE_ALLOW_PARTIAL))
