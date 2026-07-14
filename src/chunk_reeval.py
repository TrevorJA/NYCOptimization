"""chunk_reeval.py - Simulate a large chunked test ensemble and score it, metrics-only.

Re-evaluates a set of policies (decision-variable vectors) against **every chunk** of a chunked
test ensemble (``src.ensemble_generation.generate_forcing_ensemble`` with ``chunk_size > 0``),
computing objectives/robustness from **in-memory reduced metrics** — full simulation-output
timeseries are never persisted. Memory is bounded three ways: (1) each chunk is a small standalone
ensemble; (2) ``run_simulation_ensemble_batched`` (``SEARCH_REALIZATION_BATCH``) batches realizations
within a chunk, freeing timeseries per batch; (3) work is distributed across MPI ranks.

Design (reusing the re-evaluation stack):
- Work units are ``(solution, chunk)`` pairs, split across ranks by
  :func:`sensitivity_common.assign_rank_slots` (extends re-eval's solution-only split to the
  realization/chunk axis). Ranks coordinate through ``.done`` marker files, not flaky MPI collectives.
- Each unit reduces to a ``(S_chunk, M)`` base-metric matrix via :func:`src.simulation.evaluate_raw`
  (recorder -> ``/dev/null``); rows are re-keyed from the chunk's local ids to the ensemble's **global**
  realization ids. Each rank writes its rows to a long-format parquet partial.
- Rank 0 concatenates the partials, reassembles per-solution ``(N_M, M)`` matrices, and reuses
  :func:`src.reeval_core.persist_reeval_raw` (so ``reeval_raw.parquet`` / ``reeval_raw_meta.json`` /
  ``objectives_summary.csv`` are byte-compatible with the normal re-eval path) then
  :func:`src.robustness.run` for the robustness scorecards.

The re-eval ensemble (``config.REEVAL_ENSEMBLE_SPEC``, via ``NYCOPT_REEVAL_ENSEMBLE_PRESET``) must be
the test-ensemble slug: its ``realization_indices == range(N_M)`` are exactly the global ids, so the reused
persistence keys every row to its true global realization.
"""

from __future__ import annotations

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


def _rank_long_rows(
    work_slots: list[tuple[int, int]],
    dvs: np.ndarray,
    solution_ids: list[int],
    formulation: str,
    obj_set,
    chunks: list[tuple[object, list[int]]],
    realization_batch: int | None,
) -> pd.DataFrame:
    """Simulate this rank's ``(solution, chunk)`` units, returning long-format global-keyed rows."""
    from src.simulation import evaluate_raw

    frames = []
    for sol_idx, chunk_idx in work_slots:
        chunk_spec, global_ids = chunks[chunk_idx]
        sid = solution_ids[sol_idx]
        try:
            base_matrix, base_names = evaluate_raw(
                dvs[sol_idx], formulation_name=formulation,
                objective_set=obj_set, ensemble_spec=chunk_spec,
                realization_batch=realization_batch,
            )
        except Exception as exc:  # noqa: BLE001 - a failed unit contributes no rows
            print(f"[chunk-reeval] solution {sid} x chunk {chunk_idx} failed: "
                  f"{type(exc).__name__}: {exc}")
            continue
        # Vectorized long rows (row-major over local realization, objective),
        # re-keyed from the chunk's local ids to the ensemble's global ids.
        arr = np.asarray(base_matrix, dtype=float)
        r_i, m_i = arr.shape
        gid_row = np.asarray(global_ids[:r_i], dtype=int)
        frames.append(pd.DataFrame({
            "solution_id": np.full(r_i * m_i, int(sid), dtype=int),
            "realization_id": np.repeat(gid_row, m_i),
            "objective": np.tile(np.asarray(base_names, dtype=object), r_i),
            "value": arr.reshape(-1),
        }))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame({
        "solution_id": np.array([], dtype=int),
        "realization_id": np.array([], dtype=int),
        "objective": np.array([], dtype=object),
        "value": np.array([], dtype=float),
    })


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


def _merge_and_persist(
    partial_dir: Path, reeval_dir: Path, solution_ids: list[int], n_realizations: int,
    formulation: str, obj_set, seed,
) -> Path:
    """Concatenate rank chunks, reassemble per-solution matrices, and persist via the re-eval path."""
    from src.reeval_core import persist_reeval_raw
    from src import robustness

    parts = _read_partials(partial_dir)
    long_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["solution_id", "realization_id", "objective", "value"]
    )
    base_names = [o.base.name for o in obj_set]

    # Reassemble each solution's (N, M) matrix in global-id order (NaN for failed/absent cells);
    # persist_reeval_raw maps row j -> realization_indices[j] == global id j.
    raw_results = []
    for sid in solution_ids:
        sub = long_df[long_df["solution_id"] == sid]
        if sub.empty:
            raw_results.append((sid, None, None, "no rows"))
            continue
        piv = (sub.pivot_table(index="realization_id", columns="objective", values="value")
               .reindex(index=range(n_realizations), columns=base_names))
        raw_results.append((sid, piv.to_numpy(dtype=float), base_names, None))

    persist_reeval_raw(reeval_dir, raw_results, formulation, len(solution_ids), seed)

    # Pass the status-quo re-eval matrix if step 05 staged one under this same
    # reeval tag. Without it `improvement_vs_baseline` warns and silently drops --
    # which is what happened on every chunked run before this.
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
        within_sow_agg=REEVALUATION_SETTINGS.get("within_sow_aggregator", "mean"),
    )
    return reeval_dir


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
        The re-eval output directory on rank 0; ``None`` on worker ranks.
    """
    from config import (REEVAL_ENSEMBLE_SPEC, SEARCH_REALIZATION_BATCH,
                        active_scenario_name, derive_slug)
    from src.ensembles import pool_chunk_specs
    from src.reeval_core import reeval_output_dir, resolve_reeval

    if REEVAL_ENSEMBLE_SPEC is None or not REEVAL_ENSEMBLE_SPEC.is_ensemble:
        raise ValueError(
            "chunk re-eval requires NYCOPT_REEVAL_ENSEMBLE_PRESET to resolve to the chunked "
            "test ensemble (an is_ensemble spec whose realization_indices span its global index "
            "space)."
        )
    obj_set, test_spec, _ = resolve_reeval()
    test_slug = test_spec.inflow_type
    n_realizations = test_spec.n_realizations
    chunks = pool_chunk_specs(test_slug)

    dvs = np.atleast_2d(np.asarray(dvs, dtype=float))
    n_solutions = dvs.shape[0]
    if solution_ids is None:
        solution_ids = list(range(n_solutions))
    if realization_batch is None:
        realization_batch = SEARCH_REALIZATION_BATCH

    comm, rank, size = get_mpi_context()
    if reeval_dir is None:
        reeval_dir = reeval_output_dir(active_scenario_name(), derive_slug(formulation),
                                       test_spec, seed)
    partial_dir = Path(reeval_dir) / "partial"
    prepare_partial_dir(partial_dir, rank)

    # (solution, chunk) work units, split contiguously across ranks.
    work = [(s, j) for s in range(n_solutions) for j in range(len(chunks))]
    my_slots = [work[i] for i in assign_rank_slots(len(work), rank, size)]
    if rank == 0:
        print(f"[chunk-reeval] {n_solutions} solutions x {len(chunks)} chunks = {len(work)} units "
              f"across {size} rank(s); N_M={n_realizations}, batch={realization_batch}.")

    rows = _rank_long_rows(my_slots, dvs, solution_ids, formulation, obj_set, chunks,
                           realization_batch)
    _write_partial(rows, partial_dir / f"rank_{rank:03d}")
    mark_rank_done(partial_dir, rank)

    if rank != 0:
        return None
    if not await_all_done(partial_dir, size):
        raise TimeoutError("chunk re-eval: not all ranks reported done before the deadline.")
    return _merge_and_persist(partial_dir, Path(reeval_dir), solution_ids, n_realizations,
                              formulation, obj_set, seed)
