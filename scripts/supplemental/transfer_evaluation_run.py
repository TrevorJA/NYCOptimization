"""transfer_evaluation_run.py - Driver for the transfer-evaluation matrix.

Evaluates each scenario design's adopted Pareto-approximate set under the OTHER
designs' search ensembles, then reduces the resulting
(source optimization) x (target ensemble) matrix through three metrics the
study already uses: dominance over the scenario-matched FFMP baseline, the
epsilon-nondominated merged reference set per target ensemble with each
optimization's contribution to it, and hypervolume on one shared reference.

Stages, selected by ``NYCOPT_TEV_STAGE`` (set by the wrappers):

``check``     MPI task farm over a small, pre-registered sample of each
              design's OWN d0 ensemble, differenced against the objective
              columns stored in that design's ``.set``. Bounds the difference
              between the search evaluation path and this driver's path, in
              epsilon units. This is what licenses mixing stored diagonal
              values with simulated off-diagonal values inside one
              nondominated sort.
``evaluate``  MPI task farm over every (cell, solution) unit of the
              off-diagonal matrix, through
              ``src.simulation.evaluate_annual_units`` with the target passed
              as an explicit ``ensemble_spec``. Per-unit atomic writes; resume
              by resubmitting the same job.
``merge``     Reassembles per-unit artifacts into per-cell objective matrices
              and per-realization annual-unit tables, and verifies that
              recomposing from the persisted tensor reproduces the per-unit
              composed vectors exactly.
``analyze``   Computes the three readouts and writes the tables.

Settings in ``supplemental_config.py`` (``TEV_*``); no CLI value flags.
Wrappers: ``workflow/supplemental/transfer_evaluation_{check,eval,analysis}.sh``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_tev_env()

import config  # noqa: E402
from src import transfer_eval as tev  # noqa: E402
from src import transfer_stats as tstats  # noqa: E402
from src.diagnostics import problem_name_for  # noqa: E402
from src.formulations import get_n_objs, get_n_vars, get_obj_directions  # noqa: E402
from src.sensitivity_common import (  # noqa: E402
    epsilon_nondominated, get_mpi_context,
)

#: Realization count and length the adopted sets were searched at. Asserted per
#: cell rather than assumed; see configure_tev_env for why this is not 300.
SEARCH_N = 100
SEARCH_L = 10


###############################################################################
# Shared resolution
###############################################################################

def _objective_set():
    """The active annual-unit objective set."""
    from src.formulations import get_objective_set
    return get_objective_set()


def _cells(include_draws: bool = None) -> list:
    """The cell list for this run - the FULL matrix, diagonal included.

    The diagonal is simulated rather than read from the stored ``.set``
    objective columns. Those columns were produced by searches that ran at
    commit ``dc7e70b`` on 2026-08-11/12, before commit ``a1e88bd``
    (2026-08-18, "align metrics on June 1") moved ``START_DATE`` from
    1945-10-01 to 1945-12-01 and introduced ``ENSEMBLE_START_DATE``. The stored
    values are therefore on a different metric window from anything the current
    code produces, which the path-consistency check measured directly: the
    historic columns are multiples of 1/76 where the driver gives 1/77, and the
    ensemble designs differ by 1-4 epsilon. Mixing the two inside one
    nondominated sort would bias which solutions enter a merged reference set,
    so every cell goes through one evaluation path.
    """
    targets = list(scfg.TEV_TARGETS)
    if scfg.TEV_INCLUDE_DRAWS if include_draws is None else include_draws:
        targets += list(scfg.TEV_DRAW_TARGETS)
    return tev.build_cells(scfg.TEV_SET_FILES, targets, include_diagonal=True)


def _check_cells() -> list:
    """Diagonal cells, for the path-consistency check only."""
    return [c for c in tev.build_cells(
        scfg.TEV_SET_FILES, scfg.TEV_TARGETS, include_diagonal=True)
        if c.on_design]


def _sample_rows(stored_borg: np.ndarray, n_take: int) -> list:
    """Evenly spaced rows over an epsilon-normalised ordering of the front.

    One deterministic, pre-registered rule serving both jobs the check has: the
    extremes bound the path difference in the worst case, and the even spacing
    keeps the sample representative rather than adversarial. Ordering by the
    epsilon-normalised sum of the Borg objective columns is reproducible from
    the ``.set`` alone.
    """
    obj_set = _objective_set()
    eps = np.array([o.epsilon for o in obj_set], dtype=float)
    score = (np.asarray(stored_borg, dtype=float) / eps).sum(axis=1)
    order = np.argsort(score, kind="stable")
    n_rows = order.size
    if n_rows <= n_take:
        return order.tolist()
    picks = np.linspace(0, n_rows - 1, int(n_take)).round().astype(int)
    return [int(order[i]) for i in sorted(set(picks.tolist()))]


def _work_list(cells: list, sample: dict = None) -> list:
    """Cell-major (cell, row) work list.

    Cell-major so ranks working the same cell re-read one staged target through
    the page cache and hit the ``src.simulation`` model-dict cache; the spec is
    identical within a cell, so the cache key is stable and the model build
    amortises over the cell's units.
    """
    work = []
    for ci, cell in enumerate(cells):
        rows = sample[cell.key] if sample else range(_set_rows(cell))
        for row in rows:
            work.append((ci, int(row)))
    return work


_SET_ROWS_CACHE: dict = {}


def _set_rows(cell) -> int:
    """Data-row count of a cell's source set."""
    key = str(cell.set_file)
    if key not in _SET_ROWS_CACHE:
        dvs, _ = tev.load_source_set(cell.set_file, get_n_vars(scfg.TEV_FORMULATION),
                                     get_n_objs())
        _SET_ROWS_CACHE[key] = int(dvs.shape[0])
    return _SET_ROWS_CACHE[key]


###############################################################################
# The task farm (shared by `check` and `evaluate`)
###############################################################################

def _run_task_farm(cells: list, sample: dict, units_root_name: str) -> None:
    """Evaluate a work list across MPI ranks with claim scheduling.

    Ranks pull units via ``O_CREAT|O_EXCL`` claim files rather than receiving a
    contiguous slice. That matters here because the work is heterogeneous: a
    ``historic_single`` unit costs about 31 s against about 155 s for an
    N=100 x L=10 unit, so a static contiguous split would strand ranks on the
    cheap cells. Every rank scans the whole list from a rank-dependent offset,
    so no unit is orphaned by a rank that dies before claiming it.
    """
    comm, rank, size = get_mpi_context()
    is_root = rank == 0
    obj_set = _objective_set()
    n_vars, n_objs = get_n_vars(scfg.TEV_FORMULATION), get_n_objs()

    if is_root:
        print(f"[tev:{units_root_name}] {len(cells)} cell(s), {size} rank(s) in this "
              f"task", flush=True)
        for cell in cells:
            print(f"[tev:{units_root_name}]   {cell.key} "
                  f"({len(sample[cell.key]) if sample else _set_rows(cell)} units)",
                  flush=True)

    # Preconditions and provenance on rank 0 only, before any simulation.
    if is_root:
        for cell in cells:
            identity = tev.check_cell_preconditions(
                cell, n_vars, n_objs, SEARCH_N, SEARCH_L)
            cell_dir = _cell_dir(cell, units_root_name)
            cell_dir.mkdir(parents=True, exist_ok=True)
            prov = tev.cell_provenance(
                cell, identity, obj_set, _set_rows(cell),
                extra={"stage": units_root_name,
                       "n_units_planned": len(sample[cell.key]) if sample
                       else _set_rows(cell)})
            (cell_dir / "provenance.json").write_text(json.dumps(prov, indent=2,
                                                                default=str))
        print(f"[tev:{units_root_name}] preconditions passed for every cell.",
              flush=True)
    if comm is not None:
        comm.Barrier()

    work = _work_list(cells, sample)

    # Claims are scoped to the ARRAY job when there is one, so every task of a
    # submission shares a single claim space and pulls from one global pool.
    # That is what lets the matrix be spread over many small array tasks - the
    # only geometry the cluster backfills promptly under a 22,000-job backlog -
    # without two tasks evaluating the same unit. A later resubmission gets a
    # new array id and therefore a fresh claim space, so a task killed mid-unit
    # never leaves a permanently blocked claim; completion, which is read from
    # the unit files themselves, is what actually prevents rework.
    # NYCOPT_TEV_CLAIM_TAG lets SEPARATE submissions join one work pool, which
    # is how the farm is widened beyond what a single array ramps up to. The
    # per-user cap on this partition is 2,048 cores; one array reaches only a
    # few hundred because tasks start as backfill windows appear. Submitting
    # additional arrays with the same tag adds workers to the same pool, and
    # the claim files keep them from evaluating the same unit. Strides differ
    # between submissions, so a few claim collisions occur; they cost one
    # failed O_EXCL each, not a duplicated simulation.
    job_id = (os.environ.get("NYCOPT_TEV_CLAIM_TAG", "").strip()
              or os.environ.get("SLURM_ARRAY_JOB_ID")
              or os.environ.get("SLURM_JOB_ID", "local"))
    claims_dir = _units_root(units_root_name) / f"claims_{job_id}"
    claims_dir.mkdir(parents=True, exist_ok=True)
    if comm is not None:
        comm.Barrier()

    # Stride over the GLOBAL rank across the array, not the rank within one
    # task: otherwise every task would compute the same slices and collide on
    # every claim.
    n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1") or 1)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0") or 0)
    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0") or 0)
    g_rank = (task_id - task_min) * size + rank
    g_size = max(n_tasks * size, 1)

    done_by_cell = {cell.key: tev.completed_units(
        _cell_dir(cell, units_root_name), retry_failed=scfg.TEV_RETRY_FAILED)
        for cell in cells}

    # Two passes, because a naive "every rank scans the whole list" costs one
    # filesystem claim attempt per (rank, unit): at 6,330 units on 512 ranks
    # that is 3.2 million Lustre metadata operations, enough to dominate the
    # job. Pass 1 is strided, so the ranks claim disjoint units and collide
    # essentially never (about one claim per unit, total). Pass 2 refreshes the
    # done set from the filesystem and sweeps only what is genuinely left, so a
    # rank that finishes early still drains another rank's tail and the farm
    # stays dynamically balanced.
    passes = [work[g_rank::g_size], None]

    dv_cache: dict = {}
    unit_seconds = float(os.environ.get("NYCOPT_TEV_UNIT_SECONDS", "0") or 0)
    stop_epoch = float(os.environ.get("NYCOPT_TEV_STOP_EPOCH", "0") or 0)

    n_run = n_skip = n_fail = 0
    t_start = time.time()
    busy = 0.0
    stop = False
    for pass_idx, my_work in enumerate(passes):
        if stop:
            break
        if my_work is None:
            # Straggler sweep: re-read completion from the filesystem so the
            # sweep only attempts units nobody has finished, then walk the whole
            # list from a rank-dependent offset so ranks enter the tail at
            # different points.
            done_by_cell = {cell.key: tev.completed_units(
                _cell_dir(cell, units_root_name),
                retry_failed=scfg.TEV_RETRY_FAILED) for cell in cells}
            remaining = [(ci, r) for ci, r in work
                         if r not in done_by_cell[cells[ci].key]]
            if not remaining:
                break
            off = (g_rank * len(remaining)) // max(g_size, 1)
            my_work = remaining[off:] + remaining[:off]
            print(f"[tev:{units_root_name}] rank {rank}: straggler sweep over "
                  f"{len(my_work)} unfinished unit(s)", flush=True)
        for ci, row in my_work:
            cell = cells[ci]
            if row in done_by_cell[cell.key]:
                n_skip += 1
                continue
            if not tev.try_claim(claims_dir, cell.key, row):
                continue
            if tev.out_of_wall_time(unit_seconds, stop_epoch):
                print(f"[tev:{units_root_name}] rank {rank}: wall guard stop "
                      f"after {n_run} unit(s); resubmit to resume.", flush=True)
                stop = True
                break

            if str(cell.set_file) not in dv_cache:
                dv_cache[str(cell.set_file)] = tev.load_source_set(
                    cell.set_file, n_vars, n_objs)
            dvs, _stored = dv_cache[str(cell.set_file)]

            stem = tev.unit_stem(_cell_dir(cell, units_root_name), row)
            t0 = time.perf_counter()
            payload, err = tev.evaluate_unit(
                dvs[row], cell, obj_set, scfg.TEV_FORMULATION,
                realization_batch=config.SEARCH_REALIZATION_BATCH or None)
            elapsed = time.perf_counter() - t0
            busy += elapsed

            if payload is None:
                stem.parent.mkdir(parents=True, exist_ok=True)
                stem.with_suffix(".failed").write_text(str(err) + "\n")
                print(f"[tev:{units_root_name}] rank {rank} FAIL cell={cell.key} "
                      f"row={row}: {err}", flush=True)
                n_fail += 1
            else:
                meta = {
                    "cell": cell.key,
                    "row": int(row),
                    "natural": [float(v) for v in payload["natural"]],
                    "obj_names": payload["obj_names"],
                    "n_survivors": payload["n_survivors"],
                    "n_realizations": payload["n_realizations"],
                    "n_unit_years": payload["n_unit_years"],
                    "seconds": float(elapsed),
                    "rank": int(rank),
                }
                tev.atomic_write_unit(payload["units"], stem, meta)
                failed = stem.with_suffix(".failed")
                if failed.exists():
                    failed.unlink()
                tev.print_unit_line(cell.key, row, t0, rank)
            n_run += 1

    wall = time.time() - t_start
    util = busy / wall if wall > 0 else float("nan")
    print(f"[tev:{units_root_name}] rank {rank}: {n_run} evaluated "
          f"({n_fail} failed), {n_skip} already done, wall={wall:.0f}s "
          f"busy={busy:.0f}s utilization={util:.2f}", flush=True)

    # Per-rank utilization records, so the merge stage can report the achieved
    # parallel efficiency without needing the job's stdout.
    rec_dir = _units_root(units_root_name) / f"ranks_{job_id}"
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / f"rank_{g_rank:05d}.json").write_text(json.dumps({
        "rank": rank, "global_rank": g_rank, "size": size, "global_size": g_size,
        "array_task": task_id, "n_run": n_run, "n_failed": n_fail,
        "n_skipped": n_skip, "wall_s": wall, "busy_s": busy,
        "utilization": util}))


def _units_root(stage: str) -> Path:
    return scfg.TEV_UNITS_ROOT / f"{scfg.tev_prefix()}{stage}"


def _cell_dir(cell, stage: str) -> Path:
    return _units_root(stage) / cell.key


###############################################################################
# Stage: check
###############################################################################

def stage_check() -> None:
    """Evaluate a small diagonal sample and compare against the stored .set."""
    cells = _check_cells()
    n_vars, n_objs = get_n_vars(scfg.TEV_FORMULATION), get_n_objs()
    sample = {}
    for cell in cells:
        _dvs, stored = tev.load_source_set(cell.set_file, n_vars, n_objs)
        sample[cell.key] = _sample_rows(stored, scfg.TEV_CHECK_N_SOLUTIONS)
    _run_task_farm(cells, sample, "check")

    _, rank, _ = get_mpi_context()
    if rank != 0:
        return
    # Root reduces once every rank has written its units; the claim scheduler
    # means a rank only finishes when no unclaimed unit remains, so by the time
    # rank 0 falls through, its own work is done and stragglers are rare. The
    # reduction reports how many units it actually found.
    _reduce_check(cells, sample)


def _reduce_check(cells: list, sample: dict) -> None:
    """Difference driver output against the stored .set columns, in epsilon units."""
    obj_set = _objective_set()
    names = [o.name for o in obj_set]
    eps = [o.epsilon for o in obj_set]
    dirs = get_obj_directions()
    n_vars, n_objs = get_n_vars(scfg.TEV_FORMULATION), get_n_objs()

    rows, per_design = [], {}
    for cell in cells:
        _dvs, stored = tev.load_source_set(cell.set_file, n_vars, n_objs)
        picks = sample[cell.key]
        driver, kept = [], []
        for row in picks:
            frame, meta = tev.read_unit(tev.unit_stem(_cell_dir(cell, "check"), row))
            if frame is None:
                continue
            driver.append(tstats.to_borg(meta["natural"], dirs))
            kept.append(row)
        if not kept:
            print(f"[tev:check] no units found for {cell.key}", flush=True)
            continue
        table = tstats.eps_fraction_table(
            np.array(driver), stored[kept, :], eps, names)
        for entry in table:
            entry["design"] = cell.source
            entry["target_slug"] = cell.target_slug
            rows.append(entry)
        per_design[cell.source] = max(e["max_diff_over_eps"] for e in table)

    if not rows:
        sys.exit("[tev:check] no units to reduce; the evaluate step produced nothing.")

    frame = pd.DataFrame(rows)
    scfg.TEV_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = scfg.tev_table_path("tev_path_consistency")
    frame.to_csv(out, index=False)

    worst = float(frame["max_diff_over_eps"].max())
    print(f"\n[tev:check] path difference, as a fraction of each objective's epsilon:",
          flush=True)
    print(frame[["design", "objective", "max_diff_over_eps",
                 "median_diff_over_eps"]].to_string(index=False), flush=True)
    print(f"\n[tev:check] worst |delta|/eps across all objectives and designs: "
          f"{worst:.2e}", flush=True)
    print(f"[tev:check] reference magnitude (TEV_CHECK_EPS_FRAC) = "
          f"{scfg.TEV_CHECK_EPS_FRAC}", flush=True)
    print(f"[tev:check] per-design worst: "
          f"{ {k: f'{v:.2e}' for k, v in per_design.items()} }", flush=True)
    if worst > scfg.TEV_CHECK_EPS_FRAC:
        print(f"[tev:check] This is NOT pywrdrb jitter. Measured 2026-09-11: the "
              f"stored .set columns come from searches at commit dc7e70b "
              f"(2026-08-11/12), before commit a1e88bd (2026-08-18, 'align "
              f"metrics on June 1') moved START_DATE 1945-10-01 -> 1945-12-01 "
              f"and introduced ENSEMBLE_START_DATE. The historic columns are "
              f"multiples of 1/76 where the driver gives 1/77; the ensemble "
              f"designs keep 9 unit-years but sample a 2-month-shifted slice. "
              f"Consequently EVERY cell of the matrix is simulated through this "
              f"driver and no stored column is used as data. This table is "
              f"retained as the measurement of that window change.", flush=True)
    print(f"[tev:check] table -> {out}", flush=True)


###############################################################################
# Stage: evaluate
###############################################################################

def stage_evaluate() -> None:
    """MPI task farm over the matrix, optionally restricted to one cell.

    ``NYCOPT_TEV_CELL`` selects a single cell by index into the cell list. That
    exists because the cluster schedules many small jobs far sooner than one
    large one: at the time of writing a four-node ``wholenode`` request was
    estimated nine hours out while a sixty-core ``shared`` job started in ninety
    seconds. Running one cell per array task turns the matrix into nine
    concurrent small jobs.

    It is also safe to run this way. Claim files are keyed by cell, so two array
    tasks working different cells never contend, and each cell's units are
    written into its own directory. Restricting by cell is a work-assignment
    choice only: the units, and therefore the merged result, are identical
    either way.
    """
    cells = _cells()
    only = os.environ.get("NYCOPT_TEV_CELL", "").strip()
    if only:
        idx = int(only)
        if not 0 <= idx < len(cells):
            sys.exit(f"[tev:evaluate] NYCOPT_TEV_CELL={idx} out of range "
                     f"(0..{len(cells) - 1})")
        cells = [cells[idx]]
    if scfg.TEV_SMOKE:
        n_vars, n_objs = get_n_vars(scfg.TEV_FORMULATION), get_n_objs()
        sample = {}
        for cell in cells:
            _dvs, stored = tev.load_source_set(cell.set_file, n_vars, n_objs)
            sample[cell.key] = _sample_rows(stored, scfg.TEV_SMOKE_N_SOLUTIONS)
    else:
        sample = None
    _run_task_farm(cells, sample, "evaluate")


###############################################################################
# Stage: merge
###############################################################################

def stage_merge() -> None:
    """Reassemble per-unit artifacts and verify offline recomposition."""
    cells = _cells()
    obj_set = _objective_set()
    names = [o.name for o in obj_set]
    scfg.TEV_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    merged_dir = scfg.TEV_UNITS_ROOT / f"{scfg.tev_prefix()}merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    status_rows, qc = [], {"cells": {}, "valid": True}
    for cell in cells:
        n_rows = _set_rows(cell)
        natural, long_frame, status = tev.merge_cell(
            _cell_dir(cell, "evaluate"), n_rows, names)

        # Merge integrity: the same arithmetic over the same persisted data
        # must reproduce the per-unit composed vectors exactly. A mismatch
        # means the persisted tensor does not support the offline
        # recomposition this instrument promises.
        again = tev.recompose_from_units(long_frame, obj_set, n_rows)
        both = np.isfinite(natural) & np.isfinite(again)
        max_dev = float(np.nanmax(np.abs(natural[both] - again[both]))) if both.any() else 0.0
        exact = max_dev == 0.0

        np.save(merged_dir / f"{cell.key}__natural.npy", natural)
        if not long_frame.empty:
            long_frame.to_parquet(merged_dir / f"{cell.key}__units.parquet",
                                  index=False)
        pd.DataFrame(natural, columns=names).to_csv(
            merged_dir / f"{cell.key}__natural.csv", index=False)

        secs = list(status["seconds"].values())
        status_rows.append({
            "cell": cell.key, "source": cell.source,
            "target_slug": cell.target_slug,
            "n_rows": status["n_rows"], "n_present": status["n_present"],
            "n_missing": status["n_missing"], "n_failed": status["n_failed"],
            "recompose_max_dev": max_dev, "recompose_exact": exact,
            "mean_unit_seconds": float(np.mean(secs)) if secs else float("nan"),
            "total_unit_seconds": float(np.sum(secs)) if secs else 0.0,
        })
        qc["cells"][cell.key] = {
            "n_missing": status["n_missing"], "n_failed": status["n_failed"],
            "missing_rows": status["missing_rows"],
            "failed_rows": status["failed_rows"],
            "recompose_exact": exact, "recompose_max_dev": max_dev,
        }
        if status["n_missing"] or not exact:
            qc["valid"] = False
        print(f"[tev:merge] {cell.key}: {status['n_present']}/{status['n_rows']} "
              f"present, {status['n_failed']} failed, {status['n_missing']} missing, "
              f"recompose_max_dev={max_dev:.3e}", flush=True)

    frame = pd.DataFrame(status_rows)
    frame.to_csv(scfg.tev_table_path("tev_merge_status"), index=False)

    qc.update(_rank_utilization())
    scfg.tev_json_path("tev_merge_qc").write_text(json.dumps(qc, indent=2, default=str))
    print(f"[tev:merge] valid={qc['valid']}; status -> "
          f"{scfg.tev_table_path('tev_merge_status')}", flush=True)
    if not qc["valid"]:
        print("[tev:merge] NOTE: cells are incomplete or recomposition is inexact; "
              "the analyze stage will refuse to run. Resubmit the evaluate job to "
              "fill missing units (resubmitting IS the resume).", flush=True)


def _rank_utilization() -> dict:
    """Achieved rank utilization from the per-rank records."""
    out = {}
    for stage in ("evaluate", "check"):
        recs = []
        root = _units_root(stage)
        if not root.exists():
            continue
        for d in sorted(root.glob("ranks_*")):
            for f in sorted(d.glob("rank_*.json")):
                try:
                    recs.append(json.loads(f.read_text()))
                except Exception:  # noqa: BLE001
                    continue
        if not recs:
            continue
        busy = sum(r["busy_s"] for r in recs)
        wall = max(r["wall_s"] for r in recs)
        n = len(recs)
        out[f"{stage}_parallelism"] = {
            "n_ranks_reporting": n,
            "total_busy_core_s": busy,
            "max_rank_wall_s": wall,
            "aggregate_utilization": busy / (n * wall) if n and wall else float("nan"),
            "mean_rank_utilization": float(np.mean([r["utilization"] for r in recs])),
            "units_evaluated": int(sum(r["n_run"] for r in recs)),
            "units_failed": int(sum(r["n_failed"] for r in recs)),
        }
    return out


###############################################################################
# Stage: analyze
###############################################################################

def _baseline_natural(target_design: str, names: list) -> np.ndarray:
    """Scenario-matched FFMP baseline for a target ensemble, in natural units.

    A baseline is only comparable to a front scored on the SAME substrate, so
    the file is resolved per scenario. These CSVs are in natural units while
    ``.set`` columns are Borg-oriented; the conversion happens at the call
    site, never here.
    """
    path = config.baseline_objectives_csv(scfg.TEV_FORMULATION, target_design)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"[tev] no scenario-matched baseline for '{target_design}': {path}\n"
            f"[tev] Produce it with workflow/05_run_baseline.sh --search-ensemble "
            f"under that design's env file.")
    frame = pd.read_csv(path)
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise RuntimeError(
            f"[tev] baseline {path} does not carry {missing}; it was written for a "
            f"different objective set.")
    return frame.iloc[0][names].to_numpy(dtype=float)


def _cell_natural(cell, n_rows: int, names: list) -> np.ndarray:
    """Objective values for one cell, in natural units.

    Every cell, diagonal included, comes from the merged driver output. See
    :func:`_cells` for why the stored ``.set`` columns are not used.
    """
    path = (scfg.TEV_UNITS_ROOT / f"{scfg.tev_prefix()}merged"
            / f"{cell.key}__natural.npy")
    if not path.exists():
        raise FileNotFoundError(f"[tev] merged cell missing: {path}; run the merge stage.")
    return np.load(path)


def stage_analyze() -> None:
    """Compute the three readouts and write the tables."""
    qc_path = scfg.tev_json_path("tev_merge_qc")
    if not qc_path.exists():
        sys.exit("[tev:analyze] no merge QC found; run the merge stage first.")
    qc = json.loads(qc_path.read_text())
    if not qc.get("valid", False) and os.environ.get("NYCOPT_TEV_ALLOW_PARTIAL") != "1":
        sys.exit(f"[tev:analyze] merge QC reports the matrix incomplete or "
                 f"recomposition inexact ({qc_path}); refusing to analyze. "
                 f"Set NYCOPT_TEV_ALLOW_PARTIAL=1 to override deliberately.")

    obj_set = _objective_set()
    names = [o.name for o in obj_set]
    eps = np.array([o.epsilon for o in obj_set], dtype=float)
    dirs = np.array(get_obj_directions(), dtype=float)
    problem = problem_name_for(scfg.TEV_SLUG)
    n_vars, n_objs = get_n_vars(scfg.TEV_FORMULATION), get_n_objs()
    sources = sorted(scfg.TEV_SET_FILES)

    # Full 3x3, every cell from the driver (one evaluation path).
    all_cells = tev.build_cells(scfg.TEV_SET_FILES, scfg.TEV_TARGETS,
                                include_diagonal=True)
    natural = {}
    for cell in all_cells:
        natural[(cell.source, cell.target_slug)] = _cell_natural(
            cell, _set_rows(cell), names)

    target_slugs, target_designs = [], {}
    for design, draw in scfg.TEV_TARGETS:
        slug = tev.resolve_target_spec(design, draw).preset_name
        target_slugs.append(slug)
        target_designs[slug] = design

    _readout_dominance(natural, sources, target_slugs, target_designs, names, eps, dirs)
    _readout_merged(natural, sources, target_slugs, names, eps, dirs, problem,
                    n_vars, n_objs)
    _readout_hypervolume(natural, sources, target_slugs, names, eps, dirs, problem)
    _write_manifest(qc, sources, target_slugs)


def _readout_dominance(natural, sources, target_slugs, target_designs, names,
                       eps, dirs) -> None:
    """Readout 1: fraction of each set dominating the FFMP baseline."""
    rows = []
    for slug in target_slugs:
        base = _baseline_natural(target_designs[slug], names)
        for source in sources:
            obj = natural[(source, slug)]
            keep = np.isfinite(obj).all(axis=1)
            summary = tstats.dominance_summary(
                obj[keep], base, dirs, eps, tol=scfg.TEV_DOMINANCE_TOL)
            summary.update({
                "source": source, "target_slug": slug,
                "on_design": source == target_designs[slug],
                "n_evaluated": int(keep.sum()),
                "n_total": int(obj.shape[0]),
            })
            summary["objectives_beaten_hist"] = json.dumps(
                summary["objectives_beaten_hist"])
            rows.append(summary)
    frame = pd.DataFrame(rows)
    out = scfg.tev_table_path("tev_dominance_matrix")
    frame.to_csv(out, index=False)
    _baseline_table(target_designs, names)
    print(f"\n[tev:analyze] readout 1 - fraction dominating the FFMP baseline\n"
          f"{frame.pivot(index='source', columns='target_slug', values='frac_dominating').to_string()}",
          flush=True)
    print(f"[tev:analyze] -> {out}", flush=True)


def _readout_merged(natural, sources, target_slugs, names, eps, dirs, problem,
                    n_vars, n_objs) -> None:
    """Readout 2: merged reference set per ensemble, and contribution shares."""
    scfg.TEV_SETS_DIR.mkdir(parents=True, exist_ok=True)
    dv_by_source = {}
    for source in sources:
        dvs, _ = tev.load_source_set(scfg.TEV_SET_FILES[source], n_vars, n_objs)
        dv_by_source[source] = dvs

    rows, moea_rows, eps_rows = [], [], []
    for slug in target_slugs:
        # Per-source .set files in MOEAFramework v5 format, full precision.
        per_source_files, pooled_dv, pooled_obj, pooled_src = {}, [], [], []
        for source in sources:
            obj = natural[(source, slug)]
            keep = np.isfinite(obj).all(axis=1)
            dvs = dv_by_source[source][keep]
            borg = tstats.to_borg(obj[keep], dirs)
            path = tev.write_moea_set_file(
                scfg.tev_set_path(f"src-{source}__tgt-{slug}"), dvs, borg, problem)
            per_source_files[source] = path
            pooled_dv.append(dvs)
            pooled_obj.append(borg)
            pooled_src += [source] * int(keep.sum())

        pooled_dv = np.vstack(pooled_dv)
        pooled_obj = np.vstack(pooled_obj)
        pooled_src = np.array(pooled_src, dtype=object)

        # MOEAFramework performs the union; the epsilon archive is applied
        # once afterwards in the Borg box convention. The merger is called
        # WITHOUT --epsilon on purpose: it does apply epsilon dominance (see
        # transfer_eval.merge_reference_sets), but in MOEAFramework's
        # formulation rather than Borg's, and stacking two different epsilon
        # definitions would make the archive impossible to reason about.
        union_path = tev.merge_reference_sets(
            list(per_source_files.values()),
            scfg.tev_set_path(f"union__tgt-{slug}"), problem, epsilons=None)
        union_dv, union_obj = tev.load_source_set(union_path, n_vars, n_objs)
        keep_idx = epsilon_nondominated(union_obj, eps)
        merged_dv, merged_obj = union_dv[keep_idx], union_obj[keep_idx]
        merged_path = tev.write_moea_set_file(
            scfg.tev_set_path(f"merged__tgt-{slug}"), merged_dv, merged_obj, problem)

        # Attribution by decision vector: exact, unique across searches, and
        # immune to two policies sharing an objective vector.
        source_keys = {
            s: tstats.dv_key_array(pooled_dv[pooled_src == s]) for s in sources}
        credited, disjoint, n_shared = tstats.attribute_members(
            tstats.dv_key_array(merged_dv), source_keys)
        sizes = {s: int((pooled_src == s).sum()) for s in sources}
        for entry in tstats.contribution_table(credited, disjoint, sizes,
                                               merged_dv.shape[0]):
            entry.update({"target_slug": slug, "n_shared_members": n_shared,
                          "union_size": int(union_dv.shape[0])})
            rows.append(entry)

        # MOEAFramework's own Contribution indicator, as the cross-check.
        vals = tev.run_indicator("Contribution", merged_path,
                                 list(per_source_files.values()), problem,
                                 epsilons=eps,
                                 timeout_s=scfg.TEV_INDICATOR_TIMEOUT_S)
        for source, path in per_source_files.items():
            moea_rows.append({
                "target_slug": slug, "source": source,
                "moea_contribution": vals.get(str(path), float("nan")),
                "python_composition_share":
                    credited[source] / merged_dv.shape[0] if merged_dv.shape[0] else float("nan"),
            })

        # Epsilon sensitivity: a share that reorders under a neighbouring
        # epsilon is not a finding.
        for scale in scfg.TEV_EPS_SCALES:
            idx = epsilon_nondominated(union_obj, eps * scale)
            cred, _dis, _sh = tstats.attribute_members(
                tstats.dv_key_array(union_dv[idx]), source_keys)
            for source in sources:
                eps_rows.append({
                    "target_slug": slug, "source": source, "eps_scale": scale,
                    "merged_size": int(idx.size),
                    "composition_share": cred[source] / idx.size if idx.size else float("nan"),
                    "contribution_rate": cred[source] / sizes[source] if sizes[source] else float("nan"),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(scfg.tev_table_path("tev_merged_contribution"), index=False)
    _persist_cell_objectives(natural, sources, target_slugs, names, eps, dirs)
    _merged_diagnostics(natural, sources, target_slugs, names, eps, dirs)
    _validity_diagnostics(natural, sources, target_slugs, names, eps, dirs)
    pd.DataFrame(moea_rows).to_csv(
        scfg.tev_table_path("tev_contribution_crosscheck"), index=False)
    pd.DataFrame(eps_rows).to_csv(
        scfg.tev_table_path("tev_epsilon_sensitivity"), index=False)
    print(f"\n[tev:analyze] readout 2 - composition of each ensemble's merged "
          f"reference set\n"
          f"{frame.pivot(index='source', columns='target_slug', values='composition_share').to_string()}",
          flush=True)
    print(f"\n[tev:analyze] readout 2 - contribution rate (contributed / source set size)\n"
          f"{frame.pivot(index='source', columns='target_slug', values='contribution_rate').to_string()}",
          flush=True)




def _persist_cell_objectives(natural, sources, target_slugs, names, eps, dirs) -> None:
    """Tidy per-solution objectives plus merged-set membership, for the figures.

    The figure script reads tables, never the merged tensors, so every value a
    panel draws has to exist in a CSV. One row per (target ensemble, source
    design, solution): the eight natural-unit objectives, whether that solution
    survived the epsilon merge for that ensemble, and whether it survived the
    plain-Pareto merge. That is what lets a parallel-axes panel grey out the
    non-contributing solutions without recomputing an archive.
    """
    from src.solution_selection import nondominated_mask

    out = []
    for slug in target_slugs:
        pool = np.vstack([natural[(s, slug)] for s in sources])
        who = np.concatenate([[s] * natural[(s, slug)].shape[0] for s in sources])
        rows_idx = np.concatenate([np.arange(natural[(s, slug)].shape[0])
                                   for s in sources])
        eps_mask = _eps_mask(pool, dirs, eps)
        plain_mask = nondominated_mask(pool, dirs)
        frame = pd.DataFrame(pool, columns=names)
        frame.insert(0, "in_plain_pareto", plain_mask)
        frame.insert(0, "in_merged_set", eps_mask)
        frame.insert(0, "row", rows_idx)
        frame.insert(0, "source", who)
        frame.insert(0, "target_slug", slug)
        out.append(frame)
    pd.concat(out, ignore_index=True).to_csv(
        scfg.tev_table_path("tev_cell_objectives"), index=False)


def _baseline_table(target_designs, names) -> None:
    """Scenario-matched FFMP baseline per target ensemble, in natural units."""
    rows = []
    for slug, design in target_designs.items():
        try:
            vec = _baseline_natural(design, names)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[tev:analyze] baseline unavailable for {slug}: {exc}", flush=True)
            continue
        rows.append({"target_slug": slug, "scenario": design,
                     **dict(zip(names, vec))})
    if rows:
        pd.DataFrame(rows).to_csv(scfg.tev_table_path("tev_baseline"), index=False)



def _validity_diagnostics(natural, sources, target_slugs, names, eps, dirs) -> None:
    """Does the analysis survive the archives having been selected elsewhere?

    The adopted ``.set`` archives were epsilon-filtered using objective columns
    computed under the pre-``a1e88bd`` metric window, while every value analysed
    here is computed under the current one. Two consequences have to be
    measured rather than assumed:

    1. **Archive disturbance.** Re-applying the same epsilon vector to the
       current on-design values shows how much of each archive is still
       epsilon-nondominated on the substrate actually analysed. A low retention
       means the input sets are not clean epsilon archives of these values.
    2. **Does the headline pre-date the filter?** Comparing the raw
       pre-refilter union with the adopted set, on the stored columns (which
       share one window across all three designs and are therefore mutually
       comparable), shows whether a between-design difference is created by the
       epsilon re-filter or merely sharpened by it.
    """
    from src.load.reference_set import load_reference_set

    diag = {s: t for s, t in zip(sources, target_slugs)} if False else None
    on_design = {}
    for s in sources:
        for t in target_slugs:
            spec_design = _design_of_slug(t, sources)
            if spec_design == s:
                on_design[s] = t

    rows = []
    for s in sources:
        t = on_design.get(s)
        if t is None:
            continue
        nat = natural[(s, t)]
        kept = epsilon_nondominated(tstats.to_borg(nat, dirs), eps)
        rows.append({
            "source": s, "on_design_target": t, "n_archive": int(nat.shape[0]),
            "n_still_eps_nondominated": int(kept.size),
            "retained_fraction": float(kept.size / nat.shape[0]) if nat.shape[0] else float("nan"),
        })
    pd.DataFrame(rows).to_csv(scfg.tev_table_path("tev_archive_validity"), index=False)

    # Raw union vs adopted set, on the stored (single-window) columns.
    raw_rows = []
    n_vars, n_objs = get_n_vars(scfg.TEV_FORMULATION), get_n_objs()
    for s in sources:
        adopted = Path(scfg.TEV_SET_FILES[s])
        raw = adopted.with_name(f"{scfg.TEV_SLUG}_merged_raw.set")
        for tag, path in (("raw_union", raw), ("adopted", adopted)):
            if not path.exists():
                continue
            _dv, borg = load_reference_set(path, n_vars, n_objs=n_objs)
            nat = tstats.to_natural(borg, dirs)
            row = {"source": s, "archive": tag, "n": int(nat.shape[0])}
            row.update({f"median__{n}": float(np.median(nat[:, k]))
                        for k, n in enumerate(names)})
            raw_rows.append(row)
    if raw_rows:
        pd.DataFrame(raw_rows).to_csv(
            scfg.tev_table_path("tev_raw_vs_adopted"), index=False)

    if rows:
        print("\n[tev:analyze] validity - archive still epsilon-nondominated under the "
              "CURRENT metric window", flush=True)
        for r in rows:
            print(f"    {r['source']:28s} {r['n_still_eps_nondominated']:4d} of "
                  f"{r['n_archive']:4d} = {100 * r['retained_fraction']:5.1f}%", flush=True)


def _design_of_slug(slug: str, sources) -> str:
    """Which scenario design owns a target slug (for locating diagonal cells)."""
    for design, draw in list(scfg.TEV_TARGETS) + list(scfg.TEV_DRAW_TARGETS):
        if tev.resolve_target_spec(design, draw).preset_name == slug:
            return design
    return ""


def _merged_diagnostics(natural, sources, target_slugs, names, eps, dirs) -> None:
    """Is a contribution share a design effect, or arithmetic?

    Three diagnostics, because a raw composition share cannot answer that:

    1. Enrichment against each source's share of the pooled input. Under no
       design effect this is 1.0 whatever the set sizes.
    2. The same, computed on the PLAIN Pareto archive as well as the
       epsilon-box archive. If a source is enriched only after epsilon
       thinning, the result is about resolution, not dominance.
    3. Leave-one-out over the eight objectives. An eight-objective archive can
       be driven by a single axis; dropping each in turn shows which.
    """
    from src.solution_selection import nondominated_mask

    sizes = {s: int(natural[(s, target_slugs[0])].shape[0]) for s in sources}
    who = np.concatenate([[s] * sizes[s] for s in sources])

    enrich_rows, loo_rows, box_rows = [], [], []
    for slug in target_slugs:
        pool = np.vstack([natural[(s, slug)] for s in sources])

        for label, mask in (
            ("plain_pareto", nondominated_mask(pool, dirs)),
            ("epsilon_box", _eps_mask(pool, dirs, eps)),
        ):
            for row in tstats.enrichment_table(who[mask], sizes):
                row.update({"target_slug": slug, "archive": label})
                enrich_rows.append(row)

        def archive_fn(keep, _pool=pool, _slug=slug):
            sub = _pool[:, keep]
            idx = epsilon_nondominated(tstats.to_borg(sub, dirs[keep]), eps[keep])
            tot = int(idx.size)
            out = {"n_members": tot}
            for s in sources:
                share = (who[idx] == s).sum() / tot if tot else float("nan")
                out[s] = share / (sizes[s] / sum(sizes.values()))
            return out

        for row in tstats.leave_one_out_enrichment(archive_fn, len(names), names):
            row["target_slug"] = slug
            loo_rows.append(row)

        for s in sources:
            borg = tstats.to_borg(natural[(s, slug)], dirs)
            n_boxes = tstats.occupied_boxes(borg, eps)
            box_rows.append({
                "target_slug": slug, "source": s, "n_solutions": sizes[s],
                "n_distinct_boxes": n_boxes,
                "boxes_per_100_solutions": 100.0 * n_boxes / sizes[s] if sizes[s] else float("nan"),
            })

    pd.DataFrame(enrich_rows).to_csv(scfg.tev_table_path("tev_enrichment"), index=False)
    pd.DataFrame(loo_rows).to_csv(scfg.tev_table_path("tev_leave_one_out"), index=False)
    pd.DataFrame(box_rows).to_csv(scfg.tev_table_path("tev_box_occupancy"), index=False)

    ef = pd.DataFrame(enrich_rows)
    print("\n[tev:analyze] readout 2 diagnostic - enrichment vs each source's share of the "
          "pool (null = 1.00)", flush=True)
    print(ef.pivot_table(index=["archive", "source"], columns="target_slug",
                         values="enrichment").round(2).to_string(), flush=True)
    lf = pd.DataFrame(loo_rows)
    print("\n[tev:analyze] readout 2 diagnostic - leave-one-out enrichment (epsilon archive)",
          flush=True)
    print(lf.pivot_table(index="dropped", columns="target_slug",
                         values=list(sources)).round(2).to_string(), flush=True)


def _eps_mask(pool: np.ndarray, dirs, eps) -> np.ndarray:
    """Boolean epsilon-nondominated mask over a pooled natural-unit matrix."""
    mask = np.zeros(pool.shape[0], dtype=bool)
    mask[epsilon_nondominated(tstats.to_borg(pool, dirs), eps)] = True
    return mask


def _readout_hypervolume(natural, sources, target_slugs, names, eps, dirs,
                         problem) -> None:
    """Readout 3: hypervolume of each set under each ensemble.

    MOEAFramework normalises indicators against the reference set passed with
    ``--reference`` and places the hypervolume reference point at that set's
    nadir offset by ``hypervolume.delta``. Passing ONE shared reference - the
    pooled union of all nine cells - is therefore what makes the nine values
    comparable, with no hand-rolled normalisation.
    """
    files, index = [], []
    for s in sources:
        for t in target_slugs:
            path = scfg.tev_set_path(f"src-{s}__tgt-{t}")
            if path.exists():
                files.append(path)
                index.append((s, t))
    if not files:
        print("[tev:analyze] no per-cell set files; skipping hypervolume", flush=True)
        return

    # The shared reference is the plain-Pareto union of all nine cells, built by
    # MOEAFramework from the per-cell files so it carries real decision vectors
    # rather than placeholders. Its bounds fix the normalization and its nadir
    # (offset by hypervolume.delta) fixes the reference point, so passing this
    # one file to every call is what makes the nine values comparable.
    shared_ref = tev.merge_reference_sets(
        files, scfg.tev_set_path("shared_reference_all_cells"), problem,
        epsilons=None, timeout_s=scfg.TEV_INDICATOR_TIMEOUT_S)
    _rdv, _robj = tev.load_source_set(
        shared_ref, get_n_vars(scfg.TEV_FORMULATION), get_n_objs())
    pooled = _robj
    t0 = time.time()
    vals = tev.run_indicator("Hypervolume", shared_ref, files, problem,
                             epsilons=eps, timeout_s=scfg.TEV_INDICATOR_TIMEOUT_S)
    elapsed = time.time() - t0
    method = "wfg_exact" if vals else "unavailable"
    rows = [{"source": s, "target_slug": t,
             "hypervolume": vals.get(str(p), float("nan")),
             "method": method, "seconds_total": elapsed,
             "shared_reference_size": int(pooled.shape[0])}
            for (s, t), p in zip(index, files)]
    frame = pd.DataFrame(rows)
    frame.to_csv(scfg.tev_table_path("tev_hypervolume_matrix"), index=False)
    if vals:
        print(f"\n[tev:analyze] readout 3 - hypervolume on one shared reference "
              f"({elapsed:.0f}s)\n"
              f"{frame.pivot(index='source', columns='target_slug', values='hypervolume').to_string()}",
              flush=True)
    else:
        print(f"\n[tev:analyze] readout 3 - hypervolume unavailable after "
              f"{elapsed:.0f}s (8-D exact hypervolume is expensive; see the "
              f"methods note).", flush=True)


def _write_manifest(qc: dict, sources, target_slugs) -> None:
    """One provenance record for the whole analysis."""
    obj_set = _objective_set()
    manifest = {
        "instrument": "transfer_evaluation",
        "smoke": scfg.TEV_SMOKE,
        "sources": list(sources),
        "target_slugs": list(target_slugs),
        "set_files": {k: str(v) for k, v in scfg.TEV_SET_FILES.items()},
        "objective_names": [o.name for o in obj_set],
        "objective_epsilons": [o.epsilon for o in obj_set],
        "objective_directions": [o.sign for o in obj_set],
        "dominance_tol": scfg.TEV_DOMINANCE_TOL,
        "eps_scales": list(scfg.TEV_EPS_SCALES),
        "search_n": SEARCH_N, "search_l": SEARCH_L,
        "merge_qc": qc,
        "written": pd.Timestamp.utcnow().isoformat(),
    }
    path = scfg.tev_json_path("tev_manifest")
    path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n[tev:analyze] manifest -> {path}", flush=True)
    if "evaluate_parallelism" in qc:
        p = qc["evaluate_parallelism"]
        print(f"[tev:analyze] evaluate stage: {p['units_evaluated']} units on "
              f"{p['n_ranks_reporting']} ranks, aggregate utilization "
              f"{p['aggregate_utilization']:.2f}, "
              f"{p['total_busy_core_s'] / 3600:.1f} core-hours", flush=True)


###############################################################################

def main() -> None:
    stage = os.environ.get("NYCOPT_TEV_STAGE", "").strip()
    if stage == "check":
        stage_check()
    elif stage == "evaluate":
        stage_evaluate()
    elif stage == "merge":
        stage_merge()
    elif stage == "analyze":
        stage_analyze()
    else:
        sys.exit("[tev] set NYCOPT_TEV_STAGE to 'check', 'evaluate', 'merge' "
                 "or 'analyze'")


if __name__ == "__main__":
    main()
