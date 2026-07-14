"""
bench_eval_worker.py - Per-rank evaluation timer for the benchmark sweeps.

The shared measurement core of two experiments. Launched as
``mpirun -np K python bench_eval_worker.py``; every rank independently runs
1 cold + M warm evaluations through ``src.simulation.evaluate()`` — the exact
production path a Borg worker executes, not a re-implementation — and records
wall time and peak RSS per eval to its own CSV shard. K concurrent ranks on one
otherwise-idle node reproduce the memory-bandwidth contention a packed campaign
node actually experiences.

Two experiments share this worker, selected by ``NYCOPT_BENCH_EXPERIMENT``:

    packing        The Anvil packing sweep: ONE ensemble shape, ladder over the
                   ranks-per-node K. Sets the campaign's packing density.
                   (workflow/supplemental/anvil_scaling_packing.sh)
    ensemble_cost  The ensemble-cost surface: the ensemble SHAPE (N realizations
                   x L years) and the model variant (trimmed = search, full =
                   re-evaluation) are swept, each cell at its memory-feasible K.
                   Prices the campaign. (workflow/supplemental/ensemble_cost_sweep.sh)

The experiments differ only in which env knobs the calling SLURM script varies
and where the shards land; the timed code path is identical, which is the point
— a cost surface measured by a different harness than the packing sweep would
not be comparable to it.

Design choices:
    * The concurrency level K is ``MPI.COMM_WORLD.Get_size()`` — never a
      value flag (repo convention).
    * Baseline DVs for every eval: per-eval cost is set by model size x
      timesteps x scenarios, not DV values, and identical DVs make the objective
      vector byte-comparable across ranks (a free correctness check, recorded as
      ``objs_ok`` plus the first three objective values). It also detects a
      silent env-plumbing failure in the ensemble-cost sweep: if
      ``NYCOPT_USE_TRIMMED_MODEL`` never reached ``config``, the "full" cells
      would re-measure the trimmed model and produce identical objectives.
    * Lock-step memory-phase artifacts are broken by a per-rank start stagger
      (rank r sleeps ``min(r, stagger_max)`` s), recorded in the shard and
      excluded from all timings.
    * Rows are appended to the shard as each eval completes, so a walltime kill
      preserves the evals that finished. A rank killed by the OOM reaper leaves
      a partial (or no) shard — that is a measurement, not a failure; the step's
      exit code lands in the step manifest written by the SLURM script.
    * No gather / combine: the mpirun exit is the barrier and the analysis
      scripts glob the shards.

Environment (set per step by the calling SLURM script):
    NYCOPT_BENCH_EXPERIMENT          packing | ensemble_cost (default packing)
    NYCOPT_BENCH_WARM_EVALS          warm evals per rank M (default 2)
    NYCOPT_BENCH_MODE                sweep mode, recorded in the shard
    NYCOPT_SEARCH_REALIZATION_BATCH  eval batching (0 = production default)
    NYCOPT_USE_TRIMMED_MODEL         1 = trimmed (search), 0 = full (re-eval)
    NYCOPT_SCALING_KN_REALS/_YEARS   ensemble shape (ensemble_cost only)
"""
from __future__ import annotations

import csv
import os
import resource
import socket
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

EXPERIMENT = os.environ.get("NYCOPT_BENCH_EXPERIMENT", "packing")
if EXPERIMENT == "ensemble_cost":
    scfg.configure_ensemble_cost_env()  # must precede the config import
elif EXPERIMENT == "packing":
    scfg.configure_anvil_scaling_env()  # must precede the config import
else:
    sys.stderr.write(
        f"[bench] ERROR: unknown NYCOPT_BENCH_EXPERIMENT={EXPERIMENT!r}; "
        f"expected 'packing' or 'ensemble_cost'.\n"
    )
    raise SystemExit(2)

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from config import (  # noqa: E402
    SEARCH_ENSEMBLE_SPEC,
    SEARCH_REALIZATION_BATCH,
    USE_TRIMMED_MODEL,
)
from src.formulations import get_baseline_values  # noqa: E402
from src.simulation import _ensemble_window, evaluate  # noqa: E402


def _code_sha() -> str:
    """Short git SHA of the working tree the evals actually ran against.

    Recorded per shard because a SLURM job runs the code as it exists at
    START time, not submit time: a merge/checkout while the job sits in the
    queue silently changes the model under test. The analyzers refuse to pool
    shards from different SHAs, so a mid-queue code change shows up as an
    incomparable epoch instead of a contaminated average.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(PROJECT_DIR), capture_output=True,
                             text=True, timeout=10)
        sha = out.stdout.strip()
        if out.returncode != 0 or not sha:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=str(PROJECT_DIR), capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return "unknown"


def _mem_available_mb() -> float:
    """Node-wide MemAvailable from /proc/meminfo in MB (NaN if unreadable)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0  # kB -> MB
    except OSError:
        pass
    return float("nan")


def _ru_maxrss_mb() -> float:
    """Peak RSS of this process in MB (Linux ru_maxrss is in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _shard_path(spec, model_variant: str, k: int, batch: int,
                rank: int, job_id: str) -> Path:
    """Shard path for this experiment's output tree.

    The packing shard's name and columns are unchanged so ``analyze_scaling.py``
    keeps working; the ensemble-cost shards live in a separate directory keyed
    by (N, L, model, K) so neither analyzer ingests the other's data.
    """
    if EXPERIMENT == "ensemble_cost":
        return scfg.ensemble_cost_shard_path(
            spec.n_realizations, int(spec.realization_years), model_variant,
            k, rank, job_id,
        )
    return scfg.packing_shard_path(k, batch, rank, job_id)


def _formulation() -> str:
    """Formulation whose baseline DVs this experiment evaluates."""
    if EXPERIMENT == "ensemble_cost":
        return scfg.ENSEMBLE_COST_FORMULATION
    return scfg.PACKING_FORMULATION


def _stagger_max_s() -> int:
    """Cap on the per-rank start stagger for this experiment."""
    if EXPERIMENT == "ensemble_cost":
        return scfg.ENSEMBLE_COST_STAGGER_MAX_S
    return scfg.PACKING_STAGGER_MAX_S


def main() -> int:
    """Run 1 cold + M warm evals on this rank and write one CSV shard."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    k_concurrent = comm.Get_size()

    m_warm = int(os.environ.get("NYCOPT_BENCH_WARM_EVALS", "2"))
    mode = os.environ.get("NYCOPT_BENCH_MODE", "ladder")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    batch = int(SEARCH_REALIZATION_BATCH or 0)
    model_variant = "trimmed" if USE_TRIMMED_MODEL else "full"
    formulation = _formulation()
    code_sha = _code_sha()

    spec = SEARCH_ENSEMBLE_SPEC
    if spec is None or not spec.is_ensemble:
        sys.stderr.write(
            "[bench] ERROR: SEARCH_ENSEMBLE_SPEC is not a staged ensemble. "
            "Source an ensemble-design env file (workflow/envs/"
            "anvil_scaling_packing.env or workflow/envs/ensemble_cost.env) and "
            "stage workflow steps 02 and 04 first.\n"
        )
        return 2

    if EXPERIMENT == "ensemble_cost" and spec.realization_years is None:
        sys.stderr.write(
            "[bench] ERROR: the ensemble-cost sweep needs a fixed-length "
            f"ensemble; preset '{spec.preset_name}' has no realization_years.\n"
        )
        return 2

    sim_start, sim_end = _ensemble_window(spec)
    dv = np.array(get_baseline_values(formulation))

    out_csv = _shard_path(spec, model_variant, k_concurrent, batch, rank, job_id)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Decorrelate the ranks' memory-access phases before the first eval.
    stagger_s = min(rank, _stagger_max_s())
    time.sleep(stagger_s)

    fieldnames = [
        "experiment", "preset", "formulation", "model_variant",
        "n_realizations", "realization_years", "sim_start", "sim_end",
        "k_concurrent", "realization_batch", "mode", "rank", "hostname",
        "stagger_s", "eval_index", "kind", "wall_seconds", "ru_maxrss_mb",
        "mem_available_mb", "objs_ok", "obj0", "obj1", "obj2",
        "t_start_epoch", "job_id", "code_sha",
    ]
    warm: list[float] = []
    cold_s = float("nan")

    # Append each row as it completes: a walltime kill then costs only the eval
    # in flight, not the whole cell.
    with out_csv.open("w", newline="", buffering=1) as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1 + m_warm):
            t_start_epoch = time.time()
            t0 = time.perf_counter()
            objs = evaluate(dv, formulation_name=formulation)
            wall = time.perf_counter() - t0
            objs = np.asarray(objs, dtype=float)
            kind = "cold" if i == 0 else "warm"
            if kind == "warm":
                warm.append(wall)
            else:
                cold_s = wall
            writer.writerow({
                "experiment": EXPERIMENT,
                "preset": spec.preset_name,
                "formulation": formulation,
                "model_variant": model_variant,
                "n_realizations": spec.n_realizations,
                "realization_years": spec.realization_years or "full",
                "sim_start": sim_start,
                "sim_end": sim_end,
                "k_concurrent": k_concurrent,
                "realization_batch": batch,
                "mode": mode,
                "rank": rank,
                "hostname": socket.gethostname(),
                "stagger_s": stagger_s,
                "eval_index": i,
                "kind": kind,
                "wall_seconds": f"{wall:.3f}",
                "ru_maxrss_mb": f"{_ru_maxrss_mb():.1f}",
                "mem_available_mb": f"{_mem_available_mb():.1f}",
                "objs_ok": bool(np.all(np.isfinite(objs))),
                "obj0": f"{objs[0]:.6f}",
                "obj1": f"{objs[1]:.6f}",
                "obj2": f"{objs[2]:.6f}",
                "t_start_epoch": f"{t_start_epoch:.1f}",
                "job_id": job_id,
                "code_sha": code_sha,
            })
            fh.flush()

    warm_avg = sum(warm) / len(warm) if warm else float("nan")
    sys.stdout.write(
        f"[bench] rank {rank}/{k_concurrent} {spec.preset_name} {model_variant}: "
        f"cold={cold_s:.1f}s warm_avg={warm_avg:.1f}s "
        f"maxrss={_ru_maxrss_mb():.0f}MB -> {out_csv.name}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
