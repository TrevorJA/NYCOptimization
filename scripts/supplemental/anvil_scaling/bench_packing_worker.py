"""
bench_packing_worker.py - Per-rank eval timer for the Anvil packing sweep.

Stage A of the Anvil scaling experiment (see ``supplemental_config.py`` and
``workflow/supplemental/anvil_scaling_packing.sh``). Launched as
``mpirun -np K python bench_packing_worker.py``; every rank independently runs
1 cold + M warm trimmed-model ensemble evaluations through ``evaluate()`` —
the exact production path Borg workers execute — and records wall time and
peak RSS per eval to its own CSV shard. K concurrent ranks on one otherwise
idle 128-core node measure the memory-bandwidth/contention slowdown that sets
the ranks-per-node packing choice for the optimization campaigns.

Design choices:
    * The concurrency level K is ``MPI.COMM_WORLD.Get_size()`` — never a
      value flag (repo convention).
    * Baseline DVs for every eval: per-eval cost is set by model size x
      timesteps, not DV values, and identical DVs make the objective vector
      byte-comparable across ranks (a free correctness check, recorded as
      ``objs_ok`` plus the first three objective values).
    * Lock-step memory-phase artifacts are broken by a per-rank start stagger
      (rank r sleeps ``min(r, PACKING_STAGGER_MAX_S)`` s), recorded in the
      shard and excluded from all timings.
    * No gather / combine: the mpirun exit is the barrier and the analysis
      script globs the shards. A rank killed by the OOM reaper simply leaves
      no (or a partial) shard — that is a measurement, not a failure; the
      step's exit code lands in the step manifest written by the SLURM script.

Environment (set per ladder step by the SLURM script):
    NYCOPT_PACK_WARM_EVALS   warm evals per rank M (default 2)
    NYCOPT_PACK_MODE         smoke | ladder | spot (recorded in the shard)
    NYCOPT_SEARCH_REALIZATION_BATCH  eval batching (0 = production default)
"""
from __future__ import annotations

import csv
import os
import resource
import socket
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_anvil_scaling_env()  # must precede the config import

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from config import SEARCH_ENSEMBLE_SPEC, SEARCH_REALIZATION_BATCH  # noqa: E402
from src.formulations import get_baseline_values  # noqa: E402
from src.simulation import _ensemble_window, evaluate  # noqa: E402


def _code_sha() -> str:
    """Short git SHA of the working tree the evals actually ran against.

    Recorded per shard because a SLURM job runs the code as it exists at
    START time, not submit time: a merge/checkout while the job sits in the
    queue silently changes the model under test. The analyzer refuses to pool
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


def main() -> int:
    """Run 1 cold + M warm evals on this rank and write one CSV shard."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    k_concurrent = comm.Get_size()

    m_warm = int(os.environ.get("NYCOPT_PACK_WARM_EVALS", "2"))
    mode = os.environ.get("NYCOPT_PACK_MODE", "ladder")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    batch = int(SEARCH_REALIZATION_BATCH or 0)
    code_sha = _code_sha()

    spec = SEARCH_ENSEMBLE_SPEC
    if spec is None or not spec.is_ensemble:
        sys.stderr.write(
            "[packing] ERROR: SEARCH_ENSEMBLE_SPEC is not a staged ensemble. "
            "Source an ensemble-design env file (e.g. "
            "workflow/envs/anvil_scaling_packing.env) and stage steps 02-04 "
            "first.\n"
        )
        return 2

    sim_start, sim_end = _ensemble_window(spec)
    dv = np.array(get_baseline_values(scfg.PACKING_FORMULATION))

    # Decorrelate the ranks' memory-access phases before the first eval.
    stagger_s = min(rank, scfg.PACKING_STAGGER_MAX_S)
    time.sleep(stagger_s)

    rows: list[dict] = []
    for i in range(1 + m_warm):
        t_start_epoch = time.time()
        t0 = time.perf_counter()
        objs = evaluate(dv, formulation_name=scfg.PACKING_FORMULATION)
        wall = time.perf_counter() - t0
        objs = np.asarray(objs, dtype=float)
        rows.append({
            "preset": spec.preset_name,
            "formulation": scfg.PACKING_FORMULATION,
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
            "kind": "cold" if i == 0 else "warm",
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

    out_csv = scfg.packing_shard_path(k_concurrent, batch, rank, job_id)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    warm = [float(r["wall_seconds"]) for r in rows if r["kind"] == "warm"]
    warm_avg = sum(warm) / len(warm) if warm else float("nan")
    sys.stdout.write(
        f"[packing] rank {rank}/{k_concurrent}: "
        f"cold={rows[0]['wall_seconds']}s warm_avg={warm_avg:.1f}s "
        f"maxrss={rows[-1]['ru_maxrss_mb']}MB -> {out_csv.name}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
