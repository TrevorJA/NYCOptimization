"""
random_sample_mpi.py - MPI-parallel random-sample diagnostic of the active
objective set. Each MPI rank gets a slice of (baseline + N random DV vectors)
via numpy.array_split, simulates each in-memory, and computes objectives.
Rank 0 gathers and writes a single CSV identical in format to the
sequential `random_sample_salinity_test.py` output.

Why MPI over multiprocessing:
    Each Pywr-DRB sim takes ~150s with the salinity LSTM in-loop. With
    11 sims at 150s = 28 min sequentially. MPI on a single node with
    {N+1} ranks reduces wall time to ~150-200s (one sim per rank, plus
    a small gather overhead).

Required env (set by `slurm/envs/ffmp_obj7_sal.env` or equivalent):
    NYCOPT_SALINITY_ON=1
    NYCOPT_OBJECTIVES=...,salt_front_max_rm

Optional env:
    PYWRDRB_SIM_START_DATE=2018-01-01    # shorter sim window for debugging
    PYWRDRB_SIM_END_DATE=2022-12-31

Usage (interactive, single rank — for debug):
    python scripts/random_sample_mpi.py --n 10 --seed 42 --formulation ffmp

Usage (MPI):
    mpirun -np 11 python scripts/random_sample_mpi.py --n 10 --seed 42

Usage (SLURM, recommended):
    sbatch slurm/random_sample.sh
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import OUTPUTS_DIR, derive_slug, ACTIVE_OBJECTIVES, INCLUDE_SALINITY_MODEL
from src.formulations import (
    get_baseline_values,
    get_bounds,
    get_var_names,
    get_objective_set,
)
from src.simulation import dvs_to_config, run_simulation_inmemory


def _get_mpi_context():
    """Return (comm, rank, size). Falls back to (None, 0, 1) without mpi4py."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def _sample_random_dvs(formulation: str, rng: np.random.Generator,
                       n_samples: int) -> np.ndarray:
    """Uniform random samples within DV bounds. Shape (n_samples, n_vars)."""
    lows, highs = get_bounds(formulation)
    return rng.uniform(low=lows, high=highs, size=(n_samples, len(lows)))


def _run_one(sample_id: int, dv: np.ndarray, formulation: str,
             objective_set) -> dict:
    """Simulate one DV vector and return summary dict.

    Includes:
      - All active objective values (NaN on failure)
      - Salt-front time-series statistics (min/median/max RM)
      - Wall time
    """
    out = {"sample_id": sample_id, "elapsed_s": float("nan")}
    t0 = time.perf_counter()
    try:
        cfg = dvs_to_config(dv, formulation)
        data = run_simulation_inmemory(cfg, use_trimmed=True)
        objs = objective_set.compute(data)
        out["elapsed_s"] = time.perf_counter() - t0
        for name, val in zip(objective_set.names, objs):
            out[name] = float(val)
        if "salinity" in data:
            sf = data["salinity"]["salt_front_location_mu"].dropna()
            if len(sf):
                out["sf_min_RM"] = float(sf.min())
                out["sf_median_RM"] = float(sf.median())
                out["sf_max_RM"] = float(sf.max())
            else:
                out["sf_min_RM"] = out["sf_median_RM"] = out["sf_max_RM"] = float("nan")
    except Exception as e:
        tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
        out["error"] = f"{type(e).__name__}: {e} ({tb})"
        out["elapsed_s"] = time.perf_counter() - t0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10,
                        help="Number of random DV samples (default 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default 42)")
    parser.add_argument("--formulation", default="ffmp",
                        help="Formulation name (default ffmp)")
    args = parser.parse_args()

    comm, rank, size = _get_mpi_context()
    is_root = rank == 0

    if is_root:
        if not INCLUDE_SALINITY_MODEL:
            sys.exit("ERROR: NYCOPT_SALINITY_ON must be 1. "
                     "Source slurm/envs/ffmp_obj7_sal.env first.")
        if "salt_front_max_rm" not in ACTIVE_OBJECTIVES:
            sys.exit("ERROR: NYCOPT_OBJECTIVES must include 'salt_front_max_rm'. "
                     "Source slurm/envs/ffmp_obj7_sal.env first.")

    # Each rank generates the *same* sample set from the seed independently
    # (avoids comm.bcast, which is flaky with mpi4py 4.x on this OpenMPI
    # build). The DV space is small enough that re-sampling on each rank is
    # cheaper than the pickle/bcast cycle anyway.
    rng = np.random.default_rng(args.seed)
    samples = _sample_random_dvs(args.formulation, rng, args.n)

    # FFMP-family formulations always carry registered DV defaults that
    # reproduce the historical FFMP rules — include that baseline row when
    # available.
    has_baseline = True
    try:
        baseline_dv = np.asarray(get_baseline_values(args.formulation), dtype=float)
    except (ValueError, KeyError):
        has_baseline = False
        baseline_dv = None

    if has_baseline:
        sample_ids = np.array([-1] + list(range(args.n)), dtype=int)
        all_dvs = np.vstack([baseline_dv[None, :], samples])
    else:
        sample_ids = np.arange(args.n, dtype=int)
        all_dvs = samples

    if is_root:
        print(f"=== Random-sample MPI diagnostic ===", flush=True)
        print(f"  formulation: {args.formulation}", flush=True)
        print(f"  n samples:   {args.n}{' (+ baseline)' if has_baseline else ''}", flush=True)
        print(f"  seed:        {args.seed}", flush=True)
        print(f"  active obj:  {len(ACTIVE_OBJECTIVES)}", flush=True)
        print(f"  ranks:       {size}", flush=True)
        print(flush=True)

    # Each rank handles its slice. array_split distributes work as evenly
    # as possible (last few ranks get one fewer sim if size doesn't divide).
    slot_idx = np.arange(len(sample_ids), dtype=int)
    rank_slots = list(np.array_split(slot_idx, size)[rank])

    out_dir = OUTPUTS_DIR / "random_sample_tests" / derive_slug(args.formulation)
    partial_dir = out_dir / f"_partial_seed{args.seed}_n{args.n}"
    if is_root:
        out_dir.mkdir(parents=True, exist_ok=True)
        partial_dir.mkdir(parents=True, exist_ok=True)

    # Filesystem-based synchronization (avoids comm.Barrier, which hits the
    # same mpi4py/OpenMPI flakiness as bcast/gather on this build). Workers
    # poll until rank 0 has created the dir.
    if not is_root:
        for _ in range(120):  # up to 60s wait
            if partial_dir.exists():
                break
            time.sleep(0.5)

    objective_set = get_objective_set()
    rank_results = []
    t0 = time.time()
    for slot in rank_slots:
        sid = int(sample_ids[slot])
        dv = all_dvs[slot]
        r = _run_one(sid, dv, args.formulation, objective_set)
        rank_results.append(r)
        tag = "FAIL" if "error" in r else "ok"
        sf_max = r.get("salt_front_max_rm", float("nan"))
        msg = f"  [rank {rank:>2} {tag}] sid={sid:3d}"
        if "error" in r:
            msg += f"  {r['error']}"
        else:
            msg += f"  sf_max={sf_max:.2f} RM  ({r.get('elapsed_s', 0):.1f}s)"
        print(msg, flush=True)
    print(f"  [rank {rank:>2}] done {len(rank_slots)} sims in "
          f"{time.time() - t0:.1f}s", flush=True)

    # Each rank writes its own partial CSV. Done file is touched after the
    # CSV is fully flushed so rank 0 knows the partial is complete.
    if rank_results:
        partial_csv = partial_dir / f"rank_{rank:03d}.csv"
        pd.DataFrame(rank_results).to_csv(partial_csv, index=False)
    done_file = partial_dir / f"rank_{rank:03d}.done"
    done_file.touch()

    if not is_root:
        return

    # Rank 0 polls until every rank's .done file appears. Avoids final
    # comm.Barrier. Generous timeout — single sim is ~150–300s, so a rank
    # that fails to start would normally finish far inside this window.
    expected_dones = {f"rank_{r:03d}.done" for r in range(size)}
    deadline = time.time() + 1800  # 30 min total cap
    while time.time() < deadline:
        seen = {p.name for p in partial_dir.glob("rank_*.done")}
        if seen >= expected_dones:
            break
        time.sleep(2.0)
    else:
        missing = expected_dones - {p.name for p in partial_dir.glob("rank_*.done")}
        print(f"[random_sample_mpi] WARN: timeout waiting for {missing}", flush=True)

    # Concatenate all partial CSVs that exist.
    partials = sorted(partial_dir.glob("rank_*.csv"))
    if not partials:
        sys.exit("[random_sample_mpi] no partial CSVs found — all ranks failed?")
    df = pd.concat([pd.read_csv(p) for p in partials], ignore_index=True)
    df = df.sort_values("sample_id").set_index("sample_id")

    out_csv = out_dir / f"random_samples_seed{args.seed}_n{args.n}.csv"
    df.to_csv(out_csv)
    print(f"\n=== Saved {out_csv} ===", flush=True)

    # Clean up partials and done-flags now that the combined CSV is written.
    for p in partials:
        p.unlink()
    for p in partial_dir.glob("rank_*.done"):
        p.unlink()
    try:
        partial_dir.rmdir()
    except OSError:
        pass  # leave the dir if any unexpected files remain

    # Per-objective summary across random samples (baseline excluded).
    rs = df.drop(index=-1, errors="ignore")
    print("\n=== Random-sample objective spread (baseline excluded) ===", flush=True)
    for obj_name in objective_set.names:
        if obj_name not in rs.columns:
            continue
        col = rs[obj_name].dropna()
        if not len(col):
            print(f"  {obj_name}: all NaN", flush=True)
            continue
        print(f"  {obj_name}:", flush=True)
        print(f"    min   = {col.min():.4f}", flush=True)
        print(f"    p25   = {col.quantile(0.25):.4f}", flush=True)
        print(f"    median= {col.median():.4f}", flush=True)
        print(f"    p75   = {col.quantile(0.75):.4f}", flush=True)
        print(f"    max   = {col.max():.4f}", flush=True)

    # Baseline reference row.
    if -1 in df.index:
        print("\n=== Baseline FFMP objective values (sample_id=-1) ===", flush=True)
        for obj_name in objective_set.names:
            if obj_name in df.columns:
                print(f"  {obj_name:<45s} = {df.loc[-1, obj_name]:.4f}", flush=True)


if __name__ == "__main__":
    main()
