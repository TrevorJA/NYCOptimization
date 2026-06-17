"""objective_sensitivity_run.py - Random-DV objective-sensitivity diagnostic.

Runs many random decision-variable (DV) vectors through the trimmed Pywr-DRB
model on a **single historical reference trace** (``config.INFLOW_TYPE``,
default ``pub_nhmv10_BC_withObsScaled``) and records every evaluated objective
per sample. The downstream figures script
(``objective_sensitivity_figures.py``) turns this CSV into the discrimination
and redundancy diagnostics.

There is **no realization / ensemble loop** — one simulation per DV vector.
Ensemble- and scenario-noise questions are out of scope here (see §4 of
``docs/notes/methods/objective_sensitivity_experiment.md``).

Configuration lives entirely in ``supplemental_config.py`` (sample count, seed,
formulation, objective-set selection, simulation window, output paths) — there
are **no CLI value flags**. ``supplemental_config`` is imported first so its
``NYCOPT_SALINITY_ON`` env knob is in place before ``config`` is imported.

MPI mechanics mirror the prior ``random_sample_mpi.py`` harness: each rank takes
a ``numpy.array_split`` slice of (baseline + N samples), simulates each
in-memory, writes a partial CSV + a ``.done`` marker, and rank 0 concatenates
once all markers appear (filesystem barrier — avoids ``comm.bcast/gather``,
which are flaky on this OpenMPI build).

Usage (interactive, single rank — local smoke):
    python scripts/supplemental/objective_sensitivity_run.py

Usage (SLURM, recommended):
    sbatch slurm/supplemental/objective_sensitivity.sh
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

# supplemental_config MUST precede config: configure_historic_env() sets
# NYCOPT_SALINITY_ON (and the smoke window) in the environment, which config
# reads at import time to enable the salinity LSTM.
import supplemental_config as scfg  # noqa: E402

scfg.configure_historic_env()  # set experiment env before config is imported

import config  # noqa: E402
from src.formulations import get_baseline_values  # noqa: E402
from src.objectives import build_objective_set  # noqa: E402
from src.sensitivity_common import (  # noqa: E402
    assign_rank_slots,
    await_all_done,
    get_mpi_context,
    mark_rank_done,
    prepare_partial_dir,
    resolve_objective_names,
    sample_lhs_dvs,
)
from src.simulation import dvs_to_config, run_simulation_inmemory  # noqa: E402


def _run_one(sample_id: int, dv: np.ndarray, formulation: str,
             objective_set) -> dict:
    """Simulate one DV vector and return a summary row.

    Non-finite objective values (e.g. salinity/temperature when the LSTM is
    off) are recorded as-is — reported, never dropped.

    Args:
        sample_id: -1 for the baseline row, else the 0-based sample index.
        dv: Decision-variable vector.
        formulation: Formulation name.
        objective_set: ObjectiveSet to evaluate.

    Returns:
        Dict with ``sample_id``, ``elapsed_s``, each objective value, and an
        ``error`` string on failure.
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
    except Exception as e:
        tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
        out["error"] = f"{type(e).__name__}: {e} ({tb})"
        out["elapsed_s"] = time.perf_counter() - t0
    return out


def main():
    comm, rank, size = get_mpi_context()
    is_root = rank == 0

    formulation = scfg.FORMULATION

    obj_names = resolve_objective_names(scfg.OBJECTIVE_SET)
    objective_set = build_objective_set(obj_names)

    # The salt-front objective only returns real values with the salinity LSTM
    # on. Warn (don't abort) if it is requested but unavailable — it will be
    # reported as NaN, per the experiment's "report NaN, don't drop" rule. The
    # usual cause is the import-order contract (supplemental_config must precede
    # config) or a host without the PywrDRB-ML LSTM checkout.
    if is_root and "salt_front_intrusion_max_rm" in obj_names \
            and not config.INCLUDE_SALINITY_MODEL:
        print("[objective_sensitivity_run] WARN: salinity LSTM is OFF — "
              "'salt_front_intrusion_max_rm' will be NaN.", flush=True)

    # Each rank regenerates the same LHS sample from the seed independently
    # (avoids comm.bcast). The DV space is small, so re-sampling is cheaper
    # than a pickle/bcast cycle.
    samples = sample_lhs_dvs(formulation, scfg.SEED, scfg.N_SAMPLES)

    # FFMP-family formulations carry registered DV defaults reproducing the
    # historical FFMP rules — include that baseline as sample_id = -1.
    try:
        baseline_dv = np.asarray(get_baseline_values(formulation), dtype=float)
        has_baseline = True
    except (ValueError, KeyError):
        baseline_dv = None
        has_baseline = False

    if has_baseline:
        sample_ids = np.array([-1] + list(range(scfg.N_SAMPLES)), dtype=int)
        all_dvs = np.vstack([baseline_dv[None, :], samples])
    else:
        sample_ids = np.arange(scfg.N_SAMPLES, dtype=int)
        all_dvs = samples

    if is_root:
        print("=== Objective-sensitivity diagnostic (random DV sampling) ===",
              flush=True)
        print(f"  formulation:  {formulation}", flush=True)
        print(f"  n samples:    {scfg.N_SAMPLES}"
              f"{' (+ baseline)' if has_baseline else ''}", flush=True)
        print(f"  seed:         {scfg.SEED}", flush=True)
        print(f"  objective set:{scfg.OBJECTIVE_SET!r} -> {len(obj_names)} obj",
              flush=True)
        print(f"  inflow trace: {config.INFLOW_TYPE}", flush=True)
        print(f"  smoke mode:   {scfg.SMOKE}", flush=True)
        print(f"  ranks:        {size}", flush=True)
        print(flush=True)

    # Each rank handles its array_split slice.
    rank_slots = assign_rank_slots(len(sample_ids), rank, size)

    out_dir = scfg.SAMPLES_DIR
    partial_dir = out_dir / f"_partial_seed{scfg.SEED}_n{scfg.N_SAMPLES}"
    if is_root:
        out_dir.mkdir(parents=True, exist_ok=True)
    # Filesystem barrier: rank 0 creates the shard dir, workers wait for it.
    prepare_partial_dir(partial_dir, rank)

    rank_results = []
    t0 = time.time()
    for slot in rank_slots:
        sid = int(sample_ids[slot])
        r = _run_one(sid, all_dvs[slot], formulation, objective_set)
        rank_results.append(r)
        tag = "FAIL" if "error" in r else "ok"
        msg = f"  [rank {rank:>2} {tag}] sid={sid:4d}"
        msg += f"  {r['error']}" if "error" in r else f"  ({r.get('elapsed_s', 0):.1f}s)"
        print(msg, flush=True)
    print(f"  [rank {rank:>2}] done {len(rank_slots)} sims in "
          f"{time.time() - t0:.1f}s", flush=True)

    # Write this rank's partial CSV, then touch its .done marker.
    if rank_results:
        pd.DataFrame(rank_results).to_csv(partial_dir / f"rank_{rank:03d}.csv",
                                          index=False)
    mark_rank_done(partial_dir, rank)

    if not is_root:
        return

    # Rank 0 polls until every rank's .done marker appears.
    if not await_all_done(partial_dir, size):
        missing = {f"rank_{r:03d}.done" for r in range(size)} - {
            p.name for p in partial_dir.glob("rank_*.done")}
        print(f"[objective_sensitivity_run] WARN: timeout waiting for {missing}",
              flush=True)

    partials = sorted(partial_dir.glob("rank_*.csv"))
    if not partials:
        sys.exit("[objective_sensitivity_run] no partial CSVs — all ranks failed?")
    df = pd.concat([pd.read_csv(p) for p in partials], ignore_index=True)
    df = df.sort_values("sample_id").set_index("sample_id")

    out_csv = scfg.samples_csv_path()
    df.to_csv(out_csv)
    print(f"\n=== Saved {out_csv} ===", flush=True)

    # Clean up partials/markers now that the combined CSV is written.
    for p in partials:
        p.unlink()
    for p in partial_dir.glob("rank_*.done"):
        p.unlink()
    try:
        partial_dir.rmdir()
    except OSError:
        pass

    # Per-objective spread across random samples (baseline excluded).
    rs = df.drop(index=-1, errors="ignore")
    print("\n=== Random-sample objective spread (baseline excluded) ===",
          flush=True)
    for name in objective_set.names:
        if name not in rs.columns:
            continue
        col = rs[name].dropna()
        if not len(col):
            print(f"  {name}: all NaN", flush=True)
            continue
        print(f"  {name}: min={col.min():.4f} med={col.median():.4f} "
              f"max={col.max():.4f} IQR={col.quantile(.75) - col.quantile(.25):.4f}",
              flush=True)

    if -1 in df.index:
        print("\n=== Baseline FFMP objective values (sample_id=-1) ===",
              flush=True)
        for name in objective_set.names:
            if name in df.columns:
                print(f"  {name:<45s} = {df.loc[-1, name]:.4f}", flush=True)


if __name__ == "__main__":
    main()
