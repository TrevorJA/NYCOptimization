"""ensemble_objective_sensitivity_run.py - Ensemble objective-sensitivity sweep.

Evaluates many random decision-variable (DV) vectors over ONE fixed
probabilistic (Kirsch-Nowak) ensemble and stores, for each DV, the
**per-realization** value of every base objective — a matrix of shape
``(N_DV x N_realizations x N_objectives)``. Every diagnostic in the companion
figures script (ensemble-size K convergence, across-realization operator
agreement, redundancy, threshold sensitivity) is a post-hoc reduction of this
matrix, so the expensive simulation runs exactly once per DV (never per K or per
operator). See ``docs/notes/methods/ensemble_objective_sensitivity_experiment.md``.

Each DV is simulated over the full ensemble in realization batches
(``ENS_REALIZATION_BATCH``) to bound peak memory: only the scalar
per-realization metrics are kept; each batch's timeseries are freed before the
next. Salinity/temperature LSTMs are off (the active objective set uses neither;
large speedup over the ensemble).

Configuration lives entirely in ``supplemental_config.py`` (ensemble sizing, DV
count, seed, formulation, objective list, batch size, output paths) — no CLI
value flags. ``configure_ensemble_env()`` runs before ``config`` is imported so
the LSTMs are disabled at config import.

MPI: each rank takes an ``array_split`` slice of the (baseline + N) DV vectors,
simulates each, writes an ``.npz`` shard + a ``.done`` marker; rank 0 combines
the shards into the matrix HDF5 once all markers appear (filesystem barrier —
avoids the flaky ``comm.gather`` on this OpenMPI build), then deletes the shards.

Usage (interactive, single rank — local smoke):
    python scripts/supplemental/ensemble_objective_sensitivity_run.py
Usage (SLURM):
    sbatch slurm/supplemental/ensemble_objective_sensitivity.sh
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

# supplemental_config MUST precede config: configure_ensemble_env() disables the
# salinity/temperature LSTMs in the environment, which config reads at import.
import supplemental_config as scfg  # noqa: E402

scfg.configure_ensemble_env()  # set experiment env before config is imported

import config  # noqa: E402
from src.ensembles import get_ensemble_spec  # noqa: E402
from src.formulations import get_baseline_values  # noqa: E402
from src.objectives import OBJECTIVES  # noqa: E402
from src.sensitivity_common import (  # noqa: E402
    assign_rank_slots,
    await_all_done,
    get_mpi_context,
    mark_rank_done,
    prepare_partial_dir,
    resolve_objective_names,
    sample_lhs_dvs,
)
from src.simulation import (  # noqa: E402
    check_dv_feasibility,
    dvs_to_config,
    run_simulation_ensemble_batched,
)


def _eval_dv(dv: np.ndarray, formulation: str, spec, base_objs: list,
             batch_size: int) -> np.ndarray:
    """Per-realization base-metric matrix for one DV over the full ensemble.

    Delegates the realization-batching loop to the shared
    :func:`src.simulation.run_simulation_ensemble_batched` — the same path
    Borg's ``evaluate()`` uses when ``SEARCH_REALIZATION_BATCH`` is set — so the
    diagnostic and the MOEA search handle realizations identically. Each batch
    gets a distinct ``preset_name`` (handled inside the shared function) so the
    simulation layer's model-dict cache does not reuse a different batch's model.

    Args:
        dv: Decision-variable vector.
        formulation: Formulation name.
        spec: Full-ensemble :class:`EnsembleSpec`.
        base_objs: Ordered ``(name, Objective)`` pairs to evaluate per realization.
        batch_size: Realizations per simulation batch.

    Returns:
        Array of shape ``(n_realizations, n_objectives)``; failed realizations or
        objectives are ``nan``.
    """
    n_obj = len(base_objs)
    cfg = dvs_to_config(dv, formulation)

    def per_real(data) -> np.ndarray:
        # Per-realization base-metric vector; a failed metric -> NaN cell.
        row = np.full(n_obj, np.nan, dtype=float)
        for k, (_, obj) in enumerate(base_objs):
            try:
                row[k] = float(obj.compute(data))
            except Exception:
                row[k] = np.nan
        return row

    rows = run_simulation_ensemble_batched(
        cfg, spec, batch_size, per_real,
        skip_failed_batches=True,                       # failed batch -> NaN rows
        failed_value=np.full(n_obj, np.nan, dtype=float),
    )
    return np.asarray(rows, dtype=float)  # (n_real, n_obj)


def _combine_and_write(partial_dir: Path, out_path: Path, *, obj_names: list,
                       spec, formulation: str) -> None:
    """Rank-0: merge per-rank ``.npz`` shards into the matrix HDF5, then clean up."""
    shards = sorted(partial_dir.glob("rank_*.npz"))
    if not shards:
        sys.exit("[ensemble_run] no shards found — all ranks failed?")

    sample_ids, dvs, metrics = [], [], []
    for sh in shards:
        with np.load(sh) as z:
            sample_ids.append(z["sample_ids"])
            dvs.append(z["dvs"])
            metrics.append(z["metrics"])
    sample_ids = np.concatenate(sample_ids)
    dvs = np.concatenate(dvs, axis=0)
    metrics = np.concatenate(metrics, axis=0)

    order = np.argsort(sample_ids, kind="stable")
    sample_ids, dvs, metrics = sample_ids[order], dvs[order], metrics[order]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("metrics", data=metrics)           # (n_dv, n_real, n_obj)
        f.create_dataset("dv_vectors", data=dvs)            # (n_dv, n_vars)
        f.create_dataset("sample_ids", data=sample_ids)     # (n_dv,)  baseline = -1
        f.create_dataset("objective_names",
                         data=np.array(obj_names, dtype=object), dtype=str_dt)
        f.create_dataset("realization_ids",
                         data=np.asarray(spec.realization_indices, dtype=int))
        f.attrs["inflow_type"] = spec.inflow_type
        f.attrs["n_realizations"] = int(spec.n_realizations)
        f.attrs["realization_years"] = int(spec.realization_years or 0)
        f.attrs["formulation"] = formulation
        f.attrs["kn_seed"] = int(spec.seed or 0)
        f.attrs["dv_seed"] = int(scfg.ENS_SEED)
    print(f"\n=== Saved {out_path}  metrics={metrics.shape} ===", flush=True)

    for sh in shards:
        sh.unlink()
    for marker in partial_dir.glob("rank_*.done"):
        marker.unlink()
    try:
        partial_dir.rmdir()
    except OSError:
        pass


def main() -> None:
    comm, rank, size = get_mpi_context()
    is_root = rank == 0

    formulation = scfg.ENS_FORMULATION
    spec = get_ensemble_spec(scfg.ensemble_inflow_type())
    if not spec.is_ensemble:
        sys.exit(f"[ensemble_run] resolved spec '{spec.preset_name}' is not an "
                 "ensemble; check ENS_REALIZATION_YEARS / ENS_N_REALIZATIONS.")

    base_names = resolve_objective_names(scfg.ENS_OBJECTIVE_SET)
    base_objs = [(n, OBJECTIVES[n]) for n in base_names]

    # LHS DV sample (each rank regenerates it from the seed — the DV space is
    # small, so re-sampling is cheaper than a pickle/bcast cycle).
    samples = sample_lhs_dvs(formulation, scfg.ENS_SEED, scfg.ENS_N_DV)
    try:
        baseline_dv = np.asarray(get_baseline_values(formulation), dtype=float)
        sample_ids = np.array([-1] + list(range(scfg.ENS_N_DV)), dtype=int)
        all_dvs = np.vstack([baseline_dv[None, :], samples])
    except (ValueError, KeyError):
        sample_ids = np.arange(scfg.ENS_N_DV, dtype=int)
        all_dvs = samples

    if is_root:
        print("=== Ensemble objective-sensitivity sweep ===", flush=True)
        print(f"  formulation:     {formulation}", flush=True)
        print(f"  ensemble:        {spec.inflow_type} "
              f"(N={spec.n_realizations}, {spec.realization_years}-yr)", flush=True)
        print(f"  n DV vectors:    {scfg.ENS_N_DV} (+ baseline)", flush=True)
        print(f"  base objectives: {len(base_names)} -> {base_names}", flush=True)
        print(f"  realiz. batch:   {scfg.ENS_REALIZATION_BATCH}", flush=True)
        print(f"  ranks:           {size}", flush=True)
        print(flush=True)

    rank_slots = assign_rank_slots(len(sample_ids), rank, size)

    partial_dir = scfg.ENS_MATRIX_DIR / f"_partial_{scfg._ens_stem()}"
    if is_root:
        scfg.ENS_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    prepare_partial_dir(partial_dir, rank)

    n_real = spec.n_realizations
    n_obj = len(base_objs)
    local_ids = np.array([int(sample_ids[s]) for s in rank_slots], dtype=int)
    local_dvs = np.array([all_dvs[s] for s in rank_slots], dtype=float) \
        if rank_slots else np.empty((0, all_dvs.shape[1]))
    local_metrics = np.full((len(rank_slots), n_real, n_obj), np.nan, dtype=float)

    t0 = time.time()
    n_infeasible = 0
    for i, slot in enumerate(rank_slots):
        sid = int(sample_ids[slot])
        ts = time.perf_counter()
        dv = all_dvs[slot]
        # Cheap structural-feasibility probe (one realization) before the full
        # ensemble: skip a policy that is infeasible for every realization rather
        # than crashing batch after batch. Row stays all-NaN.
        feasible, err = check_dv_feasibility(dvs_to_config(dv, formulation), spec)
        if not feasible:
            n_infeasible += 1
            print(f"  [rank {rank:>2} INFEASIBLE] sid={sid:4d}  skipped "
                  f"({time.perf_counter() - ts:.1f}s): {err}", flush=True)
            continue
        try:
            local_metrics[i] = _eval_dv(dv, formulation, spec,
                                        base_objs, scfg.ENS_REALIZATION_BATCH)
            n_nan = int(np.isnan(local_metrics[i]).all(axis=1).sum())
            print(f"  [rank {rank:>2} ok] sid={sid:4d}  "
                  f"({time.perf_counter() - ts:.1f}s, {n_nan} empty realiz.)",
                  flush=True)
        except Exception:
            tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
            print(f"  [rank {rank:>2} FAIL] sid={sid:4d}  {tb}", flush=True)
    print(f"  [rank {rank:>2}] done {len(rank_slots)} DVs in "
          f"{time.time() - t0:.1f}s ({n_infeasible} infeasible skipped)", flush=True)

    np.savez(partial_dir / f"rank_{rank:03d}.npz",
             sample_ids=local_ids, dvs=local_dvs, metrics=local_metrics)
    mark_rank_done(partial_dir, rank)

    if not is_root:
        return

    if not await_all_done(partial_dir, size):
        missing = {f"rank_{r:03d}.done" for r in range(size)} - {
            p.name for p in partial_dir.glob("rank_*.done")}
        print(f"[ensemble_run] WARN: timeout waiting for {missing}", flush=True)

    _combine_and_write(partial_dir, scfg.ensemble_matrix_path(),
                       obj_names=base_names, spec=spec, formulation=formulation)


if __name__ == "__main__":
    main()
