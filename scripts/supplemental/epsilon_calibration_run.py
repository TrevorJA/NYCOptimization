"""epsilon_calibration_run.py - Per-design evaluation stage of the
epsilon-calibration experiment.

Evaluates (baseline + EPS_N_POLICIES constraint-FEASIBLE random DV vectors) on
the ACTIVE scenario design's search ensemble — the exact measure MM Borg
searches under — and persists the full per-unit annual-metric cube
``(n_dv x n_realizations x n_objectives x n_unit_years)`` for the ANNUAL-UNIT
(§2) objective registry. Every epsilon diagnostic in the companion figures
script (signal IQR, bootstrap noise floor, granularity floor, archive-size
sweep) is a post-hoc reduction of this cube, so the expensive simulation runs
exactly once per policy per design.
See ``docs/notes/methods/epsilon_calibration_experiment.md``.

Feasibility: random DV vectors are ~1% feasible under the two formal Borg
constraints, and Borg's archive only ever holds feasible solutions
(constraint-dominance), so the calibration population is drawn uniform on the
FEASIBLE region via rejection (``sample_feasible_dvs`` — pure DV arithmetic,
no simulation, so the ~100x oversampling is cheap). The realized acceptance
rate is persisted as QC.

Design selection: the scenario design is the run identity and comes from the
environment (``NYCOPT_SCENARIO_DESIGN``, sourced from ``NYCOPT_ENV_FILE`` by
the SLURM wrapper) — one sbatch job per campaign design. The single-trace
``historic`` design is evaluated through the same cube layout as N = 1 over
its consecutive water-year units.

All other settings (sample size, seed, formulation, batch size, output paths)
live in ``supplemental_config.py`` — no CLI value flags.

MPI: each rank takes an ``array_split`` slice of the DV vectors, evaluates
each, writes an ``.npz`` shard + a ``.done`` marker; rank 0 combines the
shards into the cube HDF5 once all markers appear (filesystem barrier — avoids
the flaky ``comm.gather`` on this OpenMPI build), then deletes the shards.

Usage (interactive, single rank — local smoke with EPS_SMOKE=True):
    python scripts/supplemental/epsilon_calibration_run.py
Usage (SLURM, one job per design):
    sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/eps_calib_historic.env \\
        workflow/supplemental/epsilon_calibration.sh
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

# supplemental_config MUST precede config: configure_epsilon_env() disables the
# salinity/temperature LSTMs in the environment, which config reads at import.
import supplemental_config as scfg  # noqa: E402

scfg.configure_epsilon_env()  # set experiment env before config is imported

import config  # noqa: E402
from src.formulations import get_baseline_values  # noqa: E402
from src.objectives_ensemble import (  # noqa: E402
    ENSEMBLE_OBJECTIVES,
    water_year_unit_slices,
)
from src.sensitivity_common import (  # noqa: E402
    assign_rank_slots,
    await_all_done,
    get_mpi_context,
    mark_rank_done,
    prepare_partial_dir,
    sample_feasible_dvs,
)
from src.simulation import (  # noqa: E402
    dvs_to_config,
    run_simulation_ensemble_batched,
    run_simulation_inmemory,
)


def _expected_n_units(spec) -> int:
    """Number of metric-bearing water-year units per realization.

    An ensemble realization of L water years yields exactly L - 1 units
    (``water_year_unit_slices``); the single-trace historic design's count is
    derived from the configured simulation window with the same unit rule.

    Args:
        spec: The resolved search :class:`EnsembleSpec`.

    Returns:
        Units per realization (cube's last axis length).
    """
    if spec.is_ensemble:
        return int(spec.realization_years) - 1
    idx = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    return len(water_year_unit_slices(idx))


def _eval_dv(dv: np.ndarray, formulation: str, spec, objs: list,
             n_units: int, batch_size: int) -> np.ndarray:
    """Per-unit annual-metric cube slice for one policy over the ensemble.

    Uses the same batched path Borg's ``evaluate()`` runs
    (:func:`src.simulation.run_simulation_ensemble_batched`), but keeps the
    stage-(i) annual-unit values instead of collapsing them, so
    ``compute_for_borg_from_units`` on the pooled cube reproduces the search
    scalar bit-for-bit. The historic design routes through the single-trace
    simulation as N = 1.

    Args:
        dv: Decision-variable vector (constraint-feasible).
        formulation: Formulation name.
        spec: Search :class:`EnsembleSpec`.
        objs: Ordered :class:`AnnualUnitObjective` list.
        n_units: Expected units per realization (shape guard).
        batch_size: Realizations per simulation batch (0 = one block).

    Returns:
        Array of shape ``(n_realizations, n_objectives, n_units)``; a failed
        realization batch is ``nan``.
    """
    n_obj = len(objs)
    cfg = dvs_to_config(dv, formulation)

    def per_real(data) -> np.ndarray:
        block = np.stack([np.asarray(o.annual_units(data), dtype=float)
                          for o in objs])
        if block.shape != (n_obj, n_units):
            raise ValueError(
                f"annual-unit shape {block.shape} != expected {(n_obj, n_units)}"
            )
        return block

    if not spec.is_ensemble:
        data = run_simulation_inmemory(cfg, use_trimmed=True)
        return per_real(data)[None, :, :]

    rows = run_simulation_ensemble_batched(
        cfg, spec, batch_size, per_real,
        skip_failed_batches=True,                       # failed batch -> NaN
        failed_value=np.full((n_obj, n_units), np.nan),
    )
    return np.asarray(rows, dtype=float)  # (n_real, n_obj, n_units)


def _combine_and_write(partial_dir: Path, out_path: Path, *, design: str,
                       spec, formulation: str, obj_names: list,
                       directions: list, eps_current: list,
                       sample_info: dict) -> None:
    """Rank-0: merge per-rank ``.npz`` shards into the cube HDF5, then clean up."""
    shards = sorted(partial_dir.glob("rank_*.npz"))
    if not shards:
        sys.exit("[epsilon_run] no shards found — all ranks failed?")

    sample_ids, dvs, units, secs = [], [], [], []
    for sh in shards:
        with np.load(sh) as z:
            sample_ids.append(z["sample_ids"])
            dvs.append(z["dvs"])
            units.append(z["units"])
            secs.append(z["eval_seconds"])
    sample_ids = np.concatenate(sample_ids)
    dvs = np.concatenate(dvs, axis=0)
    units = np.concatenate(units, axis=0)
    secs = np.concatenate(secs)

    order = np.argsort(sample_ids, kind="stable")
    sample_ids, dvs, units, secs = (
        sample_ids[order], dvs[order], units[order], secs[order])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    real_ids = (np.asarray(spec.realization_indices, dtype=int)
                if spec.is_ensemble else np.array([0], dtype=int))
    with h5py.File(out_path, "w") as f:
        f.create_dataset("units", data=units)         # (n_dv, n_real, n_obj, n_units)
        f.create_dataset("dv_vectors", data=dvs)      # (n_dv, n_vars)
        f.create_dataset("sample_ids", data=sample_ids)  # (n_dv,) baseline = -1
        f.create_dataset("eval_seconds", data=secs)
        f.create_dataset("objective_names",
                         data=np.array(obj_names, dtype=object), dtype=str_dt)
        f.create_dataset("directions",
                         data=np.array(directions, dtype=object), dtype=str_dt)
        f.create_dataset("epsilons_current", data=np.asarray(eps_current, float))
        f.create_dataset("realization_ids", data=real_ids)
        f.attrs["design"] = design
        f.attrs["preset_name"] = spec.preset_name
        f.attrs["inflow_type"] = spec.inflow_type
        f.attrs["is_ensemble"] = bool(spec.is_ensemble)
        f.attrs["n_realizations"] = int(spec.n_realizations)
        f.attrs["realization_years"] = int(spec.realization_years or 0)
        f.attrs["ensemble_seed"] = int(spec.seed or 0)
        f.attrs["formulation"] = formulation
        f.attrs["dv_seed"] = int(scfg.EPS_SEED)
        f.attrs["realization_batch"] = int(scfg.EPS_REALIZATION_BATCH)
        f.attrs["n_feasible_draws"] = int(sample_info["n_draws"])
        f.attrs["acceptance_rate"] = float(sample_info["acceptance_rate"])
    print(f"\n=== Saved {out_path}  units={units.shape} ===", flush=True)

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

    formulation = scfg.EPS_FORMULATION

    spec = config.SEARCH_ENSEMBLE_SPEC
    if spec is None:
        sys.exit(
            "[epsilon_run] SEARCH_ENSEMBLE_SPEC is None — the design "
            f"'{config.active_scenario_name()}' has no staged search ensemble. "
            "Run workflow steps 02-04 for this design first "
            "(NYCOPT_SCENARIO_DESIGN in the sourced env file)."
        )
    if getattr(spec, "resample_per_eval", False):
        sys.exit(
            "[epsilon_run] the active design resamples its ensemble per "
            "evaluation — every policy would see different scenarios, "
            "confounding the epsilon estimate. Calibrate on the fixed "
            "campaign designs instead."
        )
    design = config.active_scenario_name()

    # The FULL annual-unit registry (the 8 active objectives + the flood
    # diagnostics: the retired day count and its P99 variant) so
    # epsilon/operator decisions are all informed by the same cube; the
    # archive sweep in the figures script uses the active subset only.
    objs = list(ENSEMBLE_OBJECTIVES.values())
    obj_names = list(ENSEMBLE_OBJECTIVES.keys())
    directions = [o.direction for o in objs]
    eps_current = [float(o.epsilon) for o in objs]
    n_units = _expected_n_units(spec)

    # Feasible-DV sample: each rank regenerates it from the seed (deterministic;
    # avoids comm.bcast). Rejection is pure DV arithmetic — no simulation.
    t_s = time.perf_counter()
    samples, sample_info = sample_feasible_dvs(
        formulation, scfg.EPS_SEED, scfg.EPS_N_POLICIES,
        max_draws=scfg.EPS_MAX_DRAWS)
    t_sample = time.perf_counter() - t_s

    baseline_dv = np.asarray(get_baseline_values(formulation), dtype=float)
    sample_ids = np.array([-1] + list(range(scfg.EPS_N_POLICIES)), dtype=int)
    all_dvs = np.vstack([baseline_dv[None, :], samples])

    if is_root:
        print("=== Epsilon-calibration evaluation stage ===", flush=True)
        print(f"  design:          {design} ({spec.preset_name})", flush=True)
        print(f"  ensemble:        {spec.inflow_type} "
              f"(N={spec.n_realizations}, {spec.realization_years or 'full'}-yr, "
              f"{n_units} units/realization)", flush=True)
        print(f"  formulation:     {formulation}", flush=True)
        print(f"  n policies:      {scfg.EPS_N_POLICIES} (+ baseline)", flush=True)
        print(f"  feasible sample: {sample_info['n_draws']} draws, "
              f"acceptance {sample_info['acceptance_rate']:.3%} "
              f"({t_sample:.1f}s)", flush=True)
        print(f"  objectives:      {len(obj_names)} annual-unit -> {obj_names}",
              flush=True)
        print(f"  realiz. batch:   {scfg.EPS_REALIZATION_BATCH}", flush=True)
        print(f"  ranks:           {size}", flush=True)
        print(flush=True)

    rank_slots = assign_rank_slots(len(sample_ids), rank, size)

    partial_dir = scfg.EPS_CUBE_DIR / f"_partial_{scfg._eps_stem(design)}"
    if is_root:
        scfg.EPS_CUBE_DIR.mkdir(parents=True, exist_ok=True)
        # A killed prior job leaves stale shards/markers here; await_all_done
        # would return on the stale markers and the combine would merge stale
        # shards into the cube. Safe to clear unbarriered: this run's workers
        # write shards only after their first evaluation completes.
        if partial_dir.exists():
            for stale in partial_dir.glob("rank_*"):
                stale.unlink()
    prepare_partial_dir(partial_dir, rank)

    n_real = int(spec.n_realizations) if spec.is_ensemble else 1
    n_obj = len(objs)
    local_ids = np.array([int(sample_ids[s]) for s in rank_slots], dtype=int)
    local_dvs = (np.array([all_dvs[s] for s in rank_slots], dtype=float)
                 if rank_slots else np.empty((0, all_dvs.shape[1])))
    local_units = np.full((len(rank_slots), n_real, n_obj, n_units), np.nan)
    local_secs = np.full(len(rank_slots), np.nan)

    t0 = time.time()
    for i, slot in enumerate(rank_slots):
        sid = int(sample_ids[slot])
        ts = time.perf_counter()
        try:
            local_units[i] = _eval_dv(all_dvs[slot], formulation, spec, objs,
                                      n_units, scfg.EPS_REALIZATION_BATCH)
            local_secs[i] = time.perf_counter() - ts
            n_nan = int(np.isnan(local_units[i]).all(axis=(1, 2)).sum())
            print(f"  [rank {rank:>2} ok] sid={sid:4d}  "
                  f"({local_secs[i]:.1f}s, {n_nan} empty realiz.)", flush=True)
        except Exception:
            local_secs[i] = time.perf_counter() - ts
            tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
            print(f"  [rank {rank:>2} FAIL] sid={sid:4d}  {tb}", flush=True)
    print(f"  [rank {rank:>2}] done {len(rank_slots)} policies in "
          f"{time.time() - t0:.1f}s", flush=True)

    np.savez(partial_dir / f"rank_{rank:03d}.npz",
             sample_ids=local_ids, dvs=local_dvs, units=local_units,
             eval_seconds=local_secs)
    mark_rank_done(partial_dir, rank)

    if not is_root:
        return

    if not await_all_done(partial_dir, size, deadline_s=3600.0):
        missing = {f"rank_{r:03d}.done" for r in range(size)} - {
            p.name for p in partial_dir.glob("rank_*.done")}
        print(f"[epsilon_run] WARN: timeout waiting for {missing}", flush=True)

    _combine_and_write(
        partial_dir, scfg.epsilon_cube_path(design),
        design=design, spec=spec, formulation=formulation,
        obj_names=obj_names, directions=directions, eps_current=eps_current,
        sample_info=sample_info)


if __name__ == "__main__":
    main()
