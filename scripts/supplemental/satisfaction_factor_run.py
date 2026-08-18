"""satisfaction_factor_run.py - Weekly satisfaction-factor sweep (evaluation).

Re-evaluates the epsilon-calibration feasible-policy population (identical
seed/count, so cube rows align across the two experiments) on the ACTIVE
scenario design's search ensemble and stores, for the two delivery objectives
(NYC, NJ), the per-unit failing-week counts AND the §1 weekly reliability at
each candidate weekly satisfaction factor in ``SF_FACTOR_GRID``. The factor
sits inside the weekly reduction (``src/objectives.py::_weekly_delivery_ok``),
upstream of the epsilon cubes' stored counts, so bounding its influence needs
this one extra simulation pass; the factor axis itself is computed from each
realization's weekly sums at no additional simulation cost.
See ``docs/notes/methods/framing_convention_diagnostics.md`` diagnostic 2.

Design selection mirrors the epsilon calibration: the scenario design comes
from the environment (``NYCOPT_SCENARIO_DESIGN``, sourced from
``NYCOPT_ENV_FILE`` by the SLURM wrapper) — one job per campaign design. All
other settings live in ``supplemental_config.py`` (SF_* section) — no CLI
value flags.

MPI: each rank takes an ``array_split`` slice of the policies, writes an
``.npz`` shard + ``.done`` marker; rank 0 combines shards into the cube HDF5
via the filesystem barrier and deletes them.

Usage (local smoke, historic design, serial):
    NYCOPT_SF_SMOKE=1 python scripts/supplemental/satisfaction_factor_run.py
Usage (SLURM, one job per design):
    sbatch --export=ALL,NYCOPT_ENV_FILE=workflow/envs/eps_calib_fixprob.env \\
        workflow/supplemental/satisfaction_factor.sh
"""

from __future__ import annotations

import os
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

scfg.configure_epsilon_env()  # same env contract as the epsilon calibration
if scfg.SF_SMOKE:
    os.environ.setdefault("NYCOPT_SCENARIO_DESIGN", "historic")

import config  # noqa: E402
from src.formulations import get_baseline_values  # noqa: E402
from src.objectives import _delivery_entitlement, _metric_window  # noqa: E402
from src.objectives_ensemble import ffmp_year_unit_slices  # noqa: E402
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

#: Entitlement caps resolved once from config, aligned with
#: ``SF_DELIVERY_OBJECTIVES`` row order.
_CAPS: dict = {
    "demand_nyc": config.NYC_DECREE_DIVERSION_CAP_MGD,
    "demand_nj": config.NJ_DELIVERY_CAP_MGD,
}


def _expected_n_units(spec) -> int:
    """Metric-bearing water-year units per realization (cube's last axis)."""
    if spec.is_ensemble:
        return int(spec.realization_years) - 1
    idx = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    return len(ffmp_year_unit_slices(idx))


def _factor_block(data: dict, n_units: int) -> "tuple[np.ndarray, np.ndarray]":
    """Factor-swept delivery metrics for ONE realization.

    For each delivery objective and factor: the §1 weekly reliability over the
    metric window, and the per-unit-year failing-week count (weekly bins formed
    within each unit slice, matching ``_delivery_failure_weeks_annual``). Weekly
    sums are computed once per series; only the comparison repeats per factor.
    A non-finite weekly comparison counts as a failing week.

    Returns:
        ``(counts, weekly_rel)`` with shapes
        ``(n_factor, n_obj, n_units)`` and ``(n_factor, n_obj)``.
    """
    factors = scfg.SF_FACTOR_GRID
    n_obj = len(scfg.SF_DELIVERY_OBJECTIVES)
    counts = np.full((len(factors), n_obj, n_units), np.nan)
    weekly_rel = np.full((len(factors), n_obj), np.nan)
    for j, (_, dkey, lkey, reset) in enumerate(scfg.SF_DELIVERY_OBJECTIVES):
        demand = data["ibt_demands"][dkey]
        delivery = data["ibt_diversions"][lkey]
        target = _delivery_entitlement(demand, delivery, _CAPS[dkey], reset)

        wt = _metric_window(target).resample("W").sum()
        wd = _metric_window(delivery).resample("W").sum()
        for fi, f in enumerate(factors):
            ok = (wd >= f * wt)          # NaN comparison -> False (failure)
            weekly_rel[fi, j] = (float(ok.sum()) / len(ok) if len(ok) else 0.0)

        slices = ffmp_year_unit_slices(demand.index)
        if len(slices) != n_units:
            raise ValueError(
                f"unit count {len(slices)} != expected {n_units} for '{dkey}'")
        for u, sl in enumerate(slices):
            uwt = target.iloc[sl].resample("W").sum()
            uwd = delivery.iloc[sl].resample("W").sum()
            for fi, f in enumerate(factors):
                counts[fi, j, u] = float((~(uwd >= f * uwt)).sum())
    return counts, weekly_rel


def _eval_dv(dv: np.ndarray, formulation: str, spec,
             n_units: int, batch_size: int) -> "tuple[np.ndarray, np.ndarray]":
    """Factor cube slices for one policy over the full ensemble.

    Returns:
        ``(counts, weekly_rel)`` with shapes
        ``(n_real, n_factor, n_obj, n_units)`` and ``(n_real, n_factor, n_obj)``.
    """
    n_factor = len(scfg.SF_FACTOR_GRID)
    n_obj = len(scfg.SF_DELIVERY_OBJECTIVES)
    cfg = dvs_to_config(dv, formulation)

    def per_real(data) -> np.ndarray:
        counts, rel = _factor_block(data, n_units)
        # Stack into one array so the shared batched runner returns one object:
        # slot [..., -1] carries weekly_rel (broadcast over the unit axis).
        out = np.concatenate([counts, rel[:, :, None]], axis=2)
        return out                                  # (n_factor, n_obj, n_units+1)

    if not spec.is_ensemble:
        rows = per_real(run_simulation_inmemory(cfg, use_trimmed=True))[None]
    else:
        rows = run_simulation_ensemble_batched(
            cfg, spec, batch_size, per_real,
            skip_failed_batches=True,
            failed_value=np.full((n_factor, n_obj, n_units + 1), np.nan),
        )
        rows = np.asarray(rows, dtype=float)
    return rows[:, :, :, :n_units], rows[:, :, :, n_units]


def _combine_and_write(partial_dir: Path, out_path: Path, *, design: str,
                       spec, formulation: str, sample_info: dict) -> None:
    """Rank-0: merge per-rank ``.npz`` shards into the cube HDF5."""
    shards = sorted(partial_dir.glob("rank_*.npz"))
    if not shards:
        sys.exit("[sf_run] no shards found — all ranks failed?")

    sample_ids, dvs, counts, rels, secs = [], [], [], [], []
    for sh in shards:
        with np.load(sh) as z:
            sample_ids.append(z["sample_ids"])
            dvs.append(z["dvs"])
            counts.append(z["counts"])
            rels.append(z["weekly_rel"])
            secs.append(z["eval_seconds"])
    sample_ids = np.concatenate(sample_ids)
    order = np.argsort(sample_ids, kind="stable")
    sample_ids = sample_ids[order]
    dvs = np.concatenate(dvs, axis=0)[order]
    counts = np.concatenate(counts, axis=0)[order]
    rels = np.concatenate(rels, axis=0)[order]
    secs = np.concatenate(secs)[order]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    obj_names = [row[0] for row in scfg.SF_DELIVERY_OBJECTIVES]
    with h5py.File(out_path, "w") as f:
        # (n_dv, n_real, n_factor, n_obj, n_units) / (n_dv, n_real, n_factor, n_obj)
        f.create_dataset("failing_week_counts", data=counts)
        f.create_dataset("weekly_reliability", data=rels)
        f.create_dataset("dv_vectors", data=dvs)
        f.create_dataset("sample_ids", data=sample_ids)   # baseline = -1
        f.create_dataset("eval_seconds", data=secs)
        f.create_dataset("factors",
                         data=np.asarray(scfg.SF_FACTOR_GRID, dtype=float))
        f.create_dataset("objective_names",
                         data=np.array(obj_names, dtype=object), dtype=str_dt)
        f.attrs["design"] = design
        f.attrs["preset_name"] = spec.preset_name
        f.attrs["inflow_type"] = spec.inflow_type
        f.attrs["is_ensemble"] = bool(spec.is_ensemble)
        f.attrs["n_realizations"] = int(spec.n_realizations)
        f.attrs["realization_years"] = int(spec.realization_years or 0)
        f.attrs["formulation"] = formulation
        f.attrs["dv_seed"] = int(scfg.EPS_SEED)
        f.attrs["realization_batch"] = int(scfg.SF_REALIZATION_BATCH)
        f.attrs["n_feasible_draws"] = int(sample_info["n_draws"])
        f.attrs["acceptance_rate"] = float(sample_info["acceptance_rate"])
    print(f"\n=== Saved {out_path}  counts={counts.shape} ===", flush=True)

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
            "[sf_run] SEARCH_ENSEMBLE_SPEC is None — the design "
            f"'{config.active_scenario_name()}' has no staged search ensemble."
        )
    if getattr(spec, "resample_per_eval", False):
        sys.exit("[sf_run] the active design resamples per evaluation; sweep "
                 "the fixed campaign designs instead.")
    design = config.active_scenario_name()
    n_units = _expected_n_units(spec)

    samples, sample_info = sample_feasible_dvs(
        formulation, scfg.EPS_SEED, scfg.SF_N_POLICIES,
        max_draws=scfg.EPS_MAX_DRAWS)
    baseline_dv = np.asarray(get_baseline_values(formulation), dtype=float)
    sample_ids = np.array([-1] + list(range(scfg.SF_N_POLICIES)), dtype=int)
    all_dvs = np.vstack([baseline_dv[None, :], samples])

    if is_root:
        print("=== Satisfaction-factor sweep (evaluation stage) ===", flush=True)
        print(f"  design:      {design} ({spec.preset_name})", flush=True)
        print(f"  factors:     {scfg.SF_FACTOR_GRID}", flush=True)
        print(f"  n policies:  {scfg.SF_N_POLICIES} (+ baseline)"
              f"{'  [SMOKE]' if scfg.SF_SMOKE else ''}", flush=True)
        print(f"  units/real:  {n_units}; ranks: {size}", flush=True)

    rank_slots = assign_rank_slots(len(sample_ids), rank, size)
    partial_dir = scfg.SF_CUBE_DIR / f"_partial_{scfg._sf_stem(design)}"
    if is_root:
        scfg.SF_CUBE_DIR.mkdir(parents=True, exist_ok=True)
        if partial_dir.exists():
            for stale in partial_dir.glob("rank_*"):
                stale.unlink()
    prepare_partial_dir(partial_dir, rank)

    n_real = int(spec.n_realizations) if spec.is_ensemble else 1
    n_factor = len(scfg.SF_FACTOR_GRID)
    n_obj = len(scfg.SF_DELIVERY_OBJECTIVES)
    local_ids = np.array([int(sample_ids[s]) for s in rank_slots], dtype=int)
    local_dvs = (np.array([all_dvs[s] for s in rank_slots], dtype=float)
                 if rank_slots else np.empty((0, all_dvs.shape[1])))
    local_counts = np.full((len(rank_slots), n_real, n_factor, n_obj, n_units),
                           np.nan)
    local_rels = np.full((len(rank_slots), n_real, n_factor, n_obj), np.nan)
    local_secs = np.full(len(rank_slots), np.nan)

    for i, slot in enumerate(rank_slots):
        sid = int(sample_ids[slot])
        ts = time.perf_counter()
        try:
            counts, rel = _eval_dv(all_dvs[slot], formulation, spec,
                                   n_units, scfg.SF_REALIZATION_BATCH)
            local_counts[i], local_rels[i] = counts, rel
        except Exception:
            print(f"[rank {rank}] policy {sid} FAILED:\n"
                  f"{traceback.format_exc()}", flush=True)
        local_secs[i] = time.perf_counter() - ts
        print(f"[rank {rank}] policy {sid} done "
              f"({local_secs[i]:.1f}s, {i + 1}/{len(rank_slots)})", flush=True)

    np.savez_compressed(
        partial_dir / f"rank_{rank:04d}.npz",
        sample_ids=local_ids, dvs=local_dvs, counts=local_counts,
        weekly_rel=local_rels, eval_seconds=local_secs)
    mark_rank_done(partial_dir, rank)

    if is_root:
        await_all_done(partial_dir, size)
        _combine_and_write(
            partial_dir, scfg.sf_cube_path(design), design=design, spec=spec,
            formulation=formulation, sample_info=sample_info)


if __name__ == "__main__":
    main()
