"""ensemble_size_library_run.py - Layer B library build for the ensemble-size diagnostics.

Builds and persists the per-realization annual-unit LIBRARY of
``docs/notes/methods/ensemble_size_diagnostics.md`` §2: every policy of the
fixed set (``tables/policies.csv``) evaluated once on every unique realization
the Layer-B replicates need, with the stage-(i) annual metrics kept per
(realization, unit-year) so any (design, N, replicate) ensemble's objective
vector is composed offline.

Stages, selected by ``NYCOPT_ESD_STAGE`` (set by the wrappers):

``materialize``  regenerates chunk ``NYCOPT_ESD_CHUNK`` of the library plan from
    the stream-only candidate pool (``src.ensembles.materialize_subset``) into
    ``{library_slug}__chunkJJJ`` and writes the parent metadata.
``requalify``    recomputes the QC from the merged library.
``evaluate``     MPI task farm over (policy, staged ensemble, realization block)
    through ``src.simulation.evaluate_annual_units``; rank 0 merges the shards
    into ``library/unit_library_*.h5`` and runs the QC (``_library_checks``:
    composition reproduces the stored scalars; staged production members agree
    with the regenerated library rows at LP-jitter tolerance).

Settings in ``supplemental_config.py`` (``ESD_*``); no CLI value flags.
Wrappers: ``workflow/supplemental/ensemble_size_library_stage.sh`` and
``workflow/supplemental/ensemble_size_library_eval.sh``.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_esd_env()

import config  # noqa: E402
from src.ensembles import (  # noqa: E402
    get_ensemble_spec, materialize_subset, staged_ensemble_dir, with_indices_override,
)
from src.objectives_ensemble import ENSEMBLE_OBJECTIVES  # noqa: E402
from src.sensitivity_common import (  # noqa: E402
    assign_rank_slots, await_all_done, get_mpi_context, mark_rank_done,
    prepare_partial_dir,
)
from src.simulation import evaluate_annual_units  # noqa: E402

#: End-to-end (staged-vs-regenerated) tolerance, in the study's own currency:
#: every policy's COMPOSED objective on the common members must agree to this
#: fraction of the objective's epsilon (pywrdrb's LP solver carries run-to-run
#: jitter at the unit level). Unit-level differences are reported alongside,
#: never gated.
END_TO_END_EPS_FRAC: float = 0.01


###############################################################################
# Plan / policy loading
###############################################################################

def load_plan() -> dict:
    path = scfg.esd_json_path("library_plan")
    if not path.exists():
        sys.exit(f"[esd:lib] library plan missing: {path} — run ensemble_size_hazard.py first")
    return json.loads(path.read_text())


def load_policies() -> tuple[np.ndarray, list[str], list[str]]:
    """``(dvs, policy_ids, labels)`` from the persisted policy table."""
    path = scfg.esd_table_path("policies")
    if not path.exists():
        sys.exit(f"[esd:lib] policy table missing: {path} — run ensemble_size_hazard.py first")
    table = pd.read_csv(path)
    dv_cols = sorted(c for c in table.columns if c.startswith("dv"))
    return (table[dv_cols].to_numpy(dtype=float), list(table["policy_id"]),
            list(table["label"]))


###############################################################################
# Stage: materialize one chunk
###############################################################################

def _link_to_staging_root(slug: str) -> None:
    """Create the chunk dir on the staging root and symlink it into the staged tree."""
    root = scfg.ESD_STAGING_ROOT
    target = config.STAGED_ENSEMBLE_DIR / slug
    if root is None or not Path(root).exists() or target.exists():
        return
    real = Path(root) / slug
    real.mkdir(parents=True, exist_ok=True)
    target.symlink_to(real)


def _write_parent(plan: dict) -> None:
    """Parent ``_meta.json`` + ``chunk_index.json`` so the library resolves as a chunked pool."""
    parent = config.STAGED_ENSEMBLE_DIR / plan["library_slug"]
    parent.mkdir(parents=True, exist_ok=True)
    pool_meta = json.loads((staged_ensemble_dir(plan["pool_slug"]) / "_meta.json").read_text())
    entries, start = [], 0
    for c in plan["chunks"]:
        entries.append({"chunk_index": c["chunk_index"], "slug": c["slug"],
                        "global_start": start, "global_end": start + c["n_realizations"],
                        "n_realizations": c["n_realizations"]})
        start += c["n_realizations"]
    (parent / "chunk_index.json").write_text(json.dumps({
        "pool_slug": plan["library_slug"], "source_pool": plan["pool_slug"],
        "n_realizations": start, "chunk_size": plan["chunk_size"],
        "n_chunks": len(entries), "chunks": entries}, indent=2))
    (parent / "_meta.json").write_text(json.dumps({
        "slug": plan["library_slug"], "kind": "esd_library",
        "n_realizations": start, "realization_years": pool_meta["realization_years"],
        "n_years": pool_meta["realization_years"], "source_pool": plan["pool_slug"],
        "population": pool_meta.get("population"), "generator": pool_meta.get("generator"),
        "seed_domain": pool_meta.get("seed_domain"), "root_seed": pool_meta.get("root_seed"),
        "start_date": pool_meta.get("start_date"), "store_daily": True,
        "chunk_size": plan["chunk_size"], "n_chunks": len(entries),
        "note": "ensemble-size diagnostics library: regenerated pool members, chunked",
    }, indent=2))


def stage_materialize(plan: dict) -> None:
    """Regenerate one chunk (``NYCOPT_ESD_CHUNK``) of the library plan."""
    j = int(os.environ["NYCOPT_ESD_CHUNK"])
    chunk = plan["chunks"][j]
    slug = chunk["slug"]
    _write_parent(plan)
    _link_to_staging_root(slug)
    meta_path = staged_ensemble_dir(slug) / "_meta.json"
    if meta_path.exists() and os.environ.get("NYCOPT_ESD_FORCE") != "1":
        meta = json.loads(meta_path.read_text())
        if meta.get("global_realization_ids") == chunk["global_ids"]:
            print(f"[esd:materialize] chunk {j} ({slug}) already staged; skipping")
            return
    t0 = time.time()
    materialize_subset(
        plan["pool_slug"], chunk["global_ids"], slug,
        extra_meta={"kind": "esd_library_chunk", "library_slug": plan["library_slug"],
                    "chunk_index": j, "pool_draw": plan["pool_draw"]},
    )
    print(f"[esd:materialize] chunk {j}: {chunk['n_realizations']} realizations -> "
          f"{slug} in {time.time() - t0:.0f}s")


###############################################################################
# Stage: evaluate (MPI task farm)
###############################################################################

def _sources(plan: dict) -> list[dict]:
    """Every staged ensemble the library reads: library chunks + production draws."""
    out = []
    for c in plan["chunks"]:
        out.append({"key": c["slug"], "slug": c["slug"], "kind": "pool",
                    "design": None, "draw": plan["pool_draw"],
                    "global_ids": [int(g) for g in c["global_ids"]]})
    for s in plan["staged_ensembles"]:
        gids = s.get("global_ids")
        out.append({"key": s["slug"], "slug": s["slug"], "kind": "staged",
                    "design": s["design"], "draw": s["draw"],
                    "global_ids": ([int(g) for g in gids] if gids
                                   else list(range(s["n_realizations"])))})
    return out


def build_tasks(plan: dict, n_policies: int) -> list[dict]:
    """The deterministic task list: (policy, source, block of local indices)."""
    tasks = []
    for src_i, src in enumerate(_sources(plan)):
        n = len(src["global_ids"])
        for b0 in range(0, n, scfg.ESD_EVAL_BLOCK):
            local = list(range(b0, min(n, b0 + scfg.ESD_EVAL_BLOCK)))
            for p in range(n_policies):
                tasks.append({"task_id": len(tasks), "policy": p, "source": src_i,
                              "local": local})
    return tasks


def _check_staged(plan: dict) -> None:
    missing = []
    for src in _sources(plan):
        d = staged_ensemble_dir(src["slug"])
        for f in ("catchment_inflow_mgd.hdf5", "catchment_inflow_with_flood_nodes_mgd.hdf5",
                  "presimulated_releases_mgd.hdf5", "predicted_inflows_mgd.hdf5"):
            if not (d / f).exists():
                missing.append(str(d / f))
    if missing:
        sys.exit("[esd:evaluate] unstaged inputs:\n  " + "\n  ".join(missing[:12]))


def _eval_task(task: dict, dvs: np.ndarray, sources: list[dict], objs: list,
               active_idx: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """One task: the ``(R, M, U)`` tensor and the active objectives' Borg scalars."""
    src = sources[task["source"]]
    # A UNIQUE preset_name per block: src.simulation caches the built model
    # dict by preset name (+ DU signature) and the cached dict carries the
    # block's inflow_ensemble_indices, so two blocks sharing a name would
    # silently re-simulate the first block's realizations (the same reason
    # run_simulation_ensemble_batched names its batches ``__b{offset}``).
    # inflow_type (the staged dir) is unchanged.
    spec = replace(
        with_indices_override(get_ensemble_spec(src["slug"]), task["local"]),
        preset_name=f"{src['slug']}__blk{task['local'][0]}",
    )
    units, names = evaluate_annual_units(
        dvs[task["policy"]], scfg.ESD_FORMULATION, objective_set=objs,
        ensemble_spec=spec, realization_batch=0,
    )
    if names != [o.name for o in objs]:
        raise RuntimeError(f"objective order drift: {names}")
    scalars = np.array([
        objs[k].compute_for_borg_from_units(units[:, k, :].reshape(-1)) for k in active_idx
    ])
    return units, scalars


def _library_checks(units: np.ndarray, sources: list[dict], row_of: dict, policy_ids: list,
                    objs: list, active_idx: list[int], plan: dict) -> tuple[list, dict]:
    """The two library-level QC checks (also re-runnable from the merged file).

    (ii) No two blocks of one (policy, source) may hold identical unit tensors —
    the signature of the model-dict cache reusing another block's inflow
    indices. (iii) The staged production ``hazfill`` draw of the library pool,
    simulated from its staged files, must agree with the regenerated library
    rows of the same pool members: composed objectives within
    ``END_TO_END_EPS_FRAC`` x epsilon for every policy (gated), unit-level
    differences reported.

    Returns:
        ``(duplicate_blocks, end_to_end)`` — a list of ``(source, policy,
        block_start)`` triples and the end-to-end result dict.
    """
    dup_blocks = []
    for si, src in enumerate(sources):
        n = len(src["global_ids"])
        blocks = [list(range(b0, min(n, b0 + scfg.ESD_EVAL_BLOCK)))
                  for b0 in range(0, n, scfg.ESD_EVAL_BLOCK)]
        for p in range(len(policy_ids)):
            seen = []
            for bl in blocks:
                if len(bl) < 2:
                    continue
                arr = units[p, [row_of[(si, li)] for li in bl], :, :]
                if np.isnan(arr).all():
                    continue
                for prev in seen:
                    if prev.shape == arr.shape and np.array_equal(prev, arr, equal_nan=True):
                        dup_blocks.append((src["key"], p, bl[0]))
                        break
                seen.append(arr)

    e2e: dict = {"checked": False}
    pool_rows = {g: row_of[(si, li)] for si, src in enumerate(sources) if src["kind"] == "pool"
                 for li, g in enumerate(src["global_ids"])}
    for si, src in enumerate(sources):
        if src["kind"] != "staged" or src["design"] != "hazard_filling_stationary":
            continue
        if src["draw"] != plan["pool_draw"]:
            continue
        pairs = [(row_of[(si, li)], pool_rows[g]) for li, g in enumerate(src["global_ids"])
                 if g in pool_rows]
        if not pairs:
            continue
        a = units[:, [p[0] for p in pairs], :, :]
        b = units[:, [p[1] for p in pairs], :, :]
        d = np.abs(a - b)
        per_obj = {}
        worst_frac = 0.0
        for k in active_idx:
            op, eps = objs[k].unit_operator, objs[k].epsilon
            va = np.array([float(op(a[p, :, k, :].reshape(-1))) for p in range(a.shape[0])])
            vb = np.array([float(op(b[p, :, k, :].reshape(-1))) for p in range(b.shape[0])])
            frac = float(np.nanmax(np.abs(va - vb)) / eps)
            worst_frac = max(worst_frac, frac)
            per_obj[objs[k].name] = {"max_abs_composed_diff": float(np.nanmax(np.abs(va - vb))),
                                     "eps": eps, "diff_over_eps": frac,
                                     "n_units_differing": int((d[:, :, k, :] > 1e-9).sum()),
                                     "max_abs_unit_diff": float(np.nanmax(d[:, :, k, :]))}
        e2e = {"checked": True, "staged_slug": src["slug"], "n_pairs": len(pairs),
               "n_units_compared": int(d.size),
               "n_units_differing": int((d > 1e-9).sum()),
               "n_realizations_differing": int((np.nanmax(d, axis=(0, 2, 3)) > 1e-9).sum()),
               "worst_composed_diff_over_eps": worst_frac,
               "eps_frac_tolerance": END_TO_END_EPS_FRAC,
               "within_tolerance": bool(worst_frac <= END_TO_END_EPS_FRAC),
               "per_objective": per_obj}
    return dup_blocks, e2e


def stage_requalify(plan: dict) -> None:
    """Recompute QC (ii)/(iii) from the merged library and rewrite ``library_qc.json``.

    The composition check (i) needs the per-task shards and is carried over
    from the existing QC file unchanged.
    """
    path = scfg.esd_library_path()
    qc_path = scfg.esd_json_path("library_qc")
    old = json.loads(qc_path.read_text()) if qc_path.exists() else {}
    with h5py.File(path, "r") as f:
        units = f["units"][:]
        policy_ids = [s.decode() for s in f["policy_ids"][:]]
    objs = list(ENSEMBLE_OBJECTIVES.values())
    active = list(config.get_objective_set().names)
    active_idx = [[o.name for o in objs].index(a) for a in active]
    sources = _sources(plan)
    row_of, r = {}, 0
    for si, src in enumerate(sources):
        for li in range(len(src["global_ids"])):
            row_of[(si, li)] = r
            r += 1
    if r != units.shape[1]:
        sys.exit(f"[esd:requalify] library rows {units.shape[1]} != plan rows {r}")
    dup_blocks, e2e = _library_checks(units, sources, row_of, policy_ids, objs, active_idx, plan)
    failed = old.get("n_failed_tasks", 0)
    comp_ok = bool(old.get("composition_exact", False))
    qc = dict(old)
    qc.update({"duplicate_blocks": dup_blocks[:20], "n_duplicate_blocks": len(dup_blocks),
               "duplicates_ok": not dup_blocks, "end_to_end": e2e,
               "library_valid": bool(comp_ok and not dup_blocks and not failed
                                     and (not e2e["checked"] or e2e["within_tolerance"])),
               "requalified": True})
    qc_path.write_text(json.dumps(qc, indent=2))
    print(f"[esd:requalify] duplicates={len(dup_blocks)} end_to_end={e2e.get('within_tolerance')} "
          f"(worst composed diff {e2e.get('worst_composed_diff_over_eps', float('nan')):.2e} eps; "
          f"{e2e.get('n_units_differing')} / {e2e.get('n_units_compared')} units differ) "
          f"-> library_valid={qc['library_valid']}")


def _merge(partial_dir: Path, plan: dict, tasks: list[dict], dvs: np.ndarray,
           policy_ids: list[str], labels: list[str], objs: list, active_idx: list[int],
           n_ranks: int) -> Path:
    """Rank 0: merge shards into the library HDF5 and run the QC."""
    sources = _sources(plan)
    shards = sorted(partial_dir.glob("rank_*.npz"))
    if not shards:
        sys.exit("[esd:evaluate] no shards found — every rank failed?")
    done: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    for sh in shards:
        with np.load(sh, allow_pickle=True) as z:
            for tid, u, s, sec in zip(z["task_ids"], z["units"], z["scalars"], z["seconds"]):
                done[int(tid)] = (np.asarray(u, dtype=float), np.asarray(s, dtype=float), float(sec))
    failed = [t["task_id"] for t in tasks if t["task_id"] not in done]

    # Row layout: one row per (source, local index), sources in plan order.
    row_of: dict[tuple[int, int], int] = {}
    real_source, real_kind, real_design, real_draw, real_gid, real_local = [], [], [], [], [], []
    for si, src in enumerate(sources):
        for li, g in enumerate(src["global_ids"]):
            row_of[(si, li)] = len(real_source)
            real_source.append(src["key"])
            real_kind.append(src["kind"])
            real_design.append(src["design"] or "")
            real_draw.append(int(src["draw"]))
            real_gid.append(int(g))
            real_local.append(int(li))
    n_real = len(real_source)
    first = next(iter(done.values()))[0]
    n_obj, n_unit = first.shape[1], first.shape[2]
    units = np.full((len(policy_ids), n_real, n_obj, n_unit), np.nan, dtype=float)
    seconds = np.full(len(tasks), np.nan)
    for t in tasks:
        if t["task_id"] not in done:
            continue
        u, _, sec = done[t["task_id"]]
        rows = [row_of[(t["source"], li)] for li in t["local"]]
        units[t["policy"], rows, :, :] = u
        seconds[t["task_id"]] = sec

    # QC (i): composition path == driver path, exactly, per task.
    max_dev = 0.0
    for t in tasks:
        if t["task_id"] not in done:
            continue
        rows = [row_of[(t["source"], li)] for li in t["local"]]
        pooled = units[t["policy"], rows, :, :]
        again = np.array([objs[k].compute_for_borg_from_units(pooled[:, k, :].reshape(-1))
                          for k in active_idx])
        stored = done[t["task_id"]][1]
        dev = np.nanmax(np.abs(again - stored) / np.maximum(np.abs(stored), 1e-12))
        max_dev = max(max_dev, float(dev))
    composition_ok = max_dev == 0.0

    # QC (ii) duplicate blocks and (iii) end-to-end agreement.
    dup_blocks, e2e = _library_checks(units, sources, row_of, policy_ids, objs, active_idx, plan)
    duplicates_ok = not dup_blocks

    out = scfg.esd_library_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    active_names = [objs[k].name for k in active_idx]
    with h5py.File(out, "w") as f:
        f.create_dataset("units", data=units, compression="gzip", compression_opts=4)
        f.create_dataset("dv_vectors", data=dvs)
        f.create_dataset("policy_ids", data=np.array(policy_ids, dtype=object), dtype=str_dt)
        f.create_dataset("policy_labels", data=np.array(labels, dtype=object), dtype=str_dt)
        f.create_dataset("objective_names", data=np.array([o.name for o in objs], dtype=object),
                         dtype=str_dt)
        f.create_dataset("directions", data=np.array([o.direction for o in objs], dtype=object),
                         dtype=str_dt)
        f.create_dataset("epsilons", data=np.array([o.epsilon for o in objs], dtype=float))
        f.create_dataset("active_objective_names", data=np.array(active_names, dtype=object),
                         dtype=str_dt)
        f.create_dataset("real_source", data=np.array(real_source, dtype=object), dtype=str_dt)
        f.create_dataset("real_kind", data=np.array(real_kind, dtype=object), dtype=str_dt)
        f.create_dataset("real_design", data=np.array(real_design, dtype=object), dtype=str_dt)
        f.create_dataset("real_draw", data=np.array(real_draw, dtype=int))
        f.create_dataset("real_global_id", data=np.array(real_gid, dtype=int))
        f.create_dataset("real_local_id", data=np.array(real_local, dtype=int))
        f.create_dataset("task_seconds", data=seconds)
        f.attrs["library_slug"] = plan["library_slug"]
        f.attrs["pool_slug"] = plan["pool_slug"]
        f.attrs["p_ref"] = int(plan["p_ref"])
        f.attrs["formulation"] = scfg.ESD_FORMULATION
        f.attrs["eval_block"] = int(scfg.ESD_EVAL_BLOCK)
        f.attrs["n_ranks"] = int(n_ranks)
        f.attrs["n_tasks"] = int(len(tasks))
        f.attrs["n_failed_tasks"] = int(len(failed))
        f.attrs["smoke"] = bool(scfg.ESD_SMOKE)

    qc = {
        "n_policies": len(policy_ids), "n_realizations": n_real, "n_objectives": n_obj,
        "n_units": n_unit, "n_tasks": len(tasks), "n_failed_tasks": len(failed),
        "failed_task_ids": failed[:50],
        "composition_max_rel_dev": max_dev, "composition_exact": composition_ok,
        "duplicate_blocks": dup_blocks[:20], "n_duplicate_blocks": len(dup_blocks),
        "duplicates_ok": duplicates_ok,
        "end_to_end": e2e,
        "library_valid": bool(composition_ok and duplicates_ok and not failed
                              and (not e2e["checked"] or e2e["within_tolerance"])),
        "task_seconds_median": float(np.nanmedian(seconds)),
        "task_seconds_p90": float(np.nanpercentile(seconds, 90)),
        "core_hours_total": float(np.nansum(seconds) / 3600.0),
        "realization_policies": int(sum(len(t["local"]) for t in tasks)),
        "n_ranks": n_ranks,
    }
    scfg.esd_json_path("library_qc").write_text(json.dumps(qc, indent=2))
    print(f"\n=== Saved {out}  units={units.shape}  failed={len(failed)}  "
          f"composition max rel dev={max_dev:.2e} ({'OK' if composition_ok else 'MISMATCH'})  "
          f"duplicate blocks={len(dup_blocks)}  end-to-end={e2e}", flush=True)
    for sh in shards:
        sh.unlink()
    for marker in partial_dir.glob("rank_*.done"):
        marker.unlink()
    try:
        partial_dir.rmdir()
    except OSError:
        pass
    if not qc["library_valid"]:
        sys.exit(f"[esd:evaluate] LIBRARY INVALID — duplicates={len(dup_blocks)}, "
                 f"failed={len(failed)}, composition_exact={composition_ok}, "
                 f"end_to_end={e2e}. Not for analysis.")
    return out


def stage_evaluate(plan: dict) -> None:
    """MPI task farm over (policy, source, block); rank 0 merges + QC."""
    comm, rank, size = get_mpi_context()
    is_root = rank == 0
    dvs, policy_ids, labels = load_policies()
    objs = list(ENSEMBLE_OBJECTIVES.values())
    active = list(config.get_objective_set().names)
    active_idx = [[o.name for o in objs].index(a) for a in active]
    sources = _sources(plan)
    tasks = build_tasks(plan, len(policy_ids))
    _check_staged(plan)

    if is_root:
        n_rp = sum(len(t["local"]) for t in tasks)
        print("=== Ensemble-size library evaluation ===", flush=True)
        print(f"  policies: {len(policy_ids)}  sources: {len(sources)}  tasks: {len(tasks)}  "
              f"realization-policies: {n_rp}  ranks: {size}  block: {scfg.ESD_EVAL_BLOCK}  "
              f"smoke: {scfg.ESD_SMOKE}", flush=True)
        print(f"  objectives ({len(objs)} registry; {len(active)} active): "
              f"{[o.name for o in objs]}", flush=True)

    partial_dir = scfg.ESD_LIBRARY_DIR / f"_partial_{scfg.esd_prefix()}library"
    if is_root:
        scfg.ESD_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        if partial_dir.exists():
            for stale in partial_dir.glob("rank_*"):
                stale.unlink()
    prepare_partial_dir(partial_dir, rank)

    slots = assign_rank_slots(len(tasks), rank, size)
    task_ids, units_list, scalars_list, secs = [], [], [], []
    t0 = time.time()
    for slot in slots:
        task = tasks[slot]
        ts = time.perf_counter()
        try:
            u, s = _eval_task(task, dvs, sources, objs, active_idx)
            task_ids.append(task["task_id"])
            units_list.append(u)
            scalars_list.append(s)
            secs.append(time.perf_counter() - ts)
            n_nan = int(np.isnan(u).all(axis=(1, 2)).sum())
            print(f"  [rank {rank:>3} ok] task={task['task_id']:5d} policy={task['policy']} "
                  f"src={sources[task['source']]['key']} n={len(task['local'])} "
                  f"({secs[-1]:.1f}s, {n_nan} empty)", flush=True)
        except Exception:
            tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
            print(f"  [rank {rank:>3} FAIL] task={task['task_id']:5d} {tb}", flush=True)
    print(f"  [rank {rank:>3}] done {len(slots)} tasks in {time.time() - t0:.0f}s", flush=True)

    # Explicit 1-D object arrays: tail blocks are shorter than ESD_EVAL_BLOCK,
    # so the per-task tensors are ragged and must not be stacked.
    units_obj = np.empty(len(units_list), dtype=object)
    scalars_obj = np.empty(len(scalars_list), dtype=object)
    for i, (u, s) in enumerate(zip(units_list, scalars_list)):
        units_obj[i], scalars_obj[i] = u, s
    np.savez(partial_dir / f"rank_{rank:03d}.npz",
             task_ids=np.array(task_ids, dtype=int), units=units_obj,
             scalars=scalars_obj, seconds=np.array(secs, dtype=float))
    mark_rank_done(partial_dir, rank)
    if not is_root:
        return
    if not await_all_done(partial_dir, size, deadline_s=7200.0):
        missing = {f"rank_{r:03d}.done" for r in range(size)} - {
            p.name for p in partial_dir.glob("rank_*.done")}
        print(f"[esd:evaluate] WARN: timeout waiting for {missing}", flush=True)
    _merge(partial_dir, plan, tasks, dvs, policy_ids, labels, objs, active_idx, size)


def main() -> None:
    stage = os.environ.get("NYCOPT_ESD_STAGE", "").strip()
    plan = load_plan()
    if stage == "materialize":
        stage_materialize(plan)
    elif stage == "evaluate":
        stage_evaluate(plan)
    elif stage == "requalify":
        stage_requalify(plan)
    else:
        sys.exit("[esd:lib] set NYCOPT_ESD_STAGE to 'materialize', 'evaluate', or 'requalify'")


if __name__ == "__main__":
    main()
