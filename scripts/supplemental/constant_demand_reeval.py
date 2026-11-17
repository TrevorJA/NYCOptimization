"""constant_demand_reeval.py - Re-score searched policies under a constant
800 MGD NYC demand on the historic trace (supplemental sensitivity).

Every policy of the run's merged set (``sets/{slug}_merged.set``; other set
files via ``--set-files``) is re-simulated on the historic single trace and scored on the run's own
objectives. The simulation is the search's own (``src.simulation.evaluate``:
same window, trimmed model and presim, NJ demand, flow-prediction mode and
objective set) with ONE change: NYC demand is pywrdrb's ``constant_max`` (the
800 MGD FFMP maximum) instead of the time-dynamic extrapolated series
(``src.simulation.nyc_demand_sensitivity``). ``--demand historical`` runs the
unmodified search simulation as a control: its scores must reproduce the
objective columns of the set files.

When several set files are given they usually share rows, so policies are
deduplicated by exact DV vector and each unique policy is simulated once;
per-set outputs are mapped back from the unique table. The
incumbent FFMP policy is added as ``baseline`` (``--no-baseline`` drops it).
Unique policies are dealt round-robin over the MPI ranks; one evaluation is
about 24 s on an EPYC core and 110 s on a loaded Xeon core (HOPPER_NOTES.md).

Outputs, under ``outputs/{scenario}/{slug}/reeval/historic_nyc_demand_{demand}/``:

    policies.csv              one row per unique policy: policy_id, DVs, the
                              re-evaluated objectives (natural units), the set
                              files and rows it appears in, eval seconds, error
    sets/{set_stem}.set       each input set re-scored: same header (plus a
                              provenance line) and DVs, objective columns
                              replaced by the re-evaluated Borg-signed values
                              (failed rows dropped and counted in the header)
    sets/{set_stem}.csv       per-row comparison in natural units: set_row,
                              policy_id, then {obj}__search and
                              {obj}__{demand} for every objective
    meta.json                 demand source, objective names and directions,
                              input files, counts, ranks, timings, git commit

Usage (from repo root, via sbatch; never on a login node):
    sbatch workflow/supplemental/results_diagnostics/constant_demand_reeval.sh
    # control run (must reproduce the search scores):
    sbatch --export=ALL,DEMAND=historical \
        workflow/supplemental/results_diagnostics/constant_demand_reeval.sh
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from config import (  # noqa: E402
    SEARCH_ENSEMBLE_SPEC, active_scenario_name, derive_slug, get_n_vars,
    run_output_dir,
)
from src.formulations import get_baseline_values, get_objective_set, get_var_names  # noqa: E402
from src.load.reference_set import load_set_file  # noqa: E402

DEMAND_CHOICES = ("constant_max", "historical")
BASELINE_ID = "baseline"


###############################################################################
# Inputs
###############################################################################

def _mpi():
    """Return (comm, rank, size); (None, 0, 1) without mpi4py."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def _read_header(path: Path) -> list[str]:
    """Leading comment lines of a set file, verbatim (without newlines)."""
    header = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                header.append(line.rstrip("\n"))
            elif line.strip():
                break
    return header


def _header_field(header: list[str], key: str) -> list[str] | None:
    """Comma-separated value of a ``# Key: a,b,c`` header line, if present."""
    prefix = f"# {key}:"
    for line in header:
        if line.startswith(prefix):
            return [s.strip() for s in line[len(prefix):].split(",")]
    return None


def load_sets(files: list[Path], n_vars: int, obj_names: list[str]) -> list[dict]:
    """Load the given set files; check their layout against the run.

    Returns:
        One dict per set file: path, header, dv (n, n_vars), obj (n, n_obj)
        Borg-signed as stored.
    """
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"set files not found: {missing}")
    sets = []
    for path in files:
        header = _read_header(path)
        data = load_set_file(path)
        if data.size == 0:
            print(f"[setup] {path.name}: empty, skipped")
            continue
        if data.shape[1] != n_vars + len(obj_names):
            raise ValueError(
                f"{path.name}: {data.shape[1]} columns, expected "
                f"{n_vars} vars + {len(obj_names)} objectives")
        hdr_objs = _header_field(header, "Objectives")
        if hdr_objs is not None and hdr_objs != obj_names:
            raise ValueError(
                f"{path.name}: header objectives {hdr_objs} differ from the "
                f"active objective set {obj_names}; source the run's env file")
        sets.append({"path": path, "header": header,
                     "dv": data[:, :n_vars], "obj": data[:, n_vars:]})
    return sets


def unique_policies(sets: list[dict], formulation: str, with_baseline: bool):
    """Deduplicate DV rows across sets.

    Returns:
        (policy_ids, dv_unique, members): ``policy_ids[i]`` is ``"p0000"``-style
        (or ``"baseline"``), ``dv_unique[i]`` its DV vector, ``members[i]`` the
        list of ``"set_stem:row"`` it appears as. ``set["policy_index"]`` is
        filled with each row's index into the unique table.
    """
    stacked = np.vstack([s["dv"] for s in sets])
    dv_unique, inverse = np.unique(stacked, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    members: list[list[str]] = [[] for _ in range(len(dv_unique))]
    start = 0
    for s in sets:
        n = len(s["dv"])
        s["policy_index"] = inverse[start:start + n]
        for row, idx in enumerate(s["policy_index"]):
            members[idx].append(f"{s['path'].stem}:{row}")
        start += n
    policy_ids = [f"p{i:04d}" for i in range(len(dv_unique))]
    if with_baseline:
        base = np.asarray(get_baseline_values(formulation), float)[None, :]
        dv_unique = np.vstack([dv_unique, base])
        policy_ids.append(BASELINE_ID)
        members.append([])
    return policy_ids, dv_unique, members


###############################################################################
# Evaluation
###############################################################################

def evaluate_slice(ids, dv_unique, formulation, demand, rank):
    """Evaluate the policies ``ids``; return [(idx, borg_objs|None, secs, err)]."""
    from src.simulation import evaluate, nyc_demand_sensitivity

    obj_set = get_objective_set()
    ctx = (nyc_demand_sensitivity(demand) if demand != "historical"
           else contextlib.nullcontext())
    out = []
    with ctx:
        for idx in ids:
            t0 = time.time()
            try:
                objs = evaluate(dv_unique[idx], formulation_name=formulation,
                                objective_set=obj_set,
                                ensemble_spec=SEARCH_ENSEMBLE_SPEC)
                out.append((int(idx), [float(v) for v in objs],
                            time.time() - t0, None))
            except Exception as e:  # noqa: BLE001 - recorded per policy
                traceback.print_exc()
                out.append((int(idx), None, time.time() - t0,
                            f"{type(e).__name__}: {e}"))
            print(f"[rank {rank:>3}] policy {idx:>5} "
                  f"{'FAIL' if out[-1][3] else 'ok'} {out[-1][2]:.1f}s", flush=True)
    return out


###############################################################################
# Outputs
###############################################################################

def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=PROJECT_DIR, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def write_outputs(out_dir: Path, sets, policy_ids, dv_unique, members,
                  borg: np.ndarray, secs, errors, *, obj_names, signs,
                  var_names, demand, meta_extra):
    """Write policies.csv, per-set .set/.csv files and meta.json."""
    sets_out = out_dir / "sets"
    sets_out.mkdir(parents=True, exist_ok=True)
    natural = -borg * signs[None, :]          # Borg minimises: maximise objectives are stored negated
    tag = demand

    pol = pd.DataFrame(dv_unique, columns=var_names)
    pol.insert(0, "policy_id", policy_ids)
    for k, name in enumerate(obj_names):
        pol[name] = natural[:, k]
    pol["appears_in"] = [";".join(m) for m in members]
    pol["eval_seconds"] = np.round(secs, 2)
    pol["error"] = errors
    pol.to_csv(out_dir / "policies.csv", index=False)

    per_set = {}
    for s in sets:
        idx = s["policy_index"]
        ok = ~np.isnan(borg[idx]).any(axis=1)
        stem = s["path"].stem
        provenance = (
            f"# Re-evaluated {datetime.now():%Y-%m-%d}: historic trace, NYC demand "
            f"'{demand}' (constant_max = 800 MGD FFMP maximum); objective columns "
            f"replaced by the re-evaluated Borg-signed scores; "
            f"{int((~ok).sum())} failed rows dropped. Source: {s['path'].name}")
        with open(sets_out / f"{stem}.set", "w") as f:
            f.write("\n".join(s["header"] + [provenance]) + "\n")
            np.savetxt(f, np.hstack([s["dv"][ok], borg[idx][ok]]), fmt="%.6e")
        cmp = pd.DataFrame({"set_row": np.arange(len(idx)),
                            "policy_id": [policy_ids[i] for i in idx]})
        search_nat = -s["obj"] * signs[None, :]
        for k, name in enumerate(obj_names):
            cmp[f"{name}__search"] = search_nat[:, k]
            cmp[f"{name}__{tag}"] = natural[idx, k]
        cmp.to_csv(sets_out / f"{stem}.csv", index=False)
        per_set[s["path"].name] = {"rows": int(len(idx)), "failed": int((~ok).sum())}

    meta = {
        "description": ("Searched policies re-simulated on the historic trace "
                        "with the search simulation, NYC demand source "
                        f"'{demand}'; scored on the run's objectives."),
        "nyc_demand_source": demand,
        "nyc_demand_note": ("constant_max = pywrdrb constant at "
                            "max_flow_baseline_delivery_nyc (800 MGD); "
                            "historical = the search's time-dynamic series"),
        "objectives": obj_names,
        "directions": ["maximize" if s > 0 else "minimize" for s in signs],
        "objective_units": "natural (policies.csv, sets/*.csv); Borg-signed (sets/*.set)",
        "n_unique_policies": len(policy_ids),
        "sets": per_set,
        "git_commit": _git_commit(),
        "written": datetime.now().isoformat(timespec="seconds"),
        **meta_extra,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


###############################################################################
# Driver
###############################################################################

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--formulation", default="ffmp")
    ap.add_argument("--demand", choices=DEMAND_CHOICES, default="constant_max",
                    help="NYC demand source (historical = search control run)")
    ap.add_argument("--set-files", nargs="+", default=None,
                    help="set files to re-score; default "
                         "outputs/{scenario}/{slug}/sets/{slug}_merged.set")
    ap.add_argument("--out-dir", default=None,
                    help="default outputs/{scenario}/{slug}/reeval/"
                         "historic_nyc_demand_{demand}")
    ap.add_argument("--max", type=int, default=0,
                    help="evaluate only the first N unique policies (smoke test)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="do not add the incumbent FFMP policy")
    args = ap.parse_args()

    comm, rank, size = _mpi()
    t_start = time.time()

    if SEARCH_ENSEMBLE_SPEC is None or SEARCH_ENSEMBLE_SPEC.is_ensemble:
        raise SystemExit(
            "constant_demand_reeval: the active design is not the historic single "
            "trace; source workflow/envs/ffmp_obj8_historic_production.env")

    scenario = active_scenario_name()
    slug = derive_slug(args.formulation)
    set_files = ([Path(p) for p in args.set_files] if args.set_files else
                 [run_output_dir(scenario, slug, "sets") / f"{slug}_merged.set"])
    out_dir = (Path(args.out_dir) if args.out_dir else
               run_output_dir(scenario, slug, "reeval") / f"historic_nyc_demand_{args.demand}")
    if args.max > 0 and not args.out_dir:
        out_dir = out_dir.with_name(out_dir.name + "_smoke")

    obj_set = get_objective_set()
    obj_names = list(obj_set.names)
    signs = np.asarray(obj_set.directions, float)  # +1 maximise, -1 minimise
    var_names = list(get_var_names(args.formulation))
    n_vars = get_n_vars(args.formulation)

    if rank == 0:
        sets = load_sets(set_files, n_vars, obj_names)
        policy_ids, dv_unique, members = unique_policies(
            sets, args.formulation, with_baseline=not args.no_baseline)
        n_rows = sum(len(s["dv"]) for s in sets)
        n_eval = len(policy_ids) if args.max <= 0 else min(args.max, len(policy_ids))
        print(f"[setup] scenario={scenario} slug={slug} demand={args.demand}")
        print(f"[setup] {len(sets)} set files, {n_rows} rows, "
              f"{len(policy_ids)} unique policies; evaluating {n_eval} on {size} ranks")
        print(f"[setup] outputs -> {out_dir}", flush=True)
        payload = (dv_unique, n_eval)
    else:
        payload = None
    if comm is not None:
        payload = comm.bcast(payload, root=0)
    dv_unique, n_eval = payload

    # Round-robin deal: neighbouring ranks share a node, so each node gets an
    # even share; evaluations take near-equal time on one node class.
    my_ids = list(range(rank, n_eval, size))
    t0 = time.time()
    results = evaluate_slice(my_ids, dv_unique, args.formulation, args.demand, rank)
    print(f"[rank {rank:>3}] {len(my_ids)} policies in {time.time() - t0:.0f}s", flush=True)

    gathered = comm.gather(results, root=0) if comm is not None else [results]
    if rank != 0:
        return

    n_obj = len(obj_names)
    borg = np.full((len(policy_ids), n_obj), np.nan)
    secs = np.full(len(policy_ids), np.nan)
    errors: list = ["not evaluated (--max)"] * len(policy_ids)
    for chunk in gathered:
        for idx, objs, sec, err in chunk:
            secs[idx] = sec
            errors[idx] = err or ""
            if objs is not None:
                borg[idx] = objs
    n_fail = sum(bool(errors[i]) for i in range(n_eval))
    write_outputs(
        out_dir, sets, policy_ids, dv_unique, members, borg, secs, errors,
        obj_names=obj_names, signs=signs, var_names=var_names,
        demand=args.demand,
        meta_extra={"set_files": [str(p) for p in set_files], "formulation": args.formulation,
                    "scenario": scenario, "slug": slug, "mpi_ranks": size,
                    "n_evaluated": n_eval, "n_failed": n_fail,
                    "wall_seconds": round(time.time() - t_start, 1),
                    "median_eval_seconds": float(np.nanmedian(secs))},
    )
    print(f"[done] {n_eval - n_fail}/{n_eval} ok in {time.time() - t_start:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
