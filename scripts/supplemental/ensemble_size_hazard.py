"""ensemble_size_hazard.py - Layer A of the ensemble-size diagnostics + the library plan.

Selection-level (no simulation) half of
``docs/notes/methods/ensemble_size_diagnostics.md``: hazard-space
representativeness versus the ensemble size N for both matched designs, read
off the staged candidate-pool hazard images, followed by the two fixed inputs
the Layer-B library build needs — the rule-selected policy set and the
realization plan (which pool members to regenerate, in which chunk).

Blocks (all tables under ``outputs/supplemental/ensemble_size_diagnostics/tables/``):

  A-HF  ``hf_ladder.csv``       the campaign selector at every ladder N on each
                                 P = 1e6 pool image, ESD_HF_ANCHOR_DRAWS anchor
                                 plans per (pool, N), plus the matched random
                                 null: per-axis tail shares (P90, P99),
                                 stratification, joint L2-star, MST edge
                                 statistics, minimum separation — the block-D
                                 record of ``diagnose_hazard_selectors.py``.
        ``hf_selections.json``   the selected pool global ids per
                                 (pool draw, N, anchor draw) — consumed by the
                                 library plan.
  A-NP  ``np_ladder.csv``       the same selector on nested prefixes P' of pool
                                 d0 (exact i.i.d. pools of their size): the
                                 joint (N, P) adequacy surface.
  A-PS  ``ps_tail_sampling.csv`` the exact i.i.d. law of a size-N subset of the
                                 pool image: per-axis counts above the pool
                                 P90/P99, relative quantile error, and the
                                 closed-form P(>= 1 member beyond q) = 1 - q^N.
  A-CV  ``descriptor_convergence.csv``
                                 pooled-mean vs ensemble-extreme descriptors
                                 versus N (PS bands over subsets; HF plans).
  plan  ``policies.csv`` / ``policies.json``   the fixed policy set (§4.1 rule).
        ``library_plan.json``    chunks of pool members to regenerate + the
                                 staged production ensembles to read as-is.

The HF selections are made with the same primitive the step-03 selector uses
(``scengen.subsample.absolute_filling_subsample`` on the screened campaign
axes), and the N = 100 / anchor-draw-0 selection is asserted equal to the
staged production ``hazfill_stat_abs_10yr_n100_d0`` member list.

Settings in ``supplemental_config.py`` (``ESD_*``); no CLI value flags.
Wrapper: ``workflow/supplemental/ensemble_size_hazard.sh``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_esd_env()

import config  # noqa: E402
from scengen import selector_diagnostics as sd  # noqa: E402
from scengen import subsample as ss  # noqa: E402
from scengen.diagnostics import load_hazard_image  # noqa: E402
from scengen.hazard_filling import screen_hazard_axes  # noqa: E402
from scripts.supplemental.diagnose_hazard_selectors import (  # noqa: E402
    n_sweep_record,
)
from src.ensemble_size_stats import p_at_least_one_beyond  # noqa: E402
from src.ensembles import staged_ensemble_dir  # noqa: E402
from src.scenario_designs import get_scenario_design  # noqa: E402

#: Descriptor-convergence N grid: log-spaced between 10 and the largest ladder
#: rung times two, merged with the ladder itself.
_CV_GRID_POINTS = 12


###############################################################################
# Pool images
###############################################################################

def load_pool_image(draw: int) -> dict | None:
    """Load one candidate pool's hazard image restricted to the selection axes.

    Returns ``None`` (with a message) when the pool is not staged.

    Returns:
        Dict with ``H`` (``(P, m)`` on ``config.HAZARD_SELECTION_AXES``),
        ``axes``, ``global_ids``, ``slug``, ``X`` (campaign unit-box
        normalization of ``H``), ``p90``/``p99`` per axis.
    """
    slug = scfg.esd_pool_slug(draw)
    path = staged_ensemble_dir(slug) / "hazard_image.npz"
    if not path.exists():
        print(f"[esd:A] pool image not staged: {path} — skipping draw {draw}")
        return None
    img = load_hazard_image(path)
    axes_all = list(img["hazard_axes"])
    axes = list(config.HAZARD_SELECTION_AXES)
    missing = [a for a in axes if a not in axes_all]
    if missing:
        raise KeyError(f"{path}: selection axes {missing} absent from {axes_all}")
    H = np.asarray(img["H"], dtype=float)[:, [axes_all.index(a) for a in axes]]
    # The campaign selector screens the selection axes per pool; the screen
    # must retain all six on a production pool (it did on d0-d2), otherwise
    # the ladder would be scored on a different axis set than step 03 uses.
    screen = screen_hazard_axes(H, axes)
    if list(screen["retained"]) != axes:
        raise RuntimeError(
            f"{slug}: axis screen retained {screen['retained']} != selection axes {axes}"
        )
    return {
        "slug": slug, "draw": draw, "H": H, "axes": axes,
        "global_ids": np.asarray(img["realization_ids"], dtype=int),
        "X": ss.minmax_normalize(H),
        "p90": np.percentile(H, 90, axis=0),
        "p99": np.percentile(H, 99, axis=0),
    }


def anchor_seed(anchor_draw: int) -> int:
    """The design's own selector seed for ``anchor_draw`` (0 = production plan)."""
    return int(get_scenario_design("hazard_filling_stationary").selector_seed(anchor_draw))


###############################################################################
# Block A-HF: the campaign selector across the N ladder on each pool
###############################################################################

def hf_ladder(pools: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Block A-HF records + the selected global ids per (pool draw, N, anchor draw)."""
    records, selections = [], {}
    for pool in pools:
        H, X, axes = pool["H"], pool["X"], pool["axes"]
        sel_pool = selections.setdefault(str(pool["draw"]), {})
        for n in scfg.ESD_N_LADDER:
            sel_n = sel_pool.setdefault(str(n), {})
            for ad in scfg.ESD_HF_ANCHOR_DRAWS:
                rows = ss.absolute_filling_subsample(H, n, seed=anchor_seed(ad))
                sel_n[str(ad)] = [int(g) for g in pool["global_ids"][rows]]
                records.append({
                    "pool_draw": pool["draw"], "P": len(H), "n": n,
                    "selector": "lhs_nn", "seed": ad,
                    **n_sweep_record(H, X, rows, axes),
                })
            for seed in range(scfg.ESD_NULL_SEEDS):
                rows = ss.random_subsample(H, n, seed=seed)
                records.append({
                    "pool_draw": pool["draw"], "P": len(H), "n": n,
                    "selector": "random", "seed": seed,
                    **n_sweep_record(H, X, rows, axes),
                })
            print(f"[esd:A-HF] pool d{pool['draw']} N={n} done", flush=True)
    return pd.DataFrame.from_records(records), selections


def assert_production_identity(selections: dict) -> dict:
    """Check the N = 100 / anchor-draw-0 selection equals the staged production draw.

    Returns a small QC dict (written into the plan); raises when the staged
    ensemble exists and the member lists differ.
    """
    qc = {"checked": False}
    lib_draw = str(scfg.ESD_LIBRARY_POOL_DRAW)
    n_key = str(scfg.ESD_N_CAMPAIGN)
    staged = dict(scfg.ESD_STAGED_ENSEMBLES["hazard_filling_stationary"])
    slug = staged.get(scfg.ESD_LIBRARY_POOL_DRAW)
    meta_path = staged_ensemble_dir(slug) / "_meta.json" if slug else None
    if scfg.ESD_SMOKE or meta_path is None or not meta_path.exists():
        return qc
    if n_key not in selections.get(lib_draw, {}):
        return qc
    meta = json.loads(meta_path.read_text())
    if meta.get("source_pool") != scfg.esd_pool_slug(scfg.ESD_LIBRARY_POOL_DRAW):
        return qc
    ours = sorted(selections[lib_draw][n_key]["0"])
    theirs = sorted(int(g) for g in meta["global_realization_ids"])
    qc.update({"checked": True, "staged_slug": slug, "identical": ours == theirs,
               "n_common": len(set(ours) & set(theirs))})
    if ours != theirs:
        raise RuntimeError(
            f"Layer-A N=100 anchor-draw-0 selection differs from staged {slug} "
            f"({qc['n_common']}/{len(theirs)} common members)"
        )
    print(f"[esd:A-HF] production identity check PASSED against {slug}")
    return qc


###############################################################################
# Block A-NP: nested prefixes of the library pool
###############################################################################

def np_ladder(pool: dict) -> pd.DataFrame:
    """Block A-NP records on nested prefixes P' of the library pool."""
    records = []
    H_full, axes = pool["H"], pool["axes"]
    for p_prime in scfg.ESD_NP_PREFIXES:
        if p_prime > len(H_full):
            print(f"[esd:A-NP] prefix {p_prime} exceeds P={len(H_full)}; skipped")
            continue
        H = H_full[:p_prime]
        X = ss.minmax_normalize(H)
        for n in scfg.ESD_N_LADDER:
            if n > p_prime:
                continue
            for ad in scfg.ESD_HF_ANCHOR_DRAWS:
                rows = ss.absolute_filling_subsample(H, n, seed=anchor_seed(ad))
                records.append({
                    "pool_draw": pool["draw"], "P": p_prime, "n": n,
                    "selector": "lhs_nn", "seed": ad,
                    **n_sweep_record(H, X, rows, axes),
                })
        print(f"[esd:A-NP] prefix P'={p_prime} done", flush=True)
    return pd.DataFrame.from_records(records)


###############################################################################
# Block A-PS: the exact i.i.d. law of size-N subsets of the pool image
###############################################################################

def ps_tail_sampling(pool: dict) -> pd.DataFrame:
    """Per-axis tail counts and quantile errors of uniform size-N subsets."""
    H, axes = pool["H"], pool["axes"]
    P = len(H)
    pool_q = {q: np.percentile(H, q, axis=0) for q in (50.0, 90.0, 99.0)}
    records = []
    for n in scfg.ESD_N_LADDER:
        counts = {q: np.zeros((scfg.ESD_PS_SUBSETS, len(axes))) for q in scfg.ESD_TAIL_QUANTILES}
        rel_err = {q: np.zeros((scfg.ESD_PS_SUBSETS, len(axes))) for q in (50.0, 90.0, 99.0)}
        for s in range(scfg.ESD_PS_SUBSETS):
            rows = ss.random_subsample(H, n, seed=s)
            sub = H[rows]
            for q in scfg.ESD_TAIL_QUANTILES:
                counts[q][s] = (sub > np.percentile(H, q, axis=0)).sum(axis=0)
            for q, ref in pool_q.items():
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_err[q][s] = (np.percentile(sub, q, axis=0) - ref) / np.where(ref != 0, ref, np.nan)
        for k, axis in enumerate(axes):
            rec = {"n": n, "axis": axis, "P": P, "n_subsets": scfg.ESD_PS_SUBSETS}
            for q in scfg.ESD_TAIL_QUANTILES:
                c = counts[q][:, k]
                tag = f"p{int(q)}"
                rec.update({
                    f"count_{tag}_mean": float(c.mean()),
                    f"count_{tag}_sd": float(c.std(ddof=1)),
                    f"count_{tag}_p05": float(np.percentile(c, 5)),
                    f"count_{tag}_p50": float(np.percentile(c, 50)),
                    f"count_{tag}_p95": float(np.percentile(c, 95)),
                    f"share_{tag}_mean": float(c.mean() / n),
                    f"prob_zero_{tag}_empirical": float(np.mean(c == 0)),
                    f"prob_ge1_{tag}_closed_form": p_at_least_one_beyond(q / 100.0, n),
                })
            for q in (50.0, 90.0, 99.0):
                e = rel_err[q][:, k]
                rec[f"relerr_p{int(q)}_rms"] = float(np.sqrt(np.nanmean(e ** 2)))
                rec[f"relerr_p{int(q)}_p95abs"] = float(np.nanpercentile(np.abs(e), 95))
            records.append(rec)
        print(f"[esd:A-PS] N={n} done", flush=True)
    return pd.DataFrame.from_records(records)


###############################################################################
# Block A-CV: descriptor convergence (pooled mean vs ensemble extreme)
###############################################################################

def _cv_grid() -> list[int]:
    lo, hi = 10, 2 * max(scfg.ESD_N_LADDER)
    grid = set(int(round(v)) for v in np.geomspace(lo, hi, _CV_GRID_POINTS))
    grid |= set(scfg.ESD_N_LADDER)
    return sorted(grid)


def descriptor_convergence(pool: dict, selections: dict) -> pd.DataFrame:
    """Pooled-mean and ensemble-max per axis vs N: PS subset bands and HF plans."""
    H, axes = pool["H"], pool["axes"]
    records = []
    for n in _cv_grid():
        means = np.zeros((scfg.ESD_PS_SUBSETS, len(axes)))
        maxes = np.zeros_like(means)
        for s in range(scfg.ESD_PS_SUBSETS):
            sub = H[ss.random_subsample(H, n, seed=s)]
            means[s], maxes[s] = sub.mean(axis=0), sub.max(axis=0)
        for k, axis in enumerate(axes):
            for stat, arr in (("pooled_mean", means), ("ensemble_max", maxes)):
                v = arr[:, k]
                records.append({
                    "design": "fixed_probabilistic", "n": n, "axis": axis, "statistic": stat,
                    "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
                    "p95": float(np.percentile(v, 95)), "n_replicates": len(v),
                    "pool_value": float(H[:, k].mean() if stat == "pooled_mean" else H[:, k].max()),
                })
    gid_to_row = {int(g): i for i, g in enumerate(pool["global_ids"])}
    for n_key, by_plan in selections[str(pool["draw"])].items():
        n = int(n_key)
        rows_by_plan = [np.array([gid_to_row[g] for g in ids]) for ids in by_plan.values()]
        means = np.array([H[r].mean(axis=0) for r in rows_by_plan])
        maxes = np.array([H[r].max(axis=0) for r in rows_by_plan])
        for k, axis in enumerate(axes):
            for stat, arr in (("pooled_mean", means), ("ensemble_max", maxes)):
                v = arr[:, k]
                records.append({
                    "design": "hazard_filling_stationary", "n": n, "axis": axis, "statistic": stat,
                    "p05": float(np.min(v)), "p50": float(np.median(v)), "p95": float(np.max(v)),
                    "n_replicates": len(v),
                    "pool_value": float(H[:, k].mean() if stat == "pooled_mean" else H[:, k].max()),
                })
    return pd.DataFrame.from_records(records)


###############################################################################
# The fixed policy set (methods note §4.1)
###############################################################################

def _load_union() -> dict:
    """Union of the matched designs' reference sets (PS rows first)."""
    from src.solution_selection import load_natural_front

    dvs, objs, src, rows, names, dirs = [], [], [], [], None, None
    for design in ("fixed_probabilistic", "hazard_filling_stationary"):
        path = scfg.ESD_POLICY_SET_FILES[design]
        if not path.exists():
            raise FileNotFoundError(f"policy set file missing: {path}")
        dv, nat, obj_names, directions = load_natural_front(path, scfg.ESD_FORMULATION)
        dvs.append(dv)
        objs.append(nat)
        src += [design] * len(dv)
        rows += list(range(len(dv)))
        names, dirs = obj_names, directions
    return {
        "dv": np.vstack(dvs), "obj": np.vstack(objs), "source": np.array(src),
        "row": np.array(rows, dtype=int), "obj_names": list(names),
        "directions": list(dirs),
    }


def _scaled_loss(obj: np.ndarray, directions) -> np.ndarray:
    """Direction-oriented (0 = best), min-max scaled over the union."""
    signs = np.array([-1.0 if d == 1 else 1.0 for d in directions])
    loss = obj * signs[None, :]
    lo, hi = loss.min(axis=0), loss.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return (loss - lo[None, :]) / span[None, :]


def _compromise_ids(design: str) -> dict:
    """Both compromise candidates of a design from its re-eval cube (or None)."""
    from src import robustness as rob
    from src.factor_mapping import select_compromise

    leaf = (config.OUTPUTS_DIR / design / "ffmp_obj8" / "reeval" / scfg.ESD_POLICY_REEVAL_TAG)
    if not (leaf / "reeval_raw_meta.json").exists():
        print(f"[esd:plan] no re-eval cube for {design} at {leaf}; compromise rules skipped")
        return {}
    raw = rob.load_raw(leaf)
    best = select_compromise(raw, rule="best_satisficing")
    return {"best_satisficing": int(best["best_satisficing_id"]),
            "min_dist_ideal": int(best["min_dist_ideal_id"]),
            "satisficing": float(best["satisficing"])}


def select_policy_set() -> tuple[pd.DataFrame, dict]:
    """Apply the pre-registered §4.1 rule; return the policy table + provenance."""
    from src.formulations import get_baseline_values
    from src.solution_selection import best_single

    union = _load_union()
    obj, dv, names, dirs = union["obj"], union["dv"], union["obj_names"], union["directions"]
    scaled = _scaled_loss(obj, dirs)

    chosen: list[dict] = []

    def _taken(vec: np.ndarray) -> bool:
        return any(np.allclose(vec, c["dv"]) for c in chosen)

    def _add(label: str, rule: str, idx: int | None, vec: np.ndarray, extra: dict) -> None:
        chosen.append({"policy_id": f"P{len(chosen)}", "label": label, "rule": rule,
                       "union_index": idx, "dv": vec, **extra})

    _add("incumbent", "ffmp_baseline", None,
         np.asarray(get_baseline_values(scfg.ESD_FORMULATION), dtype=float),
         {"source": "incumbent", "source_row": -1})

    def _add_by_rank(label: str, rule: str, order: np.ndarray) -> None:
        for idx in order:
            idx = int(idx)
            if not _taken(dv[idx]):
                _add(label, rule, idx, dv[idx],
                     {"source": str(union["source"][idx]), "source_row": int(union["row"][idx])})
                return
        raise RuntimeError(f"rule {rule!r} exhausted the union without a new policy")

    for oname in scfg.ESD_POLICY_BEST_OBJECTIVES:
        if oname not in names:
            raise KeyError(f"objective {oname!r} not in the set files' objectives {names}")
        k = names.index(oname)
        # best_single gives the top row; the full order (ties -> lowest row)
        # lets the dedupe rule fall through to the next-best.
        top = best_single(obj, dirs, k)
        order = np.argsort(-obj[:, k] * (1.0 if dirs[k] == 1 else -1.0), kind="stable")
        assert int(order[0]) == top
        _add_by_rank(f"best_{oname}", "per_objective_best", order)

    compromise = {}
    for design in ("fixed_probabilistic", "hazard_filling_stationary"):
        ids = _compromise_ids(design)
        compromise[design] = ids
        if not ids:
            continue
        mask = union["source"] == design
        uidx = int(np.flatnonzero(mask & (union["row"] == ids["best_satisficing"]))[0])
        _add_by_rank(f"compromise_best_satisficing_{design}", "select_compromise",
                     np.array([uidx]))
    for design in ("fixed_probabilistic", "hazard_filling_stationary"):
        comp = [c for c in chosen if c["rule"] == "select_compromise" and design in c["label"]]
        if not comp:
            continue
        ref = scaled[comp[0]["union_index"]]
        d = np.linalg.norm(scaled - ref[None, :], axis=1)
        _add_by_rank(f"adjacent_to_compromise_{design}", "nearest_in_scaled_objective_space",
                     np.argsort(d, kind="stable"))
    ps_ids = compromise.get("fixed_probabilistic", {})
    if ps_ids:
        mask = union["source"] == "fixed_probabilistic"
        uidx = int(np.flatnonzero(mask & (union["row"] == ps_ids["min_dist_ideal"]))[0])
        # Fall through to the next-closest-to-ideal PS row if already chosen.
        ps_rows = np.flatnonzero(mask)
        d = np.linalg.norm(scaled[ps_rows], axis=1)
        order = ps_rows[np.argsort(d, kind="stable")]
        assert int(order[0]) == uidx or _taken(dv[uidx])
        _add_by_rank("compromise_min_dist_ideal_fixed_probabilistic", "select_compromise",
                     np.concatenate([[uidx], order]))

    chosen = chosen[:scfg.ESD_N_POLICIES]
    rows = []
    for c in chosen:
        rec = {k: v for k, v in c.items() if k != "dv"}
        idx = c["union_index"]
        for j, oname in enumerate(names):
            rec[f"search_{oname}"] = float(obj[idx, j]) if idx is not None else np.nan
        for j, v in enumerate(c["dv"]):
            rec[f"dv{j:02d}"] = float(v)
        rows.append(rec)
    table = pd.DataFrame(rows)
    provenance = {
        "rule": "docs/notes/methods/ensemble_size_diagnostics.md §4.1",
        "set_files": {k: str(v) for k, v in scfg.ESD_POLICY_SET_FILES.items()},
        "union_size": int(len(dv)),
        "reeval_tag": scfg.ESD_POLICY_REEVAL_TAG,
        "compromise": compromise,
        "objective_names": names,
        "n_policies": int(len(chosen)),
    }
    return table, provenance


###############################################################################
# The library plan
###############################################################################

def build_library_plan(selections: dict, qc: dict) -> dict:
    """Chunks of pool members to regenerate + staged ensembles to read as-is."""
    lib_draw = str(scfg.ESD_LIBRARY_POOL_DRAW)
    plans = [str(ad) for ad in scfg.ESD_HF_ANCHOR_DRAWS[:scfg.ESD_HF_LIBRARY_PLANS]]
    hf_members: dict = {}
    members = set(range(scfg.ESD_P_REF))
    for n_key, by_plan in selections[lib_draw].items():
        hf_members[n_key] = {ad: by_plan[ad] for ad in plans}
        for ad in plans:
            members.update(by_plan[ad])
    ordered = sorted(members)
    chunks = []
    for j in range(0, len(ordered), scfg.ESD_CHUNK_SIZE):
        gids = ordered[j:j + scfg.ESD_CHUNK_SIZE]
        chunks.append({
            "chunk_index": len(chunks),
            "slug": f"{scfg.esd_library_slug()}__chunk{len(chunks):03d}",
            "global_ids": gids, "n_realizations": len(gids),
        })
    staged = []
    for design, entries in scfg.ESD_STAGED_ENSEMBLES.items():
        for draw, slug in entries:
            meta_path = staged_ensemble_dir(slug) / "_meta.json"
            if scfg.ESD_SMOKE or not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            staged.append({"design": design, "draw": draw, "slug": slug,
                           "n_realizations": int(meta["n_realizations"]),
                           "global_ids": meta.get("global_realization_ids")})
    n_hf_union = len(members) - scfg.ESD_P_REF
    return {
        "library_slug": scfg.esd_library_slug(),
        "pool_slug": scfg.esd_pool_slug(scfg.ESD_LIBRARY_POOL_DRAW),
        "pool_draw": scfg.ESD_LIBRARY_POOL_DRAW,
        "p_ref": scfg.ESD_P_REF,
        "n_ladder": list(scfg.ESD_N_LADDER),
        "hf_library_plans": plans,
        "hf_members": hf_members,
        "n_unique_pool_members": len(ordered),
        "n_hf_members_outside_prefix": int(sum(1 for g in ordered if g >= scfg.ESD_P_REF)),
        "n_hf_union": n_hf_union,
        "chunk_size": scfg.ESD_CHUNK_SIZE,
        "chunks": chunks,
        "staged_ensembles": staged,
        "production_identity_qc": qc,
    }


###############################################################################
# Driver
###############################################################################

def main() -> None:
    """Run Layer A, select the policy set, and write the library plan."""
    t0 = time.time()
    scfg.ESD_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pools = [p for p in (load_pool_image(d) for d in scfg.ESD_POOL_DRAWS) if p is not None]
    if not pools:
        sys.exit("[esd:A] no candidate pool image staged")
    lib_pool = next((p for p in pools if p["draw"] == scfg.ESD_LIBRARY_POOL_DRAW), None)
    if lib_pool is None:
        sys.exit(f"[esd:A] library pool draw {scfg.ESD_LIBRARY_POOL_DRAW} is not staged")
    print(f"[esd:A] pools {[p['slug'] for p in pools]}, axes {lib_pool['axes']}, "
          f"ladder {scfg.ESD_N_LADDER}, anchor draws {scfg.ESD_HF_ANCHOR_DRAWS}, "
          f"smoke={scfg.ESD_SMOKE}", flush=True)

    ladder, selections = hf_ladder(pools)
    ladder.to_csv(scfg.esd_table_path("hf_ladder"), index=False)
    scfg.esd_json_path("hf_selections").write_text(json.dumps(selections))
    qc = assert_production_identity(selections)

    np_ladder(lib_pool).to_csv(scfg.esd_table_path("np_ladder"), index=False)
    ps_tail_sampling(lib_pool).to_csv(scfg.esd_table_path("ps_tail_sampling"), index=False)
    descriptor_convergence(lib_pool, selections).to_csv(
        scfg.esd_table_path("descriptor_convergence"), index=False)

    policies, provenance = select_policy_set()
    policies.to_csv(scfg.esd_table_path("policies"), index=False)
    scfg.esd_json_path("policies").write_text(json.dumps(provenance, indent=2))
    print("[esd:plan] policy set:\n" + policies[["policy_id", "label", "source", "source_row"]]
          .to_string(index=False), flush=True)

    plan = build_library_plan(selections, qc)
    scfg.esd_json_path("library_plan").write_text(json.dumps(plan))
    print(f"[esd:plan] library: {plan['n_unique_pool_members']} unique pool members "
          f"({scfg.ESD_P_REF} PS reference + {plan['n_hf_members_outside_prefix']} HF-only) "
          f"in {len(plan['chunks'])} chunks of <= {scfg.ESD_CHUNK_SIZE}; "
          f"{len(plan['staged_ensembles'])} staged production ensembles; "
          f"{policies.shape[0]} policies. ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
