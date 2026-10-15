"""hazard_axis_selection_run.py - Choose the hazard-filling selection axes.

Supplemental diagnostic (``docs/notes/methods/hazard_axis_selection.md``) that
picks which n in {4, 5, 6} of the candidate hazard metrics the hazard-filling
design fills, on the candidate-pool hazard images alone (no simulation):

  1. candidate axes and admissibility: the continuous metrics of the 21-metric
     image; sets with no pair at |rho_S| >= HAXSEL_PAIR_RHO on any draw and at
     least two dry-side and two wet-side metrics (``src.hazard_axis_selection``);
  2. scoring: every admissible set under ``lhs_nn`` at N on the full pools of
     both draws with S seeds, by how uniformly its selection covers the scaled
     range of every continuous metric (cluster-averaged Kolmogorov-Smirnov
     distance to uniform, the decision statistic), with the covering distance
     of the pool and the spanning-tree statistics reported; random same-N
     subsets of each pool give the null reference; the nearest-member
     effective sample size and the full-pool covering distance in a second
     pass for the leading sets;
  3. ranking, the pre-stated rule, the sensitivity, draw-agreement and
     joint-coverage tables.

Settings ``HAXSEL_*`` in ``supplemental_config.py``; env knobs
``NYCOPT_HAXSEL_{SMOKE,POOL_ROOT,WORKERS}``, ``NYCOPT_CANDIDATE_POOL_N`` and
``NYCOPT_SEARCH_N``. Outputs ``outputs/supplemental/hazard_axis_selection/tables/``
(``metrics``, ``admissible_sets``, ``wrapper_scores``, ``null_scores``,
``null_reference``, ``ranking``, ``leaders``, ``joint_check``,
``decision_sensitivity``, ``draw_agreement``, ``adopted_members_d{k}``,
``pool_quantiles_d{k}`` and ``summary.json``); figures by
``hazard_axis_selection_figures.py``. Run::

    python scripts/supplemental/hazard_axis_selection_run.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402

scfg.configure_haxsel_env()

import config  # noqa: E402
from src import hazard_axis_selection as hax  # noqa: E402


def _pool_path(draw: int) -> Path:
    root = Path(scfg.HAXSEL_POOL_ROOT) if scfg.HAXSEL_POOL_ROOT else config.STAGED_ENSEMBLE_DIR
    path = root / scfg.haxsel_pool_slug(draw) / "hazard_image.npz"
    if not path.exists():
        raise SystemExit(f"[haxsel] pool hazard image missing: {path}")
    return path


def _workers() -> int:
    if scfg.HAXSEL_WORKERS > 0:
        return scfg.HAXSEL_WORKERS
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def _build_kwargs(clusters: list, candidate: list, draw: int) -> dict:
    lo, hi = scfg.HAXSEL_BOUND_PCT
    return {"clusters": clusters, "candidate": candidate, "lo_pct": lo, "hi_pct": hi,
            "tail_pct": scfg.HAXSEL_TAIL_PCT, "cover_subsample": scfg.HAXSEL_COVER_SUBSAMPLE,
            "cover_seed": scfg.HAXSEL_COVER_SEED_BASE + draw, "cover_q": scfg.HAXSEL_COVER_Q,
            "dry_prefixes": scfg.HAXSEL_DRY_PREFIXES, "wet_prefixes": scfg.HAXSEL_WET_PREFIXES}


def _score(paths: dict, clusters: dict, candidate: list, items: list, seeds: list[int], *,
           with_leaders: bool, workers: int, label: str) -> pd.DataFrame:
    frames = []
    for d, path in paths.items():
        t1 = time.time()
        f = hax.run_stage(path, _build_kwargs(clusters[d], candidate, d), items,
                          n_select=scfg.HAXSEL_N, seeds=seeds,
                          far_distance=scfg.HAXSEL_FAR_TARGET_DISTANCE, with_leaders=with_leaders,
                          workers=workers, chunk_size=scfg.HAXSEL_CHUNK, label=f"{label} d{d}")
        f["draw"] = d
        frames.append(f)
        print(f"[haxsel] {label} d{d}: {len(items)} items in {time.time() - t1:.0f} s")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    """Run the stages and write the tables and the summary."""
    t0 = time.time()
    workers = _workers()
    paths = {d: _pool_path(d) for d in scfg.HAXSEL_DRAWS}
    scfg.HAXSEL_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: metrics, clusters and |rho| per draw (full image); the admissible sets.
    clusters, rho, names, sign, is_integer = {}, {}, None, None, None
    for d, path in paths.items():
        m = hax.metric_matrix(path)
        names, sign = list(m["names"]), m["sign"]
        clusters[d] = hax.metric_clusters(m["D"], names, cluster_rho=scfg.HAXSEL_CLUSTER_RHO)
        rho[d] = hax.spearman_abs(m["D"])
        integer = hax.integer_valued(m["D"])
        is_integer = integer if is_integer is None else (is_integer | integer)
        del m
    candidate = ~is_integer
    candidate_names = [n for n, c in zip(names, candidate) if c]
    hax.metric_table(names, sign, clusters, candidate=candidate, is_integer=is_integer,
                     facets=scfg.HAXSEL_FACETS, dry_prefixes=scfg.HAXSEL_DRY_PREFIXES,
                     wet_prefixes=scfg.HAXSEL_WET_PREFIXES).to_csv(
        scfg.haxsel_table_path("metrics"), index=False)
    admissible = hax.enumerate_admissible_sets(
        names, rho, candidate=candidate, sizes=scfg.HAXSEL_SIZES, pair_rho=scfg.HAXSEL_PAIR_RHO,
        min_dry=scfg.HAXSEL_MIN_DRY, min_wet=scfg.HAXSEL_MIN_WET,
        dry_prefixes=scfg.HAXSEL_DRY_PREFIXES, wet_prefixes=scfg.HAXSEL_WET_PREFIXES)
    counts = admissible.groupby("n").size().to_dict()
    admissible.drop(columns=["cols"]).to_csv(scfg.haxsel_table_path("admissible_sets"), index=False)
    print(f"[haxsel] candidate axes {len(candidate_names)}/{len(names)}; clusters per draw "
          f"{[len(c) for c in clusters.values()]}; admissible sets {counts} "
          f"({len(admissible)} total); workers={workers}")

    # Stage 2: every admissible set on the full pools of both draws, and the random null.
    seeds = [scfg.HAXSEL_SEED_BASE + j for j in range(scfg.HAXSEL_SEEDS)]
    items = list(zip(admissible["set_id"], admissible["cols"]))
    wrapper = _score(paths, clusters, candidate_names, items, seeds, with_leaders=False,
                     workers=workers, label="score")
    null_seeds = [scfg.HAXSEL_NULL_SEED_BASE + j for j in range(scfg.HAXSEL_NULL_DRAWS)]
    null_items = [("__null__", ())] * 1
    null = _score(paths, clusters, candidate_names, null_items, null_seeds, with_leaders=False,
                  workers=1, label="null")
    null.to_csv(scfg.haxsel_table_path("null_scores"), index=False)
    hax.null_reference(null).to_csv(scfg.haxsel_table_path("null_reference"), index=False)
    wrapper = wrapper.merge(admissible[["set_id", "n"]], on="set_id")
    ranking = hax.rank_sets(wrapper, admissible, se_multiplier=scfg.HAXSEL_TIE_SE_MULTIPLIER)

    # Stage 3: the leading sets: nearest-member n_eff and the full-pool covering distance.
    top = ranking.head(scfg.HAXSEL_LEADERS)["set_id"]
    lead = admissible.loc[admissible["set_id"].isin(top)]
    extra = _score(paths, clusters, candidate_names, list(zip(lead["set_id"], lead["cols"])), seeds,
                   with_leaders=True, workers=workers, label="leaders")
    cols = ["set_id", "draw", "seed", "ess_over_n", "cover_full_q", "cover_full_max", "cover_full_mean"]
    wrapper = wrapper.merge(extra[cols], on=["set_id", "draw", "seed"], how="left")
    wrapper.to_csv(scfg.haxsel_table_path("wrapper_scores"), index=False)

    ranking = hax.rank_sets(wrapper, admissible, se_multiplier=scfg.HAXSEL_TIE_SE_MULTIPLIER)
    adopted = hax.resolve_adoption(ranking)
    ranking.to_csv(scfg.haxsel_table_path("ranking"), index=False)
    hax.decision_sensitivity(wrapper, admissible, se_multiplier=scfg.HAXSEL_TIE_SE_MULTIPLIER,
                             adopted=adopted).to_csv(scfg.haxsel_table_path("decision_sensitivity"),
                                                     index=False)
    hax.draw_agreement(ranking, adopted=adopted).to_csv(scfg.haxsel_table_path("draw_agreement"),
                                                        index=False)
    leaders, checks = hax.joint_check(ranking, wrapper, null, adopted=adopted, top=scfg.HAXSEL_LEADERS)
    leaders.to_csv(scfg.haxsel_table_path("leaders"), index=False)
    checks.to_csv(scfg.haxsel_table_path("joint_check"), index=False)

    # Stage 4: the adopted set's members and the pool quantiles (figure inputs).
    adopted_cols = admissible.loc[admissible["set_id"] == adopted, "cols"].iloc[0]
    for d, path in paths.items():
        cache = hax.build_pool_cache(path, **_build_kwargs(clusters[d], candidate_names, d))
        members, quantiles = hax.adopted_selection_tables(
            cache, adopted_cols, n_select=scfg.HAXSEL_N, seed=scfg.HAXSEL_SEED_BASE)
        members.to_csv(scfg.haxsel_table_path(f"adopted_members_d{d}"), index=False)
        quantiles.to_csv(scfg.haxsel_table_path(f"pool_quantiles_d{d}"), index=False)
        del cache

    lead_row = ranking.iloc[0]
    row = ranking.loc[ranking["adopted"]].iloc[0]
    tail_rank = int((ranking["tail_cluster_mean"] > row["tail_cluster_mean"]).sum()) + 1
    summary = {
        "adopted": {"set_id": adopted, "n": int(row["n"]), "members": row["members"].split("|"),
                    "facets": [scfg.HAXSEL_FACETS.get(m, "") for m in row["members"].split("|")],
                    "mean": float(row["mean"]), "se": float(row["se"]), "rank": int(row["rank"]),
                    "tail_share_rank": tail_rank},
        "leader": {"set_id": lead_row["set_id"], "n": int(lead_row["n"]),
                   "mean": float(lead_row["mean"]), "se": float(lead_row["se"])},
        "tie_group": ranking.loc[ranking["tie_group"], "set_id"].tolist(),
        "rule": ("rank admissible sets by the cluster-averaged Kolmogorov-Smirnov distance of the "
                 "selected marginals to uniform over the continuous metrics, pooled over draws and "
                 "seeds (lower is better); within the leader's equivalence class "
                 f"({scfg.HAXSEL_TIE_SE_MULTIPLIER:g} standard errors of the difference) adopt the "
                 "smallest set, then the lowest estimate"),
        "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(scfg).items()
                     if k.startswith("HAXSEL_") and isinstance(v, (int, float, str, tuple))},
        "pools": {d: str(p) for d, p in paths.items()},
        "candidate_axes": candidate_names,
        "integer_valued": [n for n, c in zip(names, is_integer) if c],
        "clusters": {d: c for d, c in clusters.items()},
        "n_admissible": {int(k): int(v) for k, v in counts.items()},
        "workers": workers, "wall_seconds": round(time.time() - t0, 1),
    }
    scfg.haxsel_json_path("summary").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[haxsel] adopted {adopted} (n={row['n']}): {row['members']} "
          f"KS {row['mean']:.4f} +/- {row['se']:.4f}; leader {lead_row['set_id']} "
          f"{lead_row['mean']:.4f}; tie group {len(summary['tie_group'])}; "
          f"{summary['wall_seconds']} s")


if __name__ == "__main__":
    main()
