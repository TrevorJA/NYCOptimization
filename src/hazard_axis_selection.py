"""hazard_axis_selection.py - Choice of the hazard-filling selection axes.

Chooses which n in {4, 5, 6} of the candidate hazard metrics the hazard-filling
design fills (``docs/notes/methods/hazard_axis_selection.md``). The candidate
axes are the continuous metrics of the 21-metric hazard image (an
integer-valued metric cannot carry the N Latin hypercube strata). An admissible
set holds no two metrics with Spearman ``|rho| >= HAXSEL_PAIR_RHO`` on any
draw and at least ``HAXSEL_MIN_DRY`` dry-side and ``HAXSEL_MIN_WET`` wet-side
metrics. Every admissible set is scored under the campaign selector (``lhs_nn``
at N) on the full candidate ensembles by how uniformly its selection covers
the scaled range of every continuous metric, selected or not: the
Kolmogorov-Smirnov distance of each selected marginal to the uniform
distribution on the metric's p1-p99 range, averaged within each metric cluster
and then over the clusters (lower is better). The joint covering distance of
the pool and the spanning-tree statistics of the selection, both referenced to
random same-N subsets, are reported beside the decision. The decision rule is
pre-stated in the methods note and applied by :func:`rank_sets` and
:func:`resolve_adoption`.

Pure functions plus the worker entry points of the multiprocessing stage; the
thin driver is ``scripts/supplemental/hazard_axis_selection_run.py``. This module
never imports ``config`` (worker processes import it under ``spawn``).
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from scengen import selector_diagnostics as sd
from scengen import subsample as ss
from scengen.diagnostics import load_hazard_image, spearman_clusters
from src.hazard_metric_image import (
    descriptor_image, exact_snap_limit, hazard_tail_mask, nearest_member_ess_ratio,
)

FAMILY_DRY = "dry"
FAMILY_WET = "wet"
FAMILY_REGIME = "regime"

#: The decision statistic's column in the scored tables (lower is better).
DECISION_COLUMN = "cov_cluster_mean"

#: Columns the random null is scored on (selector-free statistics).
NULL_COLUMNS = ("cov_cluster_mean", "cov_metric_mean", "cov_side_balanced", "cov_cluster_max",
                "cover_q", "cover_max", "cover_mean", "mst_mean", "mst_min", "tail_cluster_mean")


###############################################################################
# Metrics, clusters, candidate axes and admissible sets
###############################################################################

def metric_family(name: str, *, dry_prefixes: tuple, wet_prefixes: tuple) -> str:
    """``dry``, ``wet`` or ``regime`` by name prefix."""
    if name.startswith(tuple(dry_prefixes)):
        return FAMILY_DRY
    if name.startswith(tuple(wet_prefixes)):
        return FAMILY_WET
    return FAMILY_REGIME


def metric_matrix(path: Path) -> dict:
    """The 21-metric matrix of a pool hazard image (:func:`descriptor_image`)."""
    img = load_hazard_image(path)
    return descriptor_image(img["H"], list(img["hazard_axes"]), img["supplement"],
                            img["supplement_names"])


def integer_valued(D: np.ndarray) -> np.ndarray:
    """``(k,)`` bool: metrics whose every value is a whole number."""
    return np.all(np.mod(np.asarray(D, dtype=float), 1.0) == 0.0, axis=0)


def spearman_abs(D: np.ndarray) -> np.ndarray:
    """``(k, k)`` absolute Spearman rank correlations."""
    return np.abs(np.atleast_2d(spearmanr(np.asarray(D, dtype=float)).statistic))


def metric_clusters(D: np.ndarray, names: list[str], *, cluster_rho: float) -> list[list[str]]:
    """Average-linkage Spearman clusters of the non-constant metrics.

    The partition of ``descriptor_redundancy`` (same call, without its PCA).

    Returns:
        List of clusters, each a list of metric names.
    """
    D = np.asarray(D, dtype=float)
    varying = D.std(axis=0) > 0
    kept = [n for n, v in zip(names, varying) if v]
    return spearman_clusters(D[:, varying], kept, threshold=cluster_rho)["clusters"]


def cluster_ids(clusters: list[list[str]], names: list[str]) -> np.ndarray:
    """Cluster index of every metric (``-1`` for a metric in no cluster)."""
    lookup = {m: c for c, members in enumerate(clusters) for m in members}
    return np.array([lookup.get(n, -1) for n in names], dtype=int)


def cluster_weights(clusters: list[list[str]], names: list[str]) -> np.ndarray:
    """``(k, C)`` matrix mapping per-metric values to per-cluster means.

    ``values @ W`` is the vector of cluster means over the metrics in ``names``;
    a cluster with no member in ``names`` is dropped, and a metric outside
    every cluster carries no weight.
    """
    kept = [[m for m in c if m in names] for c in clusters]
    kept = [c for c in kept if c]
    ids = cluster_ids(kept, names)
    W = np.zeros((len(names), len(kept)))
    for j, c in enumerate(ids):
        if c >= 0:
            W[j, c] = 1.0 / len(kept[c])
    return W


def cluster_families(clusters: list[list[str]], names: list[str], *, dry_prefixes: tuple,
                     wet_prefixes: tuple) -> list[str]:
    """Family of each cluster that has a member in ``names`` (clusters are family-pure)."""
    kept = [[m for m in c if m in names] for c in clusters]
    return [metric_family(c[0], dry_prefixes=dry_prefixes, wet_prefixes=wet_prefixes)
            for c in kept if c]


def metric_table(
    names: list[str], sign: np.ndarray, clusters_by_draw: dict, *, candidate: np.ndarray,
    is_integer: np.ndarray, facets: dict, dry_prefixes: tuple, wet_prefixes: tuple,
) -> pd.DataFrame:
    """One row per metric: family, facet, direction, candidacy, cluster id per draw."""
    ids = {d: cluster_ids(cl, names) for d, cl in clusters_by_draw.items()}
    return pd.DataFrame.from_records([{
        "metric": name,
        "family": metric_family(name, dry_prefixes=dry_prefixes, wet_prefixes=wet_prefixes),
        "facet": facets.get(name, ""),
        "hazard_direction": int(sign[j]),
        "integer_valued": bool(is_integer[j]),
        "candidate_axis": bool(candidate[j]),
        **{f"cluster_id__d{d}": int(ids[d][j]) for d in clusters_by_draw},
    } for j, name in enumerate(names)])


def enumerate_admissible_sets(
    names: list[str], rho_by_draw: dict, *, candidate: np.ndarray, sizes: tuple,
    pair_rho: float, min_dry: int, min_wet: int, dry_prefixes: tuple, wet_prefixes: tuple,
) -> pd.DataFrame:
    """Every set of candidate axes admissible on every draw.

    A set is admissible when its size is in ``sizes``, no two of its metrics
    have ``|rho_S| >= pair_rho`` on any draw, and it holds at least ``min_dry``
    dry-side and ``min_wet`` wet-side metrics.

    Returns:
        One row per set: ``set_id`` (``n{n}_{i}`` with ``i`` the position in
        the combinations order over the candidate axes), ``n``, ``members``
        (``|``-joined in ``names`` order), ``cols`` (column tuple), ``n_dry``,
        ``n_wet``, ``n_regime`` and ``max_abs_rho__d{k}`` per draw.
    """
    fam = [metric_family(n, dry_prefixes=dry_prefixes, wet_prefixes=wet_prefixes) for n in names]
    cand = [j for j in range(len(names)) if candidate[j]]
    rows = []
    for n in sizes:
        for i, combo in enumerate(itertools.combinations(cand, n)):
            n_dry = sum(fam[j] == FAMILY_DRY for j in combo)
            n_wet = sum(fam[j] == FAMILY_WET for j in combo)
            if n_dry < min_dry or n_wet < min_wet:
                continue
            sub = np.ix_(combo, combo)
            maxes = {d: float(np.max(rho[sub] - np.eye(n))) for d, rho in rho_by_draw.items()}
            if any(v >= pair_rho for v in maxes.values()):
                continue
            rows.append({
                "set_id": f"n{n}_{i:05d}", "n": n,
                "members": "|".join(names[j] for j in combo), "cols": tuple(combo),
                "n_dry": n_dry, "n_wet": n_wet, "n_regime": n - n_dry - n_wet,
                **{f"max_abs_rho__d{d}": v for d, v in maxes.items()},
            })
    return pd.DataFrame.from_records(rows)


###############################################################################
# Pool cache and per-selection scoring
###############################################################################

def ks_to_uniform(X: np.ndarray) -> np.ndarray:
    """Kolmogorov-Smirnov distance of every column of ``X`` to uniform on [0, 1].

    Args:
        X: ``(n, k)`` values in the unit interval (clipped scaled metrics).

    Returns:
        ``(k,)`` distances ``sup |F_n(x) - x|``.
    """
    Xs = np.sort(np.asarray(X, dtype=float), axis=0)
    n = Xs.shape[0]
    grid = np.arange(1, n + 1)[:, None] / n
    return np.maximum(grid - Xs, Xs - (grid - 1.0 / n)).max(axis=0)


def signed_exact_snap_limit(
    D: np.ndarray, sign: np.ndarray, *, tail_pct: float, lo_pct: float, hi_pct: float,
) -> np.ndarray:
    """Exact-snap tail-share limit in each metric's hazardous direction."""
    return exact_snap_limit(np.asarray(D, dtype=float) * np.asarray(sign, dtype=float),
                            tail_pct=tail_pct, lo_pct=lo_pct, hi_pct=hi_pct)


@dataclass(frozen=True)
class PoolCache:
    """One pool prepared once for repeated selections.

    Attributes:
        P: Rows.
        names: The 21 metrics (columns of ``X``).
        sign: ``+1`` where high values are hazardous, ``-1`` where low.
        X: ``(P, k)`` metrics scaled to the unit box by the robust bounds.
        cont: Column indices of the continuous metrics (the common space).
        Wc: ``(k_cont, C)`` cluster weights over the continuous metrics.
        cluster_family: Family of each of those ``C`` clusters.
        X_cover: ``(S, k_cont)`` fixed i.i.d. subsample of ``X[:, cont]`` on which
            the covering distance of every selection is measured.
        tail: ``(P, k)`` bool hazard-direction tail mask at ``tail_pct``.
        limit: ``(k,)`` signed exact-snap limits; ``W`` cluster weights over all
            metrics (the tail-share profile).
        flags: ``(P, 2)`` truncation flags; ``flag_names``.
        cover_q: Percentile of the covering-distance distribution reported.
    """

    P: int
    names: list
    sign: np.ndarray
    X: np.ndarray
    cont: np.ndarray
    Wc: np.ndarray
    cluster_family: list
    X_cover: np.ndarray
    tail: np.ndarray
    limit: np.ndarray
    W: np.ndarray
    flags: np.ndarray
    flag_names: list
    cover_q: float


def build_pool_cache(
    path: Path, *, clusters: list[list[str]], candidate: list, lo_pct: float, hi_pct: float,
    tail_pct: float, cover_subsample: int, cover_seed: int, cover_q: float,
    dry_prefixes: tuple, wet_prefixes: tuple, prefix_p: int = 0,
) -> PoolCache:
    """Load an image (its leading ``prefix_p`` rows when set), scale and mask it."""
    m = metric_matrix(Path(path))
    D, flags = m["D"], m["flags"]
    if prefix_p:
        D, flags = D[:prefix_p], flags[:prefix_p]
    names, sign = list(m["names"]), np.asarray(m["sign"], dtype=float)
    X = ss.minmax_normalize(D, lo_pct=lo_pct, hi_pct=hi_pct)
    cont = np.array([j for j, n in enumerate(names) if n in set(candidate)], dtype=int)
    cont_names = [names[j] for j in cont]
    rng = np.random.default_rng(cover_seed)
    rows = np.sort(rng.choice(len(X), size=min(cover_subsample, len(X)), replace=False))
    return PoolCache(
        P=int(len(D)), names=names, sign=sign, X=X, cont=cont,
        Wc=cluster_weights(clusters, cont_names),
        cluster_family=cluster_families(clusters, cont_names, dry_prefixes=dry_prefixes,
                                        wet_prefixes=wet_prefixes),
        X_cover=np.ascontiguousarray(X[rows][:, cont]),
        tail=hazard_tail_mask(D, sign, pct=tail_pct),
        limit=signed_exact_snap_limit(D, sign, tail_pct=tail_pct, lo_pct=lo_pct, hi_pct=hi_pct),
        W=cluster_weights(clusters, names),
        flags=np.asarray(flags, dtype=float), flag_names=list(m["flag_names"]), cover_q=cover_q,
    )


def coverage_statistics(cache: PoolCache, rows: np.ndarray, *, X_cover: np.ndarray | None = None,
                        workers: int = 1) -> dict:
    """Selector-free coverage statistics of a selection in the common space.

    Returns:
        ``cov__{metric}`` (KS distance to uniform on each continuous metric's
        scaled range), ``cov_cluster_mean`` (the decision statistic),
        ``cov_metric_mean``, ``cov_side_balanced`` (mean of the dry- and
        wet-cluster means), ``cov_cluster_max``; ``cover_q``/``cover_max``/
        ``cover_mean`` (distance from the pool subsample, or ``X_cover``, to its
        nearest selected member); ``mst_mean``/``mst_min`` of the selection;
        ``tail__{metric}`` on all 21 metrics and ``tail_cluster_mean``.
    """
    rows = np.asarray(rows, dtype=int)
    Xc = cache.X[rows][:, cache.cont]
    ks = ks_to_uniform(Xc)
    per_cluster = ks @ cache.Wc
    fam = np.asarray(cache.cluster_family)
    sides = [per_cluster[fam == f].mean() for f in (FAMILY_DRY, FAMILY_WET) if (fam == f).any()]
    d_cover, _ = cKDTree(Xc).query(cache.X_cover if X_cover is None else X_cover, k=1,
                                   workers=workers)
    mst = sd._mst_edge_stats(Xc)
    tails = cache.tail[rows].mean(axis=0)
    cont_names = [cache.names[j] for j in cache.cont]
    return {
        "cov_cluster_mean": float(per_cluster.mean()), "cov_metric_mean": float(ks.mean()),
        "cov_side_balanced": float(np.mean(sides)), "cov_cluster_max": float(per_cluster.max()),
        **{f"cov__{n}": float(v) for n, v in zip(cont_names, ks)},
        "cover_q": float(np.percentile(d_cover, cache.cover_q)),
        "cover_max": float(d_cover.max()), "cover_mean": float(d_cover.mean()),
        "mst_mean": float(mst["mst_edge_mean"]), "mst_min": float(mst["mst_edge_min"]),
        "tail_cluster_mean": float((tails @ cache.W).mean()),
        **{f"tail__{n}": float(v) for n, v in zip(cache.names, tails)},
    }


def score_selection(
    cache: PoolCache, cols: tuple, a: ss.LhsNnAssignment, *, X_S: np.ndarray, tree: cKDTree,
    far_distance: float, with_leaders: bool, X_cover_full: np.ndarray | None = None,
    ess_workers: int = 1,
) -> dict:
    """Statistics of one selection: coverage (decision and reports) plus the selector's own.

    Adds to :func:`coverage_statistics` the own-axis minimum tail share and
    attainment, ``displacement_mean``, ``frac_targets_far``, ``n_fallback``,
    the selected members' ``selected__{flag}`` fractions and, when
    ``with_leaders``, ``ess_over_n`` and the covering distance on the full
    pool (``cover_full_q``/``cover_full_max``/``cover_full_mean``).
    """
    rows = np.asarray(a.rows, dtype=int)
    cols = list(cols)
    rec = coverage_statistics(cache, rows, workers=ess_workers)
    own = np.array([rec[f"tail__{cache.names[j]}"] for j in cols])
    lim = cache.limit[cols]
    att = np.divide(own, lim, out=np.full_like(own, np.nan), where=lim > 0)
    d_near, _ = tree.query(a.targets, k=1)
    rec.update({
        "own_tail_min": float(own.min()), "own_attainment_min": float(np.nanmin(att)),
        "displacement_mean": float(np.mean(a.displacement)), "n_fallback": int(a.n_fallback),
        "frac_targets_far": float(np.mean(d_near > far_distance)),
        **{f"selected__{f}": float(v)
           for f, v in zip(cache.flag_names, cache.flags[rows].mean(axis=0))},
    })
    if with_leaders:
        rec["ess_over_n"] = nearest_member_ess_ratio(X_S, rows, workers=ess_workers)
        full = X_cover_full if X_cover_full is not None else np.ascontiguousarray(cache.X[:, cache.cont])
        d_full, _ = cKDTree(cache.X[rows][:, cache.cont]).query(full, k=1, workers=ess_workers)
        rec.update({"cover_full_q": float(np.percentile(d_full, cache.cover_q)),
                    "cover_full_max": float(d_full.max()), "cover_full_mean": float(d_full.mean())})
    return rec


def select_and_score(
    cache: PoolCache, cols: tuple, *, n_select: int, seeds: list[int], far_distance: float,
    with_leaders: bool, ess_workers: int = 1,
) -> list[dict]:
    """One ``lhs_nn`` selection per seed on the metrics ``cols``, scored.

    The kd-tree is built once per set and shared by the seeds.
    """
    X_S = np.ascontiguousarray(cache.X[:, list(cols)])
    tree = cKDTree(X_S)
    full = np.ascontiguousarray(cache.X[:, cache.cont]) if with_leaders else None
    out = []
    for seed in seeds:
        a = ss.lhs_nn_assignment(X_S, n_select, seed=seed, tree=tree)
        rec = score_selection(cache, cols, a, X_S=X_S, tree=tree, far_distance=far_distance,
                              with_leaders=with_leaders, X_cover_full=full, ess_workers=ess_workers)
        rec["seed"] = int(seed)
        out.append(rec)
    return out


def score_random_null(cache: PoolCache, *, n_select: int, seeds: list[int]) -> list[dict]:
    """Coverage statistics of random ``n_select``-subsets of the pool, one per seed."""
    out = []
    for seed in seeds:
        rows = ss.random_subsample(cache.X, n_select, seed=seed)
        rec = coverage_statistics(cache, rows)
        rec["seed"] = int(seed)
        out.append(rec)
    return out


def adopted_selection_tables(
    cache: PoolCache, cols: tuple, *, n_select: int, seed: int, quantile_points: int = 101,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The adopted set's selected members and the pool's scaled quantiles (for figures).

    Returns:
        ``members``: one row per selected member (``member_row``, ``target_index``,
        ``displacement`` and the scaled value of every metric); ``quantiles``:
        the pool's scaled values at ``quantile_points`` evenly spaced probability
        levels for every metric (``level`` plus one column per metric).
    """
    X_S = np.ascontiguousarray(cache.X[:, list(cols)])
    a = ss.lhs_nn_assignment(X_S, n_select, seed=seed, tree=cKDTree(X_S))
    members = pd.DataFrame(cache.X[a.rows], columns=cache.names)
    members.insert(0, "displacement", a.displacement)
    members.insert(0, "target_index", np.arange(len(a.rows)))
    members.insert(0, "member_row", a.rows)
    levels = np.linspace(0.0, 1.0, quantile_points)
    quantiles = pd.DataFrame(np.percentile(cache.X, 100.0 * levels, axis=0), columns=cache.names)
    quantiles.insert(0, "level", levels)
    return members, quantiles


###############################################################################
# Stage runner (in-process or worker processes)
###############################################################################

_CACHE: PoolCache | None = None


def _init_worker(path: str, build_kwargs: dict) -> None:
    """Build the pool cache once per process."""
    global _CACHE
    _CACHE = build_pool_cache(Path(path), **build_kwargs)


def _score_chunk(args: tuple) -> list[dict]:
    """Score a chunk of ``(set_id, cols)`` items on the process cache."""
    items, n_select, seeds, far_distance, with_leaders = args
    out = []
    for set_id, cols in items:
        if set_id == "__null__":
            recs = score_random_null(_CACHE, n_select=n_select, seeds=seeds)
        else:
            recs = select_and_score(_CACHE, cols, n_select=n_select, seeds=seeds,
                                    far_distance=far_distance, with_leaders=with_leaders,
                                    ess_workers=1)
        for rec in recs:
            rec["set_id"] = set_id
            out.append(rec)
    return out


def run_stage(
    path: Path, build_kwargs: dict, items: list, *, n_select: int, seeds: list[int],
    far_distance: float, with_leaders: bool, workers: int, chunk_size: int, label: str = "",
) -> pd.DataFrame:
    """Score every ``(set_id, cols)`` item on one pool, one row per seed.

    An item whose ``set_id`` is ``"__null__"`` scores random subsets instead
    of selections (its ``cols`` are ignored). ``workers <= 1`` runs in-process;
    otherwise a process pool whose workers each build the cache once. Seeds
    are fixed per item, so the result is independent of the worker count.
    """
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    args = [(chunk, n_select, list(seeds), far_distance, with_leaders) for chunk in chunks]
    results: list[list[dict]] = []
    if workers <= 1:
        _init_worker(str(path), build_kwargs)
        results = [_score_chunk(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(str(path), build_kwargs)) as ex:
            futures = [ex.submit(_score_chunk, a) for a in args]
            marks = {int(len(futures) * q) for q in (0.25, 0.5, 0.75)}
            for k, fut in enumerate(as_completed(futures), start=1):
                results.append(fut.result())
                if k in marks:
                    print(f"[haxsel] {label} {k}/{len(futures)} chunks")
    return pd.DataFrame.from_records([r for chunk in results for r in chunk])


###############################################################################
# Ranking, the decision rule and the reports
###############################################################################

def rank_sets(
    wrapper: pd.DataFrame, meta: pd.DataFrame, *, se_multiplier: float,
    column: str = DECISION_COLUMN, ascending: bool = True,
) -> pd.DataFrame:
    """Pool the replicates per set, rank, and mark the leader's equivalence class.

    Args:
        wrapper: One row per (set, draw, seed) with the scored columns.
        meta: One row per set with ``set_id``, ``n``, ``members`` and counts.
        se_multiplier: Sets within this many standard errors of the difference
            from the leader form the ``tie_group``.
        column: The statistic ranked.
        ascending: True when lower values are better (the decision statistic).

    Returns:
        One row per set sorted by rank: ``mean``, ``sd``, ``se``, ``n_rep``,
        ``rank``, ``tie_group``, per-draw means ``mean__d{d}``, and the pooled
        means of the other scored columns.
    """
    g = wrapper.groupby("set_id")
    pooled = pd.DataFrame({
        "mean": g[column].mean(), "sd": g[column].std(ddof=1), "n_rep": g[column].size(),
    })
    pooled["se"] = pooled["sd"] / np.sqrt(pooled["n_rep"])
    for d, gd in wrapper.groupby("draw"):
        pooled[f"mean__d{d}"] = gd.groupby("set_id")[column].mean()
    numeric = [c for c in wrapper.columns
               if c not in ("set_id", "draw", "seed", "n", column)
               and pd.api.types.is_numeric_dtype(wrapper[c])]
    for c in numeric:
        pooled[c] = g[c].mean()
    keep = [c for c in ("n", "members", "n_dry", "n_wet", "n_regime") if c in meta.columns]
    keep += [c for c in meta.columns if c.startswith("max_abs_rho__")]
    out = meta.set_index("set_id")[keep].join(pooled, how="inner").reset_index()
    out = out.sort_values(["mean", "n", "set_id"], ascending=[ascending, True, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    lead = out.iloc[0]
    se_diff = np.sqrt(out["se"] ** 2 + lead["se"] ** 2)
    gap = (out["mean"] - lead["mean"]) if ascending else (lead["mean"] - out["mean"])
    out["tie_group"] = gap <= se_multiplier * se_diff
    out.attrs["ascending"] = ascending
    return out


def resolve_adoption(ranking: pd.DataFrame) -> str:
    """The adopted set: smallest size in the leader's class, then the best estimate.

    Marks the ``adopted`` column and returns its ``set_id``.
    """
    ascending = ranking.attrs.get("ascending", True)
    cls = ranking.loc[ranking["tie_group"]]
    pick = cls.sort_values(["n", "mean"], ascending=[True, ascending]).iloc[0]["set_id"]
    ranking["adopted"] = ranking["set_id"] == pick
    return str(pick)


def decision_sensitivity(
    wrapper: pd.DataFrame, meta: pd.DataFrame, *, se_multiplier: float, adopted: str,
) -> pd.DataFrame:
    """The winner under each variant of the rule, against the adopted set.

    Variants: metric votes instead of cluster votes; dry- and wet-side
    balanced cluster votes; each draw alone.
    """
    variants = {"metric_mean": (wrapper, "cov_metric_mean"),
                "side_balanced": (wrapper, "cov_side_balanced")}
    for d in sorted(wrapper["draw"].unique()):
        variants[f"draw{d}_only"] = (wrapper.loc[wrapper["draw"] == d], DECISION_COLUMN)
    rows = []
    for name, (frame, column) in variants.items():
        r = rank_sets(frame, meta, se_multiplier=se_multiplier, column=column, ascending=True)
        winner = resolve_adoption(r)
        w = r.loc[r["set_id"] == winner].iloc[0]
        rows.append({"variant": name, "winner": winner, "n": int(w["n"]), "members": w["members"],
                     "winner_mean": float(w["mean"]), "equals_adopted": bool(winner == adopted),
                     "tie_group_size": int(r["tie_group"].sum())})
    return pd.DataFrame.from_records(rows)


def null_reference(null: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of every null-scored column per draw (``statistic``, ``draw``, ``mean``, ``sd``)."""
    rows = []
    for d, g in null.groupby("draw"):
        for c in NULL_COLUMNS:
            if c in g.columns:
                rows.append({"statistic": c, "draw": int(d), "mean": float(g[c].mean()),
                             "sd": float(g[c].std(ddof=1)), "n": int(len(g))})
    return pd.DataFrame.from_records(rows)


def joint_check(
    ranking: pd.DataFrame, wrapper: pd.DataFrame, null: pd.DataFrame, *, adopted: str, top: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Joint-coverage reports beside the decision.

    Returns:
        ``(leaders, checks)``: per leading set the pooled z-scores of the
        decision statistic and of the covering distance (full pool where
        measured, else the subsample) against the random null, the spanning-tree
        mean edge as a ratio to the null, and the tail-share profile mean; and
        a long table of checks: the Spearman rank correlation between the
        decision statistic and the covering distance over all sets, and whether
        any member of the equivalence class dominates the adopted set on the
        covering distance and the spanning-tree mean edge.
    """
    ref = null_reference(null).set_index(["statistic", "draw"])

    def z(column: str, null_col: str) -> pd.Series:
        parts = []
        for d, g in wrapper.groupby("draw"):
            m, s = ref.loc[(null_col, int(d)), ["mean", "sd"]]
            parts.append((g.groupby("set_id")[column].mean() - m) / s)
        return pd.concat(parts, axis=1).mean(axis=1)

    cover_col = "cover_full_q" if "cover_full_q" in wrapper.columns else "cover_q"
    lead = ranking.head(top).copy()
    zc, zq = z(DECISION_COLUMN, "cov_cluster_mean"), z("cover_q", "cover_q")
    ratio = pd.concat([
        g.groupby("set_id")["mst_mean"].mean() / ref.loc[("mst_mean", int(d)), "mean"]
        for d, g in wrapper.groupby("draw")], axis=1).mean(axis=1)
    lead["z_cov_cluster_mean"] = lead["set_id"].map(zc)
    lead["z_cover_q"] = lead["set_id"].map(zq)
    lead["mst_mean_ratio"] = lead["set_id"].map(ratio)
    if cover_col != "cover_q":
        lead["cover_full_q"] = lead["set_id"].map(wrapper.groupby("set_id")[cover_col].mean())
    checks = []
    rho = spearmanr(ranking["mean"], ranking["cover_q"]).statistic
    checks.append({"check": "spearman_cov_vs_cover_q", "value": float(rho),
                   "detail": f"over {len(ranking)} admissible sets"})
    rho_m = spearmanr(ranking["mean"], ranking["mst_mean"]).statistic
    checks.append({"check": "spearman_cov_vs_mst_mean", "value": float(rho_m),
                   "detail": f"over {len(ranking)} admissible sets"})
    cls = ranking.loc[ranking["tie_group"]]
    a = cls.loc[cls["set_id"] == adopted].iloc[0]
    others = cls.loc[cls["set_id"] != adopted]
    dominates = others.loc[(others["cover_q"] <= a["cover_q"]) & (others["mst_mean"] >= a["mst_mean"])
                           & ((others["cover_q"] < a["cover_q"]) | (others["mst_mean"] > a["mst_mean"]))]
    checks.append({"check": "class_members_dominating_adopted_on_joint", "value": float(len(dominates)),
                   "detail": "|".join(dominates["set_id"]) if len(dominates) else ""})
    checks.append({"check": "adopted_z_cov_cluster_mean", "value": float(zc[adopted]), "detail": ""})
    checks.append({"check": "adopted_z_cover_q", "value": float(zq[adopted]), "detail": ""})
    checks.append({"check": "adopted_mst_mean_ratio", "value": float(ratio[adopted]), "detail": ""})
    return lead, pd.DataFrame.from_records(checks)


def draw_agreement(ranking: pd.DataFrame, *, adopted: str) -> pd.DataFrame:
    """Agreement of the statistic between draws, one row per check."""
    ascending = ranking.attrs.get("ascending", True)
    draws = sorted(int(c.split("__d")[1]) for c in ranking.columns if c.startswith("mean__d"))
    rows = []

    def add(check: str, value: float, detail: str = "") -> None:
        rows.append({"check": check, "value": float(value), "detail": detail})

    if len(draws) == 2:
        a, b = draws
        add("spearman_between_draws", spearmanr(ranking[f"mean__d{a}"], ranking[f"mean__d{b}"]).statistic,
            f"over {len(ranking)} admissible sets")
        row = ranking.loc[ranking["set_id"] == adopted].iloc[0]
        for d in draws:
            better = (ranking[f"mean__d{d}"] < row[f"mean__d{d}"]) if ascending \
                else (ranking[f"mean__d{d}"] > row[f"mean__d{d}"])
            add(f"adopted_rank_d{d}", int(better.sum()) + 1, "rank of the adopted set on this draw alone")
    add("n_admissible", len(ranking))
    add("tie_group_size", int(ranking["tie_group"].sum()))
    return pd.DataFrame.from_records(rows)
