"""
tests/test_hazard_axis_selection.py - Pure checks of the hazard-axis selection.

Families, facets and cluster weights, the integer-valued candidacy rule, the
admissible-set enumeration (no pair at or above the rank-correlation level on
any draw, the dry/wet minimum), the Kolmogorov-Smirnov distance to uniform,
the coverage statistics of a selection, the identity of the cached-tree
selection with the campaign selector, the random null, the pre-stated tie
rule (lower is better), and the sensitivity, draw-agreement and joint-check
tables. Small synthetic inputs; no staged data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree
from scipy.stats import kstest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scengen import subsample as ss  # noqa: E402

import supplemental_config as scfg  # noqa: E402
from src import hazard_axis_selection as hax  # noqa: E402

DRY, WET = scfg.HAXSEL_DRY_PREFIXES, scfg.HAXSEL_WET_PREFIXES
NAMES = ["drought_a", "drought_b", "lowflow_c", "flood_d", "flood_e", "highflow_f", "annual_cv"]
CLUSTERS = [["drought_a", "drought_b"], ["lowflow_c"], ["flood_d"], ["flood_e"],
            ["highflow_f"], ["annual_cv"]]


def _rho(pairs: dict) -> np.ndarray:
    """|rho| matrix from ``{(i, j): rho}``; other pairs 0."""
    k = len(NAMES)
    r = np.eye(k)
    for (i, j), v in pairs.items():
        r[i, j] = r[j, i] = v
    return r


def _cache(n: int = 400, seed: int = 0) -> tuple:
    """A synthetic cache on the toy metrics; the last metric low-is-hazardous."""
    rng = np.random.default_rng(seed)
    D = rng.gamma(2.0, 1.0, size=(n, len(NAMES)))
    sign = np.array([1.0] * (len(NAMES) - 1) + [-1.0])
    flags = (rng.random((n, 2)) < 0.3).astype(float)
    X = ss.minmax_normalize(D)
    cont = np.arange(len(NAMES))
    rows = np.sort(rng.choice(n, size=100, replace=False))
    cache = hax.PoolCache(
        P=n, names=NAMES, sign=sign, X=X, cont=cont, Wc=hax.cluster_weights(CLUSTERS, NAMES),
        cluster_family=hax.cluster_families(CLUSTERS, NAMES, dry_prefixes=DRY, wet_prefixes=WET),
        X_cover=np.ascontiguousarray(X[rows]), tail=hax.hazard_tail_mask(D, sign, pct=90.0),
        limit=hax.signed_exact_snap_limit(D, sign, tail_pct=90.0, lo_pct=1.0, hi_pct=99.0),
        W=hax.cluster_weights(CLUSTERS, NAMES), flags=flags, flag_names=["onset", "termination"],
        cover_q=99.0,
    )
    return cache, D


def test_metric_family_by_prefix():
    fam = {n: hax.metric_family(n, dry_prefixes=DRY, wet_prefixes=WET) for n in NAMES}
    assert fam["drought_a"] == fam["lowflow_c"] == "dry"
    assert fam["flood_d"] == fam["highflow_f"] == "wet"
    assert fam["annual_cv"] == "regime"
    assert hax.cluster_families(CLUSTERS, NAMES, dry_prefixes=DRY, wet_prefixes=WET) == [
        "dry", "dry", "wet", "wet", "wet", "regime"]


def test_integer_valued_detects_whole_number_columns():
    D = np.column_stack([np.arange(5.0), np.arange(5.0) + 0.5, np.zeros(5)])
    np.testing.assert_array_equal(hax.integer_valued(D), [True, False, True])


def test_cluster_weights_average_within_clusters_and_drop_empty_clusters():
    W = hax.cluster_weights(CLUSTERS, NAMES)
    assert W.shape == (7, 6)
    np.testing.assert_allclose(W.sum(axis=0), 1.0)
    values = np.array([0.4, 0.2, 0.1, 0.3, 0.3, 0.1, 0.1])
    np.testing.assert_allclose(values @ W, [0.3, 0.1, 0.3, 0.3, 0.1, 0.1])
    # Restricted to a subset of metrics, clusters without a member drop out.
    Wc = hax.cluster_weights(CLUSTERS, ["drought_a", "flood_d", "annual_cv"])
    assert Wc.shape == (3, 3)


def test_ks_to_uniform_matches_scipy_and_is_zero_on_the_grid():
    rng = np.random.default_rng(3)
    X = rng.random((50, 3))
    ours = hax.ks_to_uniform(X)
    ref = [kstest(X[:, j], "uniform").statistic for j in range(3)]
    np.testing.assert_allclose(ours, ref, atol=1e-12)
    grid = (np.arange(1, 11) - 0.5) / 10
    assert hax.ks_to_uniform(grid[:, None])[0] == pytest.approx(0.05)
    assert hax.ks_to_uniform(np.full((10, 1), 0.9))[0] == pytest.approx(0.9)


def test_enumeration_pairwise_rule_and_every_draw():
    cand = np.ones(len(NAMES), dtype=bool)
    kw = dict(candidate=cand, sizes=(4,), pair_rho=0.7, min_dry=2, min_wet=2,
              dry_prefixes=DRY, wet_prefixes=WET)
    none = hax.enumerate_admissible_sets(NAMES, {0: _rho({})}, **kw)
    assert len(none) == 9 and (none["n_dry"] == 2).all() and (none["n_wet"] == 2).all()
    one = hax.enumerate_admissible_sets(NAMES, {0: _rho({(0, 1): 0.85})}, **kw)
    assert len(one) == 6
    two = hax.enumerate_admissible_sets(NAMES, {0: _rho({(0, 1): 0.85}), 1: _rho({(3, 4): 0.7})}, **kw)
    assert len(two) == 4
    assert set(two.columns) >= {"set_id", "n", "members", "cols", "max_abs_rho__d0", "max_abs_rho__d1"}
    five = hax.enumerate_admissible_sets(NAMES, {0: _rho({})}, **{**kw, "sizes": (5,)})
    assert len(five) == 15 and five["set_id"].str.startswith("n5_").all()
    cand[0] = False
    fewer = hax.enumerate_admissible_sets(NAMES, {0: _rho({})}, **{**kw, "candidate": cand})
    assert len(fewer) == 3 and not fewer["members"].str.contains("drought_a").any()


def test_signed_exact_snap_limit_low_tail():
    D = np.linspace(0.0, 1.0, 10001)[:, None]
    hi = hax.signed_exact_snap_limit(D, np.array([1.0]), tail_pct=90.0, lo_pct=1.0, hi_pct=99.0)
    lo = hax.signed_exact_snap_limit(D, np.array([-1.0]), tail_pct=90.0, lo_pct=1.0, hi_pct=99.0)
    assert hi[0] == pytest.approx((0.99 - 0.90) / 0.98)
    assert lo[0] == pytest.approx((0.10 - 0.01) / 0.98)


def test_coverage_statistics_and_score_selection_contract():
    cache, D = _cache()
    cols = (0, 2, 4)
    recs = hax.select_and_score(cache, cols, n_select=30, seeds=[0, 1], far_distance=0.25,
                                with_leaders=True)
    assert len(recs) == 2 and [r["seed"] for r in recs] == [0, 1]
    r = recs[0]
    ks = np.array([r[f"cov__{n}"] for n in NAMES])
    assert ((ks >= 0) & (ks <= 1)).all()
    assert r["cov_cluster_mean"] == pytest.approx((ks @ cache.Wc).mean())
    assert r["cov_metric_mean"] == pytest.approx(ks.mean())
    per_cluster = ks @ cache.Wc
    assert r["cov_side_balanced"] == pytest.approx(0.5 * (per_cluster[:2].mean() + per_cluster[2:5].mean()))
    assert r["cov_cluster_max"] == pytest.approx(per_cluster.max())
    assert 0 <= r["cover_q"] <= r["cover_max"] and r["cover_mean"] > 0
    assert r["mst_min"] <= r["mst_mean"]
    assert 0 < r["ess_over_n"] <= 1 and 0 <= r["frac_targets_far"] <= 1
    assert r["cover_full_q"] <= r["cover_full_max"]
    tails = np.array([r[f"tail__{n}"] for n in NAMES])
    assert r["own_tail_min"] == pytest.approx(tails[list(cols)].min())
    assert set(k for k in r if k.startswith("selected__")) == {"selected__onset", "selected__termination"}
    # Own axes are stratified by the Latin hypercube: their KS is small.
    assert ks[list(cols)].max() < 0.2


def test_cached_tree_selection_matches_campaign_selector():
    cache, D = _cache()
    cols = (1, 3, 4)
    X_S = cache.X[:, list(cols)]
    a = ss.lhs_nn_assignment(X_S, 25, seed=3, tree=cKDTree(X_S))
    np.testing.assert_array_equal(np.sort(a.rows), ss.absolute_filling_subsample(D[:, list(cols)], 25, seed=3))


def test_random_null_scores_selector_free_statistics():
    cache, D = _cache()
    recs = hax.score_random_null(cache, n_select=30, seeds=[5, 6, 7])
    assert [r["seed"] for r in recs] == [5, 6, 7]
    assert all(set(hax.NULL_COLUMNS) <= set(r) for r in recs)
    null = pd.DataFrame(recs)
    null["draw"] = 0
    ref = hax.null_reference(null)
    assert set(ref["statistic"]) == set(hax.NULL_COLUMNS) and (ref["n"] == 3).all()


def _meta() -> pd.DataFrame:
    return pd.DataFrame({
        "set_id": ["n4_a", "n4_b", "n4_c", "n5_a", "n5_b"], "n": [4, 4, 4, 5, 5],
        "members": ["a|b|c|d"] * 5, "n_dry": 2, "n_wet": 2, "n_regime": [0, 0, 0, 1, 1],
        "max_abs_rho__d0": 0.5,
    })


def _wrapper(means: dict, sd: float = 0.002, draws=(0, 1), seeds=(0, 1, 2)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for s, m in means.items():
        for d in draws:
            for seed in seeds:
                rows.append({"set_id": s, "draw": d, "seed": seed,
                             "cov_cluster_mean": m + sd * rng.standard_normal(),
                             "cov_metric_mean": m, "cov_side_balanced": m,
                             "cover_q": 1.0 - m, "mst_mean": m, "tail_cluster_mean": 0.3})
    return pd.DataFrame(rows)


def test_tie_rule_lower_is_better_and_prefers_the_smaller_set():
    meta = _meta()
    wrapper = _wrapper({"n5_a": 0.060, "n4_a": 0.0605, "n4_b": 0.08, "n5_b": 0.09, "n4_c": 0.12})
    ranking = hax.rank_sets(wrapper, meta, se_multiplier=2.0)
    assert ranking.iloc[0]["set_id"] == "n5_a" and ranking.iloc[0]["rank"] == 1
    assert set(ranking.loc[ranking["tie_group"], "set_id"]) == {"n5_a", "n4_a"}
    assert hax.resolve_adoption(ranking) == "n4_a"
    wrapper = _wrapper({"n5_a": 0.060, "n4_a": 0.080, "n4_b": 0.09, "n5_b": 0.1, "n4_c": 0.12})
    ranking = hax.rank_sets(wrapper, meta, se_multiplier=2.0)
    assert hax.resolve_adoption(ranking) == "n5_a"


def test_sensitivity_agreement_and_joint_check_tables():
    meta = _meta()
    wrapper = _wrapper({"n4_a": 0.060, "n4_b": 0.080, "n5_a": 0.070, "n5_b": 0.1, "n4_c": 0.12})
    ranking = hax.rank_sets(wrapper, meta, se_multiplier=2.0)
    adopted = hax.resolve_adoption(ranking)
    sens = hax.decision_sensitivity(wrapper, meta, se_multiplier=2.0, adopted=adopted)
    assert set(sens["variant"]) == {"metric_mean", "side_balanced", "draw0_only", "draw1_only"}
    assert sens["equals_adopted"].all()
    agree = hax.draw_agreement(ranking, adopted=adopted)
    checks = dict(zip(agree["check"], agree["value"]))
    assert checks["spearman_between_draws"] > 0.9
    assert checks["adopted_rank_d0"] == 1 and checks["adopted_rank_d1"] == 1
    null = pd.concat([
        pd.DataFrame({c: 0.3 + 0.01 * np.random.default_rng(1).standard_normal(20)
                      for c in hax.NULL_COLUMNS}).assign(draw=d, seed=range(20), set_id="__null__")
        for d in (0, 1)], ignore_index=True)
    leaders, jc = hax.joint_check(ranking, wrapper, null, adopted=adopted, top=3)
    assert len(leaders) == 3 and {"z_cov_cluster_mean", "z_cover_q", "mst_mean_ratio"} <= set(leaders.columns)
    checks = dict(zip(jc["check"], jc["value"]))
    assert checks["class_members_dominating_adopted_on_joint"] == 0
    assert checks["adopted_z_cov_cluster_mean"] < 0
