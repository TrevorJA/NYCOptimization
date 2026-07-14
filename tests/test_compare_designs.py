"""tests/test_compare_designs.py - Cross-design comparison machinery.

Builds a small SYNTHETIC multi-design re-eval tree in ``tmp_path`` -- several
campaign designs x draws x seeds, each with its own raw (S, R, M) matrix -- runs
``src.robustness`` over it to produce the per-run artifacts exactly as workflow
step 08 would, then drives ``scripts/main/compare_designs.run_comparison`` on it.

Asserts that the deliverables are produced with the right shapes:
  1. the satisficing-threshold sweep (one row per run per stringency level, plus
     the registry-default level) and its design x stringency matrix;
  2. the design-ranking rank-agreement curve (Kendall tau_b vs the default);
  3. the cross-design scorecard aggregation + design-ranking stability matrix;
  4. the nested variance components, with the DRAW as the unit of analysis
     (effective n = K, not K*S);
  5. the raw performance distributions, degeneracy flags and pooled attainability.

Also covers the two traps the machinery must not fall into: the stringency knob
must be monotone (a tighter criterion cannot make more realizations satisfice),
and a design with no outputs must warn and be skipped rather than crash.

Run:
    venv/bin/python -m pytest tests/test_compare_designs.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import src.robustness as rob  # noqa: E402
from src.scenario_designs import campaign_designs  # noqa: E402


def _load_compare_designs():
    """Import scripts/main/compare_designs.py (not an importable package path)."""
    path = PROJECT_DIR / "scripts" / "main" / "compare_designs.py"
    spec = importlib.util.spec_from_file_location("compare_designs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_designs"] = mod
    spec.loader.exec_module(mod)
    return mod


cd = _load_compare_designs()


###############################################################################
# Synthetic multi-design re-eval tree
###############################################################################

FORMULATION = "ffmp"
SLUG_BASE = f"{FORMULATION}_obj2"
TAG = "etest_synthetic"
N_SOL, N_REAL = 8, 30
DRAWS, SEEDS = (0, 1), (0, 1)

#: A: maximize / "ge"; B: minimize / "le". Thresholds sit mid-distribution so the
#: stringency sweep spans both saturated and starved ends.
META = {
    "is_ensemble": True,
    "base_names": ["A", "B"],
    "thresholds": {"A": 0.50, "B": 0.50},
    "kinds": {"A": "ge", "B": "le"},
    "directions": {"A": "maximize", "B": "minimize"},
    "realization_indices": list(range(N_REAL)),
}


def _write_run(out_dir: Path, rng: np.random.Generator, quality: float) -> None:
    """Write one run's raw matrix + meta, then score it with src.robustness.

    ``quality`` shifts the design's whole cube, so designs are separable and the
    design ranking is well-defined.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sid in range(N_SOL):
        # A spread across solutions so "best" and "median" solutions differ.
        a_loc = quality + 0.06 * sid
        b_loc = 1.0 - quality - 0.06 * sid
        for rid in range(N_REAL):
            records.append((sid, rid, "A", float(np.clip(rng.normal(a_loc, 0.15), 0, 1))))
            records.append((sid, rid, "B", float(np.clip(rng.normal(b_loc, 0.15), 0, 1))))
    pd.DataFrame(records, columns=["solution_id", "realization_id", "objective",
                                   "value"]).to_csv(
        out_dir / "reeval_raw.csv.gz", index=False, compression="gzip")
    meta = dict(META, solution_ids=list(range(N_SOL)), n_solutions=N_SOL,
                n_realizations=N_REAL)
    (out_dir / "reeval_raw_meta.json").write_text(json.dumps(meta))
    rob.run(out_dir, metrics=("satisficing_multivariate", "satisficing_univariate",
                              "laplace_mean", "maximin"))


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """A 4-design x 2-draw x 2-seed synthetic re-eval tree + the comparison outputs."""
    root = tmp_path_factory.mktemp("outputs")
    designs = campaign_designs()[:4]
    assert len(designs) >= 3, "design-ranking tau_b needs >= 3 campaign designs"
    rng = np.random.default_rng(7)
    for di, design in enumerate(designs):
        for draw in DRAWS:
            slug = SLUG_BASE if draw == 0 else f"{SLUG_BASE}_d{draw}"
            for seed in SEEDS:
                _write_run(
                    root / design / slug / "reeval" / TAG / f"seed_{seed:02d}",
                    rng, quality=0.30 + 0.10 * di + 0.01 * draw,
                )
    table_dir = root / "_comparison"
    fig_dir = root / "_figures"
    written = cd.run_comparison(
        formulation=FORMULATION, reeval_tag=TAG, outputs_root=root,
        table_dir=table_dir, fig_dir=fig_dir,
    )
    return {"root": root, "designs": designs, "written": written,
            "table_dir": table_dir, "fig_dir": fig_dir}


def _read(tree, key) -> pd.DataFrame:
    return pd.read_csv(tree["written"][key])


###############################################################################
# Discovery
###############################################################################

def test_discovery_finds_every_design_draw_seed(tree):
    runs = cd.discover_runs(FORMULATION, TAG, outputs_root=tree["root"])
    assert len(runs) == len(tree["designs"]) * len(DRAWS) * len(SEEDS)
    assert {r.design for r in runs} == set(tree["designs"])
    assert {r.draw for r in runs} == set(DRAWS)      # parsed from the _d{k} slug token
    assert {r.seed for r in runs} == set(SEEDS)      # parsed from the seed_NN subdir


def test_missing_design_warns_and_is_skipped(tree, tmp_path):
    empty = tmp_path / "empty_outputs"
    (empty / tree["designs"][0]).mkdir(parents=True)
    with pytest.warns(UserWarning):
        runs = cd.discover_runs(FORMULATION, TAG, outputs_root=empty)
    assert runs == []


###############################################################################
# 1. Threshold sweep (the deliverable)
###############################################################################

def test_threshold_sweep_shape(tree):
    sweep = _read(tree, "design_threshold_sweep")
    n_runs = len(tree["designs"]) * len(DRAWS) * len(SEEDS)
    assert len(sweep) == n_runs * (len(cd.STRINGENCY_GRID) + 1)   # grid + default
    assert sweep["is_default"].sum() == n_runs
    assert set(sweep.columns) == {"design", "draw", "seed", "stringency",
                                  "is_default", "best", "median"}
    grid = sweep.loc[~sweep["is_default"], "best"]
    assert grid.between(0.0, 1.0).all()


def test_stringency_matrix_shape(tree):
    mat = _read(tree, "design_stringency_matrix")
    # One row per (statistic, design); one column per stringency level.
    assert len(mat) == 2 * len(tree["designs"])
    assert set(mat["statistic"]) == {"best", "median"}
    qcols = [c for c in mat.columns if c not in ("design", "statistic")]
    assert len(qcols) == len(cd.STRINGENCY_GRID)


def test_sweep_is_monotone_in_stringency(tree):
    """A tighter criterion can never let MORE realizations satisfice."""
    mat = _read(tree, "design_stringency_matrix")
    mat = mat[mat["statistic"] == "best"]
    qcols = sorted((c for c in mat.columns if c not in ("design", "statistic")),
                   key=float)
    for _, row in mat.iterrows():
        v = row[qcols].to_numpy(dtype=float)
        assert np.all(np.diff(v) <= 1e-9), f"non-monotone sweep for {row['design']}"


def test_rank_agreement_shape_and_range(tree):
    agr = _read(tree, "design_rank_agreement")
    assert len(agr) == 2 * len(cd.STRINGENCY_GRID)     # best + median statistics
    assert set(agr["statistic"]) == {"best", "median"}
    assert (agr["n_designs"] == len(tree["designs"])).all()
    tau = agr["tau_b_vs_default"].dropna()
    assert tau.between(-1.0, 1.0).all()


def test_default_thresholds_located_on_the_stringency_axis(tree):
    d = _read(tree, "default_thresholds")
    assert set(d["objective"]) == {"A", "B"}
    assert d["default_stringency"].between(0.0, 1.0).all()


###############################################################################
# 2. Scorecard aggregation, design ranking stability, variance components
###############################################################################

def test_design_summary_shape(tree):
    s = _read(tree, "design_summary")
    n_runs = len(tree["designs"]) * len(DRAWS) * len(SEEDS)
    metrics = s["metric"].nunique()
    assert len(s) == n_runs * metrics
    assert cd.PRIMARY_METRIC in set(s["metric"])
    assert {"design", "draw", "seed", "metric", "best", "median",
            "higher_better"} <= set(s.columns)
    # best >= median for a higher-better metric, by construction.
    hb = s[s["higher_better"]]
    assert (hb["best"] >= hb["median"] - 1e-9).all()


def test_design_ranking_stability_is_square_over_metrics(tree):
    tau = pd.read_csv(tree["written"]["design_ranking_stability"], index_col=0)
    assert tau.shape[0] == tau.shape[1] > 1
    assert list(tau.index) == list(tau.columns)
    assert np.allclose(np.diag(tau.to_numpy(dtype=float)), 1.0)
    assert cd.PRIMARY_METRIC in tau.columns
    off = tau.to_numpy(dtype=float)[~np.eye(len(tau), dtype=bool)]
    assert np.nanmin(off) >= -1.0 and np.nanmax(off) <= 1.0


def test_variance_components_draw_is_the_unit_of_analysis(tree):
    vc = _read(tree, "design_variance_components")
    anova = vc[(vc["method"] == "anova") & (vc["statistic"] == "best")]
    assert len(anova) == 3
    comps = list(anova["component"])
    assert comps[0].startswith("design")
    assert comps[1].startswith("draw(design)")
    assert comps[2].startswith("seed(draw)")

    n_designs = len(tree["designs"])
    n_draws = n_designs * len(DRAWS)
    n_obs = n_draws * len(SEEDS)
    assert int(anova.iloc[0]["df"]) == n_designs - 1
    assert int(anova.iloc[1]["df"]) == n_draws - n_designs
    assert int(anova.iloc[2]["df"]) == n_obs - n_draws
    # The design F-test denominator is the DRAW mean-square: F = MS_design / MS_draw.
    f = float(anova.iloc[0]["f_stat"])
    assert f == pytest.approx(float(anova.iloc[0]["ms"]) / float(anova.iloc[1]["ms"]))
    assert np.isfinite(float(anova.iloc[0]["p_value"]))
    # Effective n is the draw count, and the note says so out loud.
    assert f"effective n = {n_draws} draws" in anova.iloc[0]["note"]
    assert int(anova.iloc[1]["n"]) == n_draws

    # statsmodels is installed in this venv, so the MixedLM rows must be present.
    pytest.importorskip("statsmodels")
    mixed = vc[(vc["method"] == "mixedlm") & (vc["statistic"] == "best")]
    assert len(mixed) == 2
    assert mixed["var_component"].notna().all()


###############################################################################
# 3. Raw distributions + degeneracy screen
###############################################################################

def test_performance_distributions_shape(tree):
    perf = _read(tree, "design_performance_quantiles")
    # one row per (design, objective, solution_group)
    assert len(perf) == len(tree["designs"]) * 2 * 2
    assert set(perf["solution_group"]) == {"best", "median"}
    assert {"q05", "q25", "q50", "q75", "q95"} <= set(perf.columns)
    assert (perf["q05"] <= perf["q50"] + 1e-9).all()
    assert (perf["q50"] <= perf["q95"] + 1e-9).all()


def test_degeneracy_flags_table_exists(tree):
    flags = _read(tree, "degeneracy_flags")
    assert list(flags.columns) == ["flag", "objective", "design", "value", "detail"]


def test_saturated_objective_is_flagged(tmp_path):
    """An objective every design satisfices ~always is non-discriminating."""
    root = tmp_path / "sat_outputs"
    designs = campaign_designs()[:3]
    rng = np.random.default_rng(3)
    for design in designs:
        d = root / design / SLUG_BASE / "reeval" / TAG / "seed_00"
        d.mkdir(parents=True)
        records = []
        for sid in range(4):
            for rid in range(N_REAL):
                # A is always far above its threshold -> saturated.
                records.append((sid, rid, "A", float(rng.uniform(0.95, 1.0))))
                records.append((sid, rid, "B", float(rng.uniform(0.0, 1.0))))
        pd.DataFrame(records, columns=["solution_id", "realization_id",
                                       "objective", "value"]).to_csv(
            d / "reeval_raw.csv.gz", index=False, compression="gzip")
        (d / "reeval_raw_meta.json").write_text(
            json.dumps(dict(META, solution_ids=list(range(4)))))
        rob.run(d, metrics=("satisficing_multivariate", "satisficing_univariate",
                            "laplace_mean", "maximin"))
    written = cd.run_comparison(
        formulation=FORMULATION, reeval_tag=TAG, outputs_root=root,
        table_dir=root / "_c", fig_dir=root / "_f")
    flags = pd.read_csv(written["degeneracy_flags"])
    sat = flags[flags["flag"] == "saturated_objective"]
    assert set(sat["objective"]) == {"A"}


###############################################################################
# 4. Attainability
###############################################################################

def test_attainability_has_a_pooled_row(tree):
    a = _read(tree, "design_attainability")
    assert len(a) == len(tree["designs"]) + 1
    assert "POOLED_ALL_DESIGNS" in set(a["design"])
    pooled = a[a["design"] == "POOLED_ALL_DESIGNS"].iloc[0]
    assert pooled["n_realizations"] == N_REAL
    assert 0.0 <= pooled["frac_attainable"] <= 1.0
    assert pooled["frac_unwinnable"] == pytest.approx(1 - pooled["frac_attainable"])
    # Pooling across designs can only ADD attainable realizations.
    per_design = a[a["design"] != "POOLED_ALL_DESIGNS"]
    assert pooled["n_attainable"] >= per_design["n_attainable"].max()
    assert {"unmet__A", "unmet__B"} <= set(a.columns)


###############################################################################
# Figures
###############################################################################

def test_figures_written(tree):
    for key in ("fig_threshold_sweep", "fig_design_ranking_stability",
                "fig_performance_distributions"):
        p = Path(tree["written"][key])
        assert p.exists() and p.stat().st_size > 0
