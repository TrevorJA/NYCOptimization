"""
tests/test_demand_utilization_analysis.py - The utilization axis in the analysis layer.

The NYC entitlement-utilization factor ``u`` is appended to a staged E_test as
separate ``forcing_profiles.npz`` keys (never a column of ``theta_params``).
These tests pin the plumbing that carries it from the staged npz into the
analysis layer, on metadata-only fixtures (a real ``_meta.json`` +
``forcing_profiles.npz`` pair, as ``tests/test_etest.py`` stages):

  1. ``load_etest_sample`` exposes ``u``, ``k``, ``k_clipped`` strided like
     ``theta`` when present and nothing extra for a legacy staging.
  2. ``theta_features`` returns four columns for an appended E_test and three
     for a legacy one.
  3. ``sow_demand_utilization`` joins by SOW LABEL and ``reeval_raw_meta``
     carries it (``None`` for a legacy staging).
  4. The four SI figure builders render on synthetic inputs and write their
     companion tables.

Run:
    venv/Scripts/python.exe -m pytest tests/test_demand_utilization_analysis.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import config  # noqa: E402
import src.demand_utilization as du  # noqa: E402
import src.factor_mapping as fm  # noqa: E402
import src.reeval_core as rc  # noqa: E402
from src.plotting import forcing_space as fspace  # noqa: E402
from src.plotting import demand_utilization as dup  # noqa: E402
from src.plotting.forcing_space import load_etest_sample  # noqa: E402

THETA_NAMES = ["m", "r1", "r2"]


# ---------------------------------------------------------------------------
# Staging helper
# ---------------------------------------------------------------------------

def _stage(root: Path, slug: str, *, n_theta: int, r: int, u=None, years: int = 10,
           subset_of: str | None = None) -> Path:
    """Stage ``_meta.json`` + ``forcing_profiles.npz``; ``u`` appends the factor."""
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    n = n_theta * r
    meta = {
        "slug": slug, "kind": "forcing_pool", "population": "du_forced",
        "generator": "kn", "seed_domain": "etest:kn", "theta_sampler": "lhs",
        "n_realizations": n, "n_forcing_profiles": n_theta,
        "realizations_per_profile": r, "realization_years": years, "n_years": years,
        "start_date": config.ENSEMBLE_START_DATE,
    }
    if subset_of:
        meta["subset_of"] = subset_of
    (d / "_meta.json").write_text(json.dumps(meta))
    theta = np.column_stack([
        np.linspace(-0.2, 0.25, n_theta),       # m
        np.linspace(0.05, 0.35, n_theta),       # r1
        np.linspace(0.02, 0.30, n_theta),       # r2
    ])
    payload = {
        "realization_ids": np.arange(n),
        "theta_params": np.repeat(theta, r, axis=0),
        "theta_param_names": np.array(THETA_NAMES),
        "mean_factor_a": np.ones((n, 12)),
        "theta_sampler": "lhs",
        "n_forcing_profiles": n_theta,
        "realizations_per_profile": r,
    }
    if u is not None:
        u = np.asarray(u, dtype=float)
        assert len(u) == n_theta
        k, clipped = np.where(u >= 0.99, du.DEMAND_SCALE_MAX, 1.0 + u), u >= 0.99
        payload["demand_utilization"] = np.repeat(u, r)
        payload["demand_scale_k"] = np.repeat(k, r)
        payload["demand_scale_clipped"] = np.repeat(clipped, r)
        payload["demand_utilization_bounds"] = np.array(du.DEMAND_UTILIZATION_BOUNDS)
    np.savez(d / "forcing_profiles.npz", **payload)
    return d


U6 = np.array([0.46, 0.55, 0.65, 0.75, 0.85, 0.995])


# ---------------------------------------------------------------------------
# 1 + 2. load_etest_sample / theta_features
# ---------------------------------------------------------------------------

def test_load_etest_sample_exposes_u_strided_like_theta(tmp_path):
    d = _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3, u=U6)
    s = load_etest_sample(d)
    assert s["theta"].shape == (6, 3) and s["n_sow"] == 6
    assert np.allclose(s["u"], U6)
    assert s["k"].shape == (6,) and s["k_clipped"].dtype == bool
    assert s["k_clipped"].tolist() == [False] * 5 + [True]
    assert s["theta_names"] == THETA_NAMES, "u is never a theta column"


def test_load_etest_sample_legacy_staging_has_no_u(tmp_path):
    d = _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3)
    s = load_etest_sample(d)
    assert "u" not in s and "k" not in s and "k_clipped" not in s


def test_theta_features_four_columns_when_appended(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3, u=U6)
    X, names = fm.theta_features("etest_kn_10yr_n18")
    assert names == ["em", "r1", "r2", "u"]
    assert X.shape == (6, 4)
    assert np.allclose(X[:, 3], U6)
    assert np.allclose(X[:, 0], np.exp(np.linspace(-0.2, 0.25, 6)))


def test_theta_features_three_columns_for_a_legacy_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3)
    X, names = fm.theta_features("etest_kn_10yr_n18")
    assert names == ["em", "r1", "r2"] and X.shape == (6, 3)


# ---------------------------------------------------------------------------
# 3. sow_demand_utilization + reeval_raw_meta
# ---------------------------------------------------------------------------

def test_sow_demand_utilization_joins_by_label(tmp_path, monkeypatch):
    from src.ensembles import get_ensemble_spec

    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3, u=U6)
    spec = get_ensemble_spec("etest_kn_10yr_n18")
    out = rc.sow_demand_utilization(spec, [5, 0, 2])
    assert out == {"5": pytest.approx(U6[5]), "0": pytest.approx(U6[0]),
                   "2": pytest.approx(U6[2])}
    with pytest.raises(KeyError):
        rc.sow_demand_utilization(spec, [0, 6])          # SOW 6 does not exist


def test_sow_demand_utilization_is_none_for_legacy_and_single_trace(tmp_path, monkeypatch):
    from src.ensembles import get_ensemble_spec

    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3)
    assert rc.sow_demand_utilization(get_ensemble_spec("etest_kn_10yr_n18"), [0, 1]) is None
    assert rc.sow_demand_utilization(get_ensemble_spec("historic_single"), [0]) is None
    assert rc.sow_demand_utilization(None, [0]) is None


@pytest.mark.parametrize("with_u", [True, False])
def test_reeval_raw_meta_carries_sow_demand_utilization(tmp_path, monkeypatch, with_u):
    from src.ensembles import get_ensemble_spec

    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3, u=U6 if with_u else None)
    spec = get_ensemble_spec("etest_kn_10yr_n18")
    obj = SimpleNamespace(name="A", sat_threshold=0.9, sat_kind="ge",
                          direction="maximize", unit_operator=object())
    monkeypatch.setattr(rc, "resolve_reeval", lambda *a, **k: ([obj], spec, True))
    meta = rc.reeval_raw_meta("ffmp", 1)
    assert meta["sow_labels"] == list(range(6))
    if with_u:
        assert meta["sow_demand_utilization"] == {
            str(g): pytest.approx(U6[g]) for g in range(6)}
        json.dumps(meta)                                   # JSON-serializable
    else:
        assert meta["sow_demand_utilization"] is None


# ---------------------------------------------------------------------------
# 4. Figure builders on synthetic inputs
# ---------------------------------------------------------------------------

def _ctx(tag: str, comparison_dir: Path | None = None):
    return SimpleNamespace(tag=tag, slug="ffmp_obj8",
                           comparison_dir=lambda: comparison_dir)


def _synthetic_calibration(tmp_path: Path, monkeypatch) -> None:
    k = np.linspace(du.DEMAND_SCALE_MIN, du.DEMAND_SCALE_MAX, 58)
    pd.DataFrame({"k": k, "u": 1.0 - np.exp(-0.55 * k)}).to_csv(
        tmp_path / "calib.csv", index=False)
    idx = pd.date_range("1946-01-01", "1947-12-31", freq="D")
    ref = 450.0 + 150.0 * np.sin(2 * np.pi * idx.dayofyear / 365.25)
    pd.DataFrame({"datetime": idx, "aggregate": ref}).to_csv(
        tmp_path / "ref.csv", index=False)
    monkeypatch.setattr(du, "CALIBRATION_CSV", tmp_path / "calib.csv")
    monkeypatch.setattr(du, "REFERENCE_SERIES_CSV", tmp_path / "ref.csv")


def test_fig_calibration_renders_and_writes_tables(tmp_path, monkeypatch):
    _synthetic_calibration(tmp_path, monkeypatch)
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    out = dup.fig_demand_utilization_calibration(_ctx("x"), tmp_path / "fig", table_dir)
    assert (tmp_path / "fig.png").is_file()
    clim = pd.read_csv(table_dir / "demand_utilization_climatology.csv")
    assert set(clim["u"].round(2)) == {0.45, 0.70, 0.90, 1.0}
    assert len(clim) == 4 * 12
    # u = 1 is the flat cap; every level respects the monthly-mean cap.
    assert np.allclose(clim.loc[clim["u"] == 1.0, "demand_mgd"],
                       config.NYC_DECREE_DIVERSION_CAP_MGD)
    assert (clim["demand_mgd"] <= config.NYC_DECREE_DIVERSION_CAP_MGD + 1e-9).all()
    assert out["levels"] == list(dup.CLIMATOLOGY_LEVELS)


def test_fig_calibration_names_the_driver_when_tables_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "CALIBRATION_CSV", tmp_path / "nope.csv")
    monkeypatch.setattr(du, "REFERENCE_SERIES_CSV", tmp_path / "nope2.csv")
    with pytest.raises(FileNotFoundError, match="demand_utilization_calibration.py"):
        dup.fig_demand_utilization_calibration(_ctx("x"), tmp_path / "fig", tmp_path)


def test_fig_sampling_renders_with_the_box(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3, u=U6)
    names = ["m", "r1", "psi1", "r2", "psi2"]
    lo = np.array([-0.3, 0.0, 0.0, 0.0, 0.0])
    hi = np.array([0.4, 0.5, 1.0, 0.5, 1.0])
    monkeypatch.setattr(fspace, "load_cmip6_fits", lambda *a, **k: {"fit": None})
    monkeypatch.setattr(fspace, "etest_param_box", lambda fit: (lo, hi, names))
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    out = dup.fig_demand_utilization_sampling(_ctx("etest_kn_10yr_n18"),
                                              tmp_path / "fig", table_dir)
    assert (tmp_path / "fig.png").is_file()
    rows = pd.read_csv(table_dir / "demand_utilization_sampling.csv")
    assert list(rows.columns) == ["sow", "em", "r1", "r2", "u", "k", "k_clipped"]
    assert len(rows) == 6 and out["n_clipped"] == 1


def test_fig_sampling_refuses_a_legacy_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3)
    with pytest.raises(ValueError, match="append_demand_utilization.py"):
        dup.fig_demand_utilization_sampling(_ctx("etest_kn_10yr_n18"),
                                            tmp_path / "fig", tmp_path)


def test_fig_realized_reads_the_tag_directory(tmp_path, monkeypatch):
    """The realized table is read from the re-eval tag's own staged directory:
    an E_test cut from a larger design carries its own merged tables, so no
    reader follows ``subset_of`` to a parent."""
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage(tmp_path, "etest_kn_10yr_n36", n_theta=12, r=3, u=np.linspace(0.5, 0.9, 12))
    tag = _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3, u=U6,
                 subset_of="etest_kn_10yr_n36")
    pd.DataFrame({
        "sow": np.arange(6), "u_nominal": U6, "k": 1.0 + U6, "clipped": False,
        "u_realized_mean": U6 + 0.02, "u_realized_min": U6 - 0.01,
        "u_realized_max": U6 + 0.05, "n_realizations": 3,
    }).to_csv(tag / "demand_utilization_realized_by_sow.csv", index=False)
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    out = dup.fig_demand_utilization_realized(_ctx("etest_kn_10yr_n18"),
                                              tmp_path / "fig", table_dir)
    assert (tmp_path / "fig.png").is_file()
    rows = pd.read_csv(table_dir / "demand_utilization_realized.csv")
    assert len(rows) == 6 and "em" in rows.columns
    assert out["mean_realized_minus_nominal"] == pytest.approx(0.02)


def test_fig_realized_names_the_missing_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STAGED_ENSEMBLE_DIR", tmp_path)
    _stage(tmp_path, "etest_kn_10yr_n18", n_theta=6, r=3, u=U6)
    with pytest.raises(FileNotFoundError, match="realized_by_sow"):
        dup.fig_demand_utilization_realized(_ctx("etest_kn_10yr_n18"),
                                            tmp_path / "fig", tmp_path)


def test_fig_regret_vs_utilization_renders(tmp_path):
    comp = tmp_path / "comparison"
    comp.mkdir()
    edges = np.linspace(*du.DEMAND_UTILIZATION_BOUNDS, du.UTILIZATION_REGRET_BINS + 1)
    rows = []
    for design in ("monte_carlo", "hazard_filling_stationary"):
        for seed in (0, 1):
            for b in range(du.UTILIZATION_REGRET_BINS):
                rows.append({
                    "design": design, "draw": 0, "seed": seed, "u_bin": b,
                    "u_lo": edges[b], "u_hi": edges[b + 1], "n_sow": 5,
                    "sat_best": 1.0 - 0.05 * b, "no_harm_tau_best": 0.9 - 0.03 * b,
                    "best_solution_id": 0,
                    "regret_mean__A": 0.01 * b, "gain_mean__A": 0.1 - 0.005 * b,
                    "regret_mean_best__A": 0.0, "gain_mean_best__A": 0.1,
                    "regret_mean__B": 0.2 * b, "gain_mean__B": 1.0,
                    "regret_mean_best__B": 0.0, "gain_mean_best__B": 1.0,
                })
    pd.DataFrame(rows).to_csv(comp / "design_regret_by_utilization.csv", index=False)
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    out = dup.fig_regret_vs_utilization(_ctx("x", comp), tmp_path / "fig", table_dir)
    assert (tmp_path / "fig.png").is_file()
    assert out["objectives"] == ["A", "B"] and out["n_bins"] == du.UTILIZATION_REGRET_BINS
    agg = pd.read_csv(table_dir / "regret_vs_utilization.csv")
    assert len(agg) == 2 * du.UTILIZATION_REGRET_BINS       # seeds averaged


def test_fig_regret_vs_utilization_names_compare_designs(tmp_path):
    with pytest.raises(FileNotFoundError, match="compare_designs"):
        dup.fig_regret_vs_utilization(_ctx("x", tmp_path), tmp_path / "fig", tmp_path)


def test_registry_lists_the_four_si_figures():
    from src.figures.registry import by_name

    expected = {
        "demand_utilization_calibration": {"demand_calibration"},
        "demand_utilization_sampling": {"ensemble", "demand_utilization"},
        "demand_utilization_realized": {"demand_realized"},
        "regret_vs_utilization": {"regret_by_utilization"},
    }
    for name, needs in expected.items():
        spec = by_name(name)
        assert spec.tier == "si" and set(spec.needs) == needs
        assert spec.stem.startswith("figS")
