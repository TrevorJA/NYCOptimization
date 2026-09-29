"""tests/test_hf_design_metrics.py - HF design-metric pure-function tests.

Covers the pure computation helpers of
``scripts/supplemental/hf_design_metrics_run.py`` with hand-computed synthetic
inputs (nothing routed through the staged images, except the final smoke
identity test, which is skipped when the local staged HF image is absent):

  1. scaled_coordinates: clipped equals the selector's normalization, unclipped
     keeps excursions;
  2. nearest_member: hand geometry;
  3. voronoi_masses / effective_sample_size: masses sum to 1, ESS = N for equal
     masses and 1 for a one-hot vector;
  4. coverage_summary: zero when the set is the whole candidate ensemble;
  5. mst_edges / mst_edge_stats: grid spacing and the closest pair;
  6. ks_to_uniform: <= 1/N for Latin hypercube targets;
  7. ks_two_sample: equals scipy's two-sample statistic;
  8. axis_marginal_stats: tail share of the candidate against itself and the
     beyond-box / beyond-historic counts;
  9. bound_stability: zero deviation for the full prefix;
 10. the local staged image replays exactly and the exact assignment is
     certified.

Run:
    python -m pytest tests/test_hf_design_metrics.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ks_2samp

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import supplemental_config as scfg  # noqa: E402
from scengen import subsample as ss  # noqa: E402

from scripts.supplemental.hf_design_metrics_run import (  # noqa: E402
    axis_marginal_stats, bound_stability, coverage_summary, effective_sample_size,
    geometry_record, ks_to_uniform, ks_two_sample, mst_edge_stats, mst_edges,
    nearest_member, scaled_coordinates, voronoi_masses,
)


def _image(n=200, d=3, seed=0):
    rng = np.random.default_rng(seed)
    H = rng.gamma(2.0, 1.0, size=(n, d))
    H[: n // 5, 0] = 0.0
    return H


class TestScaling:
    def test_clipped_equals_selector_normalization(self):
        H = _image()
        lo, hi = ss.robust_range_bounds(H)
        np.testing.assert_allclose(scaled_coordinates(H, lo, hi, clip=True), ss.minmax_normalize(H))

    def test_unclipped_keeps_excursions(self):
        lo, hi = np.array([0.0]), np.array([10.0])
        Z = scaled_coordinates(np.array([[-5.0], [0.0], [10.0], [20.0]]), lo, hi, clip=False)
        assert Z.ravel() == pytest.approx([-0.5, 0.0, 1.0, 2.0])


class TestGeometry:
    def test_nearest_member_hand_geometry(self):
        ref = np.array([[0.0, 0.0], [1.0, 0.0]])
        d, i = nearest_member(np.array([[0.1, 0.0], [0.9, 0.0], [0.5, 0.5]]), ref)
        assert d == pytest.approx([0.1, 0.1, np.sqrt(0.5)])
        assert i.tolist() == [0, 1, 0] or i.tolist() == [0, 1, 1]

    def test_voronoi_masses_and_ess(self):
        w = voronoi_masses(np.array([0, 0, 1, 2, 2, 2]), 4)
        assert w.sum() == pytest.approx(1.0)
        assert w.tolist() == pytest.approx([2 / 6, 1 / 6, 3 / 6, 0.0])
        assert effective_sample_size(np.full(50, 1 / 50)) == pytest.approx(50.0)
        assert effective_sample_size(np.eye(1, 50).ravel()) == pytest.approx(1.0)
        assert effective_sample_size(w) <= 4.0

    def test_coverage_zero_when_set_is_the_candidates(self):
        Z = ss.minmax_normalize(_image())
        s, arrays = geometry_record(Z, Z, quantiles=(0.5, 0.9))
        assert s["minimax_distance"] == pytest.approx(0.0)
        assert s["mean_nearest_member_distance"] == pytest.approx(0.0)
        assert arrays["weights"].sum() == pytest.approx(1.0)

    def test_coverage_summary_keys(self):
        s = coverage_summary(np.array([0.1, 0.2, 0.4]), quantiles=(0.5,))
        assert s["minimax_distance"] == pytest.approx(0.4)
        assert s["mean_nearest_member_distance"] == pytest.approx(0.7 / 3)
        assert "cover_q50" in s

    def test_mst_grid_spacing_and_closest_pair(self):
        Z = np.column_stack([np.arange(5) * 0.2, np.zeros(5)])
        e = mst_edges(Z)
        assert e == pytest.approx(np.full(4, 0.2))
        Z2 = np.vstack([Z, [[0.05, 0.0]]])
        st = mst_edge_stats(mst_edges(Z2))
        assert st["mst_edge_min"] == pytest.approx(0.05)
        assert len(mst_edges(Z2)) == 5

    def test_mst_keeps_coincident_points(self):
        Z = np.array([[0.2, 0.2], [0.2, 0.2], [0.8, 0.8]])
        e = mst_edges(Z)
        assert len(e) == 2 and e[0] == pytest.approx(0.0)


class TestMarginals:
    def test_lhs_targets_ks_within_one_over_n(self):
        for seed in range(4):
            U = ss.generate_lhs_samples(40, 3, np.zeros(3), np.ones(3), seed=seed)
            for k in range(3):
                assert ks_to_uniform(U[:, k]) <= 1 / 40 + 1e-12
        grid = (np.arange(1, 11) - 0.5) / 10
        assert ks_to_uniform(grid) == pytest.approx(0.05)

    def test_ks_two_sample_matches_scipy(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(size=37)
        y = np.sort(rng.beta(2.0, 5.0, size=120))
        assert ks_two_sample(x, y) == pytest.approx(ks_2samp(x, y).statistic)
        assert ks_two_sample(y, y) == pytest.approx(0.0)

    def test_axis_stats_tail_share_and_counts(self):
        H = _image(n=1000)
        lo, hi = ss.robust_range_bounds(H)
        Z = scaled_coordinates(H, lo, hi, clip=True)
        p90 = (np.percentile(H, 90, axis=0) - lo) / (hi - lo)
        st = axis_marginal_stats(Z[:, 1], np.sort(Z[:, 1]), p90_scaled=float(p90[1]),
                                 z_unclipped=scaled_coordinates(H, lo, hi, clip=False)[:, 1],
                                 raw=H[:, 1], raw_hist_max=float(np.percentile(H[:, 1], 95)))
        assert st["tail_share_p90"] == pytest.approx(0.10, abs=1.5 / 1000)
        assert st["ks_candidate"] == pytest.approx(0.0)
        assert st["n_beyond_box"] == int(np.sum((H[:, 1] < lo[1]) | (H[:, 1] > hi[1])))
        assert st["n_beyond_historic_max"] == int(np.sum(H[:, 1] > np.percentile(H[:, 1], 95)))
        assert st["span"] == pytest.approx(st["max"] - st["min"])


class TestBounds:
    def test_full_prefix_has_zero_deviation(self):
        H = _image(n=500)
        rows = bound_stability(H, (100, 500), lo_pct=1.0, hi_pct=99.0)
        full = [r for r in rows if r["prefix"] == 500]
        assert len(full) == 3
        assert all(r["dev_lo_span"] == 0.0 and r["dev_hi_span"] == 0.0 for r in full)
        part = [r for r in rows if r["prefix"] == 100]
        assert all(np.isfinite(r["dev_hi_span"]) for r in part)


_LOCAL_HF = PROJECT_DIR / "outputs" / "synthetic_ensembles" / "hazfill_stat_abs_10yr_n40_d0" / "hazard_image.npz"


@pytest.fixture
def smoke_settings(monkeypatch):
    """Pin the HFM settings to the local P = 300 / N = 40 smoke images.

    ``supplemental_config`` resolves ``NYCOPT_HFM_SMOKE`` once at import, so the
    values depend on which test module imports it first.
    """
    monkeypatch.setattr(scfg, "HFM_SMOKE", True)
    monkeypatch.setattr(scfg, "HFM_POOL_P", 300)
    monkeypatch.setattr(scfg, "HFM_N", 40)
    monkeypatch.setattr(scfg, "HFM_KNN_LADDER", (8, 16, 32, 64))


@pytest.mark.skipif(not _LOCAL_HF.exists(), reason="local smoke HF image not staged")
def test_smoke_identity_on_staged_image(smoke_settings):
    """The staged N = 40 selection replays exactly and its exact assignment is certified."""
    from scripts.supplemental.hf_design_metrics_run import load_candidate, load_hf, replay_selection

    cand = load_candidate(0)
    assert cand is not None
    hf = load_hf(0, cand)
    assert hf is not None and hf["n"] == 40
    rep = replay_selection(cand, hf)
    assert rep["exact"].certified
    assert rep["summary"]["exact_total"] <= rep["summary"]["greedy_total"] + 1e-12
    assert rep["summary"]["free_total"] <= rep["summary"]["exact_total"] + 1e-12
