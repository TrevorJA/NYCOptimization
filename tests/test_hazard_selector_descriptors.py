"""
tests/test_hazard_selector_descriptors.py - Descriptor blocks of the selector diagnostic.

Pure checks on the helpers ``scripts/supplemental/diagnose_hazard_selectors.py``
adds for the descriptor redundancy, axis-set comparison and truncation summary:
the descriptor matrix drops the truncation flags and orients the low-flow
minima, the hazard-direction tail mask and the exact-snap limit match hand
values, the redundancy summary clusters a correlated pair and counts
components, the nearest-member n_eff reproduces the Kish value, the named axis
sets follow ``supplemental_config.seldiag_axis_sets``, and the block-C records
assemble into the tidy tables. Small synthetic inputs; no staged data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from scengen.hazard_metrics import CANDIDATE_EVENT_METRICS, SUPPLEMENT_METRICS  # noqa: E402

import config  # noqa: E402
import supplemental_config as scfg  # noqa: E402
from scripts.supplemental import diagnose_hazard_selectors as dhs  # noqa: E402

AXES = list(CANDIDATE_EVENT_METRICS)


def _image(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (H, supplement) with 0/1 flag columns."""
    rng = np.random.default_rng(seed)
    H = rng.gamma(2.0, 1.0, size=(n, len(AXES)))
    S = rng.gamma(2.0, 1.0, size=(n, len(SUPPLEMENT_METRICS)))
    for f in scfg.SELDIAG_TRUNCATION_FLAGS:
        S[:, list(SUPPLEMENT_METRICS).index(f)] = rng.random(n) < 0.3
    return H, S


def test_descriptor_image_drops_flags_and_orients_low_flows():
    H, S = _image()
    desc = dhs.descriptor_image(H, AXES, S, SUPPLEMENT_METRICS)
    n_desc = len(AXES) + len(SUPPLEMENT_METRICS) - len(scfg.SELDIAG_TRUNCATION_FLAGS)
    assert desc["D"].shape == (len(H), n_desc) and len(desc["names"]) == n_desc
    assert desc["names"][:len(AXES)] == AXES
    assert not set(scfg.SELDIAG_TRUNCATION_FLAGS) & set(desc["names"])
    np.testing.assert_array_equal(
        desc["flags"],
        S[:, [list(SUPPLEMENT_METRICS).index(f) for f in scfg.SELDIAG_TRUNCATION_FLAGS]])
    low = {n for n, s in zip(desc["names"], desc["sign"]) if s < 0}
    assert low == set(scfg.SELDIAG_LOW_TAIL_DESCRIPTORS)


def test_hazard_tail_mask_counts_the_hazardous_side():
    x = np.arange(100.0)
    mask = dhs.hazard_tail_mask(np.column_stack([x, x]), np.array([1.0, -1.0]), pct=90.0)
    np.testing.assert_array_equal(np.flatnonzero(mask[:, 0]), np.arange(90, 100))
    np.testing.assert_array_equal(np.flatnonzero(mask[:, 1]), np.arange(0, 10))


def test_exact_snap_limit_is_the_uniform_share_above_p90():
    H = np.linspace(0.0, 1.0, 10001)[:, None]
    limit = dhs.exact_snap_limit(H, tail_pct=90.0)
    assert limit[0] == pytest.approx((0.99 - 0.90) / (0.99 - 0.01))


def test_descriptor_redundancy_clusters_and_counts_components():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2000)
    D = np.column_stack([a, a + 0.05 * rng.normal(size=2000), rng.normal(size=2000),
                         np.ones(2000)])
    red = dhs.descriptor_redundancy(D, ["a", "a_twin", "b", "flat"],
                                    threshold=0.7, variance_share=0.9)
    assert red["constant"] == ["flat"] and red["names"] == ["a", "a_twin", "b"]
    assert sorted(map(sorted, red["clusters"])) == [["a", "a_twin"], ["b"]]
    assert red["eigenvalues"].sum() == pytest.approx(3.0)
    assert 1.5 < red["participation_ratio"] < 2.5
    assert red["n_components"] == 2
    assert red["top_loadings"][0][1] in ("a", "a_twin")
    assert red["top_loadings"][1][1] == "b"


def test_nearest_member_ess_ratio_is_the_kish_value():
    X = np.array([[0.0], [0.1], [0.2], [1.0]])
    # Member rows 0 and 3: three pool rows are nearest row 0, one nearest row 3.
    ratio = dhs.nearest_member_ess_ratio(X, np.array([0, 3]))
    assert ratio == pytest.approx(1.0 / (0.75 ** 2 + 0.25 ** 2) / 2)


def test_axis_sets_follow_the_named_definition():
    sets = dhs._axis_sets(AXES)
    campaign = list(config.HAZARD_SELECTION_AXES)
    assert list(sets) == ["campaign", "full", "four_axis", "four_axis_rate", "five_axis"]
    assert sets["campaign"] == campaign and sets["full"] == AXES
    assert sets["five_axis"] == [a for a in campaign if a != "drought_severity"]
    assert sets["four_axis_rate"] == ["drought_magnitude", "drought_development_rate",
                                      "flood_peak_discharge", "flood_pulse_duration"]
    # A screened-out axis leaves every set.
    reduced = dhs._axis_sets([a for a in AXES if a != "drought_severity"])
    assert all("drought_severity" not in v for v in reduced.values())


def test_block_c_records_assemble_into_the_tidy_tables(monkeypatch):
    monkeypatch.setattr(dhs, "N_SELECT", 20)
    monkeypatch.setattr(dhs, "N_SEEDS", 2)
    H, S = _image()
    desc = dhs.descriptor_image(H, AXES, S, SUPPLEMENT_METRICS)
    sets = dhs._axis_sets(AXES)
    dim = dhs._dimension_sweep(H, AXES, sets, include_assign=False, descriptors=desc)
    assert len(dim) == 2 * len(sets)
    tails = dim[[f"tail__{n}" for n in desc["names"]]].to_numpy()
    assert ((tails >= 0) & (tails <= 1)).all()
    assert (dim["ess_over_n"] > 0).all() and (dim["ess_over_n"] <= 1).all()

    table = dhs._axis_set_comparison(dim, sets, H, AXES, desc["names"])
    for mset, axes in sets.items():
        g = table.loc[table.m_set == mset]
        assert (g.statistic == "tail_share").sum() == len(desc["names"])
        assert (g.statistic == "attainment").sum() == len(axes)
        scalars = set(g.loc[g.descriptor == "", "statistic"])
        assert scalars == {"tail_share_min_own", "attainment_min_own", "displacement_mean",
                           "frac_targets_far", "ess_over_n"}
        own = g.loc[(g.statistic == "tail_share") & (g.in_set == True), "mean"]  # noqa: E712
        tmin = g.loc[g.statistic == "tail_share_min_own", "mean"].item()
        assert tmin <= own.min() + 1e-12

    trunc = dhs._truncation_summary(H, AXES, desc, dim)
    pool = trunc.loc[trunc.population == "pool"].iloc[0]
    assert pool["drought_onset_truncated"] == pytest.approx(desc["flags"][:, 0].mean())
    assert (trunc.population == "selected").sum() == len(sets)
    assert set(trunc.loc[trunc.population == "top_decile", "axis"]) == {
        a for a in AXES if a.startswith("drought_")}
