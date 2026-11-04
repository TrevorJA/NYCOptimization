"""
tests/test_demand_utilization.py - The NYC entitlement-utilization factor u.

The axis is a calibration (u(k) on the reference record), an inverse (k(u)
by PCHIP, clipped at DEMAND_SCALE_MAX), a Latin hypercube column, and a
realized-utilization diagnostic on the FFMP-year unit. Every test plants a
synthetic reference so the properties are checked against hand values, never
against the code under test:

  1. the monthly-mean cap scales offending months to exactly the cap and
     leaves the others untouched (the Pywr-DRB rule, vectorized);
  2. u(k) is strictly monotone and bounded in (0, 1];
  3. k_from_u(u_from_k(k)) ~ k inside the unclipped range, and u = 1.0 maps to
     DEMAND_SCALE_MAX with the clipped flag;
  4. the LHS column has one value per stratum and is reproducible from the seed;
  5. realized utilization pools complete June-May FFMP years only;
  6. a missing calibration table is a loud error naming the driver, never a
     silent fallback;
  7. the seed domain is the reserved etest:*:demand stream.

Run:
    venv/Scripts/python.exe -m pytest tests/test_demand_utilization.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src import demand_utilization as du  # noqa: E402


def _synthetic_reference(low_months: bool = True) -> pd.Series:
    """A synthetic uncapped reference: ~550 MGD, seasonal cycle, lognormal noise.

    With ``low_months`` every March sits near 60 MGD, mirroring the real
    record's very wet months that stay below the cap even at ``k = 6``.
    """
    rng = np.random.default_rng(11)
    idx = pd.date_range("1946-01-01", "1975-12-31", freq="D")
    doy = idx.dayofyear.to_numpy()
    base = 550.0 + 120.0 * np.cos(2 * np.pi * (doy - 200) / 365.25)
    series = pd.Series(base * np.exp(rng.normal(0.0, 0.12, len(idx))), index=idx)
    if low_months:
        series[idx.month == 3] *= 0.11
    return series


@pytest.fixture(scope="module")
def reference() -> pd.Series:
    return _synthetic_reference()


@pytest.fixture(scope="module")
def table(reference) -> pd.DataFrame:
    return du.calibration_table_from_series(reference)


def test_cap_monthly_mean_matches_the_pywrdrb_rule(reference):
    capped = du.cap_monthly_mean(1.6 * reference, 800.0)
    scaled = 1.6 * reference
    for (_y, _m), month in capped.groupby([capped.index.year, capped.index.month]):
        raw = scaled.loc[month.index]
        if raw.mean() > 800.0:
            assert month.mean() == pytest.approx(800.0)
            np.testing.assert_allclose(month.to_numpy(),
                                       raw.to_numpy() * 800.0 / raw.mean())
        else:
            np.testing.assert_allclose(month.to_numpy(), raw.to_numpy())
    assert (capped.groupby([capped.index.year, capped.index.month]).mean() <= 800.0 + 1e-9).all()


def test_u_of_k_is_monotone_and_bounded(table, reference):
    k, u = table["k"].to_numpy(), table["u"].to_numpy()
    assert k[0] == pytest.approx(du.DEMAND_SCALE_MIN)
    assert k[-1] == pytest.approx(du.DEMAND_SCALE_MAX)
    assert np.all(np.diff(u) > 0)
    assert 0.0 < u.min() and u.max() < 1.0
    assert u[-1] > 0.9, "at k_max nearly every month is capped, so u is close to 1"
    assert du.u_from_k(1.0, table) == pytest.approx(
        float(du.cap_monthly_mean(reference, 800.0).mean()) / 800.0, rel=1e-6)


def test_saturated_tail_is_dropped_so_the_inverse_stays_defined():
    """Once every month is capped u is flat at 1.0; the table ends at that k."""
    table = du.calibration_table_from_series(_synthetic_reference(low_months=False))
    assert float(table["u"].iloc[-1]) == pytest.approx(1.0)
    assert float(table["k"].iloc[-1]) < du.DEMAND_SCALE_MAX
    assert np.all(np.diff(table["u"].to_numpy()) > 0)
    k, clipped = du.k_from_u(1.0, table)
    assert float(k) == pytest.approx(float(table["k"].iloc[-1])) and not clipped


def test_k_from_u_round_trips_inside_the_unclipped_range(table):
    ks = np.array([0.5, 0.8, 1.0, 1.37, 2.2, 4.0])
    k_back, clipped = du.k_from_u(du.u_from_k(ks, table), table)
    np.testing.assert_allclose(k_back, ks, rtol=2e-3)
    assert not clipped.any()


def test_u_one_maps_to_scale_max_with_the_clipped_flag(table):
    k, clipped = du.k_from_u(np.array([1.0, 0.9995, float(table["u"].iloc[-1])]), table)
    assert k[0] == du.DEMAND_SCALE_MAX and bool(clipped[0])
    assert k[1] == du.DEMAND_SCALE_MAX and bool(clipped[1])
    assert k[2] == pytest.approx(du.DEMAND_SCALE_MAX) and not clipped[2]
    k_scalar, c_scalar = du.k_from_u(1.0, table)
    assert float(k_scalar) == du.DEMAND_SCALE_MAX and bool(c_scalar)


def test_k_from_u_refuses_out_of_range(table):
    with pytest.raises(ValueError):
        du.k_from_u(1.01, table)
    with pytest.raises(ValueError):
        du.k_from_u(0.05, table)
    with pytest.raises(ValueError):
        du.u_from_k(du.DEMAND_SCALE_MAX + 1.0, table)


def test_lhs_column_has_one_value_per_stratum_and_is_reproducible():
    n = 50
    lo, hi = du.DEMAND_UTILIZATION_BOUNDS
    u = du.sample_utilization(n, seed=123)
    assert u.shape == (n,)
    assert u.min() >= lo and u.max() <= hi
    strata = np.floor((u - lo) / (hi - lo) * n).astype(int)
    assert sorted(strata.tolist()) == list(range(n)), "a Latin hypercube fills every stratum once"
    np.testing.assert_array_equal(u, du.sample_utilization(n, seed=123))
    assert not np.array_equal(u, du.sample_utilization(n, seed=124))


def test_realized_utilization_pools_complete_ffmp_years_only():
    idx = pd.date_range("1945-12-01", "1948-11-30", freq="D")
    values = pd.Series(100.0, index=idx)
    # Spin-up (Dec 1945 - May 1946) and the trailing partial year (Jun - Nov 1948)
    # carry poison values that must not enter the mean.
    values.loc[:"1946-05-31"] = 9999.0
    values.loc["1948-06-01":] = 9999.0
    values.loc["1946-06-01":"1947-05-31"] = 400.0
    values.loc["1947-06-01":"1948-05-31"] = 800.0
    n1 = len(values.loc["1946-06-01":"1947-05-31"])
    n2 = len(values.loc["1947-06-01":"1948-05-31"])
    expected = (400.0 * n1 + 800.0 * n2) / (n1 + n2) / 800.0
    assert du.realized_utilization(values) == pytest.approx(expected)
    with pytest.raises(ValueError):
        du.realized_utilization(values.loc[:"1946-12-31"])


def test_missing_table_is_a_loud_error_naming_the_driver(tmp_path):
    with pytest.raises(FileNotFoundError, match="demand_utilization_calibration.py"):
        du.load_calibration_table(tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError):
        du.calibration_sha256(tmp_path / "absent.csv")


def test_malformed_table_is_refused(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"k": [0.3, 0.5, 0.7], "u": [0.4, 0.3, 0.5]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        du.load_calibration_table(bad)


def test_seed_domain_is_the_reserved_demand_stream():
    from scengen.seeds import SEED_DOMAINS, design_seed

    from src.scenario_designs import SEED_ROOT

    assert du.demand_seed_domain("etest:kn") == du.DEMAND_UTILIZATION_SEED_DOMAIN
    assert du.DEMAND_UTILIZATION_SEED_DOMAIN in SEED_DOMAINS
    assert du.demand_utilization_seed("etest:kn") == design_seed(
        SEED_ROOT, du.DEMAND_UTILIZATION_SEED_DOMAIN, 0)
    assert du.demand_utilization_seed("etest:kn") != design_seed(SEED_ROOT, "etest:kn", 0)
    with pytest.raises(ValueError):
        du.demand_seed_domain("fixed")


def test_bands_and_bins_cover_the_range():
    u = np.array([0.45, 0.599, 0.6, 0.799, 0.8, 1.0])
    masks = du.utilization_band_masks(u)
    assert list(masks) == list(du.UTILIZATION_BAND_LABELS)
    stacked = np.vstack(list(masks.values()))
    assert stacked.sum(axis=0).tolist() == [1] * len(u), "every u falls in exactly one band"
    assert masks["u_lt_0.6"].tolist() == [True, True, False, False, False, False]
    assert masks["u_ge_0.8"].tolist() == [False, False, False, False, True, True]
    labels, edges = du.utilization_bin_labels(u)
    assert len(edges) == du.UTILIZATION_REGRET_BINS + 1
    assert edges[0] == du.DEMAND_UTILIZATION_BOUNDS[0] and edges[-1] == du.DEMAND_UTILIZATION_BOUNDS[1]
    assert labels[0] == 0 and labels[-1] == du.UTILIZATION_REGRET_BINS - 1


@pytest.mark.skipif(not du.CALIBRATION_CSV.exists(),
                    reason="persisted calibration table absent (run the driver)")
def test_persisted_table_reproduces_the_reference_points():
    """Sanity only: the reference-record u(k) at the quoted scales."""
    table = du.load_calibration_table()
    u = du.u_from_k(np.array([0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 6.0]), table)
    np.testing.assert_allclose(u, [0.55, 0.68, 0.79, 0.90, 0.96, 0.99, 0.998], atol=0.012)
