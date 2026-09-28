"""
tests/test_hazard_window_alignment.py - The hazard metrics and the objectives
score one window.

Both layers open on the first day after ``METRIC_EXCLUSION_MONTHS`` (June 1
of year 1 on a December-start realization) and close on May 31 of year L. The
objectives get there through ``ffmp_year_unit_slices``; the hazard image
through ``src.ensemble_generation._hazard_block`` (wet axes, a day count cut
by date) and ``scengen.hazard_metrics.scored_dry_ssi`` (dry axes, the SSI-6
series from the first month after the window); the hazard-examples figure
through ``sequence_of``, which re-derives the same series by date. These
tests pin the three to the objectives' unit slices on synthetic
December-start traces. No simulation, no staged data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

pytest.importorskip("synhydro", reason="the dry axes need SynHydro's SSI")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from scengen.hazard_metrics import _SCENARIO_STAMP_START  # noqa: E402

from config import METRIC_EXCLUSION_MONTHS  # noqa: E402
from src.objectives_ensemble import ffmp_year_unit_slices  # noqa: E402

L = 10
NODES = ("cannonsville", "pepacton", "neversink")


def _december_index(n_years: int = L) -> pd.DatetimeIndex:
    """Daily index of an L-year realization on the scenario epoch (a December 1)."""
    start = pd.Timestamp(_SCENARIO_STAMP_START)
    end = start + pd.DateOffset(years=n_years) - pd.Timedelta(days=1)
    return pd.date_range(start, end, freq="D")


def _reference(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (monthly, daily) historical reference arrays."""
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 100.0, size=78 * 12), rng.gamma(2.0, 100.0, size=78 * 365)


def _inflow_frame(idx: pd.DatetimeIndex, *, seed: int = 1, pulse=None) -> pd.DataFrame:
    """Daily catchment inflows split evenly over the NYC nodes; ``pulse`` is a
    boolean mask of days scaled fifty-fold (a flood pulse)."""
    rng = np.random.default_rng(seed)
    agg = rng.gamma(2.0, 100.0, size=len(idx))
    if pulse is not None:
        agg[pulse] *= 50.0
    return pd.DataFrame({n: agg / len(NODES) for n in NODES}, index=idx)


def test_hazard_block_scores_the_objectives_unit_window():
    """The hazard block's wet cut is the first unit slice's positional start
    and its scored span ends where the last unit slice ends; the scored SSI-6
    series starts on the first unit year's first month with one value per
    month of the L - 1 unit years."""
    from scengen.hazard_filling import daily_to_monthly
    from scengen.hazard_metrics import (compute_candidate_hazard_image,
                                        fit_reference_ssi, scored_dry_ssi)

    from src.ensemble_generation import _hazard_block

    idx = _december_index()
    slices = ffmp_year_unit_slices(idx)
    first = idx[slices[0].start]
    assert first == idx[0] + pd.DateOffset(months=METRIC_EXCLUSION_MONTHS)
    assert (first.month, first.day) == (6, 1)
    assert len(slices) == L - 1

    ref_m, ref_d = _reference()
    # A flood pulse in the last week of the exclusion window must not reach
    # the wet axes.
    pulse = (idx >= first - pd.Timedelta(days=7)) & (idx < first)
    frame = _inflow_frame(idx, pulse=pulse)
    H, axes = _hazard_block({0: frame}, [0], NODES, ref_m, ref_d, n_years=L)

    wet_cut = slices[0].start
    assert wet_cut == int((idx < first).sum())
    agg = frame.sum(axis=1).iloc[:slices[-1].stop]
    monthly = daily_to_monthly(agg, agg="mean")
    H_units, _ = compute_candidate_hazard_image(
        monthly[None, :], agg.to_numpy()[None, :], ref_m, ref_d,
        wet_exclusion_days=wet_cut,
    )
    np.testing.assert_allclose(H, H_units)

    # Opening the wet window one week earlier admits the pulse, so the two
    # cuts are not interchangeable.
    H_early, _ = compute_candidate_hazard_image(
        monthly[None, :], agg.to_numpy()[None, :], ref_m, ref_d,
        wet_exclusion_days=wet_cut - 7,
    )
    peak = axes.index("flood_peak_discharge")
    assert H_early[0, peak] > H[0, peak]

    ssi = scored_dry_ssi(fit_reference_ssi(ref_m), monthly)
    assert ssi.index[0] == first
    assert len(ssi) == 12 * len(slices)
    assert ssi.index[-1] == idx[slices[-1].stop - 1].replace(day=1)


def test_example_sequence_opens_where_the_objectives_window_opens():
    """The hazard-examples sequence, re-derived by date, starts on the first
    unit year's first month and holds one SSI-6 value per month of the
    L - 1 unit years (``sequence_of`` raises if its cut and scengen's differ)."""
    from src.plotting.hazard_examples import sequence_of

    idx = _december_index()
    slices = ffmp_year_unit_slices(idx)
    first = idx[slices[0].start]
    frame = _inflow_frame(idx, seed=2)

    seq = sequence_of(frame.sum(axis=1), 0, _reference(), L)
    assert seq.t[0] == pytest.approx((first - idx[0]).days / 365.25)
    assert len(seq.ssi) == 12 * len(slices)
    assert len(seq.year_mid) == len(slices)
