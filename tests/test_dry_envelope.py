"""
tests/test_dry_envelope.py - The dry-envelope diagnostic's level definitions.

The volume levels are derived from the CMIP6 harmonic box, so the current
E_test lower bound must be one of them, the levels must descend with the
widening margin, and the share arithmetic must be a valid fraction.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts.supplemental.dry_envelope_run import etest_share_table, volume_levels  # noqa: E402
from src.etest import E_TEST_MARGIN, E_TEST_VOLUME_MULTIPLIER_MIN  # noqa: E402


def test_levels_descend_and_include_the_margin_bound():
    levels, _env, _fit = volume_levels()
    assert levels["level"].iloc[0] == "volume_1.00"
    assert np.isclose(levels["em"].iloc[0], 1.0)
    forced = levels[levels["margin"].notna()].sort_values("margin")
    assert forced["level"].iloc[0] == "cmip6_min"
    assert (np.diff(forced["em"].to_numpy()) < 0).all()
    margin_bound = levels[levels["is_margin_bound"]]
    assert len(margin_bound) == 1
    assert np.isclose(margin_bound["margin"].iloc[0], E_TEST_MARGIN)
    assert margin_bound["em"].iloc[0] < forced["em"].iloc[0]
    assert (levels["adopted_em"] == E_TEST_VOLUME_MULTIPLIER_MIN).all()
    assert E_TEST_VOLUME_MULTIPLIER_MIN < margin_bound["em"].iloc[0]


def test_share_table_is_a_fraction_and_zero_at_the_margin_bound():
    levels, _env, _fit = volume_levels()
    share = etest_share_table(levels)
    cols = [c for c in share.columns if c.startswith("share_")]
    assert ((share[cols] >= 0) & (share[cols] <= 1)).all().all()
    at_margin = share[np.isclose(share["margin"].fillna(-1.0), E_TEST_MARGIN)]
    assert np.isclose(at_margin["share_below_margin_bound"].iloc[0], 0.0)
    adopted = share[share["lower_bound_level"] == "adopted"]
    assert len(adopted) == 1
    assert np.isclose(adopted["em_lower"].iloc[0], E_TEST_VOLUME_MULTIPLIER_MIN)
    assert adopted["share_below_margin_bound"].iloc[0] > 0
