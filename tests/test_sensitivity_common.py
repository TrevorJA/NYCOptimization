"""Unit tests for src/sensitivity_common.py shared helpers."""

import numpy as np
import pytest

from src.sensitivity_common import (
    assign_rank_slots,
    kendall_tau_b,
    sample_lhs_dvs,
)


def test_sample_lhs_dvs_shape_bounds_and_reproducibility():
    """LHS sample has the right shape, lies within bounds, and is seed-stable."""
    from src.formulations import get_bounds

    lows, highs = get_bounds("ffmp")
    n = 16
    a = sample_lhs_dvs("ffmp", seed=123, n_samples=n)
    assert a.shape == (n, len(lows))
    assert np.all(a >= np.asarray(lows) - 1e-9)
    assert np.all(a <= np.asarray(highs) + 1e-9)
    # Same seed -> identical sample.
    b = sample_lhs_dvs("ffmp", seed=123, n_samples=n)
    assert np.allclose(a, b)


def test_kendall_tau_b_agreement_and_edge_cases():
    """tau_b is +1 for identical order, -1 for reverse, nan when degenerate."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert kendall_tau_b(x, x) == pytest.approx(1.0)
    assert kendall_tau_b(x, x[::-1]) == pytest.approx(-1.0)
    # Constant vector -> undefined ranking.
    assert np.isnan(kendall_tau_b(x, np.ones_like(x)))
    # Fewer than two finite pairs -> nan.
    assert np.isnan(kendall_tau_b([1.0, np.nan], [np.nan, 2.0]))


def test_assign_rank_slots_partitions_all_items_once():
    """Rank slots partition the item range with no gaps or overlaps."""
    n_items, size = 10, 3
    slots = [assign_rank_slots(n_items, r, size) for r in range(size)]
    flat = sorted(i for s in slots for i in s)
    assert flat == list(range(n_items))
    # Balanced within one item across ranks.
    sizes = [len(s) for s in slots]
    assert max(sizes) - min(sizes) <= 1
