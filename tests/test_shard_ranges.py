"""Shard-range arithmetic for sharded candidate-pool generation.

The shard union must tile ``[0, n)`` exactly, in order, or the merged hazard
image would silently drop or duplicate realizations. Pure arithmetic — no
generator, no pywrdrb.
"""

import pytest

from src.ensemble_generation import shard_profile_range


@pytest.mark.parametrize("n,count", [(10, 1), (10, 3), (1000, 7), (1_000_000, 50)])
def test_shards_tile_exactly(n, count):
    ranges = [shard_profile_range(n, count, i) for i in range(count)]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == n
    for (_, hi), (lo, _) in zip(ranges, ranges[1:]):
        assert hi == lo  # contiguous, ordered, no gap or overlap
    sizes = [hi - lo for lo, hi in ranges]
    assert sum(sizes) == n
    assert max(sizes) - min(sizes) <= 1  # balanced


def test_invalid_shard_index_raises():
    with pytest.raises(ValueError):
        shard_profile_range(10, 3, 3)
    with pytest.raises(ValueError):
        shard_profile_range(10, 3, -1)
    with pytest.raises(ValueError):
        shard_profile_range(10, 0, 0)
