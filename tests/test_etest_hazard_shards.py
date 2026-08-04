"""Merge purity for the E_test sub-window hazard-image shards.

The sharded compute_etest_hazard_image path (one SLURM array task per chunk)
reuses the serial path's per-chunk shard files, so serial vs sharded can only
differ through the merge. _merge_shards lexsorts rows by (realization_id,
window_index) with unique keys, so the merged artifact must be byte-identical
regardless of row order within shards — proven here on a synthetic layout.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def merge_shards():
    from scripts.main.compute_etest_hazard_image import _merge_shards

    return _merge_shards


def _write_shards(out_dir, *, permute_rows, seed=11):
    """Three synthetic chunks x 2 windows x 8 axes with disjoint rid ranges."""
    rng = np.random.default_rng(seed)
    # Separate stream for row permutation so both layouts draw identical H.
    perm_rng = np.random.default_rng(seed + 999)
    axes = np.asarray([f"axis_{i}" for i in range(8)], dtype=object)
    paths = []
    for c, rid_lo in enumerate((0, 10, 20)):
        rids = np.repeat(np.arange(rid_lo, rid_lo + 10), 2)
        wins = np.tile(np.arange(2), 10)
        H = rng.normal(size=(20, 8))
        if permute_rows:
            # Different physical row order, same (rid, win) -> H mapping.
            perm = perm_rng.permutation(20)
            rids, wins, H = rids[perm], wins[perm], H[perm]
        p = out_dir / f"hazard_image_subwindows_shard_{c:03d}.npz"
        np.savez(p, H=H, realization_ids=rids, window_index=wins, hazard_axes=axes)
        paths.append(p)
    return paths


def test_merge_is_row_order_invariant_and_unlinks(tmp_path, merge_shards):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    paths_a = _write_shards(dir_a, permute_rows=False)
    paths_b = _write_shards(dir_b, permute_rows=True)

    merge_shards(paths_a, dir_a / "hazard_image_subwindows.npz", R=5)
    merge_shards(paths_b, dir_b / "hazard_image_subwindows.npz", R=5)

    a = np.load(dir_a / "hazard_image_subwindows.npz", allow_pickle=True)
    b = np.load(dir_b / "hazard_image_subwindows.npz", allow_pickle=True)

    for key in ("H", "realization_ids", "window_index", "theta_index"):
        assert a[key].tobytes() == b[key].tobytes(), key
    assert [str(x) for x in a["hazard_axes"]] == [str(x) for x in b["hazard_axes"]]
    assert int(a["window_years"]) == int(b["window_years"])

    # Rows sorted by (rid, win); theta_index = rid // R.
    rid, win = a["realization_ids"], a["window_index"]
    assert list(zip(rid, win)) == sorted(zip(rid, win))
    np.testing.assert_array_equal(a["theta_index"], rid // 5)

    # Shards consumed after the merge (both layouts).
    assert not any(p.exists() for p in paths_a + paths_b)
