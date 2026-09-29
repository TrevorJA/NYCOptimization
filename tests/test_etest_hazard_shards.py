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


def _write_shards(out_dir, *, permute_rows, seed=11, provenance=True):
    """Three synthetic chunks x 2 windows x 8 axes (+ 15 supplement columns whose
    first column encodes the row key) with disjoint rid ranges."""
    from src.ensembles import hazard_image_provenance

    rng = np.random.default_rng(seed)
    # Separate stream for row permutation so both layouts draw identical H.
    perm_rng = np.random.default_rng(seed + 999)
    axes = np.asarray([f"axis_{i}" for i in range(8)], dtype=object)
    names = np.asarray([f"supp_{i}" for i in range(15)], dtype=object)
    paths = []
    for c, rid_lo in enumerate((0, 10, 20)):
        rids = np.repeat(np.arange(rid_lo, rid_lo + 10), 2)
        wins = np.tile(np.arange(2), 10)
        H = rng.normal(size=(20, 8))
        S = rng.normal(size=(20, 15))
        S[:, 0] = 10 * rids + wins
        if permute_rows:
            # Different physical row order, same (rid, win) -> (H, S) mapping.
            perm = perm_rng.permutation(20)
            rids, wins, H, S = rids[perm], wins[perm], H[perm], S[perm]
        p = out_dir / f"hazard_image_subwindows_shard_{c:03d}.npz"
        np.savez(p, H=H, supplement=S, realization_ids=rids, window_index=wins,
                 hazard_axes=axes, supplement_names=names,
                 **(hazard_image_provenance() if provenance else {}))
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

    for key in ("H", "supplement", "realization_ids", "window_index", "theta_index"):
        assert a[key].tobytes() == b[key].tobytes(), key
    assert [str(x) for x in a["hazard_axes"]] == [str(x) for x in b["hazard_axes"]]
    assert [str(x) for x in a["supplement_names"]] == [f"supp_{i}" for i in range(15)]
    assert int(a["window_years"]) == int(b["window_years"])
    # The supplement rows travel with their (rid, win) keys.
    np.testing.assert_array_equal(a["supplement"][:, 0],
                                  10 * a["realization_ids"] + a["window_index"])

    # The merged artifact carries the scoring-convention provenance every
    # reader checks.
    from scengen.diagnostics import check_hazard_image_provenance

    check_hazard_image_provenance(a, dir_a / "hazard_image_subwindows.npz")

    # Rows sorted by (rid, win); theta_index = rid // R.
    rid, win = a["realization_ids"], a["window_index"]
    assert list(zip(rid, win)) == sorted(zip(rid, win))
    np.testing.assert_array_equal(a["theta_index"], rid // 5)

    # Shards consumed after the merge (both layouts).
    assert not any(p.exists() for p in paths_a + paths_b)


def test_merge_refuses_a_shard_without_scoring_provenance(tmp_path, merge_shards):
    """A shard lacking the provenance legs (scored under another convention)
    is refused; nothing is written and no shard is consumed."""
    paths = _write_shards(tmp_path, permute_rows=False, provenance=False)
    with pytest.raises(ValueError, match="provenance|stale"):
        merge_shards(paths, tmp_path / "hazard_image_subwindows.npz", R=5)
    assert not (tmp_path / "hazard_image_subwindows.npz").exists()
    assert all(p.exists() for p in paths)
