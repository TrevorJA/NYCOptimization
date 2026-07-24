"""Tests for storage-zone shift DVs (per-breakpoint vertical + temporal) and clamping.

Covers the `_apply_zone_shifts` pipeline via `dvs_to_config`: each curve's
_ZONE_CORNER_COUNT major breakpoints (detected on the baseline curve) are
offset vertically (additive `zone_vshift_*`) and moved along the day-of-year
axis (`zone_tshift_*`); the daily curve is rebuilt as a circular
piecewise-linear curve through the moved, offset breakpoints, then clipped to
[0, 1] and monotonicity-clamped.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.formulations import get_baseline_values, get_n_vars, get_var_names
from src.formulations.ffmp import _BREAKPOINT_COUNT
from src.simulation import (
    _ZONE_CORNER_COUNT,
    _reconstruct_breakpoint_curve,
    _zone_curve_corners,
    dvs_to_config,
    _get_cached_defaults,
)

ZONE_LEVELS = ["level1b", "level1c", "level2", "level3", "level4", "level5"]
BP = _ZONE_CORNER_COUNT


def _dv(**overrides):
    """Baseline DV vector with named overrides applied."""
    names = get_var_names("ffmp")
    dv = get_baseline_values("ffmp").copy()
    for name, value in overrides.items():
        dv[names.index(name)] = value
    return dv


def _zones(cfg):
    return {lvl: np.asarray(cfg.get_storage_zone_profile(lvl), dtype=float)
            for lvl in ZONE_LEVELS}


def test_breakpoint_count_agrees():
    # The registry index count and the apply-side corner count must match.
    assert _BREAKPOINT_COUNT == _ZONE_CORNER_COUNT == 4


def test_dv_registry():
    names = get_var_names("ffmp")
    assert get_n_vars("ffmp") == 69
    vshift = [n for n in names if n.startswith("zone_vshift_")]
    assert vshift == [f"zone_vshift_{lvl}_c{k}"
                      for lvl in ZONE_LEVELS for k in range(BP)]
    tshift = [n for n in names if n.startswith("zone_tshift_")]
    assert tshift == [f"zone_tshift_{lvl}_c{k}"
                      for lvl in ZONE_LEVELS for k in range(BP)]


def test_each_curve_has_four_corners():
    ref = _zones(_get_cached_defaults())
    for lvl in ZONE_LEVELS:
        assert len(_zone_curve_corners(ref[lvl])) == _ZONE_CORNER_COUNT, lvl


def test_baseline_reproduces_defaults():
    # The default FFMP curves are piecewise-linear through their 4 corners,
    # so the breakpoint reconstruction reproduces them exactly at baseline.
    base = _zones(dvs_to_config(get_baseline_values("ffmp"), "ffmp"))
    ref = _zones(_get_cached_defaults())
    for lvl in ZONE_LEVELS:
        assert np.allclose(base[lvl], ref[lvl], atol=1e-9), lvl


def test_reconstruct_helper_hits_nodes_and_wraps():
    x = np.array([10.0, 100.0, 200.0, 300.0])
    y = np.array([0.2, 0.5, 0.4, 0.3])
    curve = _reconstruct_breakpoint_curve(x, y, 366)
    assert curve.shape == (366,)
    for xi, yi in zip(x, y):
        assert np.isclose(curve[int(xi)], yi)
    # Linear between nodes: midpoint of [10, 100] equals the value average.
    assert np.isclose(curve[55], (0.2 + 0.5) / 2, atol=1e-6)
    # Wrap: day 0 interpolates between the last node (300) and the first (10).
    frac = (366 - 300) / ((366 - 300) + 10)
    assert np.isclose(curve[0], 0.3 + frac * (0.2 - 0.3), atol=1e-6)
    # Node order should not matter.
    shuffled = _reconstruct_breakpoint_curve(x[::-1], y[::-1], 366)
    assert np.allclose(curve, shuffled)


def test_vertical_offset_is_local_and_additive():
    """A single breakpoint's additive vertical offset moves only that node
    (and its two adjacent segments), leaving other curves untouched."""
    ref = _zones(_get_cached_defaults())["level5"]
    corners = _zone_curve_corners(ref)
    cfg = dvs_to_config(_dv(zone_vshift_level5_c1=-0.05), "ffmp")
    zones = _zones(cfg)
    offset = zones["level5"] - ref
    # Offset is exactly the DV at its own breakpoint, zero at the others.
    assert np.isclose(offset[corners[1]], -0.05)
    for k in (0, 2, 3):
        assert np.isclose(offset[corners[k]], 0.0, atol=1e-9)
    # Piecewise-linear offset: second difference vanishes off the corners.
    n = ref.size
    interior = np.setdiff1d(
        np.arange(1, n - 1),
        np.concatenate([corners, corners - 1, corners + 1]),
    )
    # atol tolerant of the ~1e-9 non-PL float noise in the stored curves.
    assert np.allclose(np.diff(offset, 2)[interior - 1], 0.0, atol=1e-6)
    # Lowering the deepest curve triggers no clamp on the others.
    other = _zones(_get_cached_defaults())
    for lvl in ["level1b", "level1c", "level2", "level3", "level4"]:
        assert np.allclose(zones[lvl], other[lvl], atol=1e-9), lvl


def test_temporal_shift_moves_only_that_breakpoint():
    """Shifting one breakpoint's day carries its value to the new position
    while the other breakpoints stay pinned; other curves are untouched."""
    ref = _zones(_get_cached_defaults())["level5"]
    corners = _zone_curve_corners(ref)
    days = 20
    cfg = dvs_to_config(_dv(zone_tshift_level5_c1=float(days)), "ffmp")
    zones = _zones(cfg)
    n = ref.size
    # The moved breakpoint carries its value to corner+days.
    assert np.isclose(zones["level5"][(corners[1] + days) % n], ref[corners[1]])
    # The other breakpoints remain pinned at their original day/value.
    for k in (0, 2, 3):
        assert np.isclose(zones["level5"][corners[k]], ref[corners[k]])
    other = _zones(_get_cached_defaults())
    for lvl in ["level1b", "level1c", "level2", "level3", "level4"]:
        assert np.allclose(zones[lvl], other[lvl], atol=1e-9), lvl


def test_temporal_shift_rounds_to_whole_days():
    ref = _zones(_get_cached_defaults())["level5"]
    corners = _zone_curve_corners(ref)
    a = _zones(dvs_to_config(_dv(zone_tshift_level5_c0=14.6), "ffmp"))["level5"]
    b = _zones(dvs_to_config(_dv(zone_tshift_level5_c0=15.0), "ffmp"))["level5"]
    assert np.allclose(a, b)


def test_clamp_enforces_monotonicity_under_extreme_shifts():
    uppers = {"level1b": 0.025, "level1c": 0.05}
    overrides = {}
    for i, lvl in enumerate(ZONE_LEVELS):
        hi = uppers.get(lvl, 0.10)
        for k in range(BP):
            overrides[f"zone_vshift_{lvl}_c{k}"] = hi if i % 2 else -0.10
            overrides[f"zone_tshift_{lvl}_c{k}"] = -30.0 if i % 2 else 30.0
    zones = _zones(dvs_to_config(_dv(**overrides), "ffmp"))
    stacked = np.vstack([zones[lvl] for lvl in ZONE_LEVELS])
    assert (np.diff(stacked, axis=0) <= 1e-12).all()
    assert stacked.min() >= 0.0 and stacked.max() <= 1.0


def test_nzone_shifts():
    names = get_var_names("ffmp_8")
    assert [n for n in names if n.startswith("zone_vshift_")] == [
        f"zone_vshift_zone_{i}_c{k}" for i in range(1, 9) for k in range(BP)
    ]
    assert [n for n in names if n.startswith("zone_tshift_")] == [
        f"zone_tshift_zone_{i}_c{k}" for i in range(1, 9) for k in range(BP)
    ]
    # Offset the deepest curve (zone_8): lowering it triggers no clamp
    # cascade onto the less-severe curves, so zone_1 stays untouched.
    dv = get_baseline_values("ffmp_8").copy()
    dv[names.index("zone_vshift_zone_8_c0")] = -0.05
    cfg = dvs_to_config(dv, "ffmp_8")
    from src.simulation import _get_cached_nzone_defaults
    ref = _get_cached_nzone_defaults(8)
    z8_ref = np.asarray(ref.get_storage_zone_profile("zone_8"), dtype=float)
    z1_ref = np.asarray(ref.get_storage_zone_profile("zone_1"), dtype=float)
    z8 = np.asarray(cfg.get_storage_zone_profile("zone_8"), dtype=float)
    z1 = np.asarray(cfg.get_storage_zone_profile("zone_1"), dtype=float)
    assert not np.allclose(z8, z8_ref)  # deepest zone moved
    assert np.allclose(z1, z1_ref)      # least-severe zone untouched
