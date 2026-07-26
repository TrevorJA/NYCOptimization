"""Tests for storage-zone shift DVs (two vertical plateaus + one temporal).

Covers the `_apply_zone_shifts` pipeline via `dvs_to_config`: each curve is a
trapezoid whose low plateau (baseline min, the void) and high plateau
(baseline max, the refill target) are shifted independently by
`zone_vshift_{level}_lower` / `zone_vshift_{level}_upper`, the values affinely
remapped between the new plateau levels, then slid by `zone_tshift_{level}`,
clipped to [0, 1], and cross-curve monotonicity-clamped. A within-curve clamp
keeps the void at or below the refill target.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.formulations import get_baseline_values, get_n_vars, get_var_names
from src.simulation import dvs_to_config, _get_cached_defaults

ZONE_LEVELS = ["level1b", "level1c", "level2", "level3", "level4", "level5"]


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


def _affine(ref, vlow, vup):
    """Expected per-curve remap (pre-roll, pre-clip, pre-cross-clamp)."""
    lo, hi = ref.min(), ref.max()
    hi_new = hi + vup
    lo_new = min(lo + vlow, hi_new)
    return lo_new + (ref - lo) / (hi - lo) * (hi_new - lo_new)


def test_dv_registry():
    names = get_var_names("ffmp")
    assert get_n_vars("ffmp") == 39
    vshift = [n for n in names if n.startswith("zone_vshift_")]
    assert vshift == [f"zone_vshift_{lvl}_{end}"
                      for lvl in ZONE_LEVELS for end in ("lower", "upper")]
    tshift = [n for n in names if n.startswith("zone_tshift_")]
    assert tshift == [f"zone_tshift_{lvl}" for lvl in ZONE_LEVELS]


def test_baseline_reproduces_defaults():
    # All-zero shifts reproduce the default curves exactly.
    base = _zones(dvs_to_config(get_baseline_values("ffmp"), "ffmp"))
    ref = _zones(_get_cached_defaults())
    for lvl in ZONE_LEVELS:
        assert np.allclose(base[lvl], ref[lvl], atol=1e-9), lvl


def test_lower_plateau_shift_deepens_void_without_lowering_refill():
    """The signature behavior: lowering the low plateau deepens the void while
    the high plateau (refill target) is untouched; other curves untouched."""
    ref = _zones(_get_cached_defaults())
    cfg = dvs_to_config(_dv(zone_vshift_level5_lower=-0.05), "ffmp")
    zones = _zones(cfg)
    # level5 is most-severe: lowering keeps it below level4, so no cross-clamp.
    assert np.allclose(zones["level5"], _affine(ref["level5"], -0.05, 0.0))
    assert np.isclose(zones["level5"].max(), ref["level5"].max())        # refill held
    assert np.isclose(zones["level5"].min(), ref["level5"].min() - 0.05)  # void deepened
    for lvl in ["level1b", "level1c", "level2", "level3", "level4"]:
        assert np.allclose(zones[lvl], ref[lvl], atol=1e-9), lvl


def test_upper_plateau_shift_lowers_refill_without_raising_void():
    """Lowering the high plateau lowers the refill target while the low plateau
    (void) is untouched."""
    ref = _zones(_get_cached_defaults())
    cfg = dvs_to_config(_dv(zone_vshift_level5_upper=-0.05), "ffmp")
    zones = _zones(cfg)
    assert np.allclose(zones["level5"], _affine(ref["level5"], 0.0, -0.05))
    assert np.isclose(zones["level5"].max(), ref["level5"].max() - 0.05)  # refill lowered
    assert np.isclose(zones["level5"].min(), ref["level5"].min())         # void held


def test_temporal_shift_rolls_whole_curve():
    ref = _zones(_get_cached_defaults())
    days = 20
    cfg = dvs_to_config(_dv(zone_tshift_level5=float(days)), "ffmp")
    zones = _zones(cfg)
    expected = np.minimum(
        np.clip(np.roll(ref["level5"], days), 0.0, 1.0), ref["level4"]
    )
    assert np.allclose(zones["level5"], expected)
    for lvl in ["level1b", "level1c", "level2", "level3", "level4"]:
        assert np.allclose(zones[lvl], ref[lvl], atol=1e-9), lvl


def test_temporal_shift_rounds_to_whole_days():
    a = _zones(dvs_to_config(_dv(zone_tshift_level5=14.6), "ffmp"))["level5"]
    b = _zones(dvs_to_config(_dv(zone_tshift_level5=15.0), "ffmp"))["level5"]
    assert np.allclose(a, b)


def test_within_curve_clamp_prevents_void_above_refill():
    """Raising the void past a lowered refill target flattens the curve at the
    refill level rather than inverting it."""
    # level1c: baseline 0.85 -> 1.0. Push void +0.10 (0.95) and refill -0.10
    # (0.90): the void would exceed the refill, so the curve flattens at 0.90.
    cfg = dvs_to_config(
        _dv(zone_vshift_level1c_lower=0.10, zone_vshift_level1c_upper=-0.10),
        "ffmp",
    )
    z = _zones(cfg)["level1c"]
    assert np.isclose(z.max(), z.min(), atol=1e-9)  # flat
    assert np.allclose(z, 0.90, atol=1e-9)


def test_clamp_enforces_monotonicity_under_extreme_shifts():
    lower_cap = {"level1b": 0.025}
    upper_cap = {"level1b": 0.0, "level1c": 0.0, "level2": 0.0}
    overrides = {}
    for i, lvl in enumerate(ZONE_LEVELS):
        overrides[f"zone_vshift_{lvl}_lower"] = (
            lower_cap.get(lvl, 0.10) if i % 2 else -0.10)
        overrides[f"zone_vshift_{lvl}_upper"] = (
            upper_cap.get(lvl, 0.10) if i % 2 else -0.10)
        overrides[f"zone_tshift_{lvl}"] = -30.0 if i % 2 else 30.0
    zones = _zones(dvs_to_config(_dv(**overrides), "ffmp"))
    stacked = np.vstack([zones[lvl] for lvl in ZONE_LEVELS])
    assert (np.diff(stacked, axis=0) <= 1e-12).all()
    assert stacked.min() >= 0.0 and stacked.max() <= 1.0


def test_nzone_shifts():
    names = get_var_names("ffmp_8")
    assert [n for n in names if n.startswith("zone_vshift_")] == [
        f"zone_vshift_zone_{i}_{end}"
        for i in range(1, 9) for end in ("lower", "upper")
    ]
    assert [n for n in names if n.startswith("zone_tshift_")] == [
        f"zone_tshift_zone_{i}" for i in range(1, 9)
    ]
    # Deepening the deepest curve's void (zone_8) triggers no clamp cascade
    # onto the less-severe curves, so zone_1 stays untouched.
    dv = get_baseline_values("ffmp_8").copy()
    dv[names.index("zone_vshift_zone_8_lower")] = -0.05
    cfg = dvs_to_config(dv, "ffmp_8")
    from src.simulation import _get_cached_nzone_defaults
    ref = _get_cached_nzone_defaults(8)
    z8_ref = np.asarray(ref.get_storage_zone_profile("zone_8"), dtype=float)
    z1_ref = np.asarray(ref.get_storage_zone_profile("zone_1"), dtype=float)
    z8 = np.asarray(cfg.get_storage_zone_profile("zone_8"), dtype=float)
    z1 = np.asarray(cfg.get_storage_zone_profile("zone_1"), dtype=float)
    assert not np.allclose(z8, z8_ref)  # deepest zone moved
    assert np.allclose(z1, z1_ref)      # least-severe zone untouched
