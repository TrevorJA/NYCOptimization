"""Tests for the flood-zone (L1a/L1b) spill-mitigation release DVs.

Covers the `_apply_flood_release_scaling` pipeline via `dvs_to_config`:
season-invariant multipliers on the default flood-zone factor rows, the
Group-7 seasonal-scaling exclusion, the Table 5 cap, the L1b <= L1a
monotonicity clamp, the delivery-factor monotonicity clamp, and the
audited bound set.
"""

import copy
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.formulations import get_baseline_values, get_n_vars, get_var_names
from src.simulation import (
    _CFS_TO_MGD,
    _apply_flood_release_scaling,
    _get_cached_defaults,
    _get_cached_nzone_defaults,
    dvs_to_config,
)

RESERVOIRS = ["cannonsville", "pepacton", "neversink"]
SEASONS = ["winter", "spring", "summer", "fall"]
LEVELS = ["level1a", "level1b", "level1c", "level2",
          "level3", "level4", "level5"]
SUMMER_IDX = np.arange(152, 244)  # Jun 1 - Aug 31 (leap DOY 153-244), 0-indexed
DEFAULT_CAPS_CFS = {"cannonsville": 4200.0, "pepacton": 2400.0,
                    "neversink": 3400.0}


def _dv(formulation="ffmp", **overrides):
    names = get_var_names(formulation)
    dv = get_baseline_values(formulation).copy()
    for name, value in overrides.items():
        dv[names.index(name)] = value
    return dv


def _factor_row(cfg, level, res):
    return np.asarray(
        cfg.get_mrf_factor_profile(f"{level}_factor_mrf_{res}", daily=True),
        dtype=float,
    )


def test_dv_registry():
    names = get_var_names("ffmp")
    assert get_n_vars("ffmp") == 69
    expected = [
        f"flood_release_scale_{zone}_{res}"
        for zone in ("l1a", "l1b")
        for res in RESERVOIRS
    ]
    start = names.index("zone_tshift_level5_c3") + 1
    assert names[start:start + 6] == expected
    assert names[start + 6] == "mrf_profile_scale_winter"
    assert not any("flood_max" in n for n in names)
    # MRF baselines are fixed constants, not DVs
    assert not any(n.startswith("mrf_cannonsville") or n.startswith("mrf_pep")
                   or n.startswith("mrf_nev") for n in names)

    names_n8 = get_var_names("ffmp_8")
    assert [n for n in names_n8
            if n.startswith("flood_release_scale_")] == expected
    assert not any("flood_max" in n for n in names_n8)


def test_recommended_bounds():
    """Lock in the audited bound set (2026-07-20/21)."""
    from src.formulations import get_formulation
    dvs = get_formulation("ffmp")["decision_variables"]
    l1a_upper = {"cannonsville": 1.35, "pepacton": 1.20, "neversink": 1.55}
    for res in RESERVOIRS:
        assert dvs[f"flood_release_scale_l1a_{res}"]["bounds"] \
            == [0.5, l1a_upper[res]]
        assert dvs[f"flood_release_scale_l1b_{res}"]["bounds"] == [0.5, 2.0]
    for season in SEASONS:
        assert dvs[f"mrf_profile_scale_{season}"]["bounds"] == [0.8, 2.6]
    # Per-breakpoint vertical offsets: per-curve upper caps applied to every
    # breakpoint; temporal shifts +/- 30 days per breakpoint.
    for k in range(4):
        assert dvs[f"zone_vshift_level1b_c{k}"]["bounds"] == [-0.10, 0.025]
        assert dvs[f"zone_vshift_level1c_c{k}"]["bounds"] == [-0.10, 0.05]
        assert dvs[f"zone_vshift_level5_c{k}"]["bounds"] == [-0.10, 0.10]
        assert dvs[f"zone_tshift_level5_c{k}"]["bounds"] == [-30.0, 30.0]
    assert dvs["nj_drought_factor_L4"]["bounds"] == [0.80, 1.0]
    assert dvs["nj_drought_factor_L5"]["bounds"] == [0.65, 1.0]
    assert dvs["mrf_target_scale_montague_level3"]["bounds"] == [0.5, 1.15]


def test_delivery_factor_monotonicity_clamp():
    """A deeper drought stage can never allow more diversion than a milder
    one: L3 >= L4 >= L5 enforced at apply time."""
    cfg = dvs_to_config(
        _dv(nyc_drought_factor_L3=0.60, nyc_drought_factor_L4=0.95,
            nyc_drought_factor_L5=0.90,
            nj_drought_factor_L4=0.80, nj_drought_factor_L5=1.0),
        "ffmp",
    )
    assert float(cfg.constants["level4_factor_delivery_nyc"]) == 0.60
    assert float(cfg.constants["level5_factor_delivery_nyc"]) == 0.60
    assert float(cfg.constants["level5_factor_delivery_nj"]) == 0.80


def test_baseline_reproduces_defaults():
    cfg = dvs_to_config(get_baseline_values("ffmp"), "ffmp")
    ref = _get_cached_defaults()
    for level in LEVELS:
        for res in RESERVOIRS:
            assert np.allclose(_factor_row(cfg, level, res),
                               _factor_row(ref, level, res)), (level, res)
    for res, cap in DEFAULT_CAPS_CFS.items():
        assert float(cfg.constants[f"flood_max_release_{res}_cfs"]) == cap


def test_multiplier_scales_whole_schedule():
    """A flood scale DV scales its zone's entire schedule (shape preserved),
    leaving other rows untouched."""
    cfg = dvs_to_config(_dv(flood_release_scale_l1a_cannonsville=1.2), "ffmp")
    ref = _get_cached_defaults()
    row = _factor_row(cfg, "level1a", "cannonsville")
    base = _factor_row(ref, "level1a", "cannonsville")
    assert np.allclose(row, 1.2 * base)
    for level in ["level1b", "level1c"]:
        assert np.allclose(_factor_row(cfg, level, "cannonsville"),
                           _factor_row(ref, level, "cannonsville")), level
    for res in ["pepacton", "neversink"]:
        assert np.allclose(_factor_row(cfg, "level1a", res),
                           _factor_row(ref, "level1a", res)), res


def test_mrf_baselines_fixed():
    """MRF baselines stay at the FFMP defaults for any DV vector."""
    cfg = dvs_to_config(_dv(mrf_profile_scale_summer=2.0), "ffmp")
    ref = _get_cached_defaults()
    for res in RESERVOIRS:
        assert float(cfg.constants[f"mrf_baseline_{res}"]) == \
            float(ref.constants[f"mrf_baseline_{res}"]), res


def test_group7_scaling_excludes_flood_rows():
    cfg = dvs_to_config(_dv(mrf_profile_scale_summer=2.0), "ffmp")
    ref = _get_cached_defaults()
    for res in RESERVOIRS:
        for level in ["level1a", "level1b"]:
            assert np.allclose(_factor_row(cfg, level, res),
                               _factor_row(ref, level, res)), (level, res)
    # Conservation rows still scale
    row = _factor_row(cfg, "level1c", "cannonsville")
    base = _factor_row(ref, "level1c", "cannonsville")
    assert np.allclose(row[SUMMER_IDX], 2.0 * base[SUMMER_IDX])
    other = np.setdiff1d(np.arange(len(base)), SUMMER_IDX)
    assert np.allclose(row[other], base[other])


def test_monotonicity_clamp_l1b_below_l1a():
    overrides = {}
    for res in RESERVOIRS:
        overrides[f"flood_release_scale_l1a_{res}"] = 0.5
        overrides[f"flood_release_scale_l1b_{res}"] = 2.0
    cfg = dvs_to_config(_dv(**overrides), "ffmp")
    for res in RESERVOIRS:
        l1a = _factor_row(cfg, "level1a", res)
        l1b = _factor_row(cfg, "level1b", res)
        assert (l1b <= l1a + 1e-12).all(), res


def test_cap_at_table5_constant():
    cfg = copy.deepcopy(_get_cached_defaults())
    baselines = {res: float(cfg.constants[f"mrf_baseline_{res}"])
                 for res in RESERVOIRS}
    params = {f"flood_release_scale_l1a_{res}": 4.0 for res in RESERVOIRS}
    _apply_flood_release_scaling(cfg, params)
    for res in RESERVOIRS:
        eff = _factor_row(cfg, "level1a", res) * baselines[res]
        cap_mgd = DEFAULT_CAPS_CFS[res] * _CFS_TO_MGD
        assert (eff <= cap_mgd + 1e-9).all(), res
    # Cannonsville L1a plateau (1500 cfs) x4 exceeds the 4200 cfs cap,
    # so the cap must actually bind somewhere.
    eff_can = _factor_row(cfg, "level1a", "cannonsville") * baselines["cannonsville"]
    assert np.isclose(eff_can.max(), DEFAULT_CAPS_CFS["cannonsville"] * _CFS_TO_MGD)


def test_scaled_profiles_step_only_on_ffmp_bin_edges():
    """Seasonally scaled schedules must only step on FFMP bin edges or FFMP
    season boundaries (Dec 1 / Apr 1 / Jun 1 / Sep 1), never mid-bin. The
    flood scales are season-invariant and contribute no edges at all."""
    overrides = {f"mrf_profile_scale_{s}": v for s, v in
                 zip(SEASONS, [1.7, 0.9, 1.2, 0.8])}
    for res in RESERVOIRS:
        overrides[f"flood_release_scale_l1a_{res}"] = 1.15
        overrides[f"flood_release_scale_l1b_{res}"] = 0.7
    cfg = dvs_to_config(_dv(**overrides), "ffmp")
    ref = _get_cached_defaults()
    season_starts = {91, 152, 244, 335}  # Apr 1, Jun 1, Sep 1, Dec 1 (0-based)
    for res in RESERVOIRS:
        for level in LEVELS:
            row = _factor_row(cfg, level, res)
            base = _factor_row(ref, level, res)
            base_edges = {i for i in range(1, 366) if base[i] != base[i - 1]}
            edges = {i for i in range(1, 366)
                     if not np.isclose(row[i], row[i - 1])}
            assert edges <= base_edges | season_starts, (level, res)


def test_nzone_baseline_and_scaling():
    cfg = dvs_to_config(get_baseline_values("ffmp_8"), "ffmp_8")
    ref = _get_cached_nzone_defaults(8)
    for i in range(9):
        for res in RESERVOIRS:
            assert np.allclose(_factor_row(cfg, f"zone_{i}", res),
                               _factor_row(ref, f"zone_{i}", res)), (i, res)

    cfg = dvs_to_config(
        _dv("ffmp_8", flood_release_scale_l1a_cannonsville=1.2), "ffmp_8"
    )
    row = _factor_row(cfg, "zone_0", "cannonsville")
    base = _factor_row(ref, "zone_0", "cannonsville")
    assert np.allclose(row, 1.2 * base)
    # zone_1 (L1b analog) and zone_2 untouched
    for zone in ["zone_1", "zone_2"]:
        assert np.allclose(_factor_row(cfg, zone, "cannonsville"),
                           _factor_row(ref, zone, "cannonsville")), zone


def test_nzone_mrf_baselines_fixed():
    cfg = dvs_to_config(_dv("ffmp_8", mrf_profile_scale_summer=2.0), "ffmp_8")
    ref = _get_cached_nzone_defaults(8)
    for res in RESERVOIRS:
        assert float(cfg.constants[f"mrf_baseline_{res}"]) == \
            float(ref.constants[f"mrf_baseline_{res}"]), res
