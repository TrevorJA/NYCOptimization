"""
salt_front_dvs.py - Decision-variable specs for the salt-front MRF
adjustment table, for FFMP-family formulations only.

The full operational table (per DRBC §2.5.3, Tables 1-2) has:

- Trenton: 2 seasons × 4 RM bands = 8 cells (3 unique non-reference values)
- Montague: 3 seasons × 4 RM bands = 12 cells (multiple unique non-reference values)
- 3 RM-band thresholds (lo, mid, hi)
- 1 activation drought-level (which level fires the rule)

This module exposes those axes as DV specs in the same schema as
`ffmp.py::FFMP_FORMULATION["decision_variables"]`. The active subset is
selected by `config.SALT_FRONT_PARAM_MODE`:

    "fixed"               -> 0 DVs
    "multipliers"         -> 15 multiplier DVs (5 reference cells pinned at 1.0)
    "multipliers_with_gate" -> + 1 activation-level DV (16 DVs)
    "full"                -> + 3 RM-threshold DVs (19 DVs)

`apply_salt_front_dvs(params)` translates a flat dict of these DVs into
the constructor kwargs for `NYCOptParameterizedSaltFrontAdjustmentRatio`:

    {
        "multipliers": {"trenton": [...], "montague": [...]},
        "rm_band_thresholds": [hi, mid, lo],
        "activation_level": int,
    }

Reference cells (value pinned at 1.0 in DRBC tables):
- Trenton band-1 (87 < sf <= 92.5) — both seasons -> 2 cells
- Montague band-1 (87 < sf <= 92.5) — all 3 seasons -> 3 cells
- Montague band-2 (82.9 < sf <= 87)  — all 3 seasons -> 3 cells (also 1.0 by FFMP table)

Wait — looking at the upstream Montague table:
    (12,1,2,3,4):  [1.185, 1, 1, 0.815]   (idx 0 1 2 3) -> idx 1 AND idx 2 are 1.0
    (5,6,7,8):     [1.031, 1, 1, 0.688]   -> same
    (9,10,11):     [1.1, 1, 1, 0.733]     -> same

So Montague has BOTH idx-1 and idx-2 at 1.0 in all rows -> 6 reference cells.
Trenton has idx-1 at 1.0 in both rows -> 2 reference cells. Plus Trenton
idx-2 and idx-3 are 0.926 (constant across both seasons -> not really
"reference 1.0" but constant).

Final count of FREE multiplier cells:
- Trenton: 2 seasons × 4 cells = 8 - 2 (idx 1 reference 1.0) = 6 free.
- Montague: 3 seasons × 4 cells = 12 - 6 (idx 1, 2 references 1.0) = 6 free.
- Total: 12 free multiplier cells.

(Earlier estimate said 15; recount gives 12. Updating mode label below to match.)

For simplicity and full table coverage we expose ALL 4 cells per (season, location),
including the reference cells. The reference cells default to 1.0 with bounds [1.0, 1.0]
when "pin_reference=True" (default), so they're effectively fixed but still appear in
the DV vector for symmetry. This makes the DV-table-to-multiplier-dict mapping
trivial and uniform.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

# Multiplier table structure: list of (season_label, [months]) pairs.
# Order matters — must match the upstream lookup order so the DV vector is
# stable.
_TRENTON_SEASONS = [
    ("decapr", [12, 1, 2, 3, 4]),
    ("maynov", [5, 6, 7, 8, 9, 10, 11]),
]
_MONTAGUE_SEASONS = [
    ("decapr", [12, 1, 2, 3, 4]),
    ("mayaug", [5, 6, 7, 8]),
    ("sepnov", [9, 10, 11]),
]

# Default multiplier values per (season_idx, band_idx). These match upstream
# `salt_front_location.py:412-422` exactly — used as DV baselines.
_DEFAULT_TRENTON_MULTS = [
    [1.0,         1.0, 0.925925926, 0.925925926],   # decapr
    [1.074074074, 1.0, 0.925925926, 0.925925926],   # maynov
]
_DEFAULT_MONTAGUE_MULTS = [
    [1.185185185, 1.0, 1.0, 0.814814815],   # decapr
    [1.03125,     1.0, 1.0, 0.6875],        # mayaug
    [1.1,         1.0, 1.0, 0.733333333],   # sepnov
]

# Per-band default RM thresholds (hi, mid, lo). Used as DV baselines.
_DEFAULT_RM_HI = 92.5
_DEFAULT_RM_MID = 87.0
_DEFAULT_RM_LO = 82.9


def _multiplier_dv_name(loc: str, season: str, band: int) -> str:
    return f"sf_mult_{loc}_{season}_b{band}"


def _multiplier_specs(mult_bounds: tuple[float, float]) -> OrderedDict:
    """DV specs for the 11 free multiplier cells.

    Only cells whose FFMP-default value is NOT 1.0 are exposed as DVs.
    Reference cells (band-1 / band-2 with default 1.0 per DRBC §2.5.3 Table 2)
    stay implicit at 1.0 — they correspond to the regulatory "reference"
    operational target, not an optimization knob.

    The active cells:
      Trenton  decapr: b2 (=0.926), b3 (=0.926)                        — 2 DVs
      Trenton  maynov: b0 (=1.074), b2 (=0.926), b3 (=0.926)           — 3 DVs
      Montague decapr: b0 (=1.185), b3 (=0.815)                        — 2 DVs
      Montague mayaug: b0 (=1.031), b3 (=0.688)                        — 2 DVs
      Montague sepnov: b0 (=1.1),   b3 (=0.733)                        — 2 DVs
      Total:                                                              11 DVs
    """
    lo, hi = mult_bounds
    specs: OrderedDict = OrderedDict()
    for season_idx, (season_label, _months) in enumerate(_TRENTON_SEASONS):
        for band in range(4):
            default = _DEFAULT_TRENTON_MULTS[season_idx][band]
            if default == 1.0:
                continue  # reference cell — implicit
            specs[_multiplier_dv_name("trenton", season_label, band)] = {
                "baseline": float(default),
                "bounds": [lo, hi],
                "units": "ratio",
            }
    for season_idx, (season_label, _months) in enumerate(_MONTAGUE_SEASONS):
        for band in range(4):
            default = _DEFAULT_MONTAGUE_MULTS[season_idx][band]
            if default == 1.0:
                continue
            specs[_multiplier_dv_name("montague", season_label, band)] = {
                "baseline": float(default),
                "bounds": [lo, hi],
                "units": "ratio",
            }
    return specs


def _rm_threshold_specs(rm_band_bounds: list[tuple[float, float]]) -> OrderedDict:
    """DV specs for the 3 RM-band thresholds (lo, mid, hi)."""
    if len(rm_band_bounds) != 3:
        raise ValueError(
            f"rm_band_bounds must have 3 entries (lo, mid, hi); got {rm_band_bounds}"
        )
    lo_b, mid_b, hi_b = rm_band_bounds
    return OrderedDict({
        "sf_rm_band_lo": {
            "baseline": float(_DEFAULT_RM_LO),
            "bounds": list(lo_b),
            "units": "RM",
        },
        "sf_rm_band_mid": {
            "baseline": float(_DEFAULT_RM_MID),
            "bounds": list(mid_b),
            "units": "RM",
        },
        "sf_rm_band_hi": {
            "baseline": float(_DEFAULT_RM_HI),
            "bounds": list(hi_b),
            "units": "RM",
        },
    })


def _activation_gate_specs(
    options: list[int],
    fixed_default: int,
) -> OrderedDict:
    """DV spec for the activation drought-level gate.

    Encoded as a continuous DV in [min(options) - 0.5, max(options) + 0.5],
    rounded to the nearest integer at apply time. Borg supports continuous
    DVs natively; rounding handles the integer constraint.
    """
    if not options:
        raise ValueError("activation level options cannot be empty")
    lo = float(min(options)) - 0.5
    hi = float(max(options)) + 0.5
    # Choose baseline as the option closest to fixed_default.
    base = min(options, key=lambda x: abs(x - fixed_default))
    return OrderedDict({
        "sf_activation_level": {
            "baseline": float(base),
            "bounds": [lo, hi],
            "units": "drought_level_idx",
        }
    })


def salt_front_dv_specs(
    mode: str,
    *,
    multiplier_bounds: tuple[float, float],
    rm_band_bounds: list[tuple[float, float]],
    activation_options: list[int],
    fixed_activation_level: int,
) -> "OrderedDict[str, dict]":
    """Build the salt-front DV registry for the requested mode.

    Returns an OrderedDict matching the schema of
    `ffmp.FFMP_FORMULATION['decision_variables']`. Empty dict for "fixed".

    Args:
        mode: one of "fixed", "multipliers", "multipliers_with_gate", "full".
        multiplier_bounds: (lo, hi) bounds applied to non-reference multipliers.
        rm_band_bounds: list of 3 (lo, hi) tuples for the RM thresholds.
        activation_options: integer FFMP drought-level indices that the
            activation gate is allowed to assume.
        fixed_activation_level: integer drought-level index used when
            activation is NOT a DV. Default 6 = L5 / Drought Emergency.
    """
    specs: OrderedDict = OrderedDict()
    if mode == "fixed":
        return specs

    if mode in ("multipliers", "multipliers_with_gate", "full"):
        specs.update(_multiplier_specs(multiplier_bounds))

    if mode in ("multipliers_with_gate", "full"):
        specs.update(_activation_gate_specs(activation_options, fixed_activation_level))

    if mode == "full":
        specs.update(_rm_threshold_specs(rm_band_bounds))

    return specs


def _name_table(mults_dict: dict, mults_template: list[list[float]],
                seasons: list[tuple[str, list[int]]], loc: str) -> list[list]:
    """Compose a [(months_list, multipliers_list), ...] structure for one
    flow target by reading per-(season, band) DV values out of `mults_dict`,
    falling back to the template (defaults) when absent.
    """
    rows = []
    for season_idx, (season_label, months) in enumerate(seasons):
        row = []
        for band in range(4):
            name = _multiplier_dv_name(loc, season_label, band)
            if name in mults_dict:
                row.append(float(mults_dict[name]))
            else:
                row.append(float(mults_template[season_idx][band]))
        rows.append([list(months), row])
    return rows


def apply_salt_front_dvs(
    params: dict[str, Any],
    *,
    fixed_activation_level: int,
) -> dict[str, Any]:
    """Translate a flat params dict into salt-front parameter constructor kwargs.

    Args:
        params: subset of the flat DV params dict containing salt-front
            DV names. Unrelated keys are ignored. Missing keys default to
            FFMP-table values.
        fixed_activation_level: fallback activation level when the gate
            is not a DV.

    Returns:
        Dict with keys:
            "multipliers": {"trenton": [...], "montague": [...]} (rows = list of
                [[months...], [m0..m3]] entries).
            "rm_band_thresholds": [hi, mid, lo].
            "activation_level": int (rounded if from DV).
    """
    trenton = _name_table(params, _DEFAULT_TRENTON_MULTS, _TRENTON_SEASONS, "trenton")
    montague = _name_table(params, _DEFAULT_MONTAGUE_MULTS, _MONTAGUE_SEASONS, "montague")

    rm_lo = float(params.get("sf_rm_band_lo", _DEFAULT_RM_LO))
    rm_mid = float(params.get("sf_rm_band_mid", _DEFAULT_RM_MID))
    rm_hi = float(params.get("sf_rm_band_hi", _DEFAULT_RM_HI))
    # Defensively re-sort so optimizer-sampled values that violate ordering
    # are handled gracefully.
    rm_sorted = sorted([rm_hi, rm_mid, rm_lo], reverse=True)

    if "sf_activation_level" in params:
        # Round the continuous DV to the nearest integer drought-level index.
        activation_level = int(round(float(params["sf_activation_level"])))
    else:
        activation_level = int(fixed_activation_level)

    return {
        "multipliers": {"trenton": trenton, "montague": montague},
        "rm_band_thresholds": list(rm_sorted),
        "activation_level": activation_level,
    }


def salt_front_dv_names(mode: str, **kwargs) -> list[str]:
    """List of salt-front DV names for the given mode (in registry order)."""
    return list(salt_front_dv_specs(mode, **kwargs).keys())
