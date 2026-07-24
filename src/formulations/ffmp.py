"""
ffmp.py - Parameterized FFMP (2017 Flexible Flow Management Program) formulation.

Defines decision variables, bounds, and baselines for the FFMP formulation
that re-optimizes existing FFMP parameters within plausible ranges, and
supports N-zone variable-resolution variants.

Salt-front-dependent flow-target adjustment DVs (FFMP-family only) are
merged in conditionally based on `config.SALT_FRONT_PARAM_MODE`. See
`salt_front_dvs.py` and `decisions/2026-04-29_salt_front_parameterization.md`.
"""

import numpy as np
from collections import OrderedDict

from .salt_front_dvs import salt_front_dv_specs


###############################################################################
# N-zone interpolation helper
###############################################################################

def _interpolate_factors(default_values, n_target):
    """Linearly interpolate a list of values to n_target points.

    Used to scale default 7-level FFMP drought factor arrays to an arbitrary
    number of drought levels in generate_ffmp_formulation(n_zones).

    Args:
        default_values: List/array of source values.
        n_target: Number of output points.

    Returns:
        List of length n_target.
    """
    x_default = np.linspace(0, 1, len(default_values))
    x_target = np.linspace(0, 1, n_target)
    return list(np.interp(x_target, x_default, default_values))


###############################################################################
# Downstream flow-target factor scaling DVs
###############################################################################

#: Default FFMP monthly flow-target factor tables (jan..dec) for the seven
#: standard drought levels. Levels 1a-2 are 1.0 (no adjustment; the Decree
#: targets apply unmodified) and are never exposed as DVs. Values match
#: ffmp_reservoir_operation_monthly_profiles.csv.
_DEFAULT_FLOW_TARGET_FACTORS = {
    "montague": np.array([
        [1.0] * 12,                                # level1a
        [1.0] * 12,                                # level1b
        [1.0] * 12,                                # level1c
        [1.0] * 12,                                # level2
        [0.942857] * 12,                           # level3
        [0.885714] * 12,                           # level4
        [0.771429] * 4 + [0.914286] * 4 + [0.857143] * 3 + [0.771429],  # level5
    ]),
    "trenton": np.array([
        [1.0] * 12,
        [1.0] * 12,
        [1.0] * 12,
        [1.0] * 12,
        [0.9] * 12,
        [0.9] * 12,
        [0.9] * 12,
    ]),
}

#: Bounds for the flow-target factor scale multipliers. The effective factor
#: (default table value x DV) is capped at 1.0 at apply time so adjusted
#: targets never exceed the Decree-fixed baseline target. The cap binds at
#: scale ~1.06-1.13 depending on the row, so 1.15 leaves every row just
#: enough headroom to reach the cap without a long flat region above it.
FLOW_TARGET_SCALE_BOUNDS = [0.5, 1.15]


def _add_flow_target_scale_dvs(dvs, loc, level_names, factor_matrix):
    """Append flow-target factor scale DVs for the drought-affected levels.

    One non-seasonal multiplier DV per drought level is exposed for each
    level whose default monthly factor row deviates from 1.0. The multiplier
    scales the level's default monthly factors across all months, so
    baseline = 1.0 reproduces the FFMP exactly. Levels with all-unity
    factors (normal operations) are never exposed — the Decree target
    applies unmodified.

    Args:
        dvs: Target DV registry (OrderedDict), mutated in place.
        loc: Location tag used in DV names ("montague" or "trenton").
        level_names: Drought level names aligned with factor_matrix rows.
        factor_matrix: (n_levels, 12) default monthly factor table.
    """
    for i, level in enumerate(level_names):
        if np.min(factor_matrix[i]) >= 1.0:
            continue
        dvs[f"mrf_target_scale_{loc}_{level}"] = {
            "baseline": 1.0,
            "bounds": list(FLOW_TARGET_SCALE_BOUNDS),
            "units": "multiplier",
        }


###############################################################################
# Salt-front DV merge helper
###############################################################################

def _merge_salt_front_dvs(dvs: OrderedDict, n_drought_levels: int = None) -> OrderedDict:
    """Append salt-front DVs to the FFMP DV registry per active config.

    Reads `config.SALT_FRONT_PARAM_MODE` and friends at call time so env
    overrides set in SLURM scripts are honored. Mutates and returns `dvs`.
    No-op when mode == "fixed" (default).

    Args:
        dvs: target DV registry (OrderedDict) to extend in place.
        n_drought_levels: drought-level count of the FFMP variant. When
            provided, the activation-gate DV's allowed levels resolve to the
            top 3 indices of this N-zone config (`[N-2, N-1, N]` for
            n_drought_levels = N+1). When None (stock FFMP), falls back to
            `config.SALT_FRONT_ACTIVATION_LEVEL_OPTIONS` (= `[4, 5, 6]` by
            default — matches the standard 7-level FFMP).
    """
    # Local import to avoid a partial-import cycle (config.py imports from
    # this module at top level).
    from config import (
        SALT_FRONT_PARAM_MODE,
        SALT_FRONT_MULTIPLIER_BOUNDS,
        SALT_FRONT_RM_BAND_BOUNDS,
        SALT_FRONT_ACTIVATION_LEVEL_OPTIONS,
        SALT_FRONT_FIXED_ACTIVATION_LEVEL,
    )
    if n_drought_levels is not None:
        # Top 3 drought-level indices for the active N-zone config. Mirrors
        # the relationship in stock FFMP where [4,5,6] are the top 3 of 7
        # levels (indices 0..6).
        activation_options = list(range(n_drought_levels - 3, n_drought_levels))
        fixed_activation = n_drought_levels - 1
    else:
        activation_options = list(SALT_FRONT_ACTIVATION_LEVEL_OPTIONS)
        fixed_activation = SALT_FRONT_FIXED_ACTIVATION_LEVEL
    extra = salt_front_dv_specs(
        SALT_FRONT_PARAM_MODE,
        multiplier_bounds=SALT_FRONT_MULTIPLIER_BOUNDS,
        rm_band_bounds=SALT_FRONT_RM_BAND_BOUNDS,
        activation_options=activation_options,
        fixed_activation_level=fixed_activation,
    )
    for name, spec in extra.items():
        if name in dvs:
            raise ValueError(
                f"Salt-front DV name '{name}' collides with an existing FFMP DV"
            )
        dvs[name] = spec
    return dvs


###############################################################################
# Standard FFMP formulation (69 DVs base, optionally extended via salt_front)
###############################################################################

# --- Flood-zone (L1a/L1b) spill-mitigation release scaling ---
# Dimensionless multipliers on the DEFAULT FFMP Tables 4a-4g flood-zone
# release schedule (e.g., L1a Cannonsville = mult x 1500 cfs), applied by
# simulation._apply_flood_release_scaling. Season-invariant — matching the
# FFMP, which holds these rows constant across its tables and seasons;
# seasonal flood policy (void scheduling, CSSO shape) is carried by the
# per-breakpoint zone-boundary shift DVs (zone_vshift_*) instead. The
# multiplier form preserves the within-year shape (L1a-absent window
# Apr 16-Jun 15, Neversink L1b step).
# The Table 5 combined-discharge caps (flood_max_release_{res}_cfs =
# 4200/2400/3400) are physical/regulatory constants and are NOT decision
# variables. L1a upper bounds are anchored to the maximum controlled
# release observed 2000-2021 (2062/842/303 cfs — the demonstrated
# release-works capacity) divided by the L1a schedule rate (1500/700/190
# cfs); 2.0 x L1b stays within that demonstrated range for all three
# reservoirs. All uppers sit below the Table 5 combined caps.
FLOOD_RELEASE_ZONES = ["l1a", "l1b"]
_FLOOD_RESERVOIRS = ["cannonsville", "pepacton", "neversink"]
_FLOOD_SCALE_UPPER = {
    ("l1a", "cannonsville"): 1.35,
    ("l1a", "pepacton"): 1.20,
    ("l1a", "neversink"): 1.55,
    ("l1b", "cannonsville"): 2.0,
    ("l1b", "pepacton"): 2.0,
    ("l1b", "neversink"): 2.0,
}
FLOOD_RELEASE_SCALE_SPECS = OrderedDict(
    (
        f"flood_release_scale_{zone}_{res}",
        {
            "baseline": 1.0,
            "bounds": [0.5, _FLOOD_SCALE_UPPER[(zone, res)]],
            "units": "multiplier",
        },
    )
    for zone in FLOOD_RELEASE_ZONES
    for res in _FLOOD_RESERVOIRS
)

# --- Per-breakpoint storage-zone boundary shifts ---
# Each storage-zone threshold curve is represented by its _BREAKPOINT_COUNT
# major breakpoints (the largest-curvature corners of the baseline curve,
# detected once by simulation._zone_curve_corners). Each breakpoint gets two
# DVs: an additive vertical offset (fraction of capacity, zone_vshift_*) and
# a temporal shift (days, zone_tshift_*). At apply time the breakpoints are
# offset/moved and the daily curve is rebuilt as a circular piecewise-linear
# curve through them (simulation._apply_zone_shifts), then clipped to [0, 1]
# and monotonicity-clamped. All-zero DVs reproduce the piecewise-linear form
# of the default curves through their breakpoints. This is where the
# formulation's seasonal flood policy lives — the FFMP's own seasonal flood
# instrument is the CSSO / zone boundary geometry (15% Nov-Feb void), not the
# release rates. Vertical upper bounds are trimmed for the top two curves,
# which sit near full storage (level1b 0.975-1.0, level1c 0.85-1.0 of
# capacity — larger upward offsets clip to 1.0).
#
# _BREAKPOINT_COUNT MUST equal simulation._ZONE_CORNER_COUNT so the DV indices
# c0..c{n-1} align one-to-one with the detected baseline corners.
_BREAKPOINT_COUNT = 4
_ZONE_VSHIFT_UPPER = {"level1b": 0.025, "level1c": 0.05}
_ZONE_VSHIFT_BOUND = 0.10
_ZONE_TSHIFT_BOUND = 30.0


def _zone_breakpoint_specs(levels, vshift_upper_by_level):
    """Build the per-breakpoint zone-shift DV specs for the given curves.

    For each curve and each of the _BREAKPOINT_COUNT breakpoints, adds an
    additive vertical-offset DV (``zone_vshift_{level}_c{k}``, fraction of
    capacity) and a temporal-shift DV (``zone_tshift_{level}_c{k}``, days).
    Baselines are 0.0 so the curve is unperturbed at the baseline vector.

    Args:
        levels: Storage-zone curve names.
        vshift_upper_by_level: Per-curve upper-bound override for the
            vertical offset (defaults to ``_ZONE_VSHIFT_BOUND``).

    Returns:
        OrderedDict of DV specs (all vertical then all temporal per curve).
    """
    specs = OrderedDict()
    for level in levels:
        upper = vshift_upper_by_level.get(level, _ZONE_VSHIFT_BOUND)
        for k in range(_BREAKPOINT_COUNT):
            specs[f"zone_vshift_{level}_c{k}"] = {
                "baseline": 0.0,
                "bounds": [-_ZONE_VSHIFT_BOUND, upper],
                "units": "fraction",
            }
        for k in range(_BREAKPOINT_COUNT):
            specs[f"zone_tshift_{level}_c{k}"] = {
                "baseline": 0.0,
                "bounds": [-_ZONE_TSHIFT_BOUND, _ZONE_TSHIFT_BOUND],
                "units": "days",
            }
    return specs

# FFMP decision variable specification.
# Each entry: {"baseline": <default value>, "bounds": [lo, hi], "units": <str>}
FFMP_FORMULATION = {
    "description": "Parameterized 2017 FFMP rule structure",
    "decision_variables": OrderedDict({

        # NOTE: The reservoir MRF baselines (122.8/64.63/48.47 MGD), the
        # Montague/Trenton baseline flow targets, and the NYC diversion cap
        # are NOT decision variables. The baselines are the fixed FFMP
        # Table 4a base rates (operational variation comes through the
        # mrf_profile_scale_* FAW-like seasonal scales below); the targets
        # and cap are 1954 Decree quantities fixed at
        # config.MONTAGUE_DECREE_TARGET_MGD, TRENTON_DECREE_TARGET_MGD,
        # and NYC_DECREE_DIVERSION_CAP_MGD.

        # --- NYC drought factors (L3, L4, L5) ---
        # L1a-L2 factors are effectively unconstrained (set to large values)
        "nyc_drought_factor_L3": {
            "baseline": 0.85,
            "bounds": [0.60, 1.0],
            "units": "fraction",
        },
        "nyc_drought_factor_L4": {
            "baseline": 0.70,
            "bounds": [0.40, 0.95],
            "units": "fraction",
        },
        "nyc_drought_factor_L5": {
            "baseline": 0.65,
            "bounds": [0.30, 0.90],
            "units": "fraction",
        },

        # --- NJ drought factors (L4, L5) ---
        # No NJ delivery objective is active, so these lower bounds are the
        # only guardrail on the Decree-party interest; they bracket the
        # negotiated FFMP values (0.90/0.80). Widen only if the NJ
        # reliability objective is activated.
        "nj_drought_factor_L4": {
            "baseline": 0.90,
            "bounds": [0.80, 1.0],
            "units": "fraction",
        },
        "nj_drought_factor_L5": {
            "baseline": 0.80,
            "bounds": [0.65, 1.0],
            "units": "fraction",
        },

        # --- Per-breakpoint storage-zone boundary shifts ---
        # (specs built by _zone_breakpoint_specs above: an additive vertical
        # offset (zone_vshift_*, fraction of capacity) and a temporal shift
        # (zone_tshift_*, days) per breakpoint, _BREAKPOINT_COUNT per curve;
        # the daily curve is rebuilt piecewise-linear through the moved,
        # offset breakpoints at apply time)
        **_zone_breakpoint_specs(
            ["level1b", "level1c", "level2", "level3", "level4", "level5"],
            _ZONE_VSHIFT_UPPER,
        ),

        # --- Flood-zone (L1a/L1b) spill-mitigation release scaling ---
        # (specs defined once in FLOOD_RELEASE_SCALE_SPECS above)
        **FLOOD_RELEASE_SCALE_SPECS,

        # --- MRF seasonal profile scaling (4 seasons) ---
        # FAW-like scaling of the conservation-release schedules. Bounds
        # match the FFMP's own Table 4a-4g FAW envelope (~1.0-2.6x base for
        # the FAW-varying zones); the 0.8 floor keeps releases near the
        # negotiated Table 4a base rates, the only protection for the
        # tailwater fishery interest (no habitat objective is active).
        "mrf_profile_scale_winter": {
            "baseline": 1.0,
            "bounds": [0.8, 2.6],
            "units": "multiplier",
        },
        "mrf_profile_scale_spring": {
            "baseline": 1.0,
            "bounds": [0.8, 2.6],
            "units": "multiplier",
        },
        "mrf_profile_scale_summer": {
            "baseline": 1.0,
            "bounds": [0.8, 2.6],
            "units": "multiplier",
        },
        "mrf_profile_scale_fall": {
            "baseline": 1.0,
            "bounds": [0.8, 2.6],
            "units": "multiplier",
        },
    }),
}

# --- Downstream flow-target factor scaling (per drought level) ---
# Exposed only for levels whose FFMP factors deviate from 1.0 (L3/L4/L5 at
# both locations) => 2 locations x 3 levels = 6 DVs. Non-seasonal: one
# multiplier scales the level's full monthly factor row.
_STANDARD_LEVELS = ["level1a", "level1b", "level1c", "level2",
                    "level3", "level4", "level5"]
for _loc in ("montague", "trenton"):
    _add_flow_target_scale_dvs(
        FFMP_FORMULATION["decision_variables"], _loc,
        _STANDARD_LEVELS, _DEFAULT_FLOW_TARGET_FACTORS[_loc],
    )

###############################################################################
# Formulation factory
###############################################################################

def generate_ffmp_formulation(n_zones=None):
    """Generate an FFMP formulation, optionally with variable zone resolution.

    With n_zones=None (default), returns the standard 69-DV formulation
    matching the 2017 FFMP's 7 drought levels (level1a..level5).

    With n_zones=N, generates an N-zone variant where:
    - N storage zone boundary curves are optimized (zone_1..zone_N)
    - N+1 drought levels (zone_0=normal, zone_1..zone_N=drought)
    - Delivery factors only included for levels where interpolated
      baseline is < the unconstrained threshold (< 100 for NYC, < 1.0 for NJ)
    - N=6 is equivalent to the standard 7-level FFMP in zone count

    Args:
        n_zones: Number of storage zone boundary curves, or None for standard.

    Returns:
        Dict with "description" and "decision_variables" keys.
    """
    if n_zones is None:
        return FFMP_FORMULATION

    # Default 7-level baselines for interpolation
    default_nyc_factors = [1_000_000, 1_000_000, 1_000_000, 1_000_000,
                           0.85, 0.70, 0.65]
    default_nj_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 0.90, 0.80]

    interp_nyc = _interpolate_factors(default_nyc_factors, n_zones + 1)
    interp_nj = _interpolate_factors(default_nj_factors, n_zones + 1)

    drought_levels = ["zone_0"] + [f"zone_{i+1}" for i in range(n_zones)]
    storage_levels = [f"zone_{i+1}" for i in range(n_zones)]

    dvs = OrderedDict()

    # MRF baselines are fixed (as in the base formulation); Montague/Trenton
    # baseline targets and the NYC diversion cap are Decree-fixed (not DVs).

    # Zone breakpoint shifts (N curves): an additive vertical offset and a
    # temporal shift per breakpoint (_BREAKPOINT_COUNT per curve). The top
    # two curves (flood-zone boundaries, near full storage) get trimmed
    # vertical upper bounds, mirroring the base formulation's level1b/level1c
    # headroom.
    _shift_upper = {storage_levels[0]: 0.025}
    if len(storage_levels) > 1:
        _shift_upper[storage_levels[1]] = 0.05
    dvs.update(_zone_breakpoint_specs(storage_levels, _shift_upper))

    # NYC delivery factors: only for levels where baseline < unconstrained threshold
    for i, level in enumerate(drought_levels):
        if interp_nyc[i] < 100:
            dvs[f"nyc_drought_factor_{level}"] = {
                "baseline": float(np.clip(interp_nyc[i], 0.30, 1.0)),
                "bounds": [0.30, 1.0],
                "units": "fraction",
            }

    # NJ delivery factors: only for levels where baseline < 1.0. Floor
    # mirrors the base formulation (no NJ objective — bounds guard the
    # Decree-party interest).
    for i, level in enumerate(drought_levels):
        if interp_nj[i] < 1.0:
            dvs[f"nj_drought_factor_{level}"] = {
                "baseline": float(np.clip(interp_nj[i], 0.65, 1.0)),
                "bounds": [0.65, 1.0],
                "units": "fraction",
            }

    # Flood-zone spill-mitigation release scaling (same DV names across all
    # N-zone variants; mapped to the two flood levels — indices below
    # flood_conservation_boundary=2 — at apply time).
    dvs.update(FLOOD_RELEASE_SCALE_SPECS)

    # MRF seasonal profile scaling (FAW-envelope bounds, as in the base
    # formulation)
    for season in ["winter", "spring", "summer", "fall"]:
        dvs[f"mrf_profile_scale_{season}"] = {
            "baseline": 1.0,
            "bounds": [0.8, 2.6],
            "units": "multiplier",
        }

    # Downstream flow-target factor scaling: interpolate the default 7-level
    # monthly factor tables to N+1 levels (per month, matching pywrdrb's
    # from_n_zones interpolation) and expose scale DVs for the
    # drought-affected zones (any month's factor < 1.0).
    x_def = np.linspace(0, 1, 7)
    x_tgt = np.linspace(0, 1, n_zones + 1)
    for loc in ("montague", "trenton"):
        default_matrix = _DEFAULT_FLOW_TARGET_FACTORS[loc]  # (7, 12)
        interp_matrix = np.column_stack([
            np.interp(x_tgt, x_def, default_matrix[:, m]) for m in range(12)
        ])  # (n_zones + 1, 12)
        _add_flow_target_scale_dvs(dvs, loc, drought_levels, interp_matrix)

    # Merge salt-front DVs (no-op when SALT_FRONT_PARAM_MODE == "fixed").
    # Safe to call here because generate_ffmp_formulation runs after module
    # import has completed, so the config-import in _merge_salt_front_dvs
    # doesn't trigger a partial-import cycle.
    # Pass n_drought_levels so the activation-gate DV (when active) resolves
    # to the top 3 indices of THIS N-zone config rather than the default
    # 7-level [4,5,6].
    _merge_salt_front_dvs(dvs, n_drought_levels=n_zones + 1)

    return {
        "description": f"Parameterized FFMP with {n_zones}-zone storage curves",
        "decision_variables": dvs,
    }
