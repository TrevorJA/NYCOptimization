"""
ffmp.py - Parameterized FFMP (2017 Flexible Flow Management Program) formulation.

Defines decision variables, bounds, and baselines for the FFMP formulation
that re-optimizes existing FFMP parameters within plausible ranges, and
supports N-zone variable-resolution variants.

Salt-front-dependent flow-target adjustment DVs (FFMP-family only) are
merged in conditionally based on `config.SALT_FRONT_PARAM_MODE`. See
`salt_front_dvs.py`.
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
#: The 0.65 floor keeps exploration conservative: it caps the reduction of
#: the FFMP's own negotiated drought-stage flow-target factors at ~35%, so
#: no searched policy proposes halving a Decree-adjacent downstream flow
#: obligation (a 0.5 floor did, which downstream states / the river master
#: would scrutinize on optics).
FLOW_TARGET_SCALE_BOUNDS = [0.65, 1.15]


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
# Standard FFMP formulation (36 DVs base, optionally extended via salt_front)
###############################################################################

# --- Flood-zone (L1a/L1b) spill-mitigation release scaling ---
# Dimensionless multipliers on the DEFAULT FFMP Tables 4a-4g flood-zone
# release schedule (e.g., L1a Cannonsville = mult x 1500 cfs), applied by
# simulation._apply_flood_release_scaling. Season-invariant — matching the
# FFMP, which holds these rows constant across its tables and seasons;
# seasonal flood policy (void scheduling, CSSO shape) is carried by the
# zone-boundary shift DVs (zone_vshift_*) instead. The
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

# --- Storage-zone boundary shifts (two vertical + one temporal per curve) ---
# Each storage-zone threshold curve is a trapezoid over the year: a low plateau
# (fall/winter void), a rising ramp, a high plateau (spring/summer refill
# target), and a falling ramp. Each curve gets three DVs: an additive shift of
# the LOW plateau (zone_vshift_{level}_lower, fraction of capacity), an additive
# shift of the HIGH plateau (zone_vshift_{level}_upper), and a temporal shift
# (zone_tshift_{level}, days) that slides the whole curve along the day-of-year
# axis. At apply time (simulation._apply_zone_shifts) the two plateau levels are
# moved independently and the curve values are affinely remapped between them
# (the two ramps re-interpolate to connect), then rolled, clipped to [0, 1], and
# cross-curve monotonicity-clamped. All-zero DVs reproduce the default curves
# exactly. Splitting the vertical shift by plateau decouples void DEPTH from the
# refill target — the FFMP's own CSSO seasonal-void lever — while preserving the
# trapezoidal shape (no new kinks); each knob maps to a visible flat segment, so
# the change stays stakeholder-legible.
#
# A curve whose baseline HIGH (refill) plateau sits at full capacity gets NO
# HIGH-plateau shift DV at all: it could only move down, and lowering the
# refill target below capacity is a permanent effective-capacity forfeit, not
# an FFMP-scale operating-rule perturbation. The FFMP treats refill-to-full by
# ~June 1 as an essential requirement (Appendix A §6: the CSSO "must be limited
# and ramped" so the reservoirs are "filled on or around June 1st every year");
# the negotiated flood lever is the seasonal VOID depth (10% in FFMP2014, 15%
# in FFMP2017), which is exactly the LOW-plateau shift. So level1b/1c/2 keep
# their refill plateaus fixed at 1.0 and are searched through void depth and
# timing only. For the remaining curves the up-cap follows baseline geometry:
# a plateau cannot be raised above capacity, so up-cap =
# min(_ZONE_VSHIFT_BOUND, 1.0 - plateau); level1b's LOW plateau sits at 0.975,
# so its zone_vshift_*_lower up-cap is 0.025. The lower bound on every
# vertical shift is -_ZONE_VSHIFT_BOUND.
_ZONE_VSHIFT_BOUND = 0.10
_ZONE_TSHIFT_BOUND = 30.0
#: Curves whose baseline refill plateau is at full capacity: HIGH plateau is
#: fixed at baseline (no zone_vshift_*_upper DV emitted).
_ZONE_UPPER_FIXED = {"level1b", "level1c", "level2"}
#: Per-curve up-cap on the LOW-plateau shift (zone_vshift_*_lower); trimmed only
#: for level1b, whose low plateau (0.975) has 0.025 of headroom to capacity.
_ZONE_VSHIFT_LOWER_CAP = {"level1b": 0.025}


def _zone_shift_specs(levels, lower_cap_by_level, upper_fixed_levels):
    """Build the zone-shift DV specs (vertical + temporal per curve).

    For each curve, adds an additive LOW-plateau shift DV
    (``zone_vshift_{level}_lower``, fraction of capacity), an additive
    HIGH-plateau shift DV (``zone_vshift_{level}_upper``) unless the curve's
    refill plateau is fixed, and a temporal-shift DV (``zone_tshift_{level}``,
    days). Baselines are 0.0 so the curve is unperturbed at the baseline
    vector.

    Args:
        levels: Storage-zone curve names.
        lower_cap_by_level: Per-curve upper-bound override for the LOW-plateau
            shift (defaults to ``_ZONE_VSHIFT_BOUND``).
        upper_fixed_levels: Curves whose refill plateau is fixed at baseline —
            no HIGH-plateau shift DV is emitted for these.

    Returns:
        OrderedDict of DV specs ([lower, upper,] temporal per curve).
    """
    specs = OrderedDict()
    for level in levels:
        specs[f"zone_vshift_{level}_lower"] = {
            "baseline": 0.0,
            "bounds": [-_ZONE_VSHIFT_BOUND,
                       lower_cap_by_level.get(level, _ZONE_VSHIFT_BOUND)],
            "units": "fraction",
        }
        if level not in upper_fixed_levels:
            specs[f"zone_vshift_{level}_upper"] = {
                "baseline": 0.0,
                "bounds": [-_ZONE_VSHIFT_BOUND, _ZONE_VSHIFT_BOUND],
                "units": "fraction",
            }
        specs[f"zone_tshift_{level}"] = {
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

        # --- NYC + NJ drought delivery factors ---
        # One rule for both Decree parties: bounds = negotiated FFMP factor
        # ± 0.15, clipped at 1.0 (no factor exceeds full delivery). Each
        # party's interest is guarded by its own reliability objective; the
        # symmetric envelope keeps every searched policy a renegotiation-scale
        # perturbation of the FFMP. L1a-L2 factors are effectively
        # unconstrained (set to large values).
        "nyc_drought_factor_L3": {
            "baseline": 0.85,
            "bounds": [0.70, 1.0],
            "units": "fraction",
        },
        "nyc_drought_factor_L4": {
            "baseline": 0.70,
            "bounds": [0.55, 0.85],
            "units": "fraction",
        },
        "nyc_drought_factor_L5": {
            "baseline": 0.65,
            "bounds": [0.50, 0.80],
            "units": "fraction",
        },
        "nj_drought_factor_L4": {
            "baseline": 0.90,
            "bounds": [0.75, 1.0],
            "units": "fraction",
        },
        "nj_drought_factor_L5": {
            "baseline": 0.80,
            "bounds": [0.65, 0.95],
            "units": "fraction",
        },

        # --- Storage-zone boundary shifts ---
        # (specs built by _zone_shift_specs above: a LOW-plateau shift
        # (zone_vshift_*_lower), a HIGH-plateau shift (zone_vshift_*_upper),
        # both fraction of capacity, and one temporal shift (zone_tshift_*,
        # days) per curve; the two plateaus move independently and the curve is
        # affinely remapped between them at apply time)
        **_zone_shift_specs(
            ["level1b", "level1c", "level2", "level3", "level4", "level5"],
            _ZONE_VSHIFT_LOWER_CAP,
            _ZONE_UPPER_FIXED,
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

    With n_zones=None (default), returns the standard 36-DV formulation
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

    # Zone shifts (N curves): a LOW-plateau shift, a HIGH-plateau shift where
    # the refill plateau is below capacity, and one temporal shift per curve.
    # The top three storage curves (flood-zone boundaries) refill to ~1.0, so
    # their refill plateaus are FIXED at baseline (no HIGH-plateau DV — same
    # rule as the base formulation's level1b/level1c/level2) and the topmost
    # curve's LOW-plateau up-cap is trimmed to 0.025, mirroring level1b's
    # headroom to capacity.
    _lower_cap = {storage_levels[0]: 0.025}
    _upper_fixed = set(storage_levels[:3])
    dvs.update(_zone_shift_specs(storage_levels, _lower_cap, _upper_fixed))

    # NYC / NJ delivery factors: the base formulation's symmetric rule —
    # bounds = interpolated FFMP factor ± 0.15, clipped at 1.0 — applied to
    # each variant's own interpolated baselines. NYC DVs exist only for
    # levels below the unconstrained threshold; NJ DVs only where the
    # interpolated baseline < 1.0.
    def _factor_spec(baseline: float) -> dict:
        b = float(np.clip(baseline, 0.0, 1.0))
        return {
            "baseline": b,
            "bounds": [round(max(b - 0.15, 0.0), 6),
                       round(min(b + 0.15, 1.0), 6)],
            "units": "fraction",
        }

    for i, level in enumerate(drought_levels):
        if interp_nyc[i] < 100:
            dvs[f"nyc_drought_factor_{level}"] = _factor_spec(interp_nyc[i])
    for i, level in enumerate(drought_levels):
        if interp_nj[i] < 1.0:
            dvs[f"nj_drought_factor_{level}"] = _factor_spec(interp_nj[i])

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
