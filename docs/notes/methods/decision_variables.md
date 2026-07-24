# Decision variables (FFMP formulation)

The decision vector parameterizes the 2017 FFMP rule structure with **69
variables** (`src/formulations/ffmp.py`; applied to the model in
`src/simulation.py`). Variable-resolution variants `ffmp_N` share the same
group structure with zone-indexed names.

| Group | DVs | Names | Baseline | Bounds | Units |
|---|---|---|---|---|---|
| NYC drought delivery factors (L3–L5) | 3 | `nyc_drought_factor_L{3,4,5}` | 0.85 / 0.70 / 0.65 | [0.6,1.0] / [0.4,0.95] / [0.3,0.9] | fraction |
| NJ drought delivery factors (L4–L5) | 2 | `nj_drought_factor_L{4,5}` | 0.90 / 0.80 | [0.8,1.0] / [0.65,1.0] | fraction |
| Per-breakpoint storage-zone vertical offsets | 24 | `zone_vshift_{level}_c{0..3}` | 0.0 | L1b [-0.10,0.025]; L1c [-0.10,0.05]; L2–L5 [-0.10,0.10] | fraction of capacity |
| Per-breakpoint storage-zone temporal shifts | 24 | `zone_tshift_{level}_c{0..3}` | 0.0 | [-30, 30] | days |
| Flood-zone spill-mitigation release scales | 6 | `flood_release_scale_{l1a,l1b}_{res}` | 1.0 | L1a: [0.5, 1.35/1.20/1.55] per reservoir; L1b: [0.5, 2.0] | multiplier |
| MRF seasonal profile scales (conservation zones) | 4 | `mrf_profile_scale_{season}` | 1.0 | [0.8, 2.6] | multiplier |
| Downstream flow-target factor scales | 6 | `mrf_target_scale_{montague,trenton}_{level}` | 1.0 | [0.5, 1.15] | multiplier |

The six storage-zone boundary curves (`level1b`…`level5`) are each
represented by their four major breakpoints (the largest-curvature corners
of the baseline curve); every breakpoint carries one vertical-offset DV and
one temporal-shift DV (6 curves × 4 breakpoints = 24 each).

Fixed (never decision variables): the reservoir MRF baselines (122.8 /
64.63 / 48.47 MGD — the FFMP Table 4a base rates; operational variation is
carried by the seasonal profile scales), the 1954 Decree quantities
(Montague and Trenton baseline targets, the 800 MGD NYC diversion cap, the
100 MGD NJ monthly-average cap), and the FFMP Table 5 maximum combined
discharge rates (4,200 / 2,400 / 3,400 cfs at Cannonsville / Pepacton /
Neversink), which are physical and regulatory limits on the release works
and spillway.

## Bounds rationale

- **MRF seasonal profile scales [0.8, 2.6]**: spans the FFMP's own
  Table 4a→4g FAW envelope (~1.0–2.6× base for the FAW-varying zones). The
  0.8 floor keeps releases near the negotiated Table 4a base rates — the
  only protection for the tailwater fishery (no habitat objective is
  active).
- **NJ factors**: no NJ delivery objective is active, so the lower bounds
  bracket the negotiated FFMP values (0.90/0.80); widen only if the NJ
  reliability objective is activated.
- **Flood L1a uppers (1.35 / 1.20 / 1.55)**: maximum controlled release
  observed 2000–2021 (2,062 / 842 / 303 cfs — demonstrated release-works
  capacity) divided by the L1a schedule rates (1,500 / 700 / 190 cfs).
  2.0 × L1b stays within that demonstrated range for all three reservoirs.
  All uppers sit below the Table 5 combined caps.
- **Zone vertical-offset uppers (L1b 0.025, L1c 0.05)**: the L1b curve sits
  at 0.975–1.0 and L1c at 0.85–1.0 of capacity; larger upward offsets clip
  to 1.0 (dead range). The per-curve upper cap applies to all four
  breakpoints; the lower bound is -0.10 throughout.
- **Flood scales season-invariant (6)**: the FFMP holds the L1a/L1b rows
  season-constant; seasonal flood freedom is reallocated to the zone-boundary
  breakpoint shifts, where the FFMP's own seasonality lives.
- **Flow-target scale upper 1.15**: the 1.0 cap on the effective factor
  binds at scale ≈ 1.06–1.13 per row; larger uppers are entirely flat.

## Feasibility clamps and Borg constraints

Three feasibility conditions are enforced twice, deliberately:

**Apply-time clamps** (`src/simulation.py`) guarantee every simulated
policy is operationally valid:

- Storage-zone curves: monotonic ordering enforced after shifts.
- Flood zones: effective L1b ≤ L1a; effective rates capped at Table 5.
- Delivery factors: `np.minimum.accumulate` over drought stages — a deeper
  stage never allows more diversion than a milder one (NYC and NJ).

**Formal Borg constraints** (`compute_constraint_violations` in
`src/simulation.py`; exposed via `src.formulations.make_constraint_function`
/ `get_n_constrs`) pose the first two conditions as constraint functions
computed from pure DV arithmetic on the cached default schedules — no
Pywr-DRB simulation. Each value is a violation magnitude (0 = feasible,
positive scales linearly with the degree of violation; magnitudes at or
below 1e-9 floor to exact 0 so float noise cannot flag infeasibility):

1. **Delivery monotonicity** — sum of positive adjacent increments in the
   NYC and NJ delivery-factor arrays (NYC L3 ≥ L4 ≥ L5, NJ L4 ≥ L5 under
   the audited bounds). Units: fraction.
2. **Flood-zone ordering** — per reservoir, the worst-day exceedance of the
   effective L1b schedule over L1a (default factor rows × scale DVs, Table 5
   cap applied), normalized by the reservoir MRF baseline and summed over
   reservoirs. Dimensionless; in the binding Apr 16 – Jun 15 equal-rate
   window it reduces to the schedule factor × (L1b scale − L1a scale)⁺.

Zone-curve crossings are deliberately **clamp-only**, not a constraint:
the per-breakpoint shift bounds make crossings ubiquitous under random
sampling, and the monotonicity clamp resolves them cleanly at apply time —
the clamped geometry is the intended policy, not a defect to search away from.

Borg applies constraint-dominance ahead of Pareto/epsilon dominance: any
feasible solution dominates every infeasible one, and infeasible solutions
rank by total |violation|. The MM Borg driver therefore skips simulation
for infeasible vectors entirely — it returns penalty objectives (1e10) plus
the violation vector, saving the ~3 min evaluation while giving the search
a direct gradient toward feasibility. The clamps stay in place, so any
vector that reaches simulation (feasible by construction) and any policy
evaluated outside the search remains operationally valid.

Accounting note: infeasible evaluations consume NFE (`maxEvaluations` is
per island) but essentially zero compute and zero simulated scenario-years;
the budget→NFE derivation must account for the feasible fraction of
evaluations.

The MOEAFramework problem JARs (workflow step 00) deliberately declare
**zero** constraints: every file they parse (solveMPI runtime snapshots,
`.set` files) contains only feasible solutions in variables + objectives
format — the archive writer strips constraint violators and never emits
constraint columns.

## Flood-zone (L1a/L1b) spill-mitigation release scaling

The FFMP's flood-operations lever is the zone-conditional release schedule of
Tables 4a–4g: when combined storage is in Zone L1, each reservoir releases at
its L1-a / L1-b / L1-c row rate. The L1-a row (1,500 / 700 / 190 cfs) and
L1-b row (600 / 300 / 150–110 cfs) are invariant across the seven FAW tables;
in Pywr-DRB they are encoded as daily factor profiles multiplied by the
reservoir MRF baseline.

The `flood_release_scale_*` DVs multiply the **default** L1a/L1b daily
schedule per (zone × reservoir), **season-invariant** — matching the FFMP,
which holds these rows constant across its seven tables and (except
Neversink's L1b step) across seasons. The profile-multiplier form preserves
the within-year shape (the L1a-absent window Apr 16–Jun 15 and Neversink's
seasonal L1b step). Seasonal flood policy is carried by the per-breakpoint
zone-boundary shifts (below): the FFMP's own seasonal flood instrument is
the CSSO / zone-boundary geometry (the ~15% Nov 1 – Feb 1 void), not the
release rates.

## Per-breakpoint storage-zone boundary shifts

Each storage-zone threshold curve is represented by its four major
breakpoints — the `_ZONE_CORNER_COUNT` = 4 largest-curvature corners of the
baseline curve (min 25 days apart, detected deterministically per curve by
`_zone_curve_corners` in `src/simulation.py`). Each breakpoint carries two
DVs: an additive vertical offset (`zone_vshift_{level}_c{k}`, fraction of
capacity) and a temporal shift (`zone_tshift_{level}_c{k}`, days). At apply
time (`_apply_zone_shifts`) each breakpoint is moved to
`(corner_day + temporal_shift) mod year` and offset to
`(baseline_value + vertical_offset)`, and the daily curve is rebuilt as a
circular piecewise-linear curve through the four moved, offset breakpoints
(`_reconstruct_breakpoint_curve`). Temporal shifts are rounded to whole days.
Properties:

- Adjusted boundaries are **piecewise-linear** through the four breakpoints —
  no new kink dates; operator-readable rule curves.
- All-zero DVs reproduce the default curves: the stored FFMP curves are
  themselves piecewise-linear through their four corners, so the
  reconstruction is exact at baseline.
- Seasonal depth control is retained — e.g., lowering the autumn/winter
  breakpoints deepens the flood void without lowering the June 1 refill
  target (the CSSO lever).

Applied before the [0, 1] clip and the cross-curve monotonicity clamp.

**Season windows (seasonal DV groups).** The `mrf_profile_scale_*` group
uses the FFMP's own Tables 4a–4g headers, not meteorological seasons: winter
Dec 1 – Mar 31, spring Apr 1 – May 31, summer Jun 1 – Aug 31, fall Sep 1 –
Nov 30 (`_SEASON_DOY_RANGES` in `src/simulation.py`, calendar-date indexed on
the 366-column profiles). The FFMP's finer date bins nest inside these
seasons, so every seasonally scaled schedule steps only on the FFMP's own bin
edges (`test_scaled_profiles_step_only_on_ffmp_bin_edges`). The flow-target
factor scales are non-seasonal (one multiplier per gauge per drought level).

Semantics enforced at apply time (`_apply_flood_release_scaling`):

- **Absolute-schedule renormalization.** The model computes the zone release
  as factor × `mrf_baseline_{res}`, and the baseline is itself a DV. Written
  factors are renormalized by (default baseline / DV baseline), so a scale of
  1.0 always yields the default cfs schedule regardless of the `mrf_{res}`
  value — each DV group has a single, interpretable effect.
- **Exclusion from the seasonal profile scales.** `mrf_profile_scale_*`
  applies only to the conservation-zone rows (L1c and below); the flood-zone
  rows are governed exclusively by `flood_release_scale_*`.
- **Table 5 cap.** Effective scheduled rates are clipped at the reservoir's
  fixed maximum combined discharge rate. Within the [0.5, 2.0] bounds the cap
  never binds (2 × 1,500 = 3,000 < 4,200 cfs, etc.); it is a guardrail.
- **Zone monotonicity.** Effective L1b release is clamped elementwise to not
  exceed effective L1a, mirroring the storage-zone shift clamp.

The separate `NYCFloodRelease` drawdown (the 7-day excess release toward the
CSSO when storage is in L1a/L1b) is unchanged and is capped at the fixed
Table 5 constants.

For `ffmp_N`, the flood zones are the drought levels with index below
`flood_conservation_boundary` (= 2 for all N): `zone_0` and `zone_1`. The DV
names are identical across all N.
