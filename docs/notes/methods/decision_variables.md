# Decision variables (FFMP formulation)

The decision vector parameterizes the 2017 FFMP rule structure with **36
variables** (`src/formulations/ffmp.py`; applied to the model in
`src/simulation.py`). Variable-resolution variants `ffmp_N` share the same
group structure with zone-indexed names. The FFMP source for every table
and section cited below is `docs/Appendix_A_FFMP-20180716-Final.pdf`.

| Group | DVs | Names | Baseline | Bounds | Units |
|---|---|---|---|---|---|
| NYC allocation reductions (L3–L5) | 3 | `nyc_allocation_reduction_L{3,4,5}` | 0.15 / 0.15 / 0.05 | [0, 0.20] / [0, 0.20] / [0, 0.10] | fraction (stage-wise reduction) |
| NJ allocation reductions (L4–L5) | 2 | `nj_allocation_reduction_L{4,5}` | 0.10 / 0.10 | [0, 0.20] / [0, 0.15] | fraction (stage-wise reduction) |
| Storage-zone low-plateau (void) shifts | 6 | `zone_vshift_{level}_lower` | 0.0 | L1b [-0.10,0.025]; L1c–L5 [-0.10,0.10] | fraction of capacity |
| Storage-zone high-plateau (refill) shifts (L3–L5 only) | 3 | `zone_vshift_{level}_upper` | 0.0 | [-0.10, 0.10] | fraction of capacity |
| Storage-zone temporal shifts (one per curve) | 6 | `zone_tshift_{level}` | 0.0 | [-30, 30] | days |
| Flood-zone spill-mitigation release scales | 6 | `flood_release_scale_{l1a,l1b}_{res}` | 1.0 | L1a: [0.5, 1.35/1.20/1.55] per reservoir; L1b: [0.5, 2.0] | multiplier |
| MRF seasonal profile scales (conservation zones) | 4 | `mrf_profile_scale_{season}` | 1.0 | [0.8, 2.6] | multiplier |
| Downstream flow-target factor scales | 6 | `mrf_target_scale_{montague,trenton}_{level}` | 1.0 | [0.65, 1.15] | multiplier |

The six storage-zone boundary curves (`level1b`…`level5`) are trapezoids: a low
plateau (fall/winter void), a rising ramp, a high plateau (spring/summer refill
target), and a falling ramp. Each curve gets a low-plateau shift and one
temporal shift; the drought curves (L3–L5), whose refill plateaus sit below
capacity, additionally get an independent high-plateau shift (6 × 2 + 3 = 15).
The flood-zone curves (L1b/L1c/L2) refill to full capacity and their refill
plateaus are **fixed at baseline** — searchable geometry there is void depth
and timing. Splitting the vertical shift by plateau decouples void depth from
the refill target while preserving the trapezoidal shape; each knob maps to a
visible flat segment, so the change stays stakeholder-legible.

The allocation-reduction DVs are **stage-wise increments**, not absolute
factors: each is the ADDITIONAL fractional reduction of the party's Decree
allocation applied on entry to that drought stage, and the effective
delivery factor at a stage is 1 minus the running sum of reductions
(`_apply_ffmp_params` in `src/simulation.py`). Baseline increments decode
to the negotiated FFMP factors exactly (NYC 0.85 / 0.70 / 0.65 at
L3/L4/L5; NJ 0.90 / 0.80 at L4/L5). Because the increments are
non-negative, a deeper stage can never allow more diversion than a milder
one — stage monotonicity is structural, with no clamp and no Borg
constraint. NYC L1a–L2 and NJ L1a–L3 remain effectively unconstrained
(model defaults).

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
- **NYC + NJ allocation reductions (one depth-preserving rule)**: lower
  bound 0 on every stage increment; upper bound = negotiated FFMP increment
  + headroom, with each party's total headroom allocated in clean 0.05
  steps so the summed uppers equal the audited worst-case cumulative
  curtailment (`NYC_MAX_TOTAL_REDUCTION` = 0.50, `NJ_MAX_TOTAL_REDUCTION`
  = 0.35 in `src/formulations/ffmp.py` — i.e. delivery factors bottom out
  at 0.50 for NYC and 0.65 for NJ). Both Decree parties may be curtailed to
  renegotiation-scale depth, each guarded by its own reliability
  objective, and no searched policy approaches near-total curtailment of
  either party. Zero reductions (full delivery at every stage) are
  reachable by design: whether curtailment earns its keep is measured by
  the reliability-vs-storage trade-off in the objectives, not imposed by
  the bounds. Every baseline increment is strictly interior to its box.
  `ffmp_N` applies the same rule to its interpolated baselines, with the
  residual headroom split uniformly across its stages.
- **Flood L1a uppers (1.35 / 1.20 / 1.55)**: maximum controlled release
  observed 2000–2021 (2,062 / 842 / 303 cfs — demonstrated release-works
  capacity) divided by the L1a schedule rates (1,500 / 700 / 190 cfs).
  2.0 × L1b stays within that demonstrated range for all three reservoirs.
  All uppers sit below the Table 5 combined caps. Note the *effective* L1b
  range is narrower than [0.5, 2.0] in the Apr 16 – Jun 15 window, where the
  L1a schedule drops to the L1b rate (the L1a-absent window): the
  flood-ordering clamp (L1b ≤ L1a) holds L1b at ≤ 1.0× there, so scales
  above 1.0 are realizable only outside that ~2-month window.
- **Fixed refill plateaus (L1b/L1c/L2)**: a curve whose baseline refill
  plateau sits at full capacity gets no high-plateau DV. Such a shift could
  only move down, and lowering the refill target below capacity is a
  permanent effective-capacity forfeit — outside the renegotiation-scale
  envelope. The FFMP treats refill-to-full by ~June 1 as an essential
  requirement (Appendix A §6: the CSSO "must be limited and ramped" so the
  reservoirs are "filled on or around June 1st every year"); its negotiated
  flood lever is the seasonal void depth (10% in FFMP2014, 15% in FFMP2017),
  carried here by the low-plateau shift. It also removes the aggressive
  drawdown artifact: `NYCFloodRelease` targets the `level1c` curve (the
  CSSO), so a lowered refill plateau would have the model evacuate all
  storage above it — through the refill window — at up to Table 5 rates.
- **Zone plateau-shift up-caps**: derived from baseline geometry — a plateau
  cannot be raised above capacity, so its up-cap = min(0.10, 1.0 − plateau
  level). L1b's low plateau sits at 0.975, so its `zone_vshift_*_lower`
  up-cap is 0.025. All other plateau up-caps are 0.10, and the lower bound
  is -0.10 throughout.
- **Flood scales season-invariant (6)**: the FFMP holds the L1a/L1b rows
  season-constant; seasonal flood freedom is reallocated to the zone-boundary
  geometry, where the FFMP's own seasonality lives — specifically the
  low-plateau (void) shift, which sets the autumn/winter void depth
  independently of the spring refill target.
- **Flow-target scales [0.65, 1.15]**: the 1.0 cap on the effective factor
  binds at scale ≈ 1.06–1.13 per row, so the 1.15 upper leaves every row just
  enough headroom to reach the cap. The 0.65 floor caps the reduction of the
  FFMP's own negotiated drought-stage flow-target factors at ~35%, keeping
  exploration conservative — no searched policy halves a Decree-adjacent
  downstream flow obligation.

## Feasibility clamps and Borg constraints

Feasibility is handled three ways, matched to where each condition lives:

- **Structural (by construction)** — delivery-stage monotonicity: the
  allocation-reduction DVs are non-negative stage increments, so the
  decoded delivery-factor arrays are non-increasing for every in-bounds
  vector. No clamp, no constraint, no feasibility signal needed.
- **Apply-time clamps** (`src/simulation.py`) guarantee every simulated
  policy is operationally valid:
  - Storage-zone curves: monotonic ordering enforced after shifts.
  - Flood zones: effective L1b ≤ L1a; effective rates capped at Table 5.
- **Formal Borg constraints** — the flood-zone condition is additionally
  posed to the optimizer, and a post-simulation reliability floor is
  appended after evaluation (below).

**Formal Borg constraint — DV-space** (`compute_constraint_violations` in
`src/simulation.py`; exposed via `src.formulations.make_constraint_function`
/ `get_n_constrs`) poses the flood-zone condition as a constraint function
computed from pure DV arithmetic on the cached default schedules — no
Pywr-DRB simulation. The value is a violation magnitude (0 = feasible,
positive scales linearly with the degree of violation; magnitudes at or
below 1e-9 floor to exact 0 so float noise cannot flag infeasibility):

1. **Flood-zone ordering** — per reservoir, the worst-day exceedance of the
   effective L1b schedule over L1a (default factor rows × scale DVs, Table 5
   cap applied), normalized by the reservoir MRF baseline and summed over
   reservoirs. Dimensionless; in the binding Apr 16 – Jun 15 equal-rate
   window it reduces to the schedule factor × (L1b scale − L1a scale)⁺.

Zone-curve crossings are deliberately **clamp-only**, not a constraint:
the zone-shift bounds make crossings common under random sampling, and
the monotonicity clamp resolves them cleanly at apply time —
the clamped geometry is the intended policy, not a defect to search away from.

**Formal Borg constraint — post-simulation**
(`src.formulations.make_post_sim_constraint_function`): a second constraint,
`nyc_reliability_floor`, enforces the stakeholder floor on NYC weekly
delivery reliability directly in the search. Unlike the DV-space
constraint it reads the COMPUTED objective vector — the MM Borg objective
wrapper (`src/mmborg.py::make_borg_objective`) appends its violation after
simulation, un-negating the Borg-minimized reliability objective (resolved
by name, `nyc_delivery_reliability_weekly`) back to the natural 0–1 scale:

2. **NYC reliability floor** — `max(0, floor − reliability)`; floor from
   `config.NYC_RELIABILITY_FLOOR` (env `NYCOPT_NYC_RELIABILITY_FLOOR`,
   default 0.5). A policy delivering below-floor weekly reliability is
   unacceptable to stakeholders regardless of the rest of the trade-off.
   Constraint-dominance excludes such policies from search and archive.

Borg applies constraint-dominance ahead of Pareto/epsilon dominance: any
feasible solution dominates every infeasible one, and infeasible solutions
rank by total |violation|. The MM Borg driver therefore skips simulation
for DV-infeasible vectors entirely — it returns penalty objectives (1e10)
plus the violation vector (post-sim slot 0.0, nothing measured), saving the
~3 min evaluation while giving the search a direct gradient toward
feasibility. Failed simulations likewise return penalty objectives with
zero violations: feasible-but-maximally-unattractive, so a failure can
never enter the archive ahead of a real solution while the constraint
channel stays truthful in magnitude. The clamps stay in place, so any
vector that reaches simulation (DV-feasible by construction) and any policy
evaluated outside the search remains operationally valid.

Accounting note: infeasible evaluations consume NFE (`maxEvaluations` is
per island) but essentially zero compute and zero simulated scenario-years;
the budget→NFE derivation must account for the feasible fraction of
evaluations. With delivery monotonicity structural, only flood-zone
ordering can reject a vector pre-simulation, so the DV-infeasible fraction
under random sampling is small.

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
seasonal L1b step). Seasonal flood policy is carried by the zone-boundary
shifts (below): the FFMP's own seasonal flood instrument is the CSSO /
zone-boundary geometry (the ~15% Nov 1 – Feb 1 void), not the release rates.

## Storage-zone boundary shifts

Each storage-zone threshold curve is a trapezoid over the year: a **low
plateau** (its baseline minimum — the fall/winter void), a rising ramp, a
**high plateau** (its baseline maximum — the spring/summer refill target), and
a falling ramp. Each curve carries an additive shift of the low plateau
(`zone_vshift_{level}_lower`, fraction of capacity) and a temporal shift
(`zone_tshift_{level}`, days); curves whose refill plateau sits below capacity
(L3–L5) also carry an additive shift of the high plateau
(`zone_vshift_{level}_upper`) — for L1b/L1c/L2 the high plateau is fixed at
baseline (full capacity). At apply time (`_apply_zone_shifts`) the two
plateau levels are moved independently to `lo_new = lo + shift_lower` and
`hi_new = hi + shift_upper`, the daily values are affinely remapped between
them — `value → lo_new + (value − lo)/(hi − lo) · (hi_new − lo_new)` — so the
two ramps re-interpolate to connect, then the curve is circularly rolled by
the temporal shift (rounded to whole days). Properties:

- **Void depth and refill target are decoupled** — e.g., a negative
  `zone_vshift_{level}_lower` deepens the autumn/winter void without lowering
  the spring refill target (the FFMP's own CSSO seasonal-void lever), which a
  single whole-curve shift could not represent.
- The **trapezoidal shape is preserved** — only the two plateau levels move and
  the ramps re-interpolate; no new kink dates. Each knob maps to a visible flat
  segment, keeping the change stakeholder-legible.
- **Within-curve clamp**: the low plateau may not exceed the high plateau
  (`lo_new ≤ hi_new`); a shift pair that would invert them flattens the curve
  to the high-plateau level (void = refill).
- All-zero DVs reproduce the default curves exactly.

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
