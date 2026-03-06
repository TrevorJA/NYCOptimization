# Objective Functions — Mathematical Descriptions

*Living document. Last updated: 2026-03-06.*

This document provides detailed mathematical definitions of all objective functions available in the optimization framework. Objectives are implemented in `src/objectives.py` as `Objective` class instances and grouped into `ObjectiveSet` collections.

---

## Notation

| Symbol | Definition |
|--------|-----------|
| $T$ | Total simulation timesteps (daily) |
| $T_w$ | Warmup period (365 days, excluded from all metrics) |
| $t$ | Daily timestep index, $t \in \{T_w+1, \ldots, T\}$ |
| $D_t^{\text{NYC}}$ | NYC water demand on day $t$ (MGD) |
| $S_t^{\text{NYC}}$ | NYC water delivery (supply) on day $t$ (MGD) |
| $Q_t^{\text{Mon}}$ | Streamflow at Delaware at Montague on day $t$ (MGD) |
| $Q_t^{\text{Tre}}$ | Streamflow at Delaware at Trenton on day $t$ (MGD) |
| $V_t^r$ | Storage volume in reservoir $r$ on day $t$ (MG) |
| $C^r$ | Total capacity of reservoir $r$ (MG) |
| $C^{\text{NYC}}$ | Total NYC system capacity = $\sum_{r \in \text{NYC}} C^r$ = 270,837 MG |
| $\text{DOY}(t)$ | Day-of-year for timestep $t$ |

NYC reservoirs: $r \in \{\text{Cannonsville}, \text{Pepacton}, \text{Neversink}\}$

---

## Active Objective Sets

### Default Set (6 objectives)

Used for primary Formulation A optimization.

| # | Name | Direction | Code ref |
|---|------|-----------|----------|
| 1 | `nyc_supply_reliability_daily` | maximize | `obj_nyc_supply_reliability_daily` |
| 2 | `nyc_drought_max_monthly_deficit_pct` | minimize | `obj_nyc_drought_max_monthly_deficit_pct` |
| 3 | `montague_flow_reliability_daily` | maximize | `obj_montague_flow_reliability_daily` |
| 4 | `trenton_flow_reliability_seasonal` | maximize | `obj_trenton_flow_reliability_seasonal` |
| 5 | `flood_risk_storage_spill_days` | minimize | `obj_flood_risk_storage_spill_days` |
| 6 | `storage_min_combined_pct` | maximize | `obj_storage_min_combined_pct` |

### Alternative Sets

| Set Name | Objectives | Variants |
|----------|-----------|----------|
| `drought_duration` | 6 | Uses `nyc_drought_max_consecutive_shortfall_days` instead of monthly deficit |
| `downstream_flood` | 6 | Uses `flood_risk_downstream_flow_days` instead of storage proxy |
| `comprehensive` | 8 | Both drought metrics + both flood metrics (all variants included) |
| `compact` | 4 | Supply reliability, Montague compliance, flood risk, min storage |

---

## Objective 1: NYC Supply Reliability

**Name:** `nyc_supply_reliability_daily`
**Direction:** Maximize
**Epsilon:** 0.005
**Range:** [0, 1]

**Definition:**
$$
f_1 = \frac{1}{T - T_w} \sum_{t=T_w+1}^{T} \mathbb{1}\left[ S_t^{\text{NYC}} \geq 0.99 \cdot D_t^{\text{NYC}} \right]
$$

The fraction of post-warmup days where NYC delivery meets at least 99% of demand. The 0.99 tolerance accounts for numerical noise in the Pywr solver (floating-point flow allocation rounding).

**Stakeholder relevance:** NYC Department of Environmental Protection. Measures basic service reliability.

**Borg form:** $-f_1$ (negated for minimization)

---

## Objective 2: NYC Maximum Monthly Deficit

**Name:** `nyc_drought_max_monthly_deficit_pct`
**Direction:** Minimize
**Epsilon:** 1.0
**Range:** [0, 100]

**Definition:**

Let months $m \in \{1, \ldots, M\}$ partition the post-warmup period. Define the daily shortage:

$$
\delta_t = \max(0, \; D_t^{\text{NYC}} - S_t^{\text{NYC}})
$$

Monthly aggregates:

$$
\Delta_m = \sum_{t \in m} \delta_t, \qquad \bar{D}_m = \sum_{t \in m} D_t^{\text{NYC}}
$$

Then:

$$
f_2 = \max_{m : \bar{D}_m > 0} \left( 100 \cdot \frac{\Delta_m}{\bar{D}_m} \right)
$$

The worst-case (maximum) monthly shortage expressed as a percentage of that month's total demand. Captures drought severity better than daily reliability, because a month with 50% delivery is more consequential than scattered individual shortfall days.

**Stakeholder relevance:** NYC DEP, drought planning. Measures worst-case supply disruption intensity.

**Borg form:** $f_2$ (already minimized)

---

## Objective 2-alt: NYC Maximum Consecutive Shortfall Days

**Name:** `nyc_drought_max_consecutive_shortfall_days`
**Direction:** Minimize
**Epsilon:** 5.0
**Range:** [0, T - T_w]

**Definition:**

Define the shortfall indicator:

$$
s_t = \mathbb{1}\left[ S_t^{\text{NYC}} < 0.99 \cdot D_t^{\text{NYC}} \right]
$$

Let $\{[a_k, b_k]\}$ be the maximal consecutive runs of $s_t = 1$. Then:

$$
f_{2'} = \max_k (b_k - a_k + 1)
$$

The longest uninterrupted streak of days where NYC delivery falls below 99% of demand. This measures drought duration rather than intensity.

**Stakeholder relevance:** NYC DEP, emergency management. Extended shortfalls trigger escalating drought responses and public health concerns.

**Borg form:** $f_{2'}$ (already minimized)

---

## Objective 3: Montague Flow Compliance

**Name:** `montague_flow_reliability_daily`
**Direction:** Maximize
**Epsilon:** 0.005
**Range:** [0, 1]

**Definition:**

$$
f_3 = \frac{1}{T - T_w} \sum_{t=T_w+1}^{T} \mathbb{1}\left[ Q_t^{\text{Mon}} \geq Q^{\text{Mon}}_{\text{target}} \right]
$$

where $Q^{\text{Mon}}_{\text{target}} = 1131.05$ MGD ($\approx 1750$ CFS), the 1954 Supreme Court Decree mandated minimum flow at Montague.

**Note:** The actual FFMP Montague target varies by drought level and season (normal = 1750 CFS, drought conditions reduce to lower targets). This metric uses the Decree baseline as a fixed threshold, which means it may overcount "violations" during declared drought periods when reduced targets are legally in effect.

**Stakeholder relevance:** All five decree parties (NYC, NYS, NJ, PA, DE), Delaware River Master. This is the most legally significant flow target.

**Borg form:** $-f_3$ (negated for minimization)

---

## Objective 4: Trenton Flow Compliance

**Name:** `trenton_flow_reliability_seasonal`
**Direction:** Maximize
**Epsilon:** 0.005
**Range:** [0, 1]

**Definition:**

Let $\mathcal{T}_{\text{active}} = \{t : \text{DOY}(t) \geq 166 \text{ or } \text{DOY}(t) \leq 74\}$ be the active period (June 15 through March 15).

$$
f_4 = \frac{1}{|\mathcal{T}_{\text{active}}|} \sum_{t \in \mathcal{T}_{\text{active}}} \mathbb{1}\left[ Q_t^{\text{Tre}} \geq Q^{\text{Tre}}_{\text{target}} \right]
$$

where $Q^{\text{Tre}}_{\text{target}} = 1938.95$ MGD ($\approx 3000$ CFS).

The Trenton target is only binding during June 15 - March 15. During March 16 - June 14, there is no minimum flow requirement at Trenton under normal FFMP operations.

**Stakeholder relevance:** NJ (drinking water intake), PA, DE. Critical for salt front management in the Delaware Estuary. When Trenton flow drops below target, salt intrusion threatens Philadelphia and Camden water intakes.

**Borg form:** $-f_4$ (negated for minimization)

---

## Objective 5: Flood Risk (Storage Proxy)

**Name:** `flood_risk_storage_spill_days`
**Direction:** Minimize
**Epsilon:** 5.0
**Range:** [0, T - T_w]

**Definition:**

$$
f_5 = \sum_{t=T_w+1}^{T} \mathbb{1}\left[ \sum_{r \in \text{NYC}} V_t^r \geq 0.95 \cdot C^{\text{NYC}} \right]
$$

Number of days where aggregate NYC storage exceeds 95% of total capacity. This is a proxy for spill risk: when reservoirs are nearly full, inflows must be released, potentially causing downstream flooding on the upper Delaware River.

**Known limitation:** This metric does not directly measure downstream stage or flood damage. It counts storage-based spill risk days but does not distinguish between controlled flood releases and actual downstream flood exceedances. The relationship between storage level and downstream flooding depends on inflow timing, release rates, and channel capacity. See `flood_risk_downstream_flow_days` for an alternative.

**Stakeholder relevance:** Upper Delaware communities (Hancock, Callicoon, Port Jervis). Flood releases from void management operations are a major source of community concern.

**Borg form:** $f_5$ (already minimized)

---

## Objective 5-alt: Flood Risk (Downstream Flow)

**Name:** `flood_risk_downstream_flow_days`
**Direction:** Minimize
**Epsilon:** 5.0
**Range:** [0, T - T_w]

**Definition:**

$$
f_{5'} = \sum_{t=T_w+1}^{T} \mathbb{1}\left[ Q_t^{\text{Mon}} \geq Q^{\text{flood}} \right]
$$

where $Q^{\text{flood}} = 16{,}148$ MGD ($\approx 25{,}000$ CFS), an approximate action stage threshold at the Montague gauge.

This directly measures downstream flood risk using simulated flow at Montague rather than storage levels. It better captures the actual flood hazard experienced by downstream communities.

**Note:** The 25,000 CFS threshold is approximate. The actual NWS action stage at Montague corresponds to a stage height, not a fixed discharge. Stage-discharge relationships should be refined with local rating curves.

**Borg form:** $f_{5'}$ (already minimized)

---

## Objective 6: Storage Resilience

**Name:** `storage_min_combined_pct`
**Direction:** Maximize
**Epsilon:** 0.5
**Range:** [0, 100]

**Definition:**

$$
f_6 = 100 \cdot \frac{\min_{t \in \{T_w+1, \ldots, T\}} \sum_{r \in \text{NYC}} V_t^r}{C^{\text{NYC}}}
$$

The minimum combined NYC storage fraction observed across the entire post-warmup period, expressed as a percentage. This captures worst-case drought vulnerability — the lowest point the system reaches.

**Stakeholder relevance:** All parties. Low storage triggers drought emergency declarations and increasingly severe operational restrictions. A system that maintains higher minimum storage is more resilient to extended dry periods.

**Borg form:** $-f_6$ (negated for minimization)

---

## Additional Available Metrics

These are implemented as `Objective` instances but not included in any default set. They can be added to custom `ObjectiveSet` configurations.

### Average Storage Percentage

**Name:** `storage_avg_combined_pct`
**Direction:** Maximize
**Epsilon:** 1.0

$$
f = 100 \cdot \frac{1}{T - T_w} \sum_{t=T_w+1}^{T} \frac{\sum_{r \in \text{NYC}} V_t^r}{C^{\text{NYC}}}
$$

Average combined NYC storage as % of capacity. Measures overall system utilization.

### Montague Deficit Severity

**Name:** `montague_flow_avg_deficit_mgd`
**Direction:** Minimize
**Epsilon:** 5.0

$$
f = \frac{\sum_{t} \max(0, Q^{\text{Mon}}_{\text{target}} - Q_t^{\text{Mon}})}{\sum_{t} \mathbb{1}[Q_t^{\text{Mon}} < Q^{\text{Mon}}_{\text{target}}]}
$$

Average daily Montague flow shortfall on days when flow is below target (MGD). Measures violation severity rather than frequency.

---

## Borg Sign Convention

Borg MOEA minimizes all objectives. For objectives with `direction = "maximize"`, the `ObjectiveSet.compute_for_borg()` method negates the raw value:

$$
f_i^{\text{Borg}} = \begin{cases}
-f_i & \text{if direction} = \text{maximize} \\
f_i & \text{if direction} = \text{minimize}
\end{cases}
$$

When reading Borg output (`.set` files, runtime files), maximize objectives must be negated back to recover the natural-direction value.

---

## Epsilon Values

Epsilon values control the resolution of the Borg epsilon-dominance archive. Solutions that differ by less than epsilon in all objectives are considered equivalent.

| Objective | Epsilon | Rationale |
|-----------|---------|-----------|
| `nyc_supply_reliability_daily` | 0.005 | 0.5% reliability resolution |
| `nyc_drought_max_monthly_deficit_pct` | 1.0 | 1 percentage point deficit resolution |
| `nyc_drought_max_consecutive_shortfall_days` | 5.0 | 5-day streak resolution |
| `montague_flow_reliability_daily` | 0.005 | 0.5% compliance resolution |
| `trenton_flow_reliability_seasonal` | 0.005 | 0.5% compliance resolution |
| `flood_risk_storage_spill_days` | 5.0 | 5-day flood risk resolution |
| `flood_risk_downstream_flow_days` | 5.0 | 5-day flood risk resolution |
| `storage_min_combined_pct` | 0.5 | 0.5% storage resolution |

---

## Open Questions

1. **Fixed vs. dynamic flow targets**: Montague and Trenton targets vary by drought level in the FFMP. Using fixed baselines may overcount violations during declared drought. Should we use drought-level-adjusted targets? This requires extracting the active drought level from simulation output.

2. **Flood metric refinement**: Neither `flood_days` nor `flood_days_downstream` captures flood damage magnitude. A stage-damage function or flow-duration-based metric might be more appropriate.

3. **Ecological flow objective**: Dropped from the current formulation. The STUDY_PLAN envisioned a composite ecological metric (thermal regime + flow variability). Using flow compliance as a proxy loses information about flow variability, timing, and magnitude that matter for ecological health.

4. **NJ diversion reliability**: Currently not tracked as an objective. NJ diversions are constrained by the model but not optimized for. Should this be an objective or a constraint?

5. **Running-average constraint**: The NYC delivery running average constraint (annual cap) is implemented in pywrdrb but not exposed as a Borg constraint or objective. Violations could occur silently.
