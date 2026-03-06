# Decision Variables — Formulation Descriptions

*Living document. Last updated: 2026-03-06.*

This document describes the decision variables for each MOEA problem formulation, including their pywrdrb implementation mapping and regulatory (FFMP/Decree) context.

---

## Formulation A: Parameterized FFMP (`ffmp`)

**Description:** Re-optimize the existing 2017 FFMP operational parameters within plausible ranges. Preserves the institutional rule structure — results are directly interpretable and implementable within the current regulatory framework.

**Total DVs:** 25

### DV Group 1: Minimum Required Flow (MRF) Baselines

These set the baseline minimum release or flow targets that each reservoir or gauge must maintain. The actual daily requirement is the baseline multiplied by drought-level-dependent daily profile factors.

| # | DV Name | Baseline | Bounds | Units | pywrdrb Method |
|---|---------|----------|--------|-------|----------------|
| 1 | `mrf_cannonsville` | 122.8 | [60, 250] | MGD | `config.update_mrf_baselines(cannonsville=...)` |
| 2 | `mrf_pepacton` | 64.63 | [30, 130] | MGD | `config.update_mrf_baselines(pepacton=...)` |
| 3 | `mrf_neversink` | 48.47 | [20, 100] | MGD | `config.update_mrf_baselines(neversink=...)` |
| 4 | `mrf_montague` | 1131.05 | [800, 1500] | MGD | `config.update_mrf_baselines(montague=...)` |
| 5 | `mrf_trenton` | 1938.95 | [1400, 2500] | MGD | `config.update_mrf_baselines(trenton=...)` |

**Regulatory context:**
- The 1954 Supreme Court Decree mandates minimum flows at Montague (1750 CFS / 1131 MGD) and minimum releases from each NYC reservoir.
- The 2017 FFMP refines these baselines and adds the Trenton target (3000 CFS / 1939 MGD, active Jun 15 - Mar 15).
- Individual reservoir MRFs (DVs 1-3) are "conservation releases" that maintain downstream habitat and baseflow.
- Montague and Trenton targets (DVs 4-5) are met through the combined effect of reservoir releases plus natural lateral inflows.

**Bound rationale:**
- Lower bounds ensure some minimum ecological baseflow.
- Upper bounds prevent excessive releases that would drain storage without commensurate downstream benefit.
- Montague and Trenton bounds bracket the range from drought-reduced targets to enhanced normal-condition targets.

---

### DV Group 2: NYC Delivery Constraints

| # | DV Name | Baseline | Bounds | Units | pywrdrb Method |
|---|---------|----------|--------|-------|----------------|
| 6 | `max_nyc_delivery` | 800 | [500, 900] | MGD | `config.update_delivery_constraints(max_nyc_delivery=...)` |

**Regulatory context:**
- The 1954 Decree allocates NYC a diversion of up to 800 MGD from the Delaware Basin via the Delaware Aqueduct.
- NYC's actual diversions are subject to a running average constraint (annual accounting period: June 1 - May 31).
- The FFMP modulates the effective cap via drought-level factors (see DV Group 3).
- **Note:** The running average constraint is implemented in pywrdrb but is NOT currently exposed as a Borg constraint. This is a potential gap.

---

### DV Group 3: NYC Drought Delivery Factors

These multiply the `max_nyc_delivery` baseline at each drought level. Only the more severe levels (L3-L5) are optimized; L1a-L2 are unconstrained (factor = 1,000,000 in pywrdrb, effectively no reduction).

| # | DV Name | Baseline | Bounds | Units | pywrdrb Mapping |
|---|---------|----------|--------|-------|-----------------|
| 7 | `nyc_drought_factor_L3` | 0.85 | [0.60, 1.0] | fraction | `drought_factors_nyc[4]` |
| 8 | `nyc_drought_factor_L4` | 0.70 | [0.40, 0.95] | fraction | `drought_factors_nyc[5]` |
| 9 | `nyc_drought_factor_L5` | 0.65 | [0.30, 0.90] | fraction | `drought_factors_nyc[6]` |

**Regulatory context:**
- Under the FFMP, NYC diversions are reduced during drought conditions:
  - **Level 1a** (normal high storage): Unconstrained
  - **Level 1b/1c** (mild concern): Unconstrained
  - **Level 2** (watch): Unconstrained
  - **Level 3** (warning): Reduced to 85% of baseline
  - **Level 4** (drought): Reduced to 70% of baseline
  - **Level 5** (emergency): Reduced to 65% of baseline
- The drought level is determined by aggregate NYC reservoir storage relative to seasonal zone thresholds (see DV Group 5).

**Implementation detail:**
The 7-element `drought_factors_nyc` array is constructed by combining non-optimized defaults for L1a-L2 with optimized values for L3-L5:
```python
nyc_factors = [1e6, 1e6, 1e6, 1e6,  # L1a, L1b, L1c, L2 (from config.constants)
               params["nyc_drought_factor_L3"],
               params["nyc_drought_factor_L4"],
               params["nyc_drought_factor_L5"]]
```

---

### DV Group 4: NJ Drought Delivery Factors

| # | DV Name | Baseline | Bounds | Units | pywrdrb Mapping |
|---|---------|----------|--------|-------|-----------------|
| 10 | `nj_drought_factor_L4` | 0.90 | [0.60, 1.0] | fraction | `drought_factors_nj[5]` |
| 11 | `nj_drought_factor_L5` | 0.80 | [0.50, 1.0] | fraction | `drought_factors_nj[6]` |

**Regulatory context:**
- NJ diversions (via the Delaware and Raritan Canal) are set at 100 MGD monthly average / 120 MGD daily max under normal conditions.
- Under the FFMP, NJ diversions are only reduced at severe drought levels:
  - **Level 1a-3**: No reduction (factor = 1.0)
  - **Level 4**: Reduced to 90% of baseline
  - **Level 5**: Reduced to 80% of baseline
- NJ's allocation is governed by a separate agreement and the 1954 Decree.

**Implementation detail:**
```python
nj_factors = [1.0, 1.0, 1.0, 1.0, 1.0,  # L1a-L3 (from config.constants)
              params["nj_drought_factor_L4"],
              params["nj_drought_factor_L5"]]
```

---

### DV Group 5: Storage Zone Vertical Shifts

These shift the seasonal storage zone threshold curves up or down. The curves define the boundaries between drought levels based on aggregate NYC storage fraction.

| # | DV Name | Baseline | Bounds | Units | pywrdrb Target |
|---|---------|----------|--------|-------|----------------|
| 12 | `zone_shift_level1b` | 0.0 | [-0.10, 0.10] | fraction | `storage_zones_df.loc["level1b"]` |
| 13 | `zone_shift_level1c` | 0.0 | [-0.10, 0.10] | fraction | `storage_zones_df.loc["level1c"]` |
| 14 | `zone_shift_level2` | 0.0 | [-0.10, 0.10] | fraction | `storage_zones_df.loc["level2"]` |
| 15 | `zone_shift_level3` | 0.0 | [-0.10, 0.10] | fraction | `storage_zones_df.loc["level3"]` |
| 16 | `zone_shift_level4` | 0.0 | [-0.10, 0.10] | fraction | `storage_zones_df.loc["level4"]` |
| 17 | `zone_shift_level5` | 0.0 | [-0.10, 0.10] | fraction | `storage_zones_df.loc["level5"]` |

**Regulatory context:**
- The FFMP defines 6 storage zones (Level 1a through Level 5) using seasonal curves that vary by day of year (366-day profiles).
- The zone boundaries determine which drought level is active, which in turn controls MRF factors, delivery reductions, and other operational modes.
- Higher zone boundaries = drought declarations triggered at higher storage = more conservative operations.
- Lower zone boundaries = drought declarations delayed = more aggressive use of storage.

**Implementation detail:**
- Each shift is additive: `new_threshold[doy] = old_threshold[doy] + shift`
- Clipped to [0, 1] after shifting.
- Monotonicity enforced: more severe levels (Level 5) must have thresholds <= less severe levels (Level 1b). After applying individual shifts, the code enforces `level_i+1 <= level_i` by taking elementwise minimum.
- A +0.10 shift raises all 366 daily values by 10% of capacity.

**Example:**
If Level 3 baseline at DOY 1 is 0.46 and `zone_shift_level3 = +0.05`:
- New Level 3 threshold at DOY 1 = 0.51
- This means the system enters Level 3 drought at a higher storage level = more conservative.

---

### DV Group 6: Flood Release Maximums

| # | DV Name | Baseline | Bounds | Units | pywrdrb Method |
|---|---------|----------|--------|-------|----------------|
| 18 | `flood_max_cannonsville` | 4200 | [2000, 8000] | CFS | `config.update_flood_limits(max_release_cannonsville=...)` |
| 19 | `flood_max_pepacton` | 2400 | [1200, 5000] | CFS | `config.update_flood_limits(max_release_pepacton=...)` |
| 20 | `flood_max_neversink` | 3400 | [1500, 7000] | CFS | `config.update_flood_limits(max_release_neversink=...)` |

**Regulatory context:**
- The FFMP Void Management / Flood Rule specifies maximum controlled release rates from each NYC reservoir.
- These limits are designed to maintain void (empty space) in reservoirs before forecasted storm events while limiting downstream flood impacts.
- **Cannonsville** (West Branch Delaware): 4200 CFS baseline. Downstream at Hale Eddy and Fishs Eddy.
- **Pepacton** (East Branch Delaware): 2400 CFS baseline. Downstream at Downsville and Fishs Eddy.
- **Neversink** (Neversink River): 3400 CFS baseline. Downstream at Bridgeville.
- Higher limits allow faster drawdown before storms but increase peak downstream flows.
- Lower limits reduce flood peaks but may leave reservoirs too full during storm events, leading to uncontrolled spill.

**Tradeoff:**
This DV group is in direct tension with Objectives 5 (flood risk) and 6 (storage resilience). Higher max releases reduce storage more aggressively (improving flood capacity but reducing drought resilience).

---

### DV Group 7: MRF Seasonal Profile Scaling

| # | DV Name | Baseline | Bounds | Units | pywrdrb Target |
|---|---------|----------|--------|-------|----------------|
| 21 | `mrf_profile_scale_winter` | 1.0 | [0.5, 2.0] | multiplier | `mrf_factors_daily_df` DOY 335-366, 1-59 |
| 22 | `mrf_profile_scale_spring` | 1.0 | [0.5, 2.0] | multiplier | `mrf_factors_daily_df` DOY 60-151 |
| 23 | `mrf_profile_scale_summer` | 1.0 | [0.5, 2.0] | multiplier | `mrf_factors_daily_df` DOY 152-243 |
| 24 | `mrf_profile_scale_fall` | 1.0 | [0.5, 2.0] | multiplier | `mrf_factors_daily_df` DOY 244-334 |

**Regulatory context:**
- The FFMP specifies daily MRF release factor profiles that vary by day of year. These factors modify the baseline MRF values seasonally (e.g., higher releases during spring/summer for ecological flows, lower during winter).
- These DVs apply a uniform multiplicative scaling to all MRF daily factors within each season:
  - **Winter:** Dec 1 - Feb 28 (DOY 335-366, 1-59)
  - **Spring:** Mar 1 - May 31 (DOY 60-151)
  - **Summer:** Jun 1 - Aug 31 (DOY 152-243)
  - **Fall:** Sep 1 - Nov 30 (DOY 244-334)

**Implementation detail:**
- Applied multiplicatively: `new_factor[doy] = old_factor[doy] * scale`
- A scale of 2.0 doubles all MRF factors in that season, roughly doubling required releases.
- A scale of 0.5 halves all MRF factors, reducing required releases.
- Modifies `config.mrf_factors_daily_df` directly.

**Known issue:** The `mrf_factors_daily_df` is initialized as a copy of `storage_zones_df` in the pywrdrb `from_defaults()` method. Both are loaded from `ffmp_reservoir_operation_daily_profiles.csv`. The CSV likely contains both storage zone and MRF factor profiles indexed by the `profile` column, but this needs runtime confirmation. If the MRF factors are not properly separated from storage zones, this scaling could produce unexpected results.

---

## Formulation B: FFMP + Enhanced Flexibility (Planned)

**Status:** Not yet implemented. Design phase.

**Estimated DVs:** 40-60

Extends Formulation A with additional degrees of freedom:

| DV Group | Description | Estimated Count |
|----------|-------------|-----------------|
| All Formulation A DVs | As above | 25 |
| Seasonal drought factors NYC | Per-season L3-L5 factors (4 seasons x 3 levels) | 12 |
| Seasonal drought factors NJ | Per-season L4-L5 factors (4 seasons x 2 levels) | 8 |
| Asymmetric zone curves | Separate fill-season and draw-season zone slopes | 6-12 |
| Reservoir-specific delivery allocation | Fractional draw from each reservoir | 2-3 |

**Regulatory interpretation:** These represent extensions to the FFMP that are plausible within the current regulatory framework. For example, seasonal drought factors could be implemented as a revision to Appendix C of the FFMP without changing the overall structure.

**pywrdrb requirements:**
- `update_delivery_constraints` already accepts per-level factors; seasonal variation requires extending the interface to accept per-level-per-season arrays.
- Zone curve asymmetry requires modifying `update_storage_zones` to accept separate ascending/descending profiles.
- Delivery allocation requires a new method or extending `update_delivery_constraints`.

---

## Formulation C: State-Aware Direct Policy Search (Planned)

**Status:** Not yet implemented. Design phase.

**Estimated DVs:** 100-300 (depends on architecture)

Replaces rule-based operations with nonlinear policy functions.

### State Inputs (candidate set)

| # | State Variable | Source in pywrdrb | Dimension |
|---|---------------|-------------------|-----------|
| 1-3 | Individual reservoir storage (fraction) | `res_storage[r] / capacity[r]` | 3 |
| 4 | Aggregate NYC storage fraction | `sum(res_storage[NYC]) / C_NYC` | 1 |
| 5-7 | Current-day inflows | `inflow_[r]` parameters | 3 |
| 8-10 | 7-day rolling avg inflows | Derived from daily inflows | 3 |
| 11 | Day of year (sin transform) | `sin(2*pi*DOY/366)` | 1 |
| 12 | Day of year (cos transform) | `cos(2*pi*DOY/366)` | 1 |
| 13 | Recent Montague flow (7-day avg) | `major_flow["delMontague"]` | 1 |
| 14 | Recent Trenton flow (7-day avg) | `major_flow["delTrenton"]` | 1 |

### Policy Actions

| # | Action | Physical Meaning | Range |
|---|--------|-----------------|-------|
| 1-3 | Individual reservoir releases | Daily release from each NYC reservoir | [MRF_min, flood_max] |
| 4 | NYC diversion fraction | Fraction of available supply diverted to NYC | [0, 1] |
| 5 | NJ diversion fraction | Fraction of available supply diverted to NJ | [0, 1] |

### RBF Architecture

For $N$ inputs, $K$ radial basis functions, $M$ outputs:

$$
a_j = w_{0,j} + \sum_{k=1}^{K} w_{k,j} \cdot \phi_k(\mathbf{x})
$$

where $\phi_k(\mathbf{x}) = \exp\left(-\sum_{i=1}^{N} \frac{(x_i - c_{k,i})^2}{r_k^2}\right)$

**Parameters per RBF:** $N$ center coordinates + 1 radius = $N + 1$
**Parameters per output:** $K + 1$ weights (including bias)
**Total:** $K(N+1) + M(K+1)$

With $N=14$, $K=6$, $M=5$: $6 \times 15 + 5 \times 7 = 90 + 35 = 125$ DVs.

### MLP Architecture

A small feedforward network: $N$ inputs -> $H$ hidden (ReLU) -> $M$ outputs (sigmoid).

**Total:** $(N+1) \times H + (H+1) \times M$

With $N=14$, $H=20$, $M=5$: $15 \times 20 + 21 \times 5 = 300 + 105 = 405$ DVs.

**pywrdrb requirements:**
- Formulation C requires intercepting pywrdrb's daily timestep to inject policy-computed releases. This likely requires a custom Pywr `Parameter` subclass that evaluates the policy function at each timestep.
- The `NYCOperationsConfig` interface is insufficient for this formulation; direct policy search bypasses the rule structure entirely.
- Current pywrdrb architecture may need extension to support custom release policies.

---

## DV Indexing Reference

Flat vector indexing for Formulation A (as used by Borg):

| Index | DV Name |
|-------|---------|
| 0 | `mrf_cannonsville` |
| 1 | `mrf_pepacton` |
| 2 | `mrf_neversink` |
| 3 | `mrf_montague` |
| 4 | `mrf_trenton` |
| 5 | `max_nyc_delivery` |
| 6 | `nyc_drought_factor_L3` |
| 7 | `nyc_drought_factor_L4` |
| 8 | `nyc_drought_factor_L5` |
| 9 | `nj_drought_factor_L4` |
| 10 | `nj_drought_factor_L5` |
| 11 | `zone_shift_level1b` |
| 12 | `zone_shift_level1c` |
| 13 | `zone_shift_level2` |
| 14 | `zone_shift_level3` |
| 15 | `zone_shift_level4` |
| 16 | `zone_shift_level5` |
| 17 | `flood_max_cannonsville` |
| 18 | `flood_max_pepacton` |
| 19 | `flood_max_neversink` |
| 20 | `mrf_profile_scale_winter` |
| 21 | `mrf_profile_scale_spring` |
| 22 | `mrf_profile_scale_summer` |
| 23 | `mrf_profile_scale_fall` |

This ordering matches `config.get_var_names("ffmp")` and the Borg DV vector.
