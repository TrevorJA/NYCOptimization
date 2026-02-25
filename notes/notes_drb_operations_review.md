# Literature Review: Delaware River Basin Regulations and NYC Reservoir Operations

Trevor Amestoy, Reed Research Group, Cornell University
February 2026

---

## 1. Overview of the DRB System

The Delaware River Basin (DRB) spans New York, Pennsylvania, New Jersey, and Delaware. NYC operates three major reservoirs in the upper basin as part of its water supply system, which together with the Rondout balancing reservoir comprise the Delaware system of the NYC water supply.

The NYC Delaware Basin reservoirs and their capacities are:

- **Pepacton** (East Branch Delaware River): 140,190 MG, the largest NYC reservoir
- **Cannonsville** (West Branch Delaware River): 95,706 MG
- **Neversink** (Neversink River): 34,941 MG
- **Rondout** (balancing reservoir): receives water from all three via the Delaware Aqueduct

Combined, the four reservoirs account for roughly 1,012 square miles of watershed and 320.4 billion gallons of capacity. The Delaware Aqueduct carries approximately half of NYC's water supply (~1.1-1.3 billion gallons per day total) 85 miles to the city. The Aqueduct has been leaking approximately 35 million gallons per day since the 1990s.


## 2. Legal Framework: The 1954 Supreme Court Decree

On June 7, 1954, the U.S. Supreme Court issued Decree 347 U.S. 995 (*New Jersey v. New York*) establishing an equitable allocation of Delaware River Basin waters. The decree defines five "Decree Parties":

1. New York State
2. New York City (treated separately due to its large diversions)
3. New Jersey
4. Pennsylvania
5. Delaware

**Key provisions:**

- NYC is authorized to divert up to 800 MGD from the Delaware Basin (when all three reservoirs are fully constructed)
- NYC must make compensating releases to maintain a minimum flow objective of 1,750 CFS at Montague, NJ
- New Jersey is authorized to divert 100 MGD without compensating releases, plus additional emergency diversions through the Delaware and Raritan Canal
- Changes to management of NYC's Delaware Basin Reservoirs require *unanimous consent* of all five decree parties

The decree created the position of Delaware River Master within the USGS. The River Master authorizes and supervises diversions, requires and directs compensating releases, and reports annually to the Supreme Court.


## 3. DRBC and the Evolution of Flow Management

The Delaware River Basin Commission (DRBC) was created in 1961, partly in response to the 1961-1967 "drought of record" which revealed a fundamental tension: NYC cannot simultaneously (1) divert its authorized 800 MGD, (2) maintain the 1,750 CFS compensating release at Montague, and (3) provide adequate ecological flows during drought.

Key milestones in flow management:

- **1983 Good Faith Agreement**: Revised yield estimates based on 1960s drought; introduced seasonal "Operational Curves"
- **DRBC Docket D-77-20** (1977-2006): Formalized ecological release management with 9 revisions
- **2007 FFMP**: First implementation of flexible, storage-based flow management
- **2011 FFMP**: Allowed NYC to deliver less than 800 MGD cap
- **2017 FFMP** (current): Most comprehensive flexible management framework
- **2023 Amendment**: Extends FFMP through May 31, 2028


## 4. The 2017 Flexible Flow Management Program (FFMP)

The 2017 FFMP is a two-part, ten-year agreement establishing the current operational framework. It is the operational regime implemented in Pywr-DRB.

### 4.1 Storage Zone-Based Drought Operations

The FFMP uses combined NYC reservoir storage levels to trigger drought management stages. Storage is evaluated daily against seasonally-varying threshold curves:

- **Level 1a (Flood)**: >95% capacity. Maximize releases for flood mitigation.
- **Level 1b-1c (Elevated)**: High storage with moderate releases.
- **Level 2 (Normal)**: ~75-95% capacity. Standard operations.
- **Level 3 (Drought Watch)**: ~60-75% capacity. Reduce NYC diversions to 85% of baseline.
- **Level 4 (Drought Warning)**: ~45-60% capacity. Reduce NYC diversions to 70% of baseline.
- **Level 5 (Drought Emergency)**: <45% capacity. Reduce NYC diversions to 65% of baseline.

The threshold curves vary seasonally (366 daily values), reflecting the need to maintain higher reserves in winter and allowing lower thresholds in summer.

### 4.2 NYC Diversion Constraints

- **Running-average constraint**: At no time shall the aggregate total quantity diverted, divided by the number of days elapsed since the preceding May 31, exceed the drought-adjusted maximum (800 MGD at L1-L2, reduced at L3+).
- **Actual recent usage**: NYC's average monthly diversion was approximately 538 MGD in 2020-2021, well below the 800 MGD cap.

### 4.3 NJ Diversion Constraints

- Daily maximum: 120 MGD
- Monthly average: 100 MGD
- L4 drought: Reduce to 90%
- L5 drought: Reduce to 80%

### 4.4 Minimum Release Requirements (MRFs)

Each reservoir has a baseline MRF that is scaled by drought level and daily/monthly profile factors:

**Baselines (MGD):**
- Cannonsville: 122.8
- Pepacton: 64.63
- Neversink: 48.47

The effective MRF at any time step is:
```
mrf_reservoir = mrf_baseline × combined_factor × daily_profile_factor
```
where the `combined_factor` blends aggregate and individual reservoir drought levels, and the `daily_profile_factor` varies seasonally (range ~0.5 to 7.5).

### 4.5 Downstream Flow Targets

- **Montague Target**: 1,131 MGD baseline (~1,750 CFS), adjusted by monthly drought factors
- **Trenton Target**: 1,939 MGD baseline (~3,000 CFS), active June 15 through March 15 during normal conditions

Meeting these targets accounts for time-of-travel lags:
- Cannonsville/Pepacton to Montague: 2-day lag
- Cannonsville/Pepacton to Trenton: 4-day lag
- Neversink to Montague: 1-day lag
- Neversink to Trenton: 3-day lag

### 4.6 Flood Control Operations

When aggregate storage exceeds the L1b/L1c threshold:
```
excess_volume = current_storage - level1c_boundary
flood_release_needed = excess_volume / 7 days
actual_flood_release = max(min(flood_release_needed - mrf_obligation, max_release), 0)
```

Maximum flood releases (CFS):
- Cannonsville: 4,200
- Pepacton: 2,400
- Neversink: 3,400

NYC endeavors to maintain 15% void space between November 1 and February 1 to help mitigate flooding.

### 4.7 FFMP Water Banks

The 2017 FFMP established four water banks:

1. **Thermal Mitigation Bank** (2,500 CFS-days annually): Releases for thermal stress mitigation during warm periods
2. **Rapid Flow Change Mitigation Bank**: Allows gradual reductions in directed releases to protect spawning habitat
3. **New Jersey Surviving Diversions Bank**: Adjusts drought operations calculations
4. **Diversion Offset Bank**: Accumulates water saved during release reductions to offset NJ diversions during drought

### 4.8 Interim Excess Release Quantity (IERQ)

Fixed at 15,468 CFS-days (non-leap) / 17,125 CFS-days (leap years). Computed based on the difference between NYC's historical peak consumption and estimated continuous safe yield. Upon request, NYC releases IERQ water to maintain 3,000 CFS at Trenton during normal conditions.


## 5. Competing Demands and Conflicts

### 5.1 The Central Tension

The fundamental conflict: during drought, NYC's authorized diversions, compensating releases at Montague, and ecological flow needs cannot all be simultaneously satisfied. The FFMP attempts to manage this through staged curtailments, but the underlying tradeoff remains.

### 5.2 Salt Front Management

The 3,000 CFS Trenton target exists to repel the salt front (7-day average location of 250 mg/L chloride) from advancing upstream toward Philadelphia and Camden water intakes. Challenges include:
- Sea level rise pushing more salt into the estuary
- Increasing land-based salinity (road salt)
- Modeling suggests current flow management may be insufficient under 0.5m+ sea level rise combined with very dry conditions

### 5.3 Ecological Concerns

- **Atlantic Sturgeon**: Critically endangered, with fewer than 250 spawning adults remaining. Designated critical habitat in 2017. Flow management through the FFMP supports minimum flows and reduces thermal/flow fluctuations.
- **Cold-Water Fisheries**: Reservoir operations strongly influence temperature regimes. The Thermal Mitigation Bank and Rapid Flow Change Mitigation Bank exist specifically to protect spawning habitat.
- **Freshwater Mussels**: Depend on migratory fish hosts and stable flows for reproduction.

### 5.4 Flood Risk

Upper Delaware communities face flood risk from reservoir spill and high-volume releases. The 2017 FFMP's 15% void target and flood monitoring nodes (Hale Eddy, Fishs Eddy, Bridgeville) represent attempts to mitigate this, but tension remains between maintaining storage for drought resilience and creating void for flood mitigation.


## 6. Implications for Multi-Objective Optimization

The FFMP framework reveals several key features relevant to optimization study design:

### 6.1 Natural Objective Set

The DRB system has clearly competing objectives that emerge from its governance structure:

- **NYC water supply reliability**: Meeting diversion demands under drought
- **Downstream flow compliance**: Montague and Trenton targets
- **Ecological flow quality**: Temperature, rapid flow change, spawning habitat
- **Flood risk mitigation**: Downstream stage at vulnerable communities
- **Storage equity/drought resilience**: Maintaining reserves for future drought
- **NJ diversion reliability**: Meeting NJ's allocation

### 6.2 Decision Variables

The NYCOperationsConfig in Pywr-DRB exposes ~31 constants plus daily/monthly profile tables, including:
- MRF baselines (3 reservoirs + 2 downstream targets)
- Drought level factors for diversions and releases
- Storage zone thresholds (seasonally varying)
- Flood release maximums
- Delivery constraints and running-average windows

### 6.3 Constraints and Institutional Realities

Any optimization must respect:
- The 1954 Supreme Court decree (hard constraints on maximum diversions)
- Unanimous consent requirement for operational changes
- Physical infrastructure limits (aqueduct capacity, release valve limits)
- Time-of-travel lags for downstream target compliance
- The annual May 31 running-average reset for NYC diversions

### 6.4 Deep Uncertainties

- Future streamflow regimes under climate change
- Demand growth or reduction (NYC has been below 800 MGD)
- Salt front dynamics under sea level rise
- Ecological flow requirements as understanding evolves
- Institutional willingness to adopt new operational rules


## 7. Key References

- Hamilton, A. L., Amestoy, T. J., & Reed, P. M. (2024). Pywr-DRB: An open-source Python model for water availability and drought risk assessment. *Environmental Modelling & Software*, 106185.
- Kolesar, P., & Serio, J. (2011). Breaking the deadlock: Improving water-release policies on the Delaware River. *Interfaces*, 41(1), 18-34.
- DRBC. (2017). Flexible Flow Management Program. https://webapps.usgs.gov/odrm/documents/ffmp/FFMP2017.pdf
- DRBC. Delaware River Basin Commission Flow Management. https://www.nj.gov/drbc/programs/flow/flow-mgmt.html
- NYSDEC. Delaware Basin Reservoir Releases. https://dec.ny.gov/nature/waterbodies/lakes-rivers/reservoir-releases/about-delaware-basin
- USGS. Delaware River Master, FFMP. https://webapps.usgs.gov/odrm/ffmp/flexible-flow-management-program
- DRBC. Salt Front Management. https://www.nj.gov/drbc/programs/flow/salt-front.html
- DRBC. Subcommittee on Ecological Flows. https://www.nj.gov/drbc/about/advisory/SEF_index.html
