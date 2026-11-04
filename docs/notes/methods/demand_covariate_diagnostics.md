# NYC Demand-Covariate Diagnostics (SI)

*Measured impact of the NYC demand regressions' flow covariate on the historic
record. Evidence: one local pass of `scripts/supplemental/demand_covariate/run.py`
(2026-11-04): the two demand series over the HIST window 1945-12-01 to 2023-11-30,
two full-model single-trace simulations of the current FFMP policy in the step-05
baseline configuration (one cached model dict, deep-copied and patched twice with
only the `demand_nyc` source swapped), the eight active annual-unit objectives, and
the `kn_50yr_n5` fixture at k = 1 before and after. Settings: `supplemental_config.py`
`DCV_*`. Tables under `outputs/supplemental/demand_covariate/tables/`, figures
(`figures.py`) under `.../figures/`. Terminology per `docs/notes/terminology.md`.*

---

## 1. The covariate

The NYC diversion regressions (quarterly OLS of monthly diversion on log monthly
inflow, River Master record Dec 1999 to Nov 2021; Pywr-DRB
`pre/extrapolate_nyc_nj_diversions.py`) see one flow quantity everywhere: the sum of
the inflows at the three NYC reservoir nodes (`nyc_inflow_covariate`). Training reads
it from the model's inflow reconstruction `pub_nhmv10_BC_withObsScaled`
(`catchment_inflow_mgd.csv`, 1945-01-01 to 2023-12-31; mean 1,317 MGD over the
record, 1,461 MGD over the training window); the historic series
(`data/diversions/diversion_nyc_extrapolated_mgd.csv`, 1945 to 2023, draw seed 1) is
extrapolated on that same record; each synthetic realization supplies its own
three-node sum from `catchment_inflow_mgd.hdf5`. Catchment inflow equals gage flow at
the three headwater reservoirs (maximum absolute difference 0 over 1945-2023), so
every path sums the identical quantity (Pywr-DRB `tests/test_diversion_covariate.py`).
NJ demand is untouched (constant 100 MGD in this study).

The superseded variant ("old", Pywr-DRB `38edd57`) trained the regressions and
extrapolated the historic series on the sum of seven USGS gauges (mean 822 MGD over
1945-2023) while the ensemble path summed reservoir inflows, so synthetic demand sat
56 to 105 MGD per month below the historic level. The gauge sum is also incomplete
before 1996 (`old_covariate_gauge_availability.csv`, `old_covariate_gauges_by_year.csv`):
no Cannonsville inflow gauge (01423000, 398 MGD mean) before Oct 1950, the Pepacton
gauge 01414000 absent for 12,485 days (most of 1950-1995), 0142400103 absent for
13,246 days, the Neversink gauge absent in 1949-1950. Summed with missing values
skipped, the old covariate's annual mean was 326 to 509 MGD in 1945-1950 against
984 MGD over the training window, which the negative slopes turned into inflated
pre-1950 demand.

## 2. Refit (`regression_fit.csv`)

| Quarter | Slope (MGD per unit ln inflow) old → new | Intercept old → new | R² old → new | Residual SD old → new | Mean ln covariate shift |
|---|---|---|---|---|---|
| DJF | −203.9 → −208.9 | 1835.6 → 1955.2 | 0.330 → 0.333 | 150.1 → 149.8 | +0.413 |
| MAM | −123.0 → −125.6 | 1346.0 → 1417.0 | 0.267 → 0.265 | 101.0 → 101.1 | +0.423 |
| JJA | −121.6 → −127.2 | 1312.9 → 1399.7 | 0.519 → 0.519 | 87.6 → 87.6 | +0.421 |
| SON | −109.4 → −112.9 | 1198.4 → 1267.1 | 0.357 → 0.357 | 137.4 → 137.3 | +0.415 |

66 months per quarter. Fit quality is unchanged: the 0.41 to 0.42 shift in mean ln
inflow is absorbed by the intercepts (+69 to +120 MGD) and slopes 2 to 5 % steeper.

## 3. Demand series on the historic record

(`demand_summary.csv`, `demand_summary_by_era.csv`, `demand_annual_mean.csv`,
`demand_monthly_means.csv`, `demand_monthly_climatology.csv`; figures
`demand_difference`, `demand_monthly_climatology`)

| 1945-12-01 to 2023-11-30 | Old (USGS gauge sum) | New (reservoir inflow) |
|---|---|---|
| Mean (MGD) | 557.5 | 549.0 |
| Utilization, complete June-May FFMP years | 0.697 | 0.687 |
| Months with mean capped at 800 (of 936) | 64 | 53 |
| Share of days above the L3 / L4 / L5 caps (680 / 560 / 520 MGD) | 0.357 / 0.566 / 0.609 | 0.339 / 0.549 / 0.590 |
| Monthly-mean p05 / p50 / p95 (MGD) | 293 / 566 / 800 | 293 / 550 / 800 |
| Zero-demand days | 1,655 | 1,818 |

By era (mean MGD old → new): before Oct 1950 630 → 543 (−87; the missing Cannonsville
gauge), Oct 1950 to 1995 572 → 564 (−8), 1996-1999 539 → 545 (+6), the observed
River Master window Dec 1999 to Nov 2021 identical at 522 (0.652 of the allocation),
Dec 2021 to Nov 2023 483 → 535 (+52). The monthly climatology is lower by 14 to
18 MGD in December to March and August and within ±8 MGD otherwise. Zero-demand days
are inherited from the nearest-neighbour daily patterns (the observed record carries
665 zero-diversion days).

## 4. Fixture at k = 1 (`fixture_utilization.csv`)

`kn_50yr_n5`, five 50-year stationary realizations, unscaled: realized utilization
0.597 to 0.619 (mean 0.610) before and 0.668 to 0.689 (mean 0.679) after; mean
demand 488 → 544 MGD. The historic-minus-fixture utilization gap falls from 0.087 to
0.008, so HIST and the ensemble designs now search at the same demand level. On the
E_test fixture `etest_kn_10yr_n12` (six SOWs, `demand_utilization_realized_by_sow.csv`)
realized minus nominal utilization moved from −0.13 to −0.07 per SOW to −0.065 to
+0.014, the two dry SOWs (e^m 0.86, 0.89) above nominal and the wet ones below
(Spearman with e^m −0.94).

## 5. NYC system outcomes, current FFMP policy

(`outcomes_summary.csv`, `drought_zone_days.csv`, `storage_annual_minimum.csv`;
metric window 1946-06-01 to 2023-11-30, after the six-month spin-up)

| | Old | New |
|---|---|---|
| Combined storage mean / min / p01 / p05 (% of capacity) | 82.2 / 3.8 / 18.0 / 41.6 | 82.4 / 4.7 / 19.5 / 41.4 |
| Days in drought watch or worse (L3+) / warning or worse (L4+) / emergency (L5) | 1,863 / 1,358 / 917 | 1,853 / 1,397 / 902 |
| Distinct L3 / L4 / L5 events | 31 / 22 / 7 | 26 / 20 / 7 |
| NYC demand / delivery mean (MGD) | 557.3 / 549.7 | 549.4 / 542.1 |
| NYC shortfall: total (MG) / % of demand / days | 213,783 / 1.36 / 1,189 | 206,469 / 1.33 / 1,072 |
| Montague days below target | 1 | 0 |
| Trenton days below target / deficit (MG) | 313 / 39,513 | 313 / 39,750 |

## 6. Storage differences (`storage_difference_windows.csv`; figures `storage_difference_windows`, `storage_1960s_drought`)

The screen reports every window where |new − old| combined storage exceeds 5 points
of capacity (the search ε of `nyc_storage_min_p01_pct`), merging exceedance runs
separated by at most 180 days. Eighteen windows qualify, 5 to 245 exceedance days
each, thirteen with the new storage higher and five lower (peak difference in points
of capacity): 1947-03 (+7.3), 1948-12 (+6.7), 1949-12 to 1950-04 (+13.5; the old
series reaches L4, the new L3), 1951-08 (+5.0), 1954-09 to 1955-01 (+6.9), 1958-10
(−5.7), 1961-09 to 1962-03 (−8.6), 1962-11 to 1963-09 (−7.0, both variants in L5),
1967-10 (+5.1), 1968-11 (+6.4), 1970-11 to 1971-03 (+7.9), 1971-10 (+5.9), 1977-09
(−5.4), 1984-12 to 1985-11 (+8.1, 245 days), 1989-10 (+5.3), 1990-10 (+5.7), 1999-03
(−5.3) and 2022-10 (−5.7). Twelve of the eighteen fall before 1996, where the old
covariate's gauge record was incomplete. The 1980-82 drought peaks at +3.7 (1981-10)
and 1995 at +2.1, below the threshold, and 2001-02 shows no difference because both
variants carry the observed diversions from Dec 1999.

The 1960s drought (figure `storage_1960s_drought`) is deeper under the new series:
storage lower from 1961-07 to 1964-04 with a peak difference of −8.6 points on
1961-11-07, FFMP-year minima 59.4 → 53.0 % (1961), 35.2 → 31.9 % (1962) and
31.5 → 28.9 % (1963), then higher at the trough, 8.0 → 9.2 % (1964) and 3.8 → 4.7 %
(1965); the annual demand is higher by 39 to 45 MGD in FFMP years 1961-1962 and lower
by 23 MGD in 1964. No window falls inside the observed River Master record, where both
variants carry the observed diversions.

## 7. Baseline objective scores (`baseline_objectives.csv`)

| Objective | ε | Old | New | Δ (new − old) | Δ / ε |
|---|---|---|---|---|---|
| nyc_delivery_reliability_annual (max) | 0.05 | 0.844 | 0.883 | +0.039 | +0.78 |
| nyc_delivery_deficit_p99_pct (min) | 10 | 40.03 | 41.00 | +0.97 | +0.10 |
| montague_flow_reliability_annual (max) | 0.05 | 0.870 | 0.870 | 0 | 0 |
| montague_flow_deficit_p99_pct (min) | 10 | 17.85 | 17.80 | −0.05 | −0.005 |
| trenton_flow_deficit_p99_pct (min) | 10 | 3.047 | 3.047 | 0 | 0 |
| downstream_flood_exceedance_annual (min) | 0.3 | 0.536 | 0.562 | +0.026 | +0.09 |
| nyc_storage_min_p01_pct (max) | 5 | 6.98 | 8.15 | +1.17 | +0.23 |
| nj_delivery_reliability_annual (max) | 0.05 | 0.870 | 0.883 | +0.013 | +0.26 |

Every difference is below one ε. The largest, NYC delivery reliability, is three more
satisfied unit-years of 77 (0.78 ε) and comes from the pre-1950 era, where the old
series was inflated by the missing gauge; the deficit, flood and storage objectives
move by a tenth to a quarter of their ε.

## 8. Reading

The historic series barely moves in level (−8.5 MGD, −0.010 utilization) while
every synthetic series rises by about 0.07 utilization, so the three scenario designs
now search at one demand level and the HIST-anchored satisficing criteria describe
the same demand the ensembles carry. Fit quality does not change: the covariate
change is a consistency correction, not a better regression. Two features of the new
series are properties of the method rather than of the covariate and are reported,
not fixed: the zero-demand days inherited from the observed daily patterns (163 more
than before), and the Dec 2021 to Nov 2023 tail, where the two covariates disagree by
52 MGD in the mean with no observed diversion to arbitrate. The record now ends
2023-12-31 with the reconstruction (the old CSV ran to 2026-08-02); the HIST window
ends 2023-11-30.
