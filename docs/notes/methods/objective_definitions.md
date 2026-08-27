# Objective Definitions for the Scenario-Design MOEA Study

*Record of the objective formulations used in the MOEA search and of the
held-out re-evaluation metric set. Terminology per `docs/notes/terminology.md`,
and citations resolve to the Zotero collection `ISYGLK35` and the notes under
`docs/notes/literature/`. Supporting diagnostics: `epsilon_calibration_experiment.md`,
`framing_convention_diagnostics.md`, `flood_objective_diagnostics.md`,
`robustness_threshold_diagnostics.md`, `regret_tolerance_diagnostics.md`.*

This note gives the mathematical definition of every objective and the rule
that reduces a simulated (timesteps × realizations) matrix to one scalar, for
each scenario design in the experimental comparison (`experimental_design.md`).
The design principle is that only the search ensemble differs between scenario
designs. The temporal metric of each objective, the annual-unit aggregation,
and the held-out re-evaluation metric are identical across designs (§3), which
is what makes the cross-design comparison commensurable.

The objective set spans the stakeholder priorities Pywr-DRB simulates. NYC
water supply, the lower-basin Decree flow obligations protecting New Jersey and
Philadelphia, downstream flood exposure, and reservoir-system resilience.

Implementation. The temporal metrics are defined in `src/objectives.py` (the
shared `Objective` class and `OBJECTIVES` registry) and the active subset is
`config.ACTIVE_OBJECTIVES`. The search-time annual-unit layer and the
re-evaluation satisficing thresholds are in `src/objectives_ensemble.py`, the
named criterion sets in `src/satisficing_criteria.py`, and the offline
re-evaluation scoring in `src/robustness.py`.

---

## 0. Conventions

- Metrics are computed on the **metric window** of each scenario, the daily
  series from six calendar months after its start (`METRIC_EXCLUSION_MONTHS = 6`),
  cut by date rather than by a fixed day count. Six months is the SSI-6
  accumulation requirement. On the December-start scenario windows the cut
  lands exactly on June 1, the FFMP operating-year boundary, and the
  hazard-selection metrics score the identical [Jun 1 year 1, May 31 year L]
  span (§2), so selection and evaluation see one window.
- Weekly resampling. The delivery reliability metric resamples by **sum**
  (weekly volumes of delivery and entitlement). The delivery deficit metric and
  the flow metrics resample by **mean** (the deficit is normalized by a
  daily-rate cap, and weekly-mean flow is the weekly-accounting basis of the
  Decree).
- **CVaR₉₀(x)** is the Conditional Value-at-Risk at the 90 % level, the mean of
  the worst (largest-deficit) 10 % of weekly values. It is coherent and far
  less variable across realizations than the single maximum (Rockafellar &
  Uryasev 2000; Fairbrother et al. 2022; Löhndorf 2016).
- Goalposts are **static**. The 1954-Decree quantities are NYC 800 MGD,
  Montague 1131.05 MGD (1750 cfs), and the NJ diversion 100 MGD baseline, and
  the **Trenton equivalent-flow objective** is 1938.95 MGD (3000 cfs), an FFMP
  and Good-Faith target rather than a Decree quantity. The time-varying live
  FFMP `mrf_target` is never used, because scoring against the live target
  would let a policy succeed by triggering drought step-downs that lower its
  own goalpost.
- **NYC and NJ delivery is a running-average right, not a daily cap.**
  pywr-drb's `FfmpNyc/NjRunningAvgParameter` let daily diversion exceed the flat
  baseline by drawing down banked allowance, so daily demand is not clipped at
  the static right. Each day's target is the realizable **entitlement**
  `E_t = min(demand_t, A_t)`, where `A_t` is the running-average allowance bank
  (`_delivery_entitlement` / `_running_avg_budget`). The bank starts at the
  static cap, accrues `cap − delivery` daily (floored at 0), and resets (NYC
  annually on Jun 1 following the model's May-31 reset, NJ monthly). The bank
  is accrued at the static cap and never at the policy's drought-scaled
  allowance, so a demand spike within the banked right is honoured, demand
  beyond it is not owed, and a policy cannot lower its own goalpost. The demand
  generator holds the monthly-average demand at or below the cap, so `A_t`
  almost always has headroom and `E_t ≈ demand_t` except at rare
  spike-vs-reset collisions.
- The **McPhail et al. (2018) T1/T2/T3 decomposition** classifies every
  aggregation. **T1** is the performance-value transform (absolute, regret,
  threshold-satisficing), **T2** the scenario subset (worst case, all, tail
  percentile, domain-satisficing), and **T3** the aggregation function (mean,
  variance, higher moments, worst case).

---

## 1. The objective set (single-realization temporal metrics)

These are the per-realization temporal quantities whose windowed-series cores
the annual-unit scheme of §2 reuses. During search every design, including
`historic`, is scored through the §2 scheme, and the historic trace enters it
as N = 1 over its 77 FFMP-year units. The active set is **8 objectives**. NJ
delivery carries independent information (redundancy screen, max |ρ_S| = 0.38
against any objective and ≤ 0.08 against Trenton). All objectives use stable
tail, percentile, or count forms rather than worst-case extremes (Quinn et al.
2017; Bonham et al. 2024). Trenton flow serves as the salinity-repulsion
goalpost, because the Trenton target repels salt intrusion and the salt-front
LSTM, unreliable in extreme drought, stays a registered diagnostic.

| # | Name (registry) | Source | Temporal aggregation | Dir | Units |
|---|-----------------|--------|----------------------|-----|-------|
| 1 | `nyc_delivery_reliability_weekly` | `delivery_nyc`, `demand_nyc` (right 800) | frac of weeks `Σ_w delivery ≥ 0.99·Σ_w E` (entitlement `E_t = min(demand,A_t)`) | MAX | frac |
| 2 | `nyc_delivery_deficit_cvar90_pct` | same | CVaR₉₀ of weekly deficit % `= 100·max(0, mean_w(E) − mean_w(delivery))/800` | MIN | % |
| 3 | `montague_flow_reliability_weekly` | `major_flow.delMontague` | frac of weeks `mean_w(flow) ≥ 1131.05` | MAX | frac |
| 4 | `montague_flow_deficit_cvar90_pct` | `delMontague` | CVaR₉₀ of `100·max(0, 1131.05 − mean_w(flow))/1131.05` | MIN | % |
| 5 | `trenton_flow_reliability_weekly` | `major_flow.delTrenton` | frac of weeks `mean_w(flow) ≥ 1938.95` | MAX | frac |
| 6 | `downstream_flood_exceedance_minor` | `flood_stage` (Hale Eddy, Fishs Eddy, Bridgeville) | mean annual `Σ_days max_gauges (stage − minor)⁺`, ft·days above NWS **minor** flood stage at the worst-affected gauge | MIN | ft·days/yr |
| 7 | `nyc_storage_p5_pct` | `res_storage[NYC]` | 5th percentile of daily `100·Σ_res storage / 270,837` | MAX | % |
| 8 | `nj_delivery_reliability_weekly` | `delivery_nj`, `demand_nj` (right 100) | frac of weeks `Σ_w delivery_nj ≥ 0.99·Σ_w E_nj` (entitlement `E_nj = min(demand_nj,A_t)`, monthly reset) | MAX | frac |

The `Objective` entries in `src/objectives.py` carry single-trace epsilons
(IQR/10 over random-DV policies on the historic trace, Reed et al. 2013) that
apply to single-realization diagnostics only. The campaign archive resolves at
the annual-unit epsilons of §2.

**Why these aggregations.**
- *Reliability frequencies (1, 3, 5, 8).* Hashimoto reliability and
  multivariate domain-satisficing, the form Herman et al. (2015) recommend,
  stable and fast-converging (Bonham et al. 2024). Montague reliability cannot
  saturate at 1.0 because FFMP step-downs intentionally drop releases below the
  target in drought, so it stays continuous.
- *CVaR₉₀ deficits (2, 4).* CVaR₉₀ is used instead of the worst-week maximum,
  which Quinn et al. (2017) flag as a high-variance, low-information signal.
  CVaR keeps the tail-risk focus but averages the worst decile, giving a
  reproducible, smooth Borg gradient. Montague flow is storm-dominated, so its
  single worst week is mostly exogenous noise and CVaR matters most there.
- *Flood exceedance above minor stage (6).* Magnitude-weighted exceedance
  (`flood_objective_diagnostics.md`). The day count is degenerate across
  policies (9 distinct values over 25 feasible policies on the historic trace)
  while the exceedance integral resolves fully and responds strictly
  monotonically to the flood-release DVs, and exceedance tracks observed annual
  flood magnitude better (Pearson 0.91 vs 0.83). The integrand is physical
  exceedance rather than monetized damage, avoiding the expectation-of-damage
  trap (Quinn et al. 2017). The max-across-gauges basis avoids triple-counting
  basin-wide events, and the metric is normalized per metric-window year so it
  is invariant to record length. The NWS **minor** (flood-onset) stage marks
  actual flooding, a more meaningful goalpost than the FFMP cautionary cutoff,
  which is also the control rule's own switching boundary. The day-count
  (`minor`/`major`/`action`) variants stay registered as diagnostics.
- *Storage p5 (7).* A low percentile is a stable vulnerability proxy, whereas
  the single-day minimum is dominated by one drought event (Quinn et al. 2017).
- *Trenton vs salinity (5) and NJ delivery (8).* New Jersey, a co-equal Decree
  party, gets direct representation so the search can discover NYC vs NJ
  robustness conflicts (Trindade et al. 2017; Hadjimichael et al. 2020).

**Diagnostics (registered, not active).** Worst-case variants
(`*_deficit_max_pct`, `nyc_storage_min_pct`), `downstream_flood_days_minor` /
`_major` / `_action`, `trenton_flow_deficit_cvar90_pct`, the salt-front metric
(`salt_front_intrusion_max_rm`), and the deferred Lordville thermal metric.
They are available for re-evaluation reporting without code changes.

Dimensionality. Eight objectives keep the epsilon-dominance archive where
hypervolume stays estimable (Reed et al. 2013), with epsilons in each metric's
native units.

---

## 2. Search aggregation, the two-layer annual-unit scheme

Two principles govern the scheme. Cross-design commensurability during search
is not required, because the held-out re-evaluation (§3) is the only
comparison point, and every operator follows published search-time practice
(objective aggregation is not a novelty focus of this study).

**Structure (Hamilton et al. 2022's two-layer vocabulary, within-record time
aggregation plus across-record noise filtering).** Each realization is
simulated continuously, the first six months are outside the metric window,
and the remainder is split into **FFMP-year units** (June 1 – May 31, the
operating year on which the FFMP's seasonal rules reset). Scenario windows are
December-aligned (December 1 of year 0 through November 30 of year L), so the
exclusion ends exactly on June 1 of year 1 and the first unit opens there. The
trailing June–November fragment of year L is discarded, leaving L − 1 units
spanning June 1 year 1 – May 31 year L, the identical window the
hazard-selection metrics score (`ffmp_year_unit_slices`). Stage (i) computes
each objective's **annual metric** on every (realization × year) unit. Stage
(ii) aggregates across the pooled **N(L − 1) unit-years** with the objective's
**unit operator**.

| # | Objective (registry) | Annual metric (per unit-year) | Unit operator (across pooled unit-years) | Dir | ε | Anchor |
|---|---|---|---|---|---|---|
| 1 | `nyc_delivery_reliability_annual` | failure-year indicator, ≥ 3 failing weeks (`Σ_w delivery < 0.99·Σ_w E`, entitlement `E_t = min(demand,A_t)`) | **frequency of non-failure years** | MAX | 0.05 | Zeff et al. 2014 Eq. 2; Trindade et al. 2017 Eq. 16; Gold et al. 2023 |
| 2 | `nyc_delivery_deficit_p99_pct` | CVaR₉₀ of weekly deficit % within the year | **worst-1st-percentile unit-year** (P99) | MIN | 10.0 | Quinn et al. 2017 (WP1), 2018; Trindade/Gold worst-1 % cost |
| 3 | `montague_flow_reliability_annual` | failure-year indicator, ≥ 3 failing weeks (`mean_w(flow) < 1131.05`) | frequency of non-failure years | MAX | 0.05 | as #1 |
| 4 | `montague_flow_deficit_p99_pct` | CVaR₉₀ of weekly Montague deficit % within the year | worst-1st-percentile unit-year | MIN | 10.0 | as #2 |
| 5 | `trenton_flow_reliability_annual` | failure-year indicator, ≥ 1 failing week vs 1938.95 MGD | frequency of non-failure years | MAX | 0.05 | as #1 |
| 6 | `downstream_flood_exceedance_annual` | `Σ_days max_gauges (stage − minor)⁺` in the year (ft·days; `flood_objective_diagnostics.md`) | **mean across unit-years** (expected annual flood exceedance) | MIN | 0.3 | Trindade expected-cost form; Quinn 2017 caution |
| 7 | `nyc_storage_min_p01_pct` | annual minimum of daily aggregate NYC storage % | **1st-percentile unit-year** | MAX | 5.0 | WP1 pattern (Quinn 2017/2018); Hamilton 2022 Q-of-max |
| 8 | `nj_delivery_reliability_annual` | failure-year indicator, ≥ 1 failing week vs the NJ delivery criterion | frequency of non-failure years | MAX | 0.05 | as #1 |

The ε column is the campaign vector [0.05, 10.0, 0.05, 10.0, 0.05, 0.3, 5.0,
0.05] in native metric units (`src/objectives_ensemble.py::_ANNUAL_REGISTRY_SPEC`,
`config.get_epsilons()`), one shared precision per objective family, derived in
`epsilon_calibration_experiment.md` (SI Text S5). ε enters Borg at runtime, so
a change needs no JAR rebuild. The P99 unit operator for the flood objective is
tie-degenerate at the pooled unit count and 12–30× noisier under bootstrap, so
it stays registered as an inactive diagnostic (`downstream_flood_days_annual_p99`).

**Why this scheme.**
- *Reliability objectives keep the threshold form where the literature keeps
  it.* Fraction-of-units frequency is the citable satisficing-in-search
  operator, the only one used in search in the WaterPaths lineage, while
  magnitude and tail objectives use mean or percentile forms, so no
  analyst-chosen satisficing level exists for them. Each annual failure
  criterion combines a **static goalpost** (§0) with a **failing-week count
  k** (`_DEFAULT_FAILURE_K`). k is 3 for NYC delivery and Montague flow, so a
  failure year is a month-scale shortfall rather than an isolated off week,
  and 1 for Trenton and NJ, where a larger k saturates the metric toward 1.0.
  The goalposts are anchored and k is a convention screened for saturation per
  design composition (`framing_convention_diagnostics.md` §1, no shipped k
  saturates in either composition, rankings stable to k ± 1, Trenton k of 1
  binding).
- *The long-record design needs no special case.* Its record is scored as
  consecutive annual units with inherited state, exactly the treatment of
  Quinn et al. (2018), who slice one continuous 1000-yr record into 1-yr units
  so that the distribution of initial conditions is representative. Every
  design therefore has the identical unit denominator, N × (L − 1)
  metric-bearing unit-years, with partial units at either end excluded.
- *Granularity and ε.* Frequency objectives have granularity 1/N(L − 1),
  while mean and percentile objectives are continuous with ε in native metric
  units.
- *Precedent floor.* Percentile operators are precedented over roughly 50–1000
  units (Quinn's WP1 used 1000). Estimator noise at the campaign unit count is
  measured directly by the ensemble-size library
  (`ensemble_size_diagnostics.md`).
- *Weekly bins re-anchor inside every unit-year.* Each Jun–May unit is
  resampled to weeks independently, so a unit holds 53 bins with one short
  (1–2-day) trailing bin that carries full weight in the failing-week counts
  and the within-year CVaR₉₀ pools (measured effect on failing-week counts
  about +2 % against a continuous weekly grid). The convention is identical
  across every design and both evaluation layers, so it cancels in every
  comparison and is recorded as a property of the per-unit accounting.

**Caveats carried explicitly.** Unit-years within a realization are dependent
(multi-year droughts appear as consecutive failure-years, which is how the
WaterPaths-lineage frequency objectives express persistence). The effective
sample size is below N(L − 1) and differs by design
(`ensemble_size_diagnostics.md` §5), disclosed rather than corrected. An annual
window cannot hold a whole multi-year drought as a single unit, so event-scale
severity enters through the hazard axes (hazard selection and scenario
discovery) rather than through the objective statistics, which are the same
annual-unit quantities in search and re-evaluation.

**Design mapping.** All three designs use this same two-layer scheme. The
**historic design** enters it as N = 1 over the consecutive FFMP-year units of
its single 78-yr trace (77 metric-bearing units; prevailing-practice reference,
Giuliani & Castelletti 2016). In McPhail terms stage (i) is T1-threshold
(reliability) or T1-absolute (magnitude and tail) and stage (ii) is T3
frequency or expectation for #1/3/5/6 and T2 tail percentile for #2/4/7. The
`hazard_filling_stationary` sample's deliberate probability distortion relative
to the generator is not corrected, and cross-design comparison rests entirely
on the common re-evaluation (§3).

---

## 3. Re-evaluation, the held-out metric set

The three designs differ only in the search ensemble. They are compared once,
by re-evaluating every resulting Pareto-approximate set on one common held-out
test ensemble E_test with one fixed metric set. Only at re-evaluation are
differences attributable to scenario design rather than to a moving measuring
stick (McPhail et al. 2020, composition moves robustness values more than
rankings).

**E_test is a designed exploration, not a probability sample, and no
robustness number here is an expectation.** E_test is a Latin hypercube over
the full range of the deeply uncertain forcing factors, with many realizations
per LHS point (`scenario_design_methods.md` §5, `src/etest.py`). Under deep
uncertainty there is no probability measure over Θ, so a satisficing fraction
over E_test is a coverage-weighted count over a designed exploration of the DU
space and is reported as such, never as an estimate of an expectation. What
makes the cross-design comparison commensurable is that E_test is identical
across all designs, not that it is probability-faithful. The held-out
re-evaluation removes the evaluation bias of scoring each design on its own
ensemble, and selection bias is not corrected, because it is the quantity the
experiment measures (Bartholomew & Kwakkel 2020). No explicit scenario
weighting is used anywhere, in search or in re-evaluation.

**The per-SOW objective matrix is persisted.** Re-evaluation persists the
(solution × SOW × objective) matrix in natural units (`reeval_raw` plus a
self-describing `reeval_raw_meta.json`). Each E_test state of the world's
R = 25 realizations contribute their stage-(i) unit-years to one pooled sample,
and the objective's §2 unit operator collapses that pool to the state's
objective value (`reeval_core.sow_objective_matrix`). Every robustness and
regret metric is scored offline from that matrix (`src/robustness.py`), so a
new metric or a changed threshold never requires re-simulating.

**The re-evaluation metric is the §2 search metric, recomputed per state of
the world.** Search and re-evaluation share one code path for the statistic,
the stage-(i) annual metrics (`annual_units()`) and the stage-(ii) unit
operators, and differ only in the pool the operator collapses.

| | Search (`_evaluate_ensemble_batched`) | Re-evaluation (`evaluate_annual_units` + `sow_objective_matrix`) |
|---|---|---|
| per realization | `annual_units()`, a vector of L − 1 annual metrics | the same `annual_units()` |
| pooling | all realizations' unit-years through the §2 unit operator to one search objective | each θ's realizations' unit-years through the same unit operator to J_i(x, θ) |

This is the standard construction of the robustness literature, in which the
performance measure inside the robustness calculation is the optimization
objective re-evaluated per state of the world (Herman et al. 2015 Eqs. 3–7;
Trindade et al. 2017; Herman et al. 2014; Quinn et al. 2018; Gold et al. 2023;
McPhail et al. 2018). What differs between search and evaluation is the
ensemble and the outer aggregation across states, never the definition of the
statistic. One metric currency also removes the annual-unit statistics'
record-length sensitivity from the comparison, because a 10-yr search window
and the 50-yr test window are pooled through the same per-unit-year operators.
Each SOW's pool (25 × 49 = 1,225 unit-years) keeps the tail operators
well-defined per state, and the per-SOW P01 is about the 12th-worst unit-year,
exactly the worst first-percentile year construction Quinn et al. (2018)
evaluate per SOW. `objectives_summary.csv` holds the mean over SOWs of the
per-SOW objective values (`sowmean__*` columns), derived from the same
persisted matrix.

### 3.1 Primary metric, multivariate satisficing (Starr's domain criterion)

**The fraction of E_test states of the world in which the policy's per-SOW
objective vector J_i(x, θ) meets all criteria of a criterion set jointly.**
This is Starr's (1962) domain criterion across the N_θ = 500 re-evaluated
states (the leading half of the 1,000 generated, `campaign_design.md` §5), the
standard measure of the Herman (2014/2015), Trindade (2017, 2019), and Gold
(2023) lineage and Herman et al. (2015)'s own recommendation. Criterion sets
are explicit subsets of one to three thresholded objectives with every other
axis unconstrained (`src/satisficing_criteria.py`, the Quinn et al. 2017
small-conjunction pattern), reported under several stakeholder framings, with
the all-axes conjunction kept only as the `reference_all8` set.

In McPhail terms T1 is satisfaction of constraints, T2 all states, and T3
frequency, applied to the search objective recomputed per state. Pooling each
θ's realizations' unit-years through the unit operator is the within-state
collapse. Natural variability inside a state enters through the statistic's
own definition, exactly as it does during search, so there is no separate
within-SOW risk-attitude knob to choose or record.

**Why the SOW is the unit.** The construction matches the Triangle lineage,
whose objectives are likewise ensemble statistics per state (Herman et al.
2014; Trindade et al. 2017; Gold et al. 2023; stated as a convention by
Bartholomew & Kwakkel 2020). Three reasons make it right here too.

- Measure separation. E_test carries no probability measure over the forcing
  space (Lamontagne et al. 2018). A pooled-realization fraction would integrate
  a designed LHS box and a fitted stochastic generator in one number, whereas
  counting states applies a coverage-weighted count to the design and leaves
  natural variability inside the per-state statistic.
- Precision. Realizations sharing a θ are not independent. The reported
  ±2.2 pp is 0.5/√500, a SOW-unit standard error on the 500 re-evaluated SOWs.
  Bonham et al. (2024)'s 50–300 convergence result is measured on a flat
  ensemble and so bounds N_θ, not R.
- One unit everywhere. The incumbent-relative regret family (§3.2b), scenario
  discovery's failure labels, and the attainability screen all consume the
  same per-SOW J_i(x, θ).

### 3.2 Secondary metrics (all reported)

| Metric | Definition | McPhail T1 / T2 / T3 | Role |
|---|---|---|---|
| **univariate satisficing** | fraction of SOWs clearing one objective's criterion (`sat_uni_sow__`) | satisfaction / all / frequency | per-objective decomposition of the primary, same unit, conjunction dropped |
| **Laplace (mean)** | mean of J_i(x, θ) across SOWs (`laplace__`) | identity / all / mean | risk-neutral reference |
| **maximin** | worst SOW's J_i(x, θ) (`maximin__`) | identity / worst case / worst case | risk-averse reference (Wald) |

Ranking agreement is summarized with Kendall's τ_b computed across the design
rankings these metrics induce, i.e. whether the metrics rank the scenario
designs the same way (Herman et al. 2015; McPhail et al. 2018, 2020).

### 3.2b Incumbent-relative regret (co-primary; answers RQ1)

RQ1 asks whether re-optimized policies improve some outcomes without degrading
others below current performance. A mean improvement can be comfortably
positive while the policy is badly worse than the status quo in a third of
futures. The regret family reports the signed incumbent advantage through its
one-sided halves (`regret_*__` and `gain_mean__`) plus unit-free frequencies,
all on the same per-SOW objective values as the primary metric.

**Reference.** The incumbent 2017 FFMP policy, evaluated in the same SOW.
McPhail et al. (2018) §3.1 license exactly this T1 (a baseline decision
alternative's performance for a given scenario in place of the best
alternative's) without naming, tabulating, or testing it, so this study makes
an admitted variant explicit. It is also Savage regret on the action set the
Decree parties actually face, retain the FFMP or adopt the candidate, a
unanimity-bound renegotiation rather than a free choice over an archive.

**Unit and construction** (`src/robustness.py`, on the per-SOW objective
values).

- `D_i(x,θ) = σ_i · [ J_i(x,θ) − J_i(b,θ) ]`, signed and oriented so positive
  means better than current operations for every objective
  (`incumbent_advantage`).
- Magnitudes in each objective's own natural units, never combined across
  objectives. `regret_mean__`, `regret_q90__` (the Herman R1/R2 tail
  statistic), `regret_cond__` (mean shortfall given a shortfall, NaN when never
  worse), and `gain_mean__` as the mandatory companion.
- Frequencies, unit-free, which carry the cross-objective and cross-design
  scalar role. `harm_freq__{obj}`, `party_harm_freq__{party}` (a disjunction
  over a Decree party's objectives, because under unanimity a loss is not
  compensable and so is never a sum), `no_harm_freq` (weak Pareto improvement
  on the incumbent), `no_harm_freq_tau`, and `n_degraded_mean`.

**Restriction to the adverse subset is McPhail's T2.** Their undesirable
deviations metric (Kwakkel et al. 2016b) decomposes as T1 regret from median,
T2 worst half, T3 sum. Ours is that construction with the reference changed
from the policy's own median to the incumbent. A mean of a clipped quantity
over all scenarios collapses toward zero when policies mostly beat the
incumbent, but a statistic computed on the sign-selected subset does not.

**No max regret.** Bonham et al. (2024) show regret families need 400+
scenarios and never converge on extreme-of-extremes operators, and McPhail et
al. document the tie-degeneracy directly. The 90th percentile over the 500
re-evaluated SOWs rests on about 50 worst states, a fixed-quantile operator far
from that degeneracy.

**Natural units, and no cross-objective magnitude scalar.** Dividing by the
incumbent's own per-state value is degenerate for this objective set. Flood
exceedance is exactly 0 in a share of states and both deficit tails are 0 in
wet ones, so the cell would be dropped, and the dropped cells are the benign
ones, biasing the estimator toward the adverse subset. Herman et al. (2015)
additionally show the normalized-deviation form selects poor-baseline solutions
as a mathematical artifact. Natural units dissolve both problems. Neither
published scale is usable, because Cohen et al. (2021) normalize on the
per-scenario span to a perfect-foresight optimum (one MOEA run per scenario)
and Sunkara et al. (2023) rescale over the alternative set, which is
design-coupled. An optional fixed scale (`incumbent_spread`, the incumbent's
q90 − q10 over E_test) is implemented as a scoring-time sensitivity and is
never the reported primary.

**The tolerance ladder.** `τ_i = k · u_i` with the unit
`u_i = max(ε_i, τ_i^floor)`. `ε_i` is the objective's annual-unit epsilon from
the campaign registry (`src/objectives_ensemble.py`), the just-noticeable
difference in exactly the annual-unit metric space the per-SOW cube lives in,
so the regret tolerance and the search resolution sit on one calibration, and
`τ_i^floor` is the measured noise floor of that objective's per-SOW estimator.
Taking the max matters because one `k` is shared across eight objectives, and
an epsilon below its own noise floor would make every rung fire on Monte Carlo
noise for that axis while being far outside the noise on the others. The
adopted whole vector is pinned as `NYCOPT_REGRET_TAU` in the production env
files and `k` is swept over `REGRET_TAU_GRID` (`scripts/main/compare_designs.py`)
for the reason the satisficing criteria are swept, because a single tolerance
could manufacture or hide the whole RQ1 answer (Quinn et al. 2020). The rules
that fix `k`, the ladder shape, the noise floor, and the discrimination band are
specified in `regret_tolerance_diagnostics.md`. No anchor may be read off the
distribution of candidate-policy regret, because that is the quantity under
test.

**Why this is not redundant with satisficing.** The satisficing criteria are
fixed scalars anchored on the incumbent's attainment
(`robustness_threshold_diagnostics.md`), whereas the regret bar is the
incumbent's performance in that SOW and moves with the forcing. Where a fixed
criterion drives the domain criterion to 0 or 1 for every policy, satisficing
ties everything (Bonham et al. 2024's saturation failure mode) and regret still
separates policies.

**Comparison rule.** Robustness and regret are read off the same policy. They
are reported as the endpoint policy's regret, the full
`(sat_multivariate_sow, no_harm_freq_tau)` cloud, the per-design non-dominated
frontier in that plane, and the per-objective natural-unit drill-down. This is
the study's analogue of Bartholomew & Kwakkel's (2020) price-of-robustness
measurement, against a fixed external incumbent per SOW rather than by
hypervolume against reference scenarios (which would reintroduce the
pooled-reference-set bias rejected in §4.3). A severity decomposition over
terciles of the dominant forcing factor `m` (|ρ_S| = 0.91–0.98 on all eight
objectives) tests whether any price is paid in the benign futures, an insurance
premium that is a finding rather than a failure.

**Degeneracy guard.** A policy scores zero regret by being the incumbent,
which is reachable because the FFMP baseline lies inside the searched DV
space. Regret is therefore never reported without `gain_mean` beside it.

### 3.3 Metrics deliberately excluded

Regret has four possible references, and exactly one is computed here.

| Reference | Question it answers | Status here |
|---|---|---|
| The incumbent policy, per SOW | how much worse off than under current rules | **computed** (§3.2b) |
| The best policy in the evaluated set, per SOW (Savage; Herman R2) | whether the wrong policy was picked from the archive | excluded |
| The same policy in a baseline SOW (Herman R1; Kasprzyk et al. 2013) | how wrong the assumptions about the future were | not computed |
| A perfect-foresight optimum, per scenario (Cohen et al. 2021) | what imperfect information cost | not computed |

Best-in-set regret is set-relative and design-coupled (dropping one design
changes every other design's score), and Bonham et al. (2024) show it converges
far more slowly than satisficing, never on max-over-time objectives like the
P99 deficit operators. Herman R1 and the Kasprzyk et al. (2013) percent
deviation are within-policy, across-SOW sensitivity measures whose reference is
the same solution's value in a baseline state of the world, not a status-quo
policy, so they answer a different question from RQ1 and are not the precedent
for the incumbent comparison (that chain is McPhail et al. 2018 §3.1 for the
reference, Herman et al. 2015 for the functional shape, Kwakkel et al. 2016b
for the adverse-subset construction). Cohen et al. (2021) baseline regret would
require one perfect-foresight MOEA run per scenario, and no perfect-foresight
optimization is performed anywhere in this study.

A search-vs-test overfitting gap is likewise not computed. Brodeur et al.
(2020) diagnose overfitting graphically and define no gap metric, and a
coverage-weighted in-sample term minus a measure-weighted out-of-sample term
measures the measure change, not overfitting (`tests/test_robustness.py`
asserts the helper's absence).

### 3.4 Attainability screen

Flag the E_test states of the world in which no policy from any design meets
the criteria. This costs zero CPU, because the (solution × SOW × objective)
matrix already exists, and it separates a design that searched badly from a
test state that is unwinnable for anyone. Every design's satisficing fraction
is bounded above by the attainable fraction, so the screen sets the ceiling
against which design differences are read (precedent, Shavazipour et al. 2021
found 23 % of their test scenarios unwinnable). It is an empirical
attainability bound over the evaluated policy pool, not a per-scenario oracle.
The codebase separates `SEARCH_ENSEMBLE_SPEC` from the common test ensemble
with a selection-bias guard (Bonham et al. 2024) that raises if they coincide.

---

## 4. Threshold sweep, scenario discovery, and the cross-design comparison rule

### 4.1 Satisficing criteria are conventions, so they are swept

No one in this lineage derives a satisficing threshold. Zeff et al. (2014)
elicited them from the Research Triangle utilities and every later number
descends from those by convention (worst-case cost 5 % in Trindade et al. 2017
and 10 % in Gold et al. 2023, restriction frequency 20 % and 10 %, Trindade's
reliability relaxed from 99 % to 98.5 % because no solution met 99 %). Exactly
one threshold in the lineage has an external anchor, Gold et al. (2023)'s peak
financial cost below 80 % of annual volumetric revenue from AWWA bond-covenant
limits.

This study therefore anchors each criterion on a Delaware River Basin Decree
or FFMP goalpost wherever one exists (§0), re-anchors criteria the status quo
itself fails per the rules of `robustness_threshold_diagnostics.md`, and
sweeps the rest. Quinn et al. (2020) make the sweep mandatory, because
robustness-rank agreement across scenario designs degrades as the satisficing
criterion becomes more stringent, so the design effect is largest at the
conservative end and any single fixed threshold could manufacture or hide the
entire result.

The main-text figure is the cross-design comparison over a grid of
stringencies, asking whether the design ranking, not the robustness value, is
invariant. The threshold-margin diagnostic draws the CDF of each objective
across E_test with the criterion as a vertical line (Gold et al. 2023, Fig. 5).
Implementation is the design-ranking stringency sweep in
`scripts/main/compare_designs.py` (`NYCOPT_COMPARE_STRINGENCY`), an offline
re-scoring of the persisted cube, with `threshold_spectrum` in
`src/robustness.py` supplying the per-run univariate spectrum.

### 4.2 Scenario discovery, a supporting analysis

Scenario discovery is a supporting analysis, not the primary comparison and
not a falsification device. The primary comparison is the re-evaluated
robustness of the resulting solutions (§3). The primary factor maps run in the
DU forcing space θ of E_test (`src/factor_mapping`,
`scripts/main/factor_mapping_run.py`, the setting of Hadjimichael et al. 2020
and Gold et al. 2022). Boosted trees (Gold et al. 2022/2023 hyperparameters)
are fit to each design's E_test failure labels, which are the focal criterion
set's conjunction on the per-SOW values, after re-evaluation. Because the
forcing factors are sampled independently by LHS, the correlated-factor
instability of factor-importance rankings (Quinn et al. 2020) does not arise
by construction.

The one hazard-space inference is the step-11 coverage-deficit mechanism test
(`scripts/main/scenario_discovery.py`). For each E_test SOW it computes the
distance in the E_test hazard image's empirical-CDF space to the nearest member
of the design's search ensemble and tests whether failure probability is
positively associated with it, read as AUC minus a random-coverage null. A
hazard-space classifier and factor map on the same labels are its supplemental
view, with the hazard axes screened for redundancy before fitting. The
prediction is that a design's policies fail on E_test in the hazard region it
under-covered during search, so hazard-filling designs should show no excess
association and `fixed_probabilistic` and `historic` should. A null is a
reportable result. Discovery in either space is reported as support for a
difference found on the primary metric, never as the basis of the comparison.

### 4.3 Cross-design comparison rule, no pooled-reference-set hypervolume

Designs are compared only by re-evaluation on the common E_test. MOEA search
metrics (hypervolume, generational distance, ε-indicator) scored against a
pooled reference set are not a defensible primary cross-design measure. They
are reported in the supplement with the reference set built per design and
read as within-design convergence diagnostics only. A pooled reference set is
biased across designs three ways. A design contributes points to the very
frontier it is scored against (contribution share is a merit diagnostic, not a
yardstick, Reed et al. 2013), designs return different solution counts
(Bartholomew & Kwakkel 2020), and noisier re-evaluated estimates contribute
more spuriously nondominated points (Shavazipour et al. 2021). Zatarain
Salazar et al. (2017) §5.3 set the protocol precedent by building reference
sets per search-ensemble level, ruling cross-level MOEA metrics incomparable,
and comparing levels only by re-evaluating on a common verification ensemble.
