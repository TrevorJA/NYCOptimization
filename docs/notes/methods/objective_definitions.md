# Objective Definitions for the Scenario-Design MOEA Study

*Last updated: 2026-06-17. Authoritative record of the objective formulations
used in the MOEA optimization; supersedes inline docstrings where they disagree.
Terminology per `docs/notes/terminology.md`; citations resolve to the Zotero
collection "Paper 3 NYC Reoptimization" (`ISYGLK35`) and the notes under
`docs/notes/literature/`. The supporting random-DV sensitivity experiment that
finalizes thresholds, epsilons, and the redundancy screen lives in
`docs/notes/methods/objective_sensitivity_experiment.md`.*

This document gives the mathematical definition of every objective and the rule
that reduces a simulated **(timesteps × realizations)** matrix to one scalar,
for each scenario design in the experimental comparison
(`docs/notes/methods/experimental_design.md`). The design principle: **only the
search-time across-realization aggregation differs between scenario designs; the
temporal metric of each objective and the held-out re-evaluation metric are
identical across all designs** (§3). That is what makes cross-design comparison
commensurable.

The objective set spans the stakeholder priorities Pywr-DRB simulates: NYC water
supply, the lower-basin Decree flow obligations protecting New Jersey and
Philadelphia, downstream flood exposure, and reservoir-system resilience.

Implementation: all metrics are defined in `src/objectives.py` (the shared
`Objective` class + `OBJECTIVES` registry); the active subset is
`config.ACTIVE_OBJECTIVES`. Swapping an objective is a config edit, never a code
change. The across-realization satisficing wrapper is in
`src/objectives_ensemble.py`.

---

## 0. Conventions

- Metrics are computed on the **post-warmup** daily series (drop the first
  `WARMUP_DAYS = 365` days).
- `resample("W")` = weekly resampling: reliability resamples by **sum**;
  flow/deficit metrics by **mean** (the weekly-accounting basis of the Decree).
- **CVaR₉₀(x)** = Conditional Value-at-Risk at the 90% level = the mean of the
  worst (largest-deficit) 10% of weekly values. Coherent and far less variable
  across realizations than the single maximum (Rockafellar & Uryasev 2000;
  applied in Fairbrother et al. 2022; Löhndorf 2016).
- Decree goalposts are the **static** 1954-Decree quantities — NYC 800 MGD,
  Montague 1131.05 MGD (= 1750 cfs), Trenton 1938.95 MGD, NJ diversion 100 MGD
  baseline — never the time-varying live FFMP `mrf_target` (scoring against the
  live target would let a policy "succeed" by triggering drought step-downs that
  lower its own goalpost).
- The **McPhail et al. (2018) T1/T2/T3 decomposition** classifies every
  aggregation: **T1** performance-value transform (absolute / regret /
  threshold-satisficing), **T2** scenario subset (worst-case / all / tail
  percentile / domain-satisficing), **T3** statistical moment (mean / variance /
  higher).

---

## 1. The objective set (single-realization / historic temporal metrics)

For the historic design there is no realization axis (R = 1), so the objective
**is** the temporal metric below. The recommended active set is **7 objectives**
(an optional 8th, NJ delivery, is pending the redundancy screen). Worst-case
extremes were replaced with stable tail/percentile/count forms (Quinn et al.
2017; Bonham et al. 2024); the salt-front objective was replaced by the Trenton
flow Decree (physically redundant — the Trenton target repels salt intrusion —
and the salt-front LSTM is unreliable in extreme drought).

| # | Name (registry) | Source | Temporal aggregation | Dir | Units | ε |
|---|-----------------|--------|----------------------|-----|-------|---|
| 1 | `nyc_delivery_reliability_weekly` | `delivery_nyc`, `demand_nyc` (cap 800) | frac of weeks `Σ_w delivery ≥ 0.99·Σ_w min(demand,800)` | MAX | frac | 0.01 |
| 2 | `nyc_delivery_deficit_cvar90_pct` | same | CVaR₉₀ of weekly deficit % `= 100·max(0, mean_w(min(demand,800)) − mean_w(delivery))/800` | MIN | % | 0.5 |
| 3 | `montague_flow_reliability_weekly` | `major_flow.delMontague` | frac of weeks `mean_w(flow) ≥ 1131.05` | MAX | frac | 0.01 |
| 4 | `montague_flow_deficit_cvar90_pct` | `delMontague` | CVaR₉₀ of `100·max(0, 1131.05 − mean_w(flow))/1131.05` | MIN | % | 0.5 |
| 5 | `trenton_flow_reliability_weekly` | `major_flow.delTrenton` | frac of weeks `mean_w(flow) ≥ 1938.95` | MAX | frac | 0.01 |
| 6 | `downstream_flood_days_minor` | `flood_stage` (Hale Eddy, Fishs Eddy, Bridgeville) | count of days any gauge `≥` its NWS **minor** flood stage | MIN | days | 3.0 |
| 7 | `nyc_storage_p5_pct` | `res_storage[NYC]` | 5th percentile of daily `100·Σ_res storage / 270,837` | MAX | % | 0.5 |
| 8 | `nj_delivery_reliability_weekly` *(optional)* | `delivery_nj`, `demand_nj` (cap 100) | frac of weeks `Σ_w delivery_nj ≥ 0.99·Σ_w min(demand_nj,100)` | MAX | frac | 0.01 |

**Why these aggregations.**
- *Reliability frequencies (1, 3, 5, 8)* — Hashimoto reliability / multivariate
  domain-satisficing, the form Herman et al. (2015) recommend; stable and
  fast-converging (Bonham et al. 2024). Montague reliability cannot saturate at
  1.0 because FFMP step-downs intentionally drop releases below the target in
  drought, so it stays continuous.
- *CVaR₉₀ deficits (2, 4)* — replace the former worst-week maxima, which Quinn
  et al. (2017) flag as high-variance, low-information signals. CVaR keeps the
  tail-risk focus but averages the worst decile → reproducible, smooth Borg
  gradient. Montague flow is storm-dominated, so its single worst week is mostly
  exogenous noise — CVaR matters most there.
- *Flood days at minor stage (6)* — count-over-threshold, the stable form that
  avoids the expectation-of-damage trap (Quinn et al. 2017). The NWS **minor**
  (flood-onset) stage marks actual flooding, a more meaningful goalpost than the
  FFMP action cutoff; `major` and `action` variants are registered for swapping.
- *Storage p5 (7)* — a low percentile is a stable vulnerability proxy; the
  single-day minimum is dominated by one drought event (Quinn et al. 2017).
- *Trenton vs salinity (5)* and *NJ delivery (8)* — give New Jersey, a co-equal
  Decree party, direct representation so the search can discover NYC↔NJ
  robustness conflicts (Trindade et al. 2017; Hadjimichael et al. 2020). NJ
  delivery is added to the active set only if the redundancy screen shows it is
  not collinear with Trenton reliability.

**Diagnostics (registered, not active):** worst-case variants
(`*_deficit_max_pct`, `nyc_storage_min_pct`), `downstream_flood_days_major` /
`_action`, `trenton_flow_deficit_cvar90_pct`, the salt-front metric
(`salt_front_intrusion_max_rm`), and the deferred Lordville thermal metric. They
are available for swapping or re-evaluation reporting without code changes.

Dimensionality: 7 keeps the epsilon-dominance archive where hypervolume stays
estimable (Reed et al. 2013); epsilons are in each metric's native units.

---

## 2. Per-scenario-design aggregation — two-layer annual-unit scheme

Two principles govern the scheme: (i) **cross-design commensurability during search
is not required** — the held-out re-evaluation (§3) is the only comparison point —
and (ii) **every operator follows published search-time practice** (objective
aggregation is not a novelty focus of this study).

**Structure (Hamilton et al. 2022's two-layer vocabulary: within-record time
aggregation + across-record noise filtering).** Each realization is simulated
continuously; the first 365 days are warm-up and excluded; the post-warm-up series is
split into **water-year units**. Stage (i): compute each objective's **annual metric**
on every (realization × year) unit. Stage (ii): aggregate across the pooled **NL
unit-years** with the objective's **unit operator**:

| # | Objective (registry) | Annual metric (per unit-year) | Unit operator (across pooled unit-years) | Dir | Anchor |
|---|---|---|---|---|---|
| 1 | `nyc_delivery_reliability_annual` | failure-year indicator: ≥1 week with `Σ_w delivery < 0.99·Σ_w min(demand,800)` | **frequency of non-failure years** | MAX | Zeff et al. 2014 Eq. 2; Trindade et al. 2017 Eq. 16; Gold et al. 2023 |
| 2 | `nyc_delivery_deficit_p99_pct` | CVaR₉₀ of weekly deficit % within the year | **worst-1st-percentile unit-year** (P99) | MIN | Quinn et al. 2017 (WP1), 2018; Trindade/Gold worst-1%-cost |
| 3 | `montague_flow_reliability_annual` | failure-year indicator: ≥1 week with `mean_w(flow) < 1131.05` | frequency of non-failure years | MAX | as #1 |
| 4 | `montague_flow_deficit_p99_pct` | CVaR₉₀ of weekly Montague deficit % within the year | worst-1st-percentile unit-year | MIN | as #2 |
| 5 | `trenton_flow_reliability_annual` | failure-year indicator vs 1938.95 MGD | frequency of non-failure years | MAX | as #1 |
| 6 | `downstream_flood_days_annual` | count of minor-flood days in the year | **mean across unit-years** (expected annual flood days); P99 variant registered pending the sensitivity experiment (expectation can mask floods — Quinn et al. 2017) | MIN | Trindade expected-cost form; Quinn 2017 caution |
| 7 | `nyc_storage_min_p01_pct` | annual minimum of daily aggregate NYC storage % | **1st-percentile unit-year** | MAX | WP1 pattern (Quinn 2017/2018); Hamilton 2022 Q-of-max |
| 8 | `nj_delivery_reliability_annual` *(optional)* | failure-year indicator vs NJ delivery criterion | frequency of non-failure years | MAX | as #1; pending redundancy screen |

**Why this scheme.**
- *Reliability objectives keep the threshold form where the literature keeps it* —
  fraction-of-units frequency is the citable satisficing-in-search operator (the only
  one used in search in the WaterPaths lineage); magnitude/tail objectives use
  mean/percentile forms, so no analyst-chosen satisficing level θ_i exists for them.
  The only thresholds are the **Decree-anchored annual failure criteria** (§0
  goalposts), screened for saturation per design composition
  (`ensemble_objective_sensitivity_experiment.md`).
- *The long-record design needs no special case:* its records are scored as
  consecutive annual units with inherited state — exactly the treatment of Quinn et
  al. (2018), who slice one continuous 1000-yr record into 1-yr units "so that the
  distribution of initial conditions … is representative." Every design therefore has
  the **identical unit denominator NL** (short: N × (L−1) metric-bearing unit-years;
  long: N′ × (L′−1); warm-up years excluded).
- *Granularity/ε:* frequency objectives have granularity 1/NL (≈10⁻³ at NL ≈ 1000+);
  mean/percentile objectives are continuous with **ε in native metric units**.
- *Precedent floor:* percentile operators are precedented only over ≳50–1000 units
  (Quinn's WP1 used 1000); NL must comfortably exceed this — verified by the
  sensitivity experiment at the campaign NL.

**Caveats carried explicitly.** Unit-years within a realization are dependent
(multi-year droughts appear as consecutive failure-years — this is how the
WaterPaths-lineage frequency objectives express persistence); effective sample size
is below NL and differs by design (disclosed with the ESS/clustered-SE machinery of
`scenario_design_methods.md` §3.2). An annual window cannot hold a whole multi-year
drought as a single unit; event-scale severity enters through the hazard axes and the
re-evaluation metrics, not the search objectives.

**Design mapping.** All ensemble designs (fixed short, long-record, resampled,
input-stratified, hazard-filling, support points) use this same two-layer scheme —
one scheme fits all naturally, even though commensurability is no longer *required*.
For the resampled design the scheme applies within each per-evaluation redraw. The
**historic design** keeps the §1 temporal metrics on its single continuous trace
(R = 1; prevailing-practice reference, Giuliani & Castelletti 2016). In McPhail
terms: stage (i) is T1-threshold (reliability) or T1-absolute (magnitude/tail);
stage (ii) is T3 = frequency/expectation for #1/3/5/6 and T2 = tail percentile for
#2/4/7.

Implementation: the annual-metric computation and unit operators live in
`src/objectives_ensemble.py` (annual-unit aggregation registry); the per-design
sample's probability distortion relative to the master (input-stratified,
hazard-filling) is deliberately not corrected — cross-design comparison rests
entirely on the common re-evaluation (§3).

---

## 3. Re-evaluation and cross-design comparison

This is the load-bearing methodological point. The six designs differ **only** in
the search ensemble and its across-realization stage (§2). They are compared
**once**, by re-evaluating *every* resulting Pareto-approximate set on **one
common held-out test ensemble with one fixed aggregation, identical across all
six designs**. The test ensemble is deliberately broad and treated as deeply
uncertain; **its empirical composition supplies the scenario probabilities
implicitly, so no explicit scenario weighting is used anywhere.** Only at
re-evaluation are differences attributable to scenario design rather than to a
moving measuring stick (McPhail et al. 2020: composition moves robustness
*values* more than *rankings*).

The re-evaluation reports, per objective, two complementary views:

1. **Robustness (satisficing fraction)** over the test ensemble — the §2 metric,
   now on the common ensemble. Primary robustness measure (Herman et al. 2015).
2. **Regret.** For policy *p* and test scenario *s*, define
   `regret_{p,s} = f*(s) − f_{p,s}` for a maximize objective (and
   `f_{p,s} − f*(s)` for a minimize objective), where `f*(s)` is the best
   performance achieved in scenario *s* across all re-evaluated policies
   (best-in-scenario). Aggregate per policy by mean (and report max) regret
   across the test ensemble. This is McPhail T1 = regret. **Goal of the study:
   identify scenario designs whose policies are simultaneously high-robustness
   and low-regret.** Caveat to honor: regret-from-best converges more slowly
   than satisficing and needs a larger test ensemble for stable rankings
   (Bonham et al. 2024).

Robustness and regret are both reported, and ranking stability across these
metrics is summarized by Kendall's τ_b (Herman et al. 2015; McPhail et al. 2018).
The search-vs-re-evaluation overfitting gap (Brodeur et al. 2020) is *not* used
as a comparison measure — it is uncommon in the MOEA literature, and the study
relies on robustness and regret instead. The supplement reports a coverage →
re-evaluated-robustness association (does a design's hazard-space coverage predict
its policies' test-set satisficing?) as a mechanism analysis
(`scenario_design_methods.md` §6b); the main text claims only what robustness and
regret show.
The codebase separates `SEARCH_ENSEMBLE_SPEC` (per design) from the common test
ensemble, with a selection-bias guard (Bonham et al. 2024) warning if they
coincide.

---

## 4. Citation table — aggregation choice → source

| Aggregation / design choice | Where used | Citation(s) |
|---|---|---|
| Reliability as weekly satisficing frequency | obj. 1, 3, 5, 8 | Hashimoto et al. 1982; Herman et al. 2015; Kasprzyk et al. 2013 |
| CVaR₉₀ in place of worst-case deficit | obj. 2, 4 | Quinn et al. 2017; Fairbrother et al. 2022; Löhndorf 2016; Rockafellar & Uryasev 2000 |
| Low percentile in place of single-day minimum | obj. 7 | Quinn et al. 2017 |
| Count-over-threshold (flood days, minor stage) | obj. 6 | Quinn et al. 2017 |
| Trenton flow replacing salinity (physical redundancy) | obj. 5 | Trindade et al. 2017; Hadjimichael et al. 2020 |
| Failure-year frequency across pooled units (search reliability) | §2 #1/3/5/8 | Zeff et al. 2014; Trindade et al. 2017; Gold et al. 2023 |
| Worst-1st-percentile unit-year (search tail objectives) | §2 #2/4/7 | Quinn et al. 2017 (WP1), 2018; Zeff/Trindade/Gold worst-1% cost |
| Consecutive annual units with inherited state (long records) | §2 | Quinn et al. 2018 |
| Two-layer time-aggregation / noise-filtering vocabulary | §2 | Hamilton et al. 2022 |
| Expectation can mask floods (P99 variant registered) | §2 #6 | Quinn et al. 2017 |
| Multivariate satisficing per stakeholder (re-evaluation) | §3 | Herman et al. 2015; McPhail et al. 2018 |
| Satisficing converges fastest (re-evaluation ensemble sizing) | §3 | Bonham et al. 2024 |
| T1/T2/T3 decomposition of every aggregation | §0–§2 | McPhail et al. 2018 |
| Resampling reduces overfitting; per-eval noise | design 4 | Trindade et al. 2017; Brodeur et al. 2020 |
| Composition moves values more than rankings; hold re-eval metric fixed | §3 | McPhail et al. 2020 |
| Regret (best-in-scenario) as a comparison metric | §3 | McPhail et al. 2018; Herman et al. 2015 |
| Epsilon-dominance, dimensionality, diagnostics | §1 | Reed et al. 2013; Hadka & Reed 2013 |

---

## 5. Open items

1. Finalize the optional 8th objective (`nj_delivery_reliability_weekly`) after
   the redundancy screen (`objective_sensitivity_experiment.md`).
2. From the two-arm ensemble sensitivity experiment
   (`ensemble_objective_sensitivity_experiment.md`): (a) confirm the Decree-anchored
   annual failure criteria (#1/3/5/8) do not saturate under either a probabilistic
   or a hazard-filled composition (adjust the failure definition — e.g., ≥k failing
   weeks — if they do); (b) pick the flood-days unit operator (mean vs P99, #6);
   (c) set native-unit epsilons for the mean/percentile objectives and confirm P99
   stability at the campaign NL; (d) validate the annual-unit choice against
   realization-level rankings.
3. The salt-front (`salt_front_intrusion_max_rm`) and Lordville thermal metrics
   remain registered diagnostics; both are out of the active search set.
