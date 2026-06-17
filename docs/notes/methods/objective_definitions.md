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

## 2. Per-scenario-design aggregation (single family: satisficing-fraction)

Every objective in a multi-realization design reduces a (timesteps ×
realizations) matrix in **two stages**: (i) apply the §1 temporal metric to each
realization *r* → a per-realization scalar `m_r`; (ii) collapse `{m_r}` across
realizations. The across-realization stage uses **one operator family for every
design — the satisficing fraction**:

```
obj_i = (1 / R) · Σ_r  1[ m_{r,i} meets θ_i ]
```

with per-objective satisficing level `θ_i` (`ge` for maximize-base metrics, `le`
for minimize-base), and non-finite `m_{r,i}` counting as **unsatisfied**. All
resulting objectives are maximize (higher satisficing fraction is better). In
McPhail terms the across-realization stage is T1 = threshold, T2 =
domain-satisficing over all realizations, T3 = expected value (a frequency).
Note `θ_i` (analyst-chosen acceptable levels, set in
`objective_sensitivity_experiment.md`) are distinct from the Decree thresholds
inside the temporal metrics.

A single across-realization family (rather than tuning the operator per design)
gives the cleanest commensurability story and isolates the scenario-design
effect from any operator effect.

| Design | Realization axis | Across-realization stage | Notes |
|--------|------------------|--------------------------|-------|
| 1. Historic record | none (R=1) | **none** — objective = the raw temporal `m_1` | Satisficing degenerates at R=1, so historic uses the continuous temporal metric directly. A prevailing-practice reference, not a budget-matched comparison; its objectives are in different units (continuous) than the ensemble designs' (fractions), which is fine because comparison happens only at re-evaluation (§3). |
| 2. Fixed probabilistic (N short) | fixed draw of N | satisficing fraction over N | Intended sweet spot — satisficing converges with the fewest scenarios (Bonham et al. 2024). |
| 3. Fixed probabilistic, long records | few long records | satisficing fraction over the few records | Coarse denominator (3 records → {0,⅓,⅔,1}); flagged explicitly. Within-record extremes are already carried by the long temporal window. |
| 4. Resampled (redrawn every FE) | redrawn each evaluation | satisficing fraction within that evaluation's K-draw | No fixed ensemble → a Monte-Carlo estimator that varies between evaluations of the same policy (Trindade et al. 2017; Brodeur et al. 2020). K is sized so this noise is below each objective's epsilon. Comparisons involving this design rely entirely on re-evaluation. |
| 5. Input-stratified (LHS params) | fixed structured sample | satisficing fraction (uniform average over the sample) | The sample's probability distortion relative to the master ensemble is immaterial to the search objective and is not corrected; cross-design comparison rests entirely on the common re-evaluation (§3). |
| 6. Hazard-filling (space-filling) | fixed sample | satisficing fraction (uniform average over the sample) | Same as design 5: search aggregation is a plain uniform average over the designed sample; no probability correction. |

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
relies on robustness and regret instead.
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
| Multivariate satisficing per stakeholder | §2 | Herman et al. 2015; McPhail et al. 2018 |
| Single across-realization family; satisficing converges fastest | §2 | Bonham et al. 2024 |
| T1/T2/T3 decomposition of every aggregation | §0–§2 | McPhail et al. 2018 |
| Resampling reduces overfitting; per-eval noise | design 4 | Trindade et al. 2017; Brodeur et al. 2020 |
| Composition moves values more than rankings; hold re-eval metric fixed | §3 | McPhail et al. 2020 |
| Regret (best-in-scenario) as a comparison metric | §3 | McPhail et al. 2018; Herman et al. 2015 |
| Epsilon-dominance, dimensionality, diagnostics | §1 | Reed et al. 2013; Hadka & Reed 2013 |

---

## 5. Open items

1. Finalize the optional 8th objective (`nj_delivery_reliability_weekly`) after
   the redundancy screen (`objective_sensitivity_experiment.md`).
2. Fix the per-objective satisficing levels `θ_i` and epsilons from the
   sensitivity-experiment results.
3. The salt-front (`salt_front_intrusion_max_rm`) and Lordville thermal metrics
   remain registered diagnostics; both are out of the active search set.
