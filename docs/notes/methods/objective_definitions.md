# Objective Definitions for the Scenario-Design MOEA Study

*Last updated: 2026-08-05. Authoritative record of the objective formulations
used in the MOEA optimization and of the held-out re-evaluation metric set;
supersedes inline docstrings where they disagree. Terminology per
`docs/notes/terminology.md`; citations resolve to the Zotero collection
"Paper 3 NYC Reoptimization" (`ISYGLK35`) and the notes under
`docs/notes/literature/`. Supporting diagnostics: the epsilon-calibration
experiment (`epsilon_calibration_experiment.md`) and the framing-convention
diagnostics (`framing_convention_diagnostics.md`).*

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
change. The search-time annual-unit layer and the re-evaluation satisficing
layer are both in `src/objectives_ensemble.py`; the offline re-evaluation
scoring is in `src/robustness.py`.

---

## 0. Conventions

- Metrics are computed on the **metric window** of each scenario: the daily
  series from six calendar months after its start (`METRIC_EXCLUSION_MONTHS = 6`),
  cut by date, never by a fixed day count. Six months is the SSI-6 accumulation
  requirement, so the hazard-selection metrics exclude the same interval and
  selection and evaluation score the identical window.
- `resample("W")` = weekly resampling: the delivery metrics resample by **sum**
  (weekly volumes of delivery and entitlement); the flow metrics resample by
  **mean** (the weekly-accounting basis of the Decree).
- **CVaR₉₀(x)** = Conditional Value-at-Risk at the 90% level = the mean of the
  worst (largest-deficit) 10% of weekly values. Coherent and far less variable
  across realizations than the single maximum (Rockafellar & Uryasev 2000;
  applied in Fairbrother et al. 2022; Löhndorf 2016).
- Goalposts are **static**: the 1954-Decree quantities — NYC 800 MGD, Montague
  1131.05 MGD (= 1750 cfs), NJ diversion 100 MGD baseline — plus the **Trenton
  equivalent-flow objective** 1938.95 MGD (= 3000 cfs), an FFMP / Good-Faith
  target rather than a Decree quantity. Never the time-varying live FFMP
  `mrf_target` (scoring against the live target would let a policy "succeed" by
  triggering drought step-downs that lower its own goalpost).
- **NYC/NJ delivery is a running-*average* right, not a daily cap.** pywr-drb's
  `FfmpNyc/NjRunningAvgParameter` let daily diversion exceed the flat baseline by
  drawing down banked allowance, so daily demand is **not** clipped at the static
  right. Each day's target is the realizable **entitlement**
  `E_t = min(demand_t, A_t)`, where `A_t` is the running-average allowance bank
  (`_delivery_entitlement` / `_running_avg_budget`): it starts at the static cap,
  accrues `cap − delivery` daily (floored at 0), and resets (NYC annually on
  Jun 1 following the model's May-31 reset; NJ monthly). The bank is accrued at
  the **static** cap — never the policy's drought-scaled allowance — so a demand
  spike within the banked right is honored, demand beyond it is not owed, and a
  policy cannot lower its own goalpost. The demand generator already holds the
  monthly-average demand ≤ the cap, so `A_t` almost always has headroom and
  `E_t ≈ demand_t` except at rare spike-vs-reset collisions.
- The **McPhail et al. (2018) T1/T2/T3 decomposition** classifies every
  aggregation: **T1** performance-value transform (absolute / regret /
  threshold-satisficing), **T2** scenario subset (worst-case / all / tail
  percentile / domain-satisficing), **T3** aggregation function (mean / variance /
  higher moments / worst-case).

---

## 1. The objective set (single-realization / historic temporal metrics)

These metrics are the per-realization quantities scored at re-evaluation (§3).
During search, every design — including historic — is scored through the §2
annual-unit scheme; the historic trace enters it as N = 1 over its 76
water-year units. The active set is **8 objectives**; NJ delivery carries
independent information (redundancy screen: max |ρ_S| = 0.38 against any
objective, ≤ 0.08 against Trenton). All objectives use stable
tail/percentile/count forms rather than worst-case extremes (Quinn et al.
2017; Bonham et al. 2024); Trenton flow serves as the salinity-repulsion
goalpost (the Trenton target repels salt intrusion, and the salt-front LSTM —
unreliable in extreme drought — stays a registered diagnostic).

| # | Name (registry) | Source | Temporal aggregation | Dir | Units | ε |
|---|-----------------|--------|----------------------|-----|-------|---|
| 1 | `nyc_delivery_reliability_weekly` | `delivery_nyc`, `demand_nyc` (right 800) | frac of weeks `Σ_w delivery ≥ 0.99·Σ_w E` (entitlement `E_t = min(demand,A_t)`) | MAX | frac | 0.07 |
| 2 | `nyc_delivery_deficit_cvar90_pct` | same | CVaR₉₀ of weekly deficit % `= 100·max(0, mean_w(E) − mean_w(delivery))/800` | MIN | % | 1.5 |
| 3 | `montague_flow_reliability_weekly` | `major_flow.delMontague` | frac of weeks `mean_w(flow) ≥ 1131.05` | MAX | frac | 0.02 |
| 4 | `montague_flow_deficit_cvar90_pct` | `delMontague` | CVaR₉₀ of `100·max(0, 1131.05 − mean_w(flow))/1131.05` | MIN | % | 1.5 |
| 5 | `trenton_flow_reliability_weekly` | `major_flow.delTrenton` | frac of weeks `mean_w(flow) ≥ 1938.95` | MAX | frac | 0.0003 |
| 6 | `downstream_flood_exceedance_minor` | `flood_stage` (Hale Eddy, Fishs Eddy, Bridgeville) | mean annual `Σ_days max_gauges (stage − minor)⁺` — ft·days above NWS **minor** flood stage at the worst-affected gauge | MIN | ft·days/yr | 0.01 |
| 7 | `nyc_storage_p5_pct` | `res_storage[NYC]` | 5th percentile of daily `100·Σ_res storage / 270,837` | MAX | % | 1.5 |
| 8 | `nj_delivery_reliability_weekly` | `delivery_nj`, `demand_nj` (right 100) | frac of weeks `Σ_w delivery_nj ≥ 0.99·Σ_w E_nj` (entitlement `E_nj = min(demand_nj,A_t)`, monthly reset) | MAX | frac | 0.007 |

Epsilons are the calibrated values in `src/objectives.py`: ε ≈ IQR/10 of each
objective's spread across N = 500 random-DV policies on the historic reference
trace (Reed et al. 2013), rounded to clean steps. The §2 annual-unit registry
(`src/objectives_ensemble.py`) carries its **own, separate** campaign epsilons
from the epsilon-calibration experiment
(`epsilon_calibration_experiment.md`).

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
- *flood exceedance above minor stage (6)* — magnitude-weighted exceedance
  (`flood_objective_diagnostics.md`): the day count is degenerate across
  policies (9 distinct values over 25 feasible policies on the historic trace)
  while the exceedance integral resolves fully and responds strictly
  monotonically to
  the flood-release DVs; exceedance also tracks observed annual flood magnitude
  better (Pearson 0.91 vs 0.83) without the expectation-of-damage trap — the
  integrand is physical exceedance, not monetized damage (Quinn et al. 2017).
  The max-across-gauges basis avoids triple-counting basin-wide events.
  Normalized per metric-window year so the metric is invariant to record
  length. The NWS **minor** (flood-onset) stage marks actual flooding, a more
  meaningful goalpost than the FFMP cautionary cutoff — which is also the
  control rule's own switching boundary; the day-count (`minor`/`major`/
  `action`) variants stay registered as diagnostics.
- *Storage p5 (7)* — a low percentile is a stable vulnerability proxy; the
  single-day minimum is dominated by one drought event (Quinn et al. 2017).
- *Trenton vs salinity (5)* and *NJ delivery (8)* — give New Jersey, a co-equal
  Decree party, direct representation so the search can discover NYC↔NJ
  robustness conflicts (Trindade et al. 2017; Hadjimichael et al. 2020).

**Diagnostics (registered, not active):** worst-case variants
(`*_deficit_max_pct`, `nyc_storage_min_pct`), `downstream_flood_days_minor` /
`_major` / `_action`, `trenton_flow_deficit_cvar90_pct`, the salt-front metric
(`salt_front_intrusion_max_rm`), and the deferred Lordville thermal metric. They
are available for swapping or re-evaluation reporting without code changes.

Dimensionality: 8 keeps the epsilon-dominance archive where hypervolume stays
estimable (Reed et al. 2013); epsilons are in each metric's native units.

---

## 2. Per-scenario-design search aggregation — two-layer annual-unit scheme

Two principles govern the scheme: (i) **cross-design commensurability during search
is not required** — the held-out re-evaluation (§3) is the only comparison point —
and (ii) **every operator follows published search-time practice** (objective
aggregation is not a novelty focus of this study).

**Structure (Hamilton et al. 2022's two-layer vocabulary: within-record time
aggregation + across-record noise filtering).** Each realization is simulated
continuously; the first six months are outside the metric window and excluded; the
remainder is split into **water-year units**. Scenario windows are October-aligned, so
the remainder begins April 1 of the first water year and the first WHOLE water-year
unit is WY2. Stage (i): compute each objective's **annual metric**
on every (realization × year) unit. Stage (ii): aggregate across the pooled **NL
unit-years** with the objective's **unit operator**:

| # | Objective (registry) | Annual metric (per unit-year) | Unit operator (across pooled unit-years) | Dir | Anchor |
|---|---|---|---|---|---|
| 1 | `nyc_delivery_reliability_annual` | failure-year indicator: ≥ k = 3 failing weeks (`Σ_w delivery < 0.99·Σ_w E`; entitlement `E_t = min(demand,A_t)`) | **frequency of non-failure years** | MAX | Zeff et al. 2014 Eq. 2; Trindade et al. 2017 Eq. 16; Gold et al. 2023 |
| 2 | `nyc_delivery_deficit_p99_pct` | CVaR₉₀ of weekly deficit % within the year | **worst-1st-percentile unit-year** (P99) | MIN | Quinn et al. 2017 (WP1), 2018; Trindade/Gold worst-1%-cost |
| 3 | `montague_flow_reliability_annual` | failure-year indicator: ≥ k = 3 failing weeks (`mean_w(flow) < 1131.05`) | frequency of non-failure years | MAX | as #1 |
| 4 | `montague_flow_deficit_p99_pct` | CVaR₉₀ of weekly Montague deficit % within the year | worst-1st-percentile unit-year | MIN | as #2 |
| 5 | `trenton_flow_reliability_annual` | failure-year indicator: ≥ k = 1 failing week vs 1938.95 MGD | frequency of non-failure years | MAX | as #1 |
| 6 | `downstream_flood_exceedance_annual` | `Σ_days max_gauges (stage − minor)⁺` in the year (ft·days; `flood_objective_diagnostics.md`) | **mean across unit-years** (expected annual flood exceedance; the P99 unit operator is tie-degenerate at NL = 900 and 12–30× noisier — registered diagnostic; Quinn et al. 2017 expectation-masking caution answered by the operator screen) | MIN | Trindade expected-cost form; Quinn 2017 caution |
| 7 | `nyc_storage_min_p01_pct` | annual minimum of daily aggregate NYC storage % | **1st-percentile unit-year** | MAX | WP1 pattern (Quinn 2017/2018); Hamilton 2022 Q-of-max |
| 8 | `nj_delivery_reliability_annual` | failure-year indicator (k = 1) vs NJ delivery criterion | frequency of non-failure years | MAX | as #1 |

**Why this scheme.**
- *Reliability objectives keep the threshold form where the literature keeps it* —
  fraction-of-units frequency is the citable satisficing-in-search operator (the only
  one used in search in the WaterPaths lineage); magnitude/tail objectives use
  mean/percentile forms, so no analyst-chosen satisficing level θ_i exists for them.
  Each annual failure criterion combines a **static goalpost** (§0) with a
  **failure-week count k** (k = 3 for NYC delivery and Montague flow; k = 1 for
  Trenton and NJ; `_DEFAULT_FAILURE_K`). The goalposts are anchored; k is a
  convention, screened for saturation per design composition
  (`framing_convention_diagnostics.md` §1: no shipped k saturates in either
  composition, rankings stable to k ± 1; Trenton k = 1 binding).
- *The long-record design needs no special case:* its records are scored as
  consecutive annual units with inherited state — exactly the treatment of Quinn et
  al. (2018), who slice one continuous 1000-yr record into 1-yr units "so that the
  distribution of initial conditions … is representative." Every design therefore has
  the **identical unit denominator NL** (short: N × (L−1) metric-bearing unit-years;
  long: N′ × (L′−1); partial units at either end excluded).
- *Granularity/ε:* frequency objectives have granularity 1/NL (≈10⁻³ at NL ≈ 1000+);
  mean/percentile objectives are continuous with **ε in native metric units**.
- *Precedent floor:* percentile operators are precedented only over ≳50–1000 units
  (Quinn's WP1 used 1000); NL must comfortably exceed this — estimator noise at
  the campaign NL is measured by the epsilon-calibration bootstrap and the
  framing-convention operator screen.

**Caveats carried explicitly.** Unit-years within a realization are dependent
(multi-year droughts appear as consecutive failure-years — this is how the
WaterPaths-lineage frequency objectives express persistence); effective sample size
is below NL and differs by design; this is disclosed rather than corrected. An annual window cannot hold a whole multi-year
drought as a single unit; event-scale severity enters through the hazard axes
(hazard selection and scenario discovery), not through the objective statistics
— which are the same annual-unit quantities in search and re-evaluation.

**Design mapping.** All three designs use this same two-layer scheme. The
**historic design** enters it as N = 1 over the consecutive water-year units of
its single continuous trace (76 metric-bearing units; prevailing-practice
reference, Giuliani & Castelletti 2016). In McPhail
terms: stage (i) is T1-threshold (reliability) or T1-absolute (magnitude/tail);
stage (ii) is T3 = frequency/expectation for #1/3/5/6 and T2 = tail percentile for
#2/4/7.

Implementation: the annual-metric computation and unit operators live in
`src/objectives_ensemble.py` (annual-unit aggregation registry); the
`hazard_filling` sample's deliberate probability distortion relative to the
generator is not corrected — cross-design comparison rests entirely on the common
re-evaluation (§3).

---

## 3. Re-evaluation: the held-out metric set

The three designs differ **only** in the search ensemble and its across-realization
stage (§2). They are compared **once**, by re-evaluating *every* resulting
Pareto-approximate set on **one common held-out test ensemble E_test with one
fixed metric set, identical across all designs**. Only at re-evaluation are
differences attributable to scenario design rather than to a moving measuring
stick (McPhail et al. 2020: composition moves robustness *values* more than
*rankings*).

**E_test is a designed exploration, not a probability sample — and no robustness
number here is an expectation.** E_test is a Latin hypercube over the *full* range
of the deeply-uncertain forcing factors, with many realizations per LHS point
(`scenario_design_methods.md` §5). Under deep uncertainty there is no probability
measure over Θ — that is what "deep" means — so a satisficing fraction over E_test
is a **coverage-weighted count over a designed exploration of the DU space**, and
is reported as such, never as an estimate of 𝔼[·]. **What makes the cross-design
comparison commensurable is that E_test is *identical across all designs*, not
that it is probability-faithful.** (This also means the held-out re-evaluation
does not "restore the true measure" for a hazard-filling design; it removes the
*evaluation* bias of scoring each design on its own ensemble. Selection bias is
not corrected — see §5.)

**No explicit scenario weighting is used anywhere**, in search or in
re-evaluation.

**The per-SOW objective matrix is persisted.** Re-evaluation persists the
**(solution × SOW × objective)** matrix in natural units
(`reeval_raw.parquet` + a self-describing `reeval_raw_meta.json`): each E_test
state of the world's R = 25 realizations contribute their stage-(i) unit-years
to one pooled sample, and the objective's §2 unit operator collapses that pool
to the state's objective value. Every robustness and regret metric is scored
*offline* from that matrix (`src/robustness.py`), so a new metric or a changed
threshold never requires re-simulating. Precision is governed by N_θ, not by
N_test.

**The re-evaluation metric IS the §2 search metric, recomputed per state of the
world.** Search and re-evaluation share one code path for the statistic — the
stage-(i) annual metrics (`annual_units()`) and the stage-(ii) unit operators —
and differ only in the pool the operator collapses:

| | Search (`_evaluate_ensemble_batched`) | Re-evaluation (`evaluate_annual_units` + `sow_objective_matrix`) |
|---|---|---|
| per realization | `annual_units()` → vector of L−1 **annual** metrics | the same `annual_units()` |
| pooling | ALL realizations' unit-years → §2 unit operator → one search objective | EACH θ's realizations' unit-years → the same unit operator → J_i(x, θ) |

This is the standard construction of the robustness literature: the performance
measure inside the robustness calculation is the optimization objective
re-evaluated per state of the world (Herman et al. 2015 Eqs. 3–7, whose
`F(x)_{i,j}` is objective *i* in SOW *j*; Trindade et al. 2017, whose
satisficing is computed on the objective values "as calculated with
Eqs. (16)–(20)"; Herman et al. 2014; Quinn et al. 2018; Gold et al. 2023;
McPhail et al. 2018, whose `f(x_i, S)` is titled "objectives and performance
metrics"). What differs between search and evaluation is the ensemble and the
outer aggregation across states — never the definition of the statistic. One
metric currency also removes the annual-unit statistics' record-length
sensitivity from the comparison: a 10-yr search window and the 50-yr test
window are pooled through the same per-unit-year operators.

Each SOW's pool (25 × 49 ≈ 1,225 unit-years) keeps the tail operators
well-defined per state: the per-SOW P01 is ≈ the 12th-worst unit-year, exactly
the "worst first-percentile year" construction Quinn et al. (2018) evaluate per
SOW. Commensurability across designs is unchanged — E_test and the metric set
are identical for every design.

`objectives_summary.csv` holds the mean over SOWs of the per-SOW objective
values (`sowmean__*` columns) — a risk-neutral summary derived from the same
persisted matrix, never a second simulation.

### 3.1 Primary metric: multivariate satisficing (Starr's domain criterion)

**The fraction of E_test STATES OF THE WORLD in which the policy's per-SOW
objective vector `J_i(x,θ)` meets ALL criteria jointly** — Starr's (1962)
domain criterion across the N_θ = 1,000 states, the standard measure of the
Herman (2014/2015) / Trindade (2017, 2019) / Gold (2023) lineage, and Herman
et al. (2015)'s own recommendation ("a carefully elicited multivariate
satisficing measure of robustness allows stakeholders to achieve their
problem-specific performance requirements").

In McPhail terms: T1 = satisfaction-of-constraints, T2 = all states, T3 =
frequency, applied to the search objective recomputed per state. Pooling each
θ's realizations' unit-years through the unit operator IS the within-state
collapse — natural variability inside a state enters through the statistic's
own definition, exactly as it does during search, so there is no separate
within-SOW risk-attitude knob to choose or record.

**Why the SOW is the unit.** The construction matches the Triangle lineage,
whose objectives are likewise ensemble statistics per state (reliability as a
probability over realizations, cost as a quantile across them; Herman et al.
2014; Trindade et al. 2017; Gold et al. 2023; stated as a convention by
Bartholomew & Kwakkel 2020). Three reasons make it right here too:

- **Measure separation.** E_test carries no probability measure over the forcing
  space (Lamontagne et al. 2018). A pooled-realization fraction would integrate
  a DESIGNED LHS box and a FITTED stochastic generator in one number; counting
  states applies a coverage-weighted count to the design and leaves natural
  variability inside the per-state statistic.
- **Precision.** Realizations sharing a θ are not independent. The reported
  ±1.6 pp is `0.5/√1000`, a SOW-unit standard error. Bonham et al. (2024)'s
  50–300 convergence result is measured on a FLAT ensemble and so bounds N_θ,
  not R.
- **One unit everywhere.** The incumbent-regret family (§3.2b), scenario
  discovery's failure labels, and the attainability screen all consume the same
  per-SOW `J_i(x,θ)`; there is no second unit for a reader to track.

### 3.2 Secondary metrics (all reported)

| Metric | Definition | McPhail T1 / T2 / T3 | Role |
|---|---|---|---|
| **univariate satisficing** | fraction of SOWs clearing one objective's criterion (`sat_uni_sow__`) | satisfaction / all / frequency | per-objective decomposition of the primary — same unit, conjunction dropped |
| **Laplace (mean)** | mean of `J_i(x,θ)` across SOWs (`laplace__`) | identity / all / mean | risk-neutral reference |
| **maximin** | worst SOW's `J_i(x,θ)` (`maximin__`) | identity / worst-case / worst-case | risk-averse reference (Wald) |

Ranking agreement is summarized with **Kendall's τ_b computed across the *design*
rankings** these metrics induce — i.e. do the metrics rank the scenario
designs the same way? (Herman et al. 2015; McPhail et al. 2018, 2020.)

### 3.2b Incumbent-relative regret (co-primary; answers RQ2)

RQ2 asks whether policies improve some outcomes *without degrading others below
current performance*: a mean improvement can be comfortably positive while the
policy is badly worse than the status quo in a third of futures. The regret
family reports the signed incumbent advantage through its one-sided halves
(`regret_*__` and `gain_mean__`) plus unit-free frequencies, all on the same
per-SOW objective values as the primary metric.

**Reference.** The incumbent 2017 FFMP policy, evaluated **in the same SOW**.
McPhail et al. (2018) §3.1 license exactly this T1 — "Alternative metrics that are
based on the relative performance of decision alternatives use some type of
baseline performance *for a given scenario* instead of the performance of the best
decision alternative" — but never name, tabulate, or test it, so this study makes
an admitted variant explicit rather than importing one. It is also Savage regret
on the action set the Decree parties actually face, `{retain the FFMP, adopt the
candidate}`: a unanimity-bound renegotiation, not a free choice over an archive.

**Unit and construction** (`src/robustness.py`; the per-SOW objective values):

- `D_i(x,θ) = σ_i · [ J_i(x,θ) − J_i(b,θ) ]` — signed, oriented so positive =
  better than current operations for every objective (`incumbent_advantage`).
- Magnitudes, in each objective's **own natural units**, never combined across
  objectives: `regret_mean__`, `regret_q90__` (the Herman R1/R2 tail statistic),
  `regret_cond__` (mean shortfall *given* a shortfall; NaN when never worse), and
  `gain_mean__` as the mandatory companion.
- Frequencies, **unit-free**, which carry the cross-objective and cross-design
  scalar role: `harm_freq__{obj}`, `party_harm_freq__{party}` (a disjunction over
  a Decree party's objectives — under unanimity a loss is not compensable, so it
  is never a sum), `no_harm_freq` (weak Pareto improvement on the incumbent),
  `no_harm_freq_tau`, and `n_degraded_mean`.

**Restriction to the adverse subset is McPhail's T2, not ad-hoc clipping.** Their
"undesirable deviations" metric (Kwakkel et al. 2016b) decomposes as T1 = regret
from median / T2 = worst-half / T3 = sum; ours is that construction with the
reference changed from the policy's own median to the incumbent. A *mean of a
clipped quantity over all scenarios* does collapse toward zero when policies
mostly beat the incumbent — the objection that produced the unclipped signed
metric — but a statistic computed on the sign-selected subset does not.

**No max regret.** Bonham et al. (2024): regret families need 400+ scenarios and
never converge on extreme-of-extremes operators. McPhail et al. document the
tie-degeneracy directly. The 90th percentile over 1,000 SOWs is well resolved.

**Natural units, and no cross-objective magnitude scalar.** Dividing by the
incumbent's own per-state value is degenerate for this objective set: flood
exceedance is exactly 0 in a share of states and both deficit tails are 0 in wet
ones, so the cell is NaN'd and *silently dropped* — and the dropped cells are
the benign ones, biasing the estimator toward the adverse subset. Herman et al.
(2015) additionally show the normalized-deviation form selects poor-baseline
solutions as a "mathematical artifact". Natural units dissolve both rather than
patching them. Neither published scale is usable: Cohen et al. (2021) normalize
on the per-scenario span to a perfect-foresight optimum (one MOEA run per
scenario, out of budget), and Sunkara et al. (2023) rescale over the alternative
set, which is design-coupled. McPhail et al. give no normalization guidance at
all. An optional fixed scale (`incumbent_spread`, the incumbent's q90−q10 over
E_test) is implemented as a scoring-time sensitivity so a reviewer asking for a
dimensionless regret can be answered without re-simulating; it is never the
reported primary.

**The tolerance ladder.** `τ_i = k · u_i` with the unit `u_i =
max(ε_i, τ_i^floor)`: `ε_i` is the objective's ANNUAL-UNIT epsilon from the
campaign registry (`src/objectives_ensemble.py`) — the epsilon-calibration
experiment's just-noticeable difference in exactly the annual-unit metric space
the per-SOW cube lives in, so the regret tolerance and the search resolution
sit on ONE calibration — and `τ_i^floor` is the measured noise floor of that
objective's per-SOW estimator. Taking the max is not cosmetic: one `k` is
shared across eight objectives, and an epsilon *below* its own noise floor
makes every rung fire on Monte Carlo noise for that axis while being far
outside the noise on the others, so a single rung silently means two different
things. `k` is swept (`REGRET_TAU_GRID` in `compare_designs.py`) rather than
fixed, for the reason the satisficing criteria are swept: a single tolerance
could manufacture or hide the whole RQ2 answer (Quinn et al. 2020).

**Both `k` and the ladder shape are pre-registration quantities**, and the rules
that fix them — plus the noise floor, the discrimination band, the empirical
nulls behind the non-inferiority margin, and the assay-sensitivity control — are
specified in `regret_tolerance_diagnostics.md`. The rule that matters most: no
anchor may be read off the distribution of candidate-policy regret, because that
is the quantity under test.

**Why this is not redundant with satisficing.** The satisficing criteria are fixed
scalars anchored on the incumbent's *historic* attainment
(`robustness_threshold_diagnostics.md` §0b rule 1); the regret bar is the
incumbent's performance *in that SOW*, and moves with the forcing. Where a fixed
criterion drives the domain criterion to 0 (or to 1) for every policy, satisficing
ties everything — Bonham et al. (2024)'s saturation failure mode — and regret
still separates policies.

**Comparison rule.** Robustness and regret are read off the **same** policy, or
the claim "more robust without more regret" is about nothing. Reported as: the
draw-level endpoint policy's regret; the full `(sat_multivariate_sow,
no_harm_freq_tau)` cloud; the per-design non-dominated frontier in that plane; and
the per-objective natural-unit drill-down. This is the study's analogue of
Bartholomew & Kwakkel's (2020) price-of-robustness measurement, but against a
fixed external incumbent per SOW rather than by hypervolume against reference
scenarios, which would reintroduce the pooled-reference-set bias rejected in §4.3.
A **severity decomposition** over terciles of the dominant forcing factor `m`
(|ρ_S| = 0.91–0.98 on all eight objectives) tests whether any price is paid in the
benign futures — an insurance premium, which is a finding, not a failure.

**Degeneracy guard.** A policy scores zero regret by *being* the incumbent, which
is reachable because the FFMP baseline lies inside the searched DV space. Regret is
therefore never reported without `gain_mean` beside it.

### 3.3 Metrics deliberately excluded

Regret has four possible references. Exactly one is computed here, and naming all
four is the cleanest way to state what is excluded and why — no published paper
lays them side by side.

| Reference | Question it answers | Status here |
|---|---|---|
| The incumbent policy, per SOW | "How much worse off than under current rules?" | **computed** (§3.2b) |
| The best policy in the evaluated set, per SOW (Savage; Herman R2) | "Did we pick the wrong policy from the archive?" | excluded |
| The same policy in a baseline SOW (Herman R1; Kasprzyk et al. 2013) | "How wrong were our assumptions about the future?" | not computed |
| A perfect-foresight optimum, per scenario (Cohen et al. 2021) | "What did imperfect information cost?" | not computed |

**Best-in-set regret** is set-relative and design-coupled (dropping one design
changes every other design's score), and Bonham et al. (2024) show it converges
far more slowly than satisficing — never, on max-over-time objectives like our P99
deficit operators. Herman et al. (2015) found their two regret metrics "tended to
agree with each other", so it is also likely to be largely redundant with the
incumbent-referenced metric we do compute. **Herman R1 / Kasprzyk et al. (2013)
percent deviation** is a *within-policy, across-SOW* sensitivity measure — their
Eq. (9) reference is the same solution's value in a baseline *state of the world*,
not a status-quo policy — and it answers a different question from RQ2; note that
this note and the manuscript previously miscited it as the precedent for the
status-quo comparison, which it is not (the correct chain is McPhail et al. 2018
§3.1 for the reference, Herman et al. 2015 for the functional shape, Kwakkel et
al. 2016b for the adverse-subset construction). **Cohen et al. (2021) baseline
regret** would require one perfect-foresight MOEA run per scenario; no
perfect-foresight optimization is performed anywhere in this study, and Cohen is
cited as motivation only.

A **search-vs-test "overfitting gap"** is likewise not computed: Brodeur et al.
(2020) diagnose overfitting graphically and define no gap metric, and a
coverage-weighted in-sample term minus a measure-weighted out-of-sample term
measures the measure change, not overfitting. (`src/robustness.py`
deliberately contains no such helper; `tests/test_robustness.py` asserts its
absence.)

### 3.4 Attainability screen (free)

Flag the E_test states of the world in which **no policy from any design** meets
the criteria. This costs zero CPU — the (solution × SOW × objective) matrix
already exists — and it separates *"this design searched badly"* from *"this
test state is unwinnable for anyone."* Every design's satisficing fraction is
bounded above by the attainable fraction, so the screen also sets the ceiling
against which design differences should be read. Precedent: Shavazipour et al.
(2021) found 23% of their test scenarios unwinnable by any policy. Stated
honestly: this is an **empirical attainability bound over the evaluated policy
pool**, not a per-scenario oracle (an oracle would require the perfect-foresight
optimization rejected in §3.3).

The codebase separates `SEARCH_ENSEMBLE_SPEC` (per design) from the common test
ensemble, with a selection-bias guard (Bonham et al. 2024) that raises a hard
error if they coincide.

---

## 4. Threshold sweep, optional mechanism analysis, and the cross-design comparison rule

### 4.1 Satisficing criteria are conventions — so they are swept (main-text figure)

No one in this lineage derives a satisficing threshold. Zeff et al. (2014)
elicited them from the Research Triangle utilities; every later number descends
from those by convention and drifts without stated reason (worst-case cost 5% in
Trindade et al. 2017 → 10% in Gold et al. 2023; restriction frequency 20% → 10%;
Trindade relaxed reliability 99% → 98.5% because no solution met 99%). Exactly one
threshold in the lineage has an external anchor: Gold et al. (2023)'s peak
financial cost < 80% of annual volumetric revenue, from AWWA bond-covenant limits.

This study therefore (i) anchors each criterion on a **Delaware River Basin
Decree / FFMP goalpost** wherever one exists (§0) and (ii) **sweeps the rest**.
Quinn et al. (2020) makes the sweep mandatory rather than cosmetic:
robustness-rank agreement **across scenario designs degrades as the satisficing
criterion becomes more stringent** — "the more conservative one wants to be in
finding robust policies, the harder it is to choose this consistently across
experimental designs." The design effect is therefore largest at the conservative
end, and any single fixed threshold could **manufacture or hide the entire
result**.

**Main-text figure:** the cross-design comparison over a grid of thresholds, with
the question being whether the **design ranking** — not the robustness value — is
invariant. Threshold-margin diagnostic: the CDF of each objective across E_test
with the criterion drawn as a vertical dashed line (Gold et al. 2023, Fig. 5).
Sweeping thresholds for *design-ranking* stability has not been done in this
lineage; it is a contribution, not a robustness check. (Implementation: the
design-ranking stringency sweep in `scripts/main/compare_designs.py`, driven by
`NYCOPT_COMPARE_STRINGENCY` — an offline re-scoring of the persisted cube;
`threshold_spectrum` in `src/robustness.py` supplies the per-run univariate
spectrum.)

### 4.2 Scenario discovery in the DU factor space — optional supporting analysis

This is a supporting analysis, not the primary comparison and not a falsification
device. The primary comparison is the re-evaluated robustness of the resulting
solutions (§3). Where it is run, boosted trees (Gold et al. 2022/2023
hyperparameters) are fit to each design's E_test failure realizations — labelled
by the **conjunction** of the satisficing criteria, i.e. the §3.1 primary — **in
the DU factor (forcing-parameter) space of E_test**, after re-evaluation. No
scenario discovery is performed in hazard space.

Its role is to characterize *which deeply uncertain conditions* drive failure and
whether the two designs' policies fail under different regions of the forcing
envelope. It is reported as support for a difference found on the primary metric,
never as the basis of the comparison. Because the forcing factors are sampled
independently by LHS, the correlated-factor instability of factor-importance
rankings (Quinn et al. 2020) does not arise by construction.

### 4.3 Cross-design comparison rule: no pooled-reference-set hypervolume

**Designs are compared only by re-evaluation on the common E_test.** MOEA search
metrics (hypervolume, generational distance, ε-indicator) scored against a
*pooled* reference set are **not** a defensible primary cross-design measure; they
are demoted to the supplement, where the reference set is built **per design** and
the metrics are read as within-design convergence diagnostics only.

A pooled reference set is biased across designs three ways: a design
contributes points to the very frontier it is scored against (contribution
share is a merit diagnostic, not a yardstick — Reed et al. 2013); designs
return different solution counts (Bartholomew & Kwakkel 2020); and noisier
re-evaluated estimates contribute more spuriously nondominated points
(Shavazipour et al. 2021). Protocol precedent: Zatarain Salazar et al. (2017)
§5.3 build reference sets per search-ensemble level, rule cross-level MOEA
metrics incomparable, and compare levels only by re-evaluating on a common
independent verification ensemble.

---

## 5. Threats to validity, named up front

- **Degeneracy check — raw performance is co-reported with every robustness
  number.** A robustness metric can be stable, optimizable, and still perverse.
  Huang et al. (2025)'s undesirable-deviations metric is driven to zero by making
  performance *uniformly bad* (a water-supply deficit near 1.0 is "robust" because
  it is consistently terrible); Bonham et al. (2024)'s vulnerability-satisficing
  saturates so that everything ties. Every robustness number is therefore reported
  alongside the **raw performance distribution (median + spread)** of the same
  objectives on E_test (Gold et al. 2023, Fig. 5; Huang et al. 2025, Fig. 8). This
  is the check that stops a design from "winning" on its own metric while being
  catastrophically bad in reality.
- **The null result the study must beat (Eker & Kwakkel 2018).** Diversity-based
  scenario selection did **not** beat random selection: "selecting the scenarios
  based on policy relevance and diversity does not lead to significantly more
  favorable results … compared to an arbitrary set of scenarios." The
  `fixed_probabilistic` → `hazard_filling` contrast **is** that
  benchmark. Our differentiators: their diversity is in *outcome* space on a
  benchmark problem (the Lake Problem) with little scenario→outcome leverage, ours
  is in *hazard* space on a real system; and our comparison rests on replicated
  ensemble draws with the draw as the unit of analysis, versus their counting of
  solutions above a group median from a single draw.
- **Search-measure mismatch is a systematic penalty, not a wash (Giuliani &
  Castelletti 2016, Fig. 4b–f).** Policies designed under across-scenario
  aggregation Φ_j and scored under Φ_k are **dominated** by policies correctly
  designed under Φ_k — in all five panels. Hazard-filling searches under a
  deliberately distorted measure and is scored under the test measure, so Giuliani
  predicts it will be penalized. **That is the null this study must beat**, and it
  is stated as such rather than left for a reviewer to find.
- **Selection bias is not corrected — it is the quantity being measured
  (Bartholomew & Kwakkel 2020).** Their conclusions contain the reviewer's
  objection pre-written: a design that selects scenarios from a chosen region
  "intrinsically biases subsequent results towards solutions that do well in this
  region. But there is no a-priori reason to assume that these resulting solutions
  might not be vulnerable in a different way." The answer: the distortion is
  deliberate, stated, and coverage-motivated; the held-out re-evaluation corrects
  **evaluation** bias; **selection** bias is *not* corrected and is precisely what
  the experiment measures.

---

## 6. Citation table — aggregation choice → source

| Aggregation / design choice | Where used | Citation(s) |
|---|---|---|
| Reliability as weekly satisficing frequency | obj. 1, 3, 5, 8 | Hashimoto et al. 1982; Herman et al. 2015; Kasprzyk et al. 2013 |
| CVaR₉₀ in place of worst-case deficit | obj. 2, 4 | Quinn et al. 2017; Fairbrother et al. 2022; Löhndorf 2016; Rockafellar & Uryasev 2000 |
| Low percentile in place of single-day minimum | obj. 7 | Quinn et al. 2017 |
| Magnitude-weighted exceedance above flood stage (exceedance, minor stage) | obj. 6 | Quinn et al. 2017; `flood_objective_diagnostics.md` |
| Trenton flow replacing salinity (physical redundancy) | obj. 5 | Trindade et al. 2017; Hadjimichael et al. 2020 |
| ε ≈ IQR/10 calibration | §1 | Reed et al. 2013; Hadka & Reed 2013 |
| Failure-year frequency across pooled units (search reliability) | §2 #1/3/5/8 | Zeff et al. 2014; Trindade et al. 2017; Gold et al. 2023 |
| Worst-1st-percentile unit-year (search tail objectives) | §2 #2/4/7 | Quinn et al. 2017 (WP1), 2018; Zeff/Trindade/Gold worst-1% cost |
| Consecutive annual units with inherited state (long records) | §2 | Quinn et al. 2018 |
| Two-layer time-aggregation / noise-filtering vocabulary | §2 | Hamilton et al. 2022 |
| Expectation can mask floods (P99 variant registered) | §2 #6 | Quinn et al. 2017 |
| Multivariate satisficing / domain criterion (PRIMARY re-eval metric) | §3.1 | Starr 1962; Herman et al. 2014, 2015; Trindade et al. 2017; Gold et al. 2022, 2023 |
| T1/T2/T3 decomposition; Laplace / maximin secondaries | §0, §3.2 | McPhail et al. 2018; Giuliani & Castelletti 2016 |
| A baseline decision alternative is a licensed regret reference, per scenario | §3.2b | McPhail et al. 2018 §3.1 |
| Regret shape: per-objective deviation → 90th percentile over SOWs | §3.2b | Herman et al. 2015 (R1/R2, Eqs. 3–6) |
| Restriction to the adverse subset (T2 = worst-half) | §3.2b | Kwakkel et al. 2016b (undesirable deviations) |
| Savage regret on the realized action set {retain, adopt} | §3.2b | Savage 1951 |
| Weak dominance over the status quo as a decision condition | §3.2b | Cohen et al. 2021 (applied as a filter) |
| Ratio normalization fails when the reference approaches zero | §3.2b | Herman et al. 2015; Eker & Kwakkel 2018 |
| Compensatory aggregation hides party-level failure → disjunction | §3.2b | Sunkara et al. 2023 |
| Multi-objective robustness resists a single scalar | §3.2b | Watson & Kasprzyk 2017 |
| The price of robustness (search-phase robustness costs elsewhere) | §3.2b | Bartholomew & Kwakkel 2020; Bertsimas & Sim 2004 |
| Extreme training scenarios can skew multi-objective trade-offs | §3.2b, §5 | Huang et al. 2025 |
| Kendall's τ_b for ranking agreement across metrics | §3.2 | Herman et al. 2015; McPhail et al. 2018, 2020 |
| Satisficing converges fastest; regret-from-best does not | §3.3 | Bonham et al. 2024 |
| Perfect-foresight-per-scenario regret is unscalable (motivation only) | §3.3 | Cohen et al. 2021 |
| Overfitting diagnosed graphically, not by a gap metric | §3.3 | Brodeur et al. 2020 |
| Some test scenarios are unwinnable by any policy | §3.4 | Shavazipour et al. 2021 |
| Satisficing thresholds are elicited conventions | §4.1 | Zeff et al. 2014; Trindade et al. 2017; Gold et al. 2023 |
| Design-ranking agreement degrades with stringency (→ sweep) | §4.1 | Quinn et al. 2020 |
| Threshold-margin CDF diagnostic | §4.1 | Gold et al. 2023 (Fig. 5) |
| Boosted-tree scenario discovery (DU factor space) | §4.2 | Gold et al. 2022, 2023; Quinn et al. 2020 |
| Per-level reference sets; compare only on a common verification ensemble | §4.3 | Zatarain Salazar et al. 2017 |
| Contribution share is a merit diagnostic, not a yardstick | §4.3 | Reed et al. 2013; Zatarain Salazar et al. 2016 |
| Cardinality asymmetry / noise-induced spurious dominance | §4.3 | Bartholomew & Kwakkel 2020; Shavazipour et al. 2021 |
| Robustness metrics can be degenerate → co-report raw performance | §5 | Huang et al. 2025; Bonham et al. 2024; Gold et al. 2023 |
| Diversity-based selection did not beat random selection | §5 | Eker & Kwakkel 2018 |
| Search-measure mismatch penalizes the mismatched policy | §5 | Giuliani & Castelletti 2016 |
| Composition moves values more than rankings; hold re-eval metric fixed | §3 | McPhail et al. 2020 |

---

## 7. Open items

1. Validate the annual-unit choice against realization-level rankings
   (long-record set; machinery specified only if pursued).
2. Set the **centre** of the §4.1 threshold grid (Decree/FFMP anchors where they
   exist; elicited-convention defaults elsewhere) and the grid's span.
3. The salt-front (`salt_front_intrusion_max_rm`) and Lordville thermal metrics
   remain registered diagnostics; both are out of the active search set.
