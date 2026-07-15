# Objective Definitions for the Scenario-Design MOEA Study

*Last updated: 2026-07-13. Authoritative record of the objective formulations
used in the MOEA optimization and of the held-out re-evaluation metric set;
supersedes inline docstrings where they disagree. Terminology per
`docs/notes/terminology.md`; citations resolve to the Zotero collection
"Paper 3 NYC Reoptimization" (`ISYGLK35`) and the notes under
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
change. The search-time annual-unit layer and the re-evaluation satisficing
layer are both in `src/objectives_ensemble.py`; the offline re-evaluation
scoring is in `src/robustness.py`.

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
  percentile / domain-satisficing), **T3** aggregation function (mean / variance /
  higher moments / worst-case).

---

## 1. The objective set (single-realization / historic temporal metrics)

For the historic design there is no realization axis (R = 1), so the objective
**is** the temporal metric below. These same metrics are the per-realization
quantities scored at re-evaluation (§3). The recommended active set is
**7 objectives** (an optional 8th, NJ delivery, is pending the redundancy
screen). Worst-case extremes were replaced with stable tail/percentile/count
forms (Quinn et al. 2017; Bonham et al. 2024); the salt-front objective was
replaced by the Trenton flow Decree (physically redundant — the Trenton target
repels salt intrusion — and the salt-front LSTM is unreliable in extreme
drought).

| # | Name (registry) | Source | Temporal aggregation | Dir | Units | ε |
|---|-----------------|--------|----------------------|-----|-------|---|
| 1 | `nyc_delivery_reliability_weekly` | `delivery_nyc`, `demand_nyc` (cap 800) | frac of weeks `Σ_w delivery ≥ 0.99·Σ_w min(demand,800)` | MAX | frac | 0.07 |
| 2 | `nyc_delivery_deficit_cvar90_pct` | same | CVaR₉₀ of weekly deficit % `= 100·max(0, mean_w(min(demand,800)) − mean_w(delivery))/800` | MIN | % | 1.5 |
| 3 | `montague_flow_reliability_weekly` | `major_flow.delMontague` | frac of weeks `mean_w(flow) ≥ 1131.05` | MAX | frac | 0.02 |
| 4 | `montague_flow_deficit_cvar90_pct` | `delMontague` | CVaR₉₀ of `100·max(0, 1131.05 − mean_w(flow))/1131.05` | MIN | % | 1.5 |
| 5 | `trenton_flow_reliability_weekly` | `major_flow.delTrenton` | frac of weeks `mean_w(flow) ≥ 1938.95` | MAX | frac | 0.0003 |
| 6 | `downstream_flood_days_minor` | `flood_stage` (Hale Eddy, Fishs Eddy, Bridgeville) | count of days any gauge `≥` its NWS **minor** flood stage | MIN | days | 1.0 |
| 7 | `nyc_storage_p5_pct` | `res_storage[NYC]` | 5th percentile of daily `100·Σ_res storage / 270,837` | MAX | % | 1.5 |
| 8 | `nj_delivery_reliability_weekly` *(optional)* | `delivery_nj`, `demand_nj` (cap 100) | frac of weeks `Σ_w delivery_nj ≥ 0.99·Σ_w min(demand_nj,100)` | MAX | frac | 0.007 |

Epsilons are the calibrated values in `src/objectives.py`: ε ≈ IQR/10 of each
objective's spread across N = 500 random-DV policies on the historic reference
trace (Reed et al. 2013), rounded to clean steps. The §2 annual-unit registry
(`src/objectives_ensemble.py`) carries its **own, separate** epsilons, still
placeholders pending the ensemble sensitivity experiment.

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

## 2. Per-scenario-design search aggregation — two-layer annual-unit scheme

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

**Design mapping.** Both ensemble designs (`fixed_probabilistic` and
`hazard_filling`) use this same two-layer scheme — one scheme fits both naturally,
even though commensurability is no longer *required*. The
**historic design** keeps the §1 temporal metrics on its single continuous trace
(R = 1; prevailing-practice reference, Giuliani & Castelletti 2016). In McPhail
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

**The full matrix is persisted.** Unlike search, which keeps only objective
scores, re-evaluation persists the entire (solution × realization × objective)
matrix in natural units, plus each realization's **SOW id** (its θ / LHS-point
index). Every robustness metric is scored *offline* from that matrix, so a new
metric never requires re-simulating, and **both units of robustness are available
without additional compute**: the **SOW unit** (collapse each θ's realizations,
then apply the Starr criterion across the N_θ SOWs — the Herman 2014 / Trindade
2017 / Gold 2022–23 standard) and the **realization unit**. Precision is governed
by N_θ, not by N_test.

**The re-evaluation metric is not the §2 search metric — by design.** Search and
re-evaluation compute structurally different quantities on different code paths:

| | Search (`_evaluate_ensemble_batched`) | Re-evaluation (`_ensemble_base_matrix`) |
|---|---|---|
| per realization | `annual_units()` → vector of L−1 **annual** metrics | `base.compute()` → **one whole-trace scalar** (the §1 temporal metric) |
| across realizations | pool NL unit-years → §2 **unit operator** (frequency / P99 / mean) | fraction of realizations clearing the criteria (`SatisficingAgg`) |

The §2 unit operators are **never invoked at re-evaluation**. The re-evaluation
metric is **realization-level satisficing on the §1 whole-trace metrics**: each
E_test realization yields one scalar per objective, and robustness is the
fraction of realizations that clear the criteria. This metric does **not** need
to equal the search metric — it needs only to be **identical across all designs**,
which is exactly what makes the comparison commensurable.

The re-evaluation drivers (`src/reevaluate.py`, `src/reevaluate_mpi.py`) persist
the raw **(solution × realization × objective)** cube in natural units
(`reeval_raw.parquet` + a self-describing `reeval_raw_meta.json`). Every metric
below is therefore an *offline re-scoring* of that cube (`src/robustness.py`),
not a re-simulation — the whole set is free once the cube exists.

### 3.1 Primary metric: multivariate satisficing (Starr's domain criterion)

**The fraction of E_test realizations in which the policy meets ALL criteria
jointly.** The realization is the unit: a policy satisfices in a realization if
every criterion holds on that realization's whole-trace §1 metrics. This is
Starr's (1962) domain criterion, and it is the standard robustness unit of the
Herman et al. (2014, 2015) / Trindade et al. (2017) / Gold et al. (2022, 2023)
lineage, where the SOW is likewise the unit — hence directly citable and directly
comparable to published numbers. McPhail T1 = satisfaction-of-constraints,
T2 = all scenarios, T3 = frequency.

### 3.2 Secondary metrics (all reported)

| Metric | Definition | McPhail T1 / T2 / T3 | Role |
|---|---|---|---|
| **univariate satisficing** | fraction of realizations clearing one objective's criterion | satisfaction / all / frequency | per-objective decomposition of the primary |
| **Laplace (mean)** | mean performance across realizations | identity / all / mean | risk-neutral reference |
| **maximin** | worst realization | identity / worst-case / worst-case | risk-averse reference |
| **improvement over status quo** | per-realization deviation from the **default FFMP policy** on the *same* E_test | regret-vs-baseline / all / mean (max also reported) | "is it better than current operations?" |

*Improvement over status quo* is design-independent and requires no extra
optimization: the default FFMP baseline is already simulated on E_test by
workflow step 05, and the reference does not move when designs are added or
dropped. Precedent: Kasprzyk et al. (2013) percent-deviation-from-baseline.
(Implementation: `improvement_vs_baseline` in `src/robustness.py`.)

Ranking agreement is summarized with **Kendall's τ_b computed across the *design*
rankings** these metrics induce — i.e. do the metrics rank the scenario
designs the same way? (Herman et al. 2015; McPhail et al. 2018, 2020.)

### 3.3 Metrics deliberately excluded

**Regret-from-best (best-in-set regret) is not used.** Two reasons.
(a) *It is set-relative and design-coupled.* With `f*(s)` = the best performance
in scenario *s* over the pooled re-evaluated policies, dropping one design from
the pool changes every other design's regret. It is therefore not a
design-independent quantity, and design-independence is the minimum requirement
of a cross-design comparison statistic.
(b) *It does not converge on our objectives.* Bonham et al. (2024) find
regret-from-best needs 400+ scenarios to stabilize (satisficing: 50–300) and,
on a max-over-time objective, **never** converges; they explicitly caution
against using it in isolation. Our two deficit objectives are worst-1st-percentile
(P99) operators, so regret on them is exactly the extreme-of-extremes estimator
Bonham warns about.

**Cohen et al. (2021) baseline regret is not used, and no perfect-foresight
optimization is performed anywhere in this study.** It requires one
perfect-foresight MOEA run per scenario (97 optimizations, 3,233 CPU-h in
Cohen), which is formulation-specific and does not scale to a candidate pool of
10⁵–10⁶ scenarios. Cohen et al. (2021) is cited as **motivation** for the
contribution, never as a metric we compute.

**The search-vs-test "overfitting gap" is not used.** Two reasons.
(a) *There is no such metric to cite.* Brodeur et al. (2020) diagnose overfitting
**graphically** — cost distributions over the training and the held-out test
ensembles plotted side by side. They define no gap equation, report no gap
magnitude, and rank nothing by a gap. Citing Brodeur for a defined gap metric
would not survive review.
(b) *It is structurally invalid here.* When the in-sample term is
coverage-weighted (hazard-filling's deliberately distorted measure) and the
out-of-sample term is measure-weighted (E_test's natural composition), their
difference is a difference of two expectations **under two different measures**.
It is an artifact of the measure change, not an overfitting quantity — and it
would *grow* with exactly the coverage this study advocates. Brodeur's own caveat
is the citation: they restrict all claims to *relative* rankings within each
period and never interpret the absolute train-vs-test difference, precisely
because their two ensembles are not drawn from the same distribution.
(`src/robustness.py` retains an in-sample-minus-re-eval helper as a diagnostic
only; it is not a comparison metric and is not reported as one.)

### 3.4 Attainability screen (free)

Flag the E_test realizations in which **no policy from any design** meets the
criteria. This costs zero CPU — the (solution × realization × objective) cube
already exists — and it separates *"this design searched badly"* from *"this test
scenario is unwinnable for anyone."* Every design's satisficing fraction is
bounded above by the attainable fraction, so the screen also sets the ceiling
against which design differences should be read. Precedent: Shavazipour et al.
(2021) found 23% of their test scenarios unwinnable by any policy. Stated
honestly: this is an **empirical attainability bound over the evaluated policy
pool**, not a per-scenario oracle (an oracle would require the perfect-foresight
optimization rejected in §3.3).

The codebase separates `SEARCH_ENSEMBLE_SPEC` (per design) from the common test
ensemble, with a selection-bias guard (Bonham et al. 2024) warning if they
coincide.

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
lineage; it is a contribution, not a robustness check. (Implementation:
`threshold_spectrum` in `src/robustness.py`; `NYCOPT_SAT_THRESHOLDS` re-scores
the cube without re-simulating.)

### 4.2 Scenario discovery in hazard space — optional supporting analysis

This is a supporting analysis, not the primary comparison and not a falsification
device. The primary comparison is the re-evaluated robustness of the resulting
solutions (§3). Where it is run, boosted trees (Gold et al. 2022/2023
hyperparameters) are fit to each design's E_test failure realizations — labelled
by the **conjunction** of the satisficing criteria, i.e. the §3.1 primary — **in
the hazard space of E_test**, not only in its forcing-parameter (input) space.

Its role is to characterize *where* policies fail and, if a robustness difference
is observed, to offer supporting evidence for the coverage → robustness mechanism
(the expectation being that a design's policies fail in the hazard region it
under-covered). It is reported as support for a difference found on the primary
metric, never as the basis of the comparison, because a coverage-vs-failure
association alone cannot separate the claimed mechanism from intrinsic scenario
difficulty.

Caveat: correlated hazard axes destabilize factor-importance rankings (Quinn et
al. 2020 report Sobol first-order sums going negative under correlated factors),
so the hazard axes are redundancy-screened before this analysis is run.

### 4.3 Cross-design comparison rule: no pooled-reference-set hypervolume

**Designs are compared only by re-evaluation on the common E_test.** MOEA search
metrics (hypervolume, generational distance, ε-indicator) scored against a
*pooled* reference set are **not** a defensible primary cross-design measure; they
are demoted to the supplement, where the reference set is built **per design** and
the metrics are read as within-design convergence diagnostics only.

Protocol precedent: **Zatarain Salazar et al. (2017)** §5.3 / Figs. 12–13
optimize the same problem at three search-ensemble sizes and rule that (i) the
reference set is built **per level**, (ii) cross-level MOEA metrics are
**incomparable**, and (iii) levels are compared **only** by re-evaluating on a
common independent verification ensemble. Their Fig. 13 gates reference-set
contribution on held-out performance — the nearest published thing to a
winner's-curse correction.

Three compounding reasons a pooled reference set is biased across designs:
1. **Contributor bias** — a design contributes points to the very frontier it is
   scored against. Contribution share is reported as a *merit* diagnostic (Reed et
   al. 2013; Zatarain Salazar et al. 2016), not as a neutral yardstick.
2. **Cardinality asymmetry** — designs return different numbers of solutions.
   Bartholomew & Kwakkel (2020) name this and do not correct it; Shavazipour et
   al. (2021) mitigate only by reporting % retained rather than counts.
3. **Noise-induced spurious dominance** — Shavazipour et al. (2021): "some
   solutions are dominated because of the random values set by the model … not
   because of the existence of any better solutions." A design whose re-evaluated
   estimates are **noisier** contributes more spuriously nondominated points and is
   therefore **flattered** by the pooled sort.

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
  is in *hazard* space on a real system; and our comparison statistic has far more
  power (K ensemble draws × S seeds, with the draw as the unit of analysis, versus
  their counting of solutions above a group median).
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
| Count-over-threshold (flood days, minor stage) | obj. 6 | Quinn et al. 2017 |
| Trenton flow replacing salinity (physical redundancy) | obj. 5 | Trindade et al. 2017; Hadjimichael et al. 2020 |
| ε ≈ IQR/10 calibration | §1 | Reed et al. 2013; Hadka & Reed 2013 |
| Failure-year frequency across pooled units (search reliability) | §2 #1/3/5/8 | Zeff et al. 2014; Trindade et al. 2017; Gold et al. 2023 |
| Worst-1st-percentile unit-year (search tail objectives) | §2 #2/4/7 | Quinn et al. 2017 (WP1), 2018; Zeff/Trindade/Gold worst-1% cost |
| Consecutive annual units with inherited state (long records) | §2 | Quinn et al. 2018 |
| Two-layer time-aggregation / noise-filtering vocabulary | §2 | Hamilton et al. 2022 |
| Expectation can mask floods (P99 variant registered) | §2 #6 | Quinn et al. 2017 |
| Multivariate satisficing / domain criterion (PRIMARY re-eval metric) | §3.1 | Starr 1962; Herman et al. 2014, 2015; Trindade et al. 2017; Gold et al. 2022, 2023 |
| T1/T2/T3 decomposition; Laplace / maximin secondaries | §0, §3.2 | McPhail et al. 2018; Giuliani & Castelletti 2016 |
| Percent-deviation-from-baseline (improvement over status quo) | §3.2 | Kasprzyk et al. 2013 |
| Kendall's τ_b for ranking agreement across metrics | §3.2 | Herman et al. 2015; McPhail et al. 2018, 2020 |
| Satisficing converges fastest; regret-from-best does not | §3.3 | Bonham et al. 2024 |
| Perfect-foresight-per-scenario regret is unscalable (motivation only) | §3.3 | Cohen et al. 2021 |
| Overfitting diagnosed graphically, not by a gap metric | §3.3 | Brodeur et al. 2020 |
| Some test scenarios are unwinnable by any policy | §3.4 | Shavazipour et al. 2021 |
| Satisficing thresholds are elicited conventions | §4.1 | Zeff et al. 2014; Trindade et al. 2017; Gold et al. 2023 |
| Design-ranking agreement degrades with stringency (→ sweep) | §4.1 | Quinn et al. 2020 |
| Threshold-margin CDF diagnostic | §4.1 | Gold et al. 2023 (Fig. 5) |
| Boosted-tree scenario discovery; correlated-factor caveat | §4.2 | Gold et al. 2022, 2023; Quinn et al. 2020 |
| Per-level reference sets; compare only on a common verification ensemble | §4.3 | Zatarain Salazar et al. 2017 |
| Contribution share is a merit diagnostic, not a yardstick | §4.3 | Reed et al. 2013; Zatarain Salazar et al. 2016 |
| Cardinality asymmetry / noise-induced spurious dominance | §4.3 | Bartholomew & Kwakkel 2020; Shavazipour et al. 2021 |
| Robustness metrics can be degenerate → co-report raw performance | §5 | Huang et al. 2025; Bonham et al. 2024; Gold et al. 2023 |
| Diversity-based selection did not beat random selection | §5 | Eker & Kwakkel 2018 |
| Search-measure mismatch penalizes the mismatched policy | §5 | Giuliani & Castelletti 2016 |
| Composition moves values more than rankings; hold re-eval metric fixed | §3 | McPhail et al. 2020 |

---

## 7. Open items

1. Finalize the optional 8th objective (`nj_delivery_reliability_weekly`) after
   the redundancy screen (`objective_sensitivity_experiment.md`).
2. From the two-design ensemble sensitivity experiment
   (`ensemble_objective_sensitivity_experiment.md`): (a) confirm the Decree-anchored
   annual failure criteria (#1/3/5/8) do not saturate under either a probabilistic
   or a hazard-filled composition (adjust the failure definition — e.g., ≥k failing
   weeks — if they do); (b) pick the flood-days unit operator (mean vs P99, #6);
   (c) set native-unit epsilons for the §2 mean/percentile objectives and confirm P99
   stability at the campaign NL; (d) validate the annual-unit choice against
   realization-level rankings.
3. Set the **centre** of the §4.1 threshold grid (Decree/FFMP anchors where they
   exist; elicited-convention defaults elsewhere) and the grid's span.
4. Fix the E_test hazard-axis set (redundancy screen) before the §4.2 scenario
   discovery is run.
5. The salt-front (`salt_front_intrusion_max_rm`) and Lordville thermal metrics
   remain registered diagnostics; both are out of the active search set.
</content>
</invoke>
