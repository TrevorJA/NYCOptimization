# NYC Reservoir Re-Optimization: Project Summary

*Entry point for new readers. Details live in `docs/notes/`; this page states what the
study is, what is decided, and what is still open. The manuscript at
`docs/manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md` is the authoritative
description of the method. Where a note and the code disagree, the code is the record
of what exists and the note is a proposal.*

---

## The study in one paragraph

We re-optimize the operating rules of the four NYC reservoirs in the Delaware River
Basin (the FFMP rule structure, 36 decision variables) with the multi-master Borg MOEA
coupled to the Pywr-DRB simulation model. The methodological contribution is not the
re-optimization itself but a controlled test of how the streamflow scenario ensemble
used to evaluate candidate policies *during search* is constructed. The proposed design
— **hazard filling** — selects scenarios from a large candidate pool of short synthetic
streamflow sequences so that the retained scenarios cover a multi-dimensional **hazard
space** (six selection axes computed on each sequence: drought magnitude, severity,
onset rate, and recovery rate from SSI-6 run theory, plus flood peak discharge and pulse
duration from peaks over threshold), deliberately over-representing the severe corners
where reservoir policies are decided. It is compared against the discipline's default,
an independent and identically distributed sample from the same stochastic generator.
Both designs' Pareto-approximate policies are re-evaluated on a common held-out, deeply
uncertain test ensemble, and re-evaluated robustness is the sole basis of comparison.

## Research questions

1. **RQ1.** Do policies within the FFMP rule structure exist that, relative to current
   operations evaluated under the same deeply uncertain conditions, simultaneously
   improve outcomes for multiple stakeholders (NYC supply reliability, Montague and
   Trenton flow-target compliance, downstream flood exposure, minimum reservoir storage)
   without degrading any other outcome beyond stated tolerances?
2. **RQ2 (core).** Does constructing the search ensemble by space-filling coverage of the
   hazard space, rather than by i.i.d. sampling from the same stochastic generator,
   change the robustness of the resulting Pareto-approximate policies under held-out,
   deeply uncertain re-evaluation?

Numbering follows the manuscript (Section 1, P7). See `notes/research_questions.md`,
`notes/research_contributions.md`.

## Scenario designs compared (RQ2)

Three designs, all drawn from one **stationary population** (Kirsch–Nowak fitted to the
historic record, no climate perturbation). Registry: `src/scenario_designs.py`. HF is the
manuscript's shorthand for `hazard_filling_stationary` and MC for `monte_carlo`.

| Design | Construction | Role |
|---|---|---|
| `historic` | The observed record, one continuous 78-yr trace (Dec 1945 – Nov 2023), scored as 77 FFMP-year units | Prevailing-practice reference (Giuliani 2016; Herman 2020); unmatched |
| `monte_carlo` | N × L realizations drawn i.i.d. from the stationary generator; frozen across the search | The random-sampling control (Quinn 2017; Zatarain Salazar 2017) |
| `hazard_filling_stationary` | LHS anchors in absolute, robust range-scaled hazard space (p1/p99 bounds), snapped to the nearest member of its own i.i.d. candidate pool | **Proposed method** |

**The controlled contrast.** `monte_carlo` → `hazard_filling_stationary` holds
the generator, population law, N, and L fixed and varies *only the selection rule*: does
hazard coverage beat random sampling? Because the candidate pool is sampled i.i.d., a
uniform random size-N subset of it has exactly the law of N fresh i.i.d. draws, which
makes `monte_carlo` the *exact statistical control* for
`hazard_filling_stationary`. This is the Eker & Kwakkel (2018) null benchmark
(diversity-based selection did not beat random selection) raised to hazard space, on a
real system, with seed-replicated searches and draw-dependence measured by
re-evaluation.

**Why absolute hazard space.** The selector fills the hazard space in absolute,
robust range-scaled magnitude units (per-axis p1/p99 bounds; sample extremes are
non-convergent order statistics, so full-range bounds would tie the geometry to the
pool size and its outliers) rather than empirical-CDF/rank units. Because the hazard
marginals of a stochastic generator are strongly right-skewed, filling the range
uniformly draws selected members from the sparse severe corners far more often than
their pool frequency, so severe drought and flood conditions are over-represented in the
search ensemble relative to their probability under the generator. This is the deliberate
distribution shift RQ2 tests. A rank-space variant (`hazard_filling_stationary_cdf`) is
registered only as a non-campaign sensitivity.

Full construction recipe: `notes/methods/scenario_design_methods.md`. Gap statement:
`notes/literature/scenario_design.md`. Nearest antecedents, each differentiated there:
Cohen et al. 2021 (training-scenario properties → robustness, but problem-driven regret
selection needing one perfect-foresight optimization per scenario), Bonham et al. 2024
(space-filling subsampling, but post-hoc ranking), Zatarain Salazar et al. 2017 (1-D,
probability-preserving flow stratification).

## Pipeline

1. **Generation** — the stationary Kirsch–Nowak generator produces the `monte_carlo`
   ensemble directly and the `hazard_filling_stationary` candidate pool (P = 10⁶ per
   draw). The pool is sampled i.i.d., and only its hazard image plus seeds are stored;
   realizations regenerate deterministically on demand (chunked storage for large pools).
2. **Hazard metrics + redundancy handling** — drought and flood descriptors per
   sequence; the screen (degenerate drop + near-duplicate prune at |ρ_S| ≥ 0.95)
   retains all eight candidates, with rank-correlation structure reported as a
   diagnostic. The campaign **selection axes** are a fixed six-descriptor subset
   (drought magnitude, severity, onset rate, recovery rate; flood peak discharge, pulse
   duration — `config.HAZARD_SELECTION_AXES`); drought duration and flood rise rate stay
   computed and reported but do not enter the snap distance.
3. **Selection (hazard filling only)** — Latin hypercube anchors in absolute, robust
   range-scaled hazard space, snapped to the nearest unused pool member. The snap is
   intrinsic: hazard coordinates are emergent properties of a realized sequence, so a
   hazard-space design must *select from* a pool, whereas an i.i.d. design *generates*
   directly.
4. **Search** — MM Borg over FFMP decision variables; objectives evaluated on the
   design's ensemble (workflow steps 00–06).
5. **Re-evaluation** — every final Pareto set re-simulated on the common held-out test
   ensemble, which is never the source of any search ensemble. The (solution × SOW ×
   objective) matrix is persisted in natural units, and robustness metrics are scored
   offline from it (steps 08–11), so a new metric never requires re-simulating.

Operational how-to: `workflow/README.md` and the step scripts `workflow/00–14_*.sh`.

## Objectives

Eight active objectives (NYC delivery reliability + P99 deficit tail, Montague
reliability + P99 deficit tail, Trenton reliability, expected annual downstream flood
exceedance (ft·days above NWS minor flood stage), NYC storage annual-minimum P01, NJ
delivery reliability), computed by the two-layer annual-unit scheme (annual metric per
realization × FFMP-year unit (Jun 1 – May 31, the FFMP operating year); a
per-objective unit operator over the pooled unit-years) and documented in
`notes/methods/objective_definitions.md`. **These annual-unit statistics are the study's
single metric currency**: search pools all realizations' unit-years; robustness and
regret recompute the same statistics per E_test state of the world. The annual-unit
epsilons [0.05, 10, 0.05, 10, 0.05, 0.3, 5.0, 0.05] are calibrated per
`notes/methods/epsilon_calibration_experiment.md` (max over the two ensemble designs,
the historic design excluded and disclosed). The satisficing thresholds and the named
criterion-set placements are provisional until the threshold diagnostic and the
re-anchoring audit run on the 500-SOW production cube
(`notes/methods/robustness_threshold_diagnostics.md`; `TODO.md`).

## Comparison controls

- **Budget**: both matched designs run at N = 300, L = 10 yr — 3,000 scenario-years per
  evaluation — at equal NFE, so per-evaluation cost, scenario-years, and
  wall-clock are identical and equal-NFE coincides with equal-scenario-years. The common
  (N, L) is required: if L differed, the selection rule would be confounded with record
  length.
- **The i.i.d. pool is load-bearing.** A uniform random size-N subset of an i.i.d. pool
  has exactly the law of N fresh i.i.d. draws, which is what makes `monte_carlo`
  the exact control for `hazard_filling_stationary`. A structured (e.g. LHS) pool would
  void the control. Enforced by an invariant test.
- **Seed-stream disjointness**: the candidate pool and the test ensemble generate from
  namespaced seed domains, so no design and the test ensemble ever share realizations.
- **Replication**: one searched ensemble draw × S = 2 MOEA seeds per matched design, set
  against the compute balance. A draw is the design's construction re-run from scratch
  with a fresh seed; three are staged, the search runs on draw 0, and the seed is the
  unit of analysis. The comparison is conditional on that draw, and draw-dependence is
  measured by re-evaluating each design's final Pareto set on its own two other draws
  (SI). Seed 1 of every design is continued to 750k NFE and reported from its runtime
  archive at 500k. `historic` runs the same two seeds. Full specification and budget:
  `notes/methods/campaign_design.md`.
- **Single comparison point**: cross-design metrics computed only on held-out
  re-evaluation.

See `notes/methods/experimental_design.md`.

## The test ensemble (E_test)

E_test is the **only carrier of deep uncertainty** in the study and the **largest
ensemble by a wide margin**: generated as N_θ = 1,000 LHS points over the full range of
the deeply-uncertain climate-forcing factors (the CMIP6 harmonic hypercube) × R = 25
realizations × L_test = 50 yr (25,000 realizations, `etest_kn_50yr_n25000`), of which
the campaign re-evaluates the leading 500 SOWs (12,500 realizations, 625k
scenario-years, `etest_kn_50yr_n25000_first25ch`; a chunk-prefix subset of the randomly
ordered design, sized from the literature for a 3-axis forcing space:
`notes/methods/campaign_design.md` §5). Each LHS point is a state of the world, and its
realizations sample natural variability within it; the 50-yr records (vs L = 10 in
search) test sustained operation — storage carryover across consecutive droughts and
matured entitlement banking. Re-evaluation runs the **trimmed model**, like search: the
policy-independent non-NYC releases are presimulated once per realization and reused
for every Pareto set. Because the search ensembles are drawn from the unperturbed
stationary generator while E_test spans a forced climate envelope, the re-evaluation is
a **generalization test**: does hazard coverage of the natural-variability manifold
produce policies that generalize to conditions never presented during search? E_test is
structurally distinct from both search designs by construction; whether the design
ranking depends on the region of the test space it emphasizes is measured by the
composition-sensitivity re-scoring (hazard-restricted and envelope-restricted subsets of
the persisted matrix), not assumed.

E_test is sampled by **LHS, not i.i.d.**: the i.i.d. rule applies only to the candidate
pool, where it underwrites the exact control. E_test is never a control (its 500-SOW
campaign prefix is a subsample of the measuring stick, not a control construction), so
it should *cover* the deeply-uncertain space rather than sample it in proportion to a
measure. It follows that **no robustness number is an expectation** — under deep
uncertainty there is no probability measure over the forcing space, so a satisficing
fraction over E_test is a coverage-weighted count over a designed exploration, and the
comparison is commensurable because E_test is *identical across designs*, not because
it is probability-faithful. The campaign uses one construction (Kirsch–Nowak over the
wide DU box); rankings are conditional on it, a declared limitation; a structurally
different second construction (multi-site HMM) is registered but parked outside the
campaign — the DRB-fitted HMM is near-memoryless, so it would vary the generator family
without adding a persistence stress. **The (solution × SOW × objective) matrix is
persisted** in natural units — each state's 25 realizations pool their unit-years
through the objectives' own unit operators — so every robustness and regret metric is
scored offline without re-simulating.

## Comparison metrics

Two families, both transformations of the same per-SOW annual-unit objective values
(the search objectives recomputed per state of the world — Herman et al. 2014, 2015;
Trindade et al. 2017; McPhail et al. 2018). The RQ2 endpoint is the re-evaluated
**multivariate Starr satisficing fraction**: the all-criteria conjunction counted
over the 500 re-evaluated states. This keeps the designed DU box and the fitted stochastic
generator from being integrated into one number, matches the precision the SOW count
supports, and puts RQ2 on the same unit as the regret family below. The run-level
scalar is the maximum satisficing fraction attained in the run's re-evaluated set,
reported with its per-objective satisficing decomposition (the maximum-over-a-set
bias is disclosed). Secondary metrics are univariate satisficing, the
coverage-weighted mean (Laplace), and maximin.

The RQ1 endpoint is **incumbent-relative regret**: how much worse a candidate policy is
than the status-quo 2017 FFMP policy *in the same state of the world*. Magnitudes are
reported per objective in natural units and never combined; the unit-free harm
frequencies — per objective, per Decree party, and the joint no-harm frequency at a
tolerance τ_i = k · max(ε_i, floor_i) swept over k — carry the cross-design summary. It
is a fixed, design-independent reference that McPhail et al. (2018) license and no
published water-resources study formalizes.
**No set-relative (best-in-set), baseline-SOW, or perfect-foresight regret is computed.**
The two families are complementary rather than redundant: the satisficing criteria are
fixed scalars anchored on the incumbent's historic attainment, whereas the regret bar
moves with the forcing, so regret still discriminates where the domain criterion
saturates. Together they test the working hypothesis — that hazard filling buys
robustness without paying the price of robustness (Bartholomew & Kwakkel 2020; Bertsimas
& Sim 2004) in regret against current operations — with both axes read off the *same*
policy.
Ranking agreement across metrics is summarized by Kendall's τ_b. A **criterion sweep**
reports whether the design difference holds across the range of defensible satisficing
thresholds rather than at one arbitrary point. Hazard-space coverage statistics are
reported only as method verification that the selector fills the space as intended,
never as a comparison result.

## Status

**In place:** the end-to-end pipeline (smoke-verified), the measured campaign cost
basis (173.8 s per N = 100 evaluation trimmed, full model 1.16×, and 21,850 SU per
N = 100 / 500k-NFE production search on 8 × 128, from which every campaign number
scales), the P = 10⁶ candidate pools for draws 0–2, E_test with its presim pass, and
the incumbent-on-E_test matrix. The hazard-filling selection has no tail-share
threshold; the minimum per-axis share above the pool P90 is reported as a property of
the selector on the pool's joint geometry (`notes/methods/hazard_selector_diagnostics.md`).

**Not yet staged:** the N = 300 search ensembles (draws 0–2 of both matched designs)
and their step-05 baselines (`TODO.md` §1).

**Decided:** the three designs above; a single stationary search population with deep
uncertainty carried only in E_test; N = 300, L = 10 yr at equal NFE (the smallest ladder
size at which the i.i.d. control resolves every objective's paired difference within
ε/2); 500k NFE per search as the reporting budget, with seed 1 continued to 750k and
reported from its runtime archive at 500k (the comparison is recomputable at earlier
budgets); the production MM Borg geometry (12 Anvil nodes, 4 islands × 382 workers,
1,533 ranks, 128 per node with a 150-realization batch); one searched draw × S = 2 seeds;
absolute range-scaled hazard-space selection on the six campaign selection axes from a
P = 10⁶ candidate pool; E_test generated at N_θ = 1,000 LHS SOWs × R = 25 × L_test = 50 yr
and re-evaluated on its leading 500 SOWs (trimmed-model re-evaluation); the calibrated
annual-unit epsilon vector; comparison metrics = multivariate Starr satisficing
(primary) with Laplace and maximin as secondary anchors, and incumbent-relative regret
(per-objective regret and gain magnitudes, harm frequencies, and the no-harm frequency)
as the co-primary RQ1 family; search aggregation = two-layer annual-unit scheme; the
framing conventions (failure-week counts, flood unit operator = mean, 0.99 weekly
satisfaction factor); forcing space retains historical persistence (claims scoped
accordingly).

**Budget.** The campaign is priced on the measured production basis (21,850 SU per
N = 100 / 500k search, scaled by (N/100)^0.951) against the remaining Anvil balance of
about 600,000 SU: ~340k SU for the four matched searches, ~10k for the `historic`
reference, ~5k staging, and ~66k for the 500-SOW E_test re-evaluation at the 2,000-policy
cap, about 423k in total with a reserve of ~177k (122k if the unmeasured 8 → 12 node
scaling costs the full 17 % carried as its upper bound). The model basis, 1.53× higher,
does not fit and is the stress case; the first seed-1 runs price the campaign before
seed 2 is submitted. A third seed does not fit. The variable-resolution `ffmp_N` sweep is
a conditional SI extension that runs only on whatever SU remains at the end of the
campaign. Table: `notes/methods/campaign_design.md` §6.

**Remaining before campaign launch:** stage the search ensembles at N = 300 (steps
02–04 for draws 0–2 of both matched designs, step 05 baselines) with
`validate_staged_seasonality` build QC; re-verify the ε floors on the N = 300 ensembles;
run the one-node batched-search memory smoke; then seed 1 of every design, which prices
the campaign. Re-run the satisficing-threshold diagnostic and the criteria re-anchoring
audit on the production cubes and adopt final placements; re-run the regret-tolerance
pass A on the incumbent cube (and pass B after the E_test re-evaluation). Tracked in
`TODO.md`.

**Open decisions:** the satisficing criterion values and sweep-grid centre; whether and
under which design the variable-resolution `ffmp_N` sweep runs on leftover SU (an SI
extension, not a manuscript research question).

## Document index

| Topic | Doc |
|---|---|
| Manuscript (authoritative method) | `manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md` |
| Research questions / contributions | `notes/research_questions.md`, `notes/research_contributions.md` |
| Experimental design (controls, replication) | `notes/methods/experimental_design.md` |
| Campaign at scale (N, draws, seeds, NFE, geometry, budget) | `notes/methods/campaign_design.md` |
| Ensemble construction recipe | `notes/methods/scenario_design_methods.md` |
| Hazard-selector diagnostics | `notes/methods/hazard_selector_diagnostics.md` |
| Forcing-space parameterization (E_test) | `notes/methods/forcing_parameterization.md` |
| Objective definitions | `notes/methods/objective_definitions.md` |
| Epsilon calibration | `notes/methods/epsilon_calibration_experiment.md` |
| Framing-convention diagnostics | `notes/methods/framing_convention_diagnostics.md` |
| Terminology (controlled vocabulary) | `notes/terminology.md` |
| Literature hub + topic notes | `notes/literature/README.md`, `notes/literature/scenario_design.md` |
| Workflow / HPC operation | `../workflow/README.md`, `../workflow/envs/README.md` |
