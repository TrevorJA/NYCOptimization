# NYC Reservoir Re-Optimization: Project Summary

*Entry point for new readers. Last updated 2026-08-05. Details live in `docs/notes/`;
this page states what the study is, what is decided, and what is still open. The
manuscript at `docs/manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md` is the
authoritative description of the method. Where a note and the code disagree, the code
is the record of what exists and the note is a proposal.*

---

## The study in one paragraph

We re-optimize the operating rules of the four NYC reservoirs in the Delaware River
Basin (the FFMP rule structure, 36 decision variables) with the multi-master Borg MOEA
coupled to the Pywr-DRB simulation model. The methodological contribution is not the
re-optimization itself but a controlled test of how the streamflow scenario ensemble
used to evaluate candidate policies *during search* is constructed. The proposed design
— **hazard filling** — selects scenarios from a large candidate pool of short synthetic
streamflow sequences so that the retained scenarios cover a multi-dimensional **hazard
space** (drought, low-flow, and high-flow metrics computed on each sequence),
deliberately over-representing the severe corners where reservoir policies are decided.
It is compared against the discipline's default, an independent and identically
distributed sample from the same stochastic generator. Both designs' Pareto-approximate
policies are re-evaluated on a common held-out, deeply uncertain test ensemble, and
re-evaluated robustness is the sole basis of comparison.

## Research questions

1. **RQ1 (core).** Does constructing the search ensemble by hazard-space coverage,
   rather than by i.i.d. sampling from the same stochastic generator, change the
   robustness of the resulting Pareto-approximate policies under held-out, deeply
   uncertain re-evaluation?
2. **RQ2.** Can re-optimizing the FFMP parameters improve NYC/basin outcomes (supply
   reliability, Montague/Trenton flow targets, downstream flooding, storage resilience)
   relative to current operations?
3. **RQ3.** Does a variable-resolution FFMP structure with more storage zones (`ffmp_N`)
   improve performance or robustness?

See `notes/research_questions.md`, `notes/research_contributions.md`.

## Scenario designs compared (RQ1)

Three designs, all drawn from one **stationary population** (Kirsch–Nowak fitted to the
historic record, no climate perturbation). Registry: `src/scenario_designs.py`.

| Design | Construction | Role |
|---|---|---|
| `historic` | The observed record, one continuous ~77-yr trace | Prevailing-practice reference (Giuliani 2016; Herman 2020); unmatched, K = 1 |
| `fixed_probabilistic` | N × L realizations drawn i.i.d. from the stationary generator; frozen across the search | The random-sampling control (Quinn 2017; Zatarain Salazar 2017) |
| `hazard_filling` | LHS anchors in absolute, robust range-scaled hazard space (p1/p99 bounds), snapped to the nearest member of its own i.i.d. candidate pool | **Proposed method** |

**The controlled contrast.** `fixed_probabilistic` → `hazard_filling` holds the
generator, population law, N, and L fixed and varies *only the selection rule*: does
hazard coverage beat random sampling? Because the candidate pool is sampled i.i.d., a
uniform random size-N subset of it has exactly the law of N fresh i.i.d. draws, which
makes `fixed_probabilistic` the *exact statistical control* for `hazard_filling`. This
is the Eker & Kwakkel (2018) null benchmark (diversity-based selection did not beat
random selection) raised to hazard space, on a real system, with a replication scheme
that separates ensemble-construction variance from search variance.

**Why absolute hazard space.** The selector fills the hazard space in absolute,
robust range-scaled magnitude units (per-axis p1/p99 bounds; sample extremes are
non-convergent order statistics, so full-range bounds would tie the geometry to the
pool size and its outliers) rather than empirical-CDF/rank units. Because the hazard
marginals of a stochastic generator are strongly right-skewed, filling the range
uniformly draws selected members from the sparse severe corners far more often than
their pool frequency, so severe drought and flood conditions are over-represented in the
search ensemble relative to their probability under the generator. This is the deliberate
distribution shift RQ1 tests. A rank-space variant is registered only as a non-campaign
sensitivity.

Full construction recipe: `notes/methods/scenario_design_methods.md`. Gap statement:
`notes/literature/scenario_design.md`. Nearest antecedents, each differentiated there:
Cohen et al. 2021 (training-scenario properties → robustness, but problem-driven regret
selection needing one perfect-foresight optimization per scenario), Bonham et al. 2024
(space-filling subsampling, but post-hoc ranking), Zatarain Salazar et al. 2017 (1-D,
probability-preserving flow stratification).

## Pipeline

1. **Generation** — the stationary Kirsch–Nowak generator produces the `fixed_probabilistic`
   ensemble directly and the `hazard_filling` candidate pool. The pool is sampled i.i.d.,
   and only its hazard image plus seeds are stored; realizations regenerate
   deterministically on demand (chunked storage for large pools).
2. **Hazard metrics + redundancy handling** — drought and flood descriptors per
   sequence; the screen (degenerate drop + near-duplicate prune at |ρ_S| ≥ 0.95)
   retains all eight candidates, with rank-correlation structure reported as a
   diagnostic. The campaign **selection axes** are a fixed six-descriptor subset
   (deficit volume, peak depth, onset rate, recovery rate; peak magnitude, pulse
   duration — `config.HAZARD_SELECTION_AXES`); duration and rise rate stay
   computed and reported but do not enter the snap distance.
3. **Selection (hazard filling only)** — Latin hypercube anchors in absolute, robust
   range-scaled hazard space, snapped to the nearest unused pool member. The snap is intrinsic: hazard
   coordinates are emergent properties of a realized sequence, so a hazard-space design
   must *select from* a pool, whereas an i.i.d. design *generates* directly.
4. **Search** — MM Borg over FFMP decision variables; objectives evaluated on the
   design's ensemble (workflow steps 00–06).
5. **Re-evaluation** — every final Pareto set re-simulated on the common held-out test
   ensemble, which is never the source of any search ensemble. The full (solution ×
   realization × objective) matrix is persisted in natural units, and robustness metrics
   are scored offline from it (steps 08–09), so a new metric never requires re-simulating.

Operational how-to: `workflow/README.md` and the step scripts `workflow/00–09_*.sh`.

## Objectives

Eight active objectives (NYC delivery reliability + CVaR₉₀ deficit, Montague reliability
+ CVaR₉₀ deficit, Trenton reliability, downstream flood exceedance (ft·days above NWS
minor flood stage), NYC storage 5th percentile, NJ delivery reliability), defined in
`src/objectives.py` and documented in `notes/methods/objective_definitions.md`. During
search, each objective's per-realization temporal metric is collapsed across
realizations by a two-layer annual-unit scheme (annual metric per realization ×
water-year unit; a per-objective unit operator over the pooled unit-years). The
annual-unit epsilons are calibrated per `notes/methods/epsilon_calibration_experiment.md`
(max over the two ensemble designs, the historic arm excluded and disclosed); the
satisficing-threshold vector is measured
(`notes/methods/robustness_threshold_diagnostics.md`) and awaits adoption into the
registry.

## Comparison controls

- **Budget**: both matched designs run at N = 100, L = 10 yr — 1,000 scenario-years per
  evaluation — at equal NFE, so per-evaluation cost, scenario-years, and
  wall-clock are identical and equal-NFE coincides with equal-scenario-years. The common
  (N, L) is required: if L differed, the selection rule would be confounded with record
  length.
- **The i.i.d. pool is load-bearing.** A uniform random size-N subset of an i.i.d. pool
  has exactly the law of N fresh i.i.d. draws, which is what makes `fixed_probabilistic`
  the exact control for `hazard_filling`. A structured (e.g. LHS) pool would void the
  control. Enforced by an invariant test.
- **Seed-stream disjointness**: the candidate pool and the test ensemble generate from
  namespaced seed domains, so no design and the test ensemble ever share realizations.
- **Replication**: K = 3 ensemble draws × S = 2 MOEA seeds per matched design, set
  against the compute allocation. A draw is the design's construction re-run from scratch
  with a fresh seed, and is the unit of analysis; draw- and seed-level results are
  reported transparently. `historic` has K = 1.
- **Single comparison point**: cross-design metrics computed only on held-out
  re-evaluation.

See `notes/methods/experimental_design.md`.

## The test ensemble (E_test)

E_test is the **only carrier of deep uncertainty** in the study and the **largest
ensemble by a wide margin**: N_θ = 1,000 LHS points over the full range of the
deeply-uncertain climate-forcing factors (the CMIP6 harmonic hypercube) × R = 25
realizations × L_test = 50 yr — 25,000 realizations, 1.25M scenario-years. Each LHS
point is a state of the world, and its realizations sample natural variability within
it; the 50-yr records (vs L = 10 in search) test sustained operation — storage
carryover across consecutive droughts and matured entitlement banking. Re-evaluation
runs the **trimmed model**, like search: the policy-independent non-NYC releases are
presimulated once per realization and reused for every Pareto set. Because the search ensembles are drawn
from the unperturbed stationary generator while E_test spans a forced climate envelope,
the re-evaluation is a **generalization test**: does hazard coverage of the
natural-variability manifold produce policies that generalize to conditions never
presented during search? E_test is structurally distinct from both search designs, so it
does not favor either.

E_test is sampled by **LHS, not i.i.d.**: the i.i.d. rule applies only to the candidate
pool, where it underwrites the exact control. E_test is never subsampled and is never a
control, so it should *cover* the deeply-uncertain space rather than sample it in
proportion to a measure. It follows that **no robustness number is an expectation** —
under deep uncertainty there is no probability measure over the forcing space, so a
satisficing fraction over E_test is a coverage-weighted count over a designed
exploration, and the comparison is commensurable because E_test is *identical across
designs*, not because it is probability-faithful. The campaign uses one construction
(Kirsch–Nowak over the wide DU box); rankings are conditional on it, a declared
limitation; a structurally different second construction (multi-site HMM) is registered
but parked outside the campaign — the DRB-fitted HMM is near-memoryless, so it would
vary the generator family without adding a persistence stress. **The full (solution × realization × objective)
matrix is persisted** in natural units with each realization's SOW id, so any robustness
metric — at the SOW unit or the realization unit — is scored offline without
re-simulating.

## Comparison metrics

Two families. The RQ1 endpoint is the re-evaluated **multivariate Starr satisficing
fraction** of the policies a design produces, counted on the **SOW unit**: the 25
realizations sharing a forcing point are collapsed by their mean, and the
all-criteria conjunction is counted over the 1,000 states. This keeps the designed
DU box and the fitted stochastic generator from being integrated into one number,
matches the precision the SOW count supports, and puts RQ1 on the same unit as the
regret family below; the pooled realization unit is co-reported as a sensitivity.
The run-level scalar is the maximum satisficing fraction attained in the run's
re-evaluated set, reported with its per-objective satisficing decomposition (the
maximum-over-a-set bias is disclosed).
Secondary metrics are univariate satisficing, the coverage-weighted mean (Laplace),
maximin, and signed improvement-over-status-quo.

The RQ2 endpoint is **incumbent-relative regret**: how much worse a candidate policy is
than the status-quo 2017 FFMP policy *in the same state of the world*. Magnitudes are
reported per objective in natural units and never combined; the unit-free harm
frequencies — per objective, per Decree party, and the joint no-harm frequency at a swept
tolerance — carry the cross-design summary. It is a fixed, design-independent reference
that McPhail et al. (2018) license and no published water-resources study formalizes.
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
reported only as method verification that the selector administers the intended treatment
at strength, never as a comparison result.

## Status

**In place:** the end-to-end pipeline (smoke-verified); measured campaign costs
(173.8 s/eval trimmed, full model 1.16×, ~33,400 SU per 500k-NFE search); all
production inputs staged, verified, and adequacy-gated — candidate pools for all
three draws, both matched designs' search ensembles, E_test with its one-time
presim pass, and the baseline-on-E_test matrix.

**Decided:** the three designs above; a single stationary search population with deep
uncertainty carried only in E_test; N = 100, L = 10 yr at equal NFE; 500k NFE per search
(attained budget justified post hoc from the runtime archive, and the comparison
recomputable at earlier budgets); the production MM Borg geometry (8 Anvil nodes,
4 islands × 254 workers, 1,021 ranks, ~32.6 h/search); K = 3 draws × S = 2 seeds; absolute
range-scaled hazard-space selection on the six campaign selection axes from a P = 10⁶
candidate pool; E_test at N_θ = 1,000 LHS SOWs × R = 25 × L_test = 50 yr (trimmed-model
re-evaluation); the calibrated annual-unit epsilon vector; comparison
metrics = multivariate Starr satisficing (primary) with Laplace, maximin, and signed
improvement-over-status-quo as anchors; search aggregation = two-layer annual-unit
scheme; the framing conventions (failure-week counts, flood unit operator = mean,
0.99 weekly satisfaction factor) measured and confirmed; forcing space retains
historical persistence (claims scoped accordingly).

**Total Anvil allocation = 750,000 SU.** The full campaign (two matched designs × K = 3 ×
S = 2 at 500k NFE, plus the cheap `historic` reference, generation, and the E_test
re-evaluation at ~80,000 SU) is approximately 503,000 SU (67%), leaving a ~247,000-SU
reserve. First call on the reserve is an additional draw for both matched designs
(~134k); the RQ3 variable-resolution sweep (~200k) is deprioritized and runs only on
whatever SU remains at the end of the campaign.

**Remaining before campaign launch:** adopt the measured satisficing thresholds into
the registry; the confirmatory search under the adopted epsilon vector; the Anvil
shakeout (hazard-filling step-06 smoke + pilot go/no-go). Tracked in `TODO.md`.

**Open decisions:** the satisficing criterion values and sweep-grid centre; the
scenario design under which the RQ3 variable-resolution sweep is run, if the leftover
SU permits it. See `notes/methods/experimental_design.md` for the open-questions list.

## Document index

| Topic | Doc |
|---|---|
| Manuscript (authoritative method) | `manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md` |
| Research questions / contributions | `notes/research_questions.md`, `notes/research_contributions.md` |
| Experimental design (controls, replication) | `notes/methods/experimental_design.md` |
| Ensemble construction recipe | `notes/methods/scenario_design_methods.md` |
| Forcing-space parameterization (E_test) | `notes/methods/forcing_parameterization.md` |
| Objective definitions | `notes/methods/objective_definitions.md` |
| Framing-convention diagnostics | `notes/methods/framing_convention_diagnostics.md` |
| Terminology (controlled vocabulary) | `notes/terminology.md` |
| Literature hub + topic notes | `notes/literature/README.md`, `notes/literature/scenario_design.md` |
| Workflow / HPC operation | `../workflow/README.md`, `../workflow/envs/README.md` |
