# NYC Reservoir Re-Optimization: Project Summary

*Entry point for new readers. Last updated 2026-07-15. Details live in `docs/notes/`;
this page states what the study is, what is decided, and what is still open. The
manuscript at `docs/manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md` is the
authoritative description of the method. Where a note and the code disagree, the code
is the record of what exists and the note is a proposal.*

---

## The study in one paragraph

We re-optimize the operating rules of the four NYC reservoirs in the Delaware River
Basin (the FFMP rule structure, 24 decision variables) with the multi-master Borg MOEA
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
| `hazard_filling` | LHS anchors in absolute, range-scaled hazard space, snapped to the nearest member of its own i.i.d. candidate pool | **Proposed method** |

**The controlled contrast.** `fixed_probabilistic` → `hazard_filling` holds the
generator, population law, N, and L fixed and varies *only the selection rule*: does
hazard coverage beat random sampling? Because the candidate pool is sampled i.i.d., a
uniform random size-N subset of it has exactly the law of N fresh i.i.d. draws, which
makes `fixed_probabilistic` the *exact statistical control* for `hazard_filling`. This
is the Eker & Kwakkel (2018) null benchmark (diversity-based selection did not beat
random selection) raised to hazard space, on a real system, with a replication scheme
that separates ensemble-construction variance from search variance.

**Why absolute hazard space.** The selector fills the hazard space in absolute,
range-scaled magnitude units rather than empirical-CDF/rank units. Because the hazard
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
2. **Hazard metrics + screening** — drought/low-flow/high-flow indices per sequence; a
   redundancy screen selects 3–4 low-collinearity axes.
3. **Selection (hazard filling only)** — Latin hypercube anchors in absolute, range-scaled
   hazard space, snapped to the nearest unused pool member. The snap is intrinsic: hazard
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

Seven active objectives (NYC delivery reliability + CVaR₉₀ deficit, Montague reliability
+ CVaR₉₀ deficit, Trenton reliability, downstream minor-flood days, NYC storage 5th
percentile), defined in `src/objectives.py` and documented in
`notes/methods/objective_definitions.md`. During search, each objective's per-realization
temporal metric is collapsed across realizations by a two-layer annual-unit scheme
(annual metric per realization × water-year unit; a per-objective unit operator over the
pooled unit-years). Thresholds and the annual-unit epsilons are placeholders pending the
sensitivity experiments.

## Comparison controls

- **Budget**: both matched designs run at N = 100, L = 10 yr — 1,000 scenario-years per
  evaluation — at equal NFE, so per-evaluation cost, warm-up, scenario-years, and
  wall-clock are identical and equal-NFE coincides with equal-scenario-years. The common
  (N, L) is required: if L differed, the selection rule would be confounded with record
  length.
- **The i.i.d. pool is load-bearing.** A uniform random size-N subset of an i.i.d. pool
  has exactly the law of N fresh i.i.d. draws, which is what makes `fixed_probabilistic`
  the exact control for `hazard_filling`. A structured (e.g. LHS) pool would void the
  control. Enforced by an invariant test.
- **Seed-stream disjointness**: the candidate pool and the test ensemble generate from
  namespaced seed domains, so no design and the test ensemble ever share realizations.
- **Replication**: K = 3 ensemble draws × S = 2 MOEA seeds per matched design (targets,
  revisable from a pilot). A draw is the design's construction re-run from scratch with a
  fresh seed, and is the unit of analysis (mixed-effects framing). `historic` has K = 1.
- **Single comparison point**: cross-design metrics computed only on held-out
  re-evaluation, with pooled and leave-one-out reference sets.

See `notes/methods/experimental_design.md`.

## The test ensemble (E_test)

E_test is the **only carrier of deep uncertainty** in the study and the **largest
ensemble by a wide margin**. It is a **Latin hypercube over the full range of the
deeply-uncertain climate-forcing factors** (the CMIP6 harmonic hypercube), with **many
realizations per LHS point** — each LHS point is a state of the world, and its
realizations sample natural variability within it. Because the search ensembles are drawn
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
limitation, and a structurally different second construction (multi-site HMM) is
registered as an optional sensitivity. **The full (solution × realization × objective)
matrix is persisted** in natural units with each realization's SOW id, so any robustness
metric — at the SOW unit or the realization unit — is scored offline without
re-simulating.

## Comparison metrics

The primary endpoint is the re-evaluated **multivariate Starr satisficing fraction** of
the policies a design produces; the run-level scalar is the maximum satisficing fraction
attained in the run's re-evaluated set (with leave-one-out reference-set correction and
robustness-space hypervolume as bounding co-reports). Secondary metrics are univariate
satisficing, the coverage-weighted mean (Laplace), maximin, and signed
improvement-over-status-quo (a fixed-reference, design-independent quantity, and the only
regret-type metric used — **no set-relative or perfect-foresight regret is computed**).
Ranking agreement across metrics is summarized by Kendall's τ_b. A **criterion sweep**
reports whether the design difference holds across the range of defensible satisficing
thresholds rather than at one arbitrary point. Hazard-space coverage statistics are
reported only as method verification that the selector administers the intended treatment
at strength, never as a comparison result.

## Status

**Working now:** end-to-end smoke runs; the stationary designs' generation and single-
realization regeneration verified; chunked-pool machinery implemented; the Anvil
packing/scaling experiment and the ensemble cost-surface experiment complete (measured
campaign cost: 173.8 s/eval trimmed, full model 1.16×, ~33,300 SU per 500k-NFE search).

**Decided:** the three designs above; a single stationary search population with deep
uncertainty carried only in E_test; N = 100, L = 10 yr at equal NFE; 500k NFE target
(revisable once initial runs reveal convergence); K = 3 draws × S = 2 seeds; absolute
range-scaled hazard-space selection; comparison metrics = multivariate Starr satisficing
(primary) with Laplace, maximin, and signed improvement-over-status-quo as anchors, **no
set-relative or perfect-foresight regret**; search aggregation = two-layer annual-unit
scheme; forcing space retains historical persistence (claims scoped accordingly).

**Total Anvil allocation = 750,000 SU.** The full campaign (two matched designs × K = 3 ×
S = 2 at 500k NFE, plus the cheap `historic` reference and re-evaluation) is on the order
of 415,000 SU, leaving reserve for the RQ3 variable-resolution sweep, any additional
draws a power calculation indicates, and the optional second E_test.

**Not yet in place (gates the RQ1 campaign):** the production-scale candidate pool; the
held-out test ensemble E_test at production size; the multi-draw (K > 1) generation; the
final satisficing thresholds and epsilons; the pilot minimum-detectable-effect
calculation that fixes K.

**Open decisions:** E_test sizing (N_theta × R × L_test, against the SU allocation) and
whether to stand up the optional second E_test construction; the hazard-axis set (from
the redundancy screen on the production pool); the pool size P; the flood unit operator
(mean vs P99) and failure-criterion values (from the sensitivity experiment); the
scenario design under which the RQ3 variable-resolution sweep is run. See
`notes/methods/experimental_design.md` for the open-questions list.

## Document index

| Topic | Doc |
|---|---|
| Manuscript (authoritative method) | `manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md` |
| Research questions / contributions | `notes/research_questions.md`, `notes/research_contributions.md` |
| Experimental design (controls, replication) | `notes/methods/experimental_design.md` |
| Ensemble construction recipe | `notes/methods/scenario_design_methods.md` |
| Forcing-space parameterization (E_test) | `notes/methods/forcing_parameterization.md` |
| Objective definitions | `notes/methods/objective_definitions.md` |
| Objective sensitivity experiments | `notes/methods/objective_sensitivity_experiment.md`, `notes/methods/ensemble_objective_sensitivity_experiment.md` |
| Terminology (controlled vocabulary) | `notes/terminology.md` |
| Literature hub + topic notes | `notes/literature/README.md`, `notes/literature/scenario_design.md` |
| Workflow / HPC operation | `../workflow/README.md`, `../workflow/envs/README.md` |
