# Experimental Design: Comparison of Scenario Designs for Optimization

*Terminology per `docs/notes/terminology.md`. Construction recipes in `scenario_design_methods.md`. Objective formulations in `objective_definitions.md`. Citations per the literature notes indexed by `docs/notes/literature/scenario_design.md`.*

---

## Purpose

The study tests whether the composition of the streamflow scenario set used during many-objective search affects the quality and robustness of the resulting Pareto-approximate policies.

**Each scenario design is constructed by its own published recipe, with its own seed stream.** No design is derived from another's data, and none is subsampled from a shared pool — that would misdescribe every method it claims to represent, since no published study builds its search ensemble that way. An independent optimization is performed with each design, and every resulting Pareto-approximate set is re-evaluated on a single large held-out test ensemble. Re-evaluated performance is the sole basis of cross-design comparison.

---

## Scenario designs compared

Six designs across two **populations** — the law from which a design's realizations are drawn. Population and selection rule are independent choices; the comparison holds one fixed while varying the other.

### Stationary population (Kirsch–Nowak fit to the historic record)

| Design | Construction | Representative literature |
|---|---|---|
| `historic` | The observed record, simulated as one continuous trace | Giuliani et al. (2016); Herman et al. (2020) |
| `fixed_probabilistic` | *N* × *L* realizations generated i.i.d. from the stationary generator; frozen across the search | Quinn et al. (2017); Zatarain Salazar et al. (2017) |
| `resampled_probabilistic` | Own stationary pool; *N* realizations redrawn at every function evaluation | Brodeur et al. (2020); Trindade et al. (2017) and Gold et al. (2023) for the re-randomization principle only |
| `hazard_filling_stationary` *(proposed)* | Space-filling subsample, in hazard space, of its own stationary candidate pool | Contribution |

### DU-forced population (CMIP6 harmonic forcing hypercube)

| Design | Construction | Representative literature |
|---|---|---|
| `input_stratified` | Latin hypercube over the harmonic forcing parameters; *R* realizations **generated** per design point | Quinn et al. (2020); Bartholomew & Kwakkel (2020); Eker & Kwakkel (2018); Watson & Kasprzyk (2017) |
| `hazard_filling_du` *(proposed)* | Space-filling subsample, in hazard space, of its own DU candidate pool | Contribution |

The historical record cannot be matched in computational size and serves as a reference for prevailing practice rather than a controlled comparison.

---

## Rationale

**Why hazard-filling runs in both populations.** Whichever single population were chosen, one of the two key comparisons would be confounded. A stationary-only pool gives an exact control against `fixed_probabilistic` but leaves `input_stratified` with no input space to stratify — its forcing parameters are fixed by the historic fit. A DU-only pool gives an exact control against `input_stratified` but confounds `fixed_probabilistic`, which is a different population. Running the method in both dissolves the problem and invents no design to serve a control.

The literature supports both, because **hazard-space design does not require a deeply-uncertain input space**. Zatarain Salazar et al. (2017) subsamples a *stationary* 10,000-trace Kirsch–Nowak pool by a realized-flow metric and uses the subsample during search. Herman et al. (2016) amplifies drought severity by a stationary weighted bootstrap. Zaniolo et al. (2023) and FIND (Zaniolo et al. 2024) control drought frequency, intensity, and duration in SSI space with no climate input space at all. Input stratification, by contrast, cannot exist without one.

**Historical record.** Optimization over a single observed record remains common in applied reservoir studies and anchors the comparison to prevailing practice.

**Fixed probabilistic ensemble.** Random sampling from a stationary stochastic generator is the standard, well-characterized approach throughout the regional water-supply literature. It is the baseline against which structured alternatives are judged — and, because a uniform random subset of an i.i.d. pool has exactly the law of an i.i.d. sample of the same size, it is also the *exact statistical control* for `hazard_filling_stationary`.

**Resampled probabilistic ensemble.** Re-randomizing the scenario set at every function evaluation tests whether *freezing* the search ensemble causes overfitting, separately from the effect of which scenarios are chosen. Because objective values then vary across evaluations of the same policy, comparisons involving this design rely entirely on the final re-evaluation.

**Input-stratified ensemble.** Stratifying scenarios across deeply uncertain generator parameters is the most common recent approach in the decision-making-under-deep-uncertainty literature. Contrasting it with `hazard_filling_du` isolates the central claim. The hypothesis, supported by Quinn et al. (2020) and Guo et al. (2018), is that distinct parameter sets produce hydrologically redundant realizations, so stratification applied in hazard space yields better coverage of the conditions that stress the system.

**Hazard-filling ensembles.** The proposed method. Scenarios are selected from a candidate pool so that their hazard-metric coordinates are approximately uniform and well separated. Hazard-filling is the only design that needs a pool, and this is intrinsic rather than incidental: hazard coordinates are emergent properties of a realized sequence, so a hazard-space design must *select from* a pool, whereas an input-space design *generates to* its design points.

**Deliberate exclusions.** Adaptive scenario selection (Giudici et al. 2020) modifies the optimization algorithm rather than the ensemble. One-dimensional stratification on a single flow statistic (Zatarain Salazar et al. 2017) is the nearest published precedent of the proposed method and is treated as the limiting case that the multi-dimensional design generalizes, not as a separate design. Each exclusion is acknowledged in the discussion.

---

## Controlled contrasts

| Contrast | Held fixed | Question |
|---|---|---|
| `fixed_probabilistic` → `hazard_filling_stationary` | generator, population law, *N*, *L* | Does hazard coverage beat random sampling? |
| `input_stratified` → `hazard_filling_du` | forcing space, *N*, *L* | Does hazard coverage beat **input** coverage? *(the central claim)* |
| `hazard_filling_stationary` → `hazard_filling_du` | selection rule, *N*, *L* | What does the DU forcing space add? |

Each contrast varies exactly one thing, and every claim has an exact within-population control.

---

## Controls for fair comparison

1. **Computational budget.** All matched designs run at *N* = 100, *L* = 10 yr — 1,000 scenario-years per evaluation — at equal NFE. Because *N* and *L* are common, per-evaluation simulation cost, warm-up, scenario-years, and wall-clock are *identical*, so equal-NFE and equal-scenario-years coincide. There is one budget condition, not two arms, and no confound between ensemble composition and search effort. The common *(N, L)* is required rather than convenient: if *L* differed across designs, the selection rule would be confounded with record length. See `scenario_design_methods.md` §6 for why *N* = 100 is the smallest defensible fill and why long records are not viable at a fixed per-evaluation budget.

2. **Convergence reporting.** Within each design, search progress is monitored through the algorithm's runtime dynamics and reported per seed as a diagnostic. These internal quantities are never compared across designs, since objective values computed on different search ensembles — or on resampled draws — are not commensurable. Convergence assessment is not a stopping rule, so no design benefits from a tuned termination criterion.

3. **Single point of comparison.** Cross-design quality comparison occurs only once, by re-evaluating every final Pareto-approximate set on the common held-out test ensemble and recomputing nondominated sets from the re-evaluated objective values for all designs alike.

4. **Seed-stream disjointness.** Each design, each draw, and the test ensemble generate from a namespaced seed domain, so no two designs and no design and the test ensemble ever share realizations. This matters now that every design *generates* rather than selecting indices from shared data.

---

## Replication

Two sources of variability are distinguished: the random construction of the ensemble and the stochasticity of the search algorithm. A **draw** is the design's construction re-run from scratch with a fresh seed — one definition for every design. For `fixed_probabilistic` a draw is a fresh i.i.d. sample; for `input_stratified`, a fresh LHS design; for the hazard-filling designs, a fresh selector anchor plan (the selector is deterministic given its anchor seed, so *K* draws measure anchor-placement variance — the design's construction variance); for `resampled_probabilistic`, a fresh pool.

`historic` has *K* = 1: composition variance is zero by construction. Every other design has *K* draws × *S* seeds.

The unit of analysis for between-design tests is the **draw**; seeds within a draw are pseudoreplicates, so effective *n* ≈ *K*, not *KS*. Model: outcome ~ design (fixed) + draw(design) (random) + seed(draw) (random). Target *K* = 10, *S* = 2–3 — more draws, fewer seeds. Because draws are now independent *generations* rather than re-indexings of shared data, *K* must be fixed before generation.

---

## Evaluation

All Pareto-approximate sets are re-evaluated on one large held-out test ensemble, never used during any search and never the source of any search ensemble.

**E_test is the largest ensemble in the study by a wide margin, and is built to be maximally uncertainty-encompassing**: a Latin hypercube over the *full* range of the deeply-uncertain forcing factors, with many realizations per LHS point (each LHS point is a **state of the world**; its realizations sample natural variability within it). It is sampled by LHS, not i.i.d. — the i.i.d. rule applies only to the candidate pools, where it underwrites the distributional-equivalence control; E_test is never subsampled and is never a control, so it should *cover* the deeply-uncertain space rather than sample it in proportion to a measure. Construction and sizing: `scenario_design_methods.md` §5.

Consequently **no robustness number is an expectation.** Under deep uncertainty there is no probability measure over the forcing space, so a satisficing fraction over E_test is a coverage-weighted count over a *designed exploration*. The cross-design comparison is commensurable because E_test is **identical across all designs**, not because it is probability-faithful.

**The full re-evaluation matrix is persisted** — every (solution × realization × objective) value in natural units, plus each realization's SOW id — so any robustness metric can be scored offline without re-simulating, and both the SOW unit and the realization unit are available at no additional compute cost.

Design rankings are conditional on the test-ensemble design, since robustness is only defined relative to the conditions over which it is measured (McPhail et al. 2018; Quinn et al. 2020). The campaign uses **one** construction (Kirsch–Nowak over the wide DU box), so this conditioning is **declared, not bounded** — a stated limitation. A structurally different second construction (multi-site HMM) is registered as an optional variant; standing it up would let ranking stability across test-ensemble constructions be measured rather than assumed, which is Quinn et al. (2020)'s recommendation. That is a scope decision, not a technical blocker.

The primary comparison measure is the **multivariate Starr satisficing fraction** on the test ensemble, with **Laplace (mean)**, **maximin**, and **signed improvement over the status-quo FFMP policy** as secondary anchors, and ranking agreement across them summarized by Kendall's τ_b over the *design* rankings (`objective_definitions.md` §3). **No regret is computed**: best-in-set regret is design-coupled (dropping one design changes every other design's score) and Cohen-style baseline regret would require a perfect-foresight optimization per scenario. Standard errors are clustered by forcing parameter (θ), since realizations within a state of the world are not independent.

Following McPhail et al. (2020) — scenario composition moves robustness *values* more than *rankings* — the comparison also reports whether the designs produce materially different **policies** (decision-variable and simulated-behavior differences among selected compromise solutions), not only different indicator scores. Because a robustness scalar can be stable, optimizable and still perverse, raw performance distributions are co-reported alongside every robustness number.

Two analyses carry the argument: a **satisficing-threshold sweep** (rank agreement across designs degrades as the criterion tightens — Quinn et al. 2020 — so a single threshold could manufacture or hide the entire effect), and **scenario discovery in hazard space** as the mechanism test (a design's policies should fail in the hazard region it under-covered).

Several designs depart from the scenario probabilities of their population. Hazard-filling does so deliberately; `input_stratified` and `historic` do so by construction. Objective values computed during search therefore estimate different quantities across designs. This departure is the design choice under study, not an artifact, and the held-out re-evaluation is the common basis of comparison.

---

## Open questions

1. **Test-ensemble design** — which deeply uncertain factors it spans, its size, and how realizations are allocated across them. Size matters doubly because nondominated sets are recomputed from re-evaluated values.
2. **Hazard-metric axes** — pending the redundancy screen on the production candidate pools. The retained *m* determines whether *N* = 100 fills adequately.
3. **Sizing not yet fixed** — the *N*<sub>θ</sub>/*R* split for `input_stratified`, the candidate pool sizes *P*, and the draw and seed counts *K*, *S* against the compute budget.
4. **The flood-days unit operator** (mean vs P99) and the frequency objectives' annual failure criteria, set by the ensemble objective-sensitivity experiment.
5. **Figure plan** for the results — not yet drafted.
