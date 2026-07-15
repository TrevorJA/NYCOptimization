# Experimental Design: Comparison of Scenario Designs for Optimization

*Terminology per `docs/notes/terminology.md`. Construction recipes in `scenario_design_methods.md`. Objective formulations in `objective_definitions.md`. Citations per the literature notes indexed by `docs/notes/literature/scenario_design.md`. The manuscript draft (`docs/manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md`) is the authoritative specification; this note tracks it.*

---

## Purpose

The study tests whether constructing the search ensemble by hazard-space coverage, rather than by independent sampling from the same stochastic generator, changes the robustness of the resulting Pareto-approximate policies under held-out, deeply uncertain re-evaluation. This is RQ1, the core question.

Each scenario design is constructed by its own published recipe with its own seed stream. No design is derived from another's data. An independent optimization is performed with each design, and every resulting Pareto-approximate set is re-evaluated on a single large held-out test ensemble E_test. Re-evaluated robustness of the resulting solutions is the sole basis of cross-design comparison.

---

## Scenario designs compared

Three designs, all drawn from a single stationary population (the Kirsch–Nowak generator fitted to the reconstructed observed record and run without climate perturbation). Deep uncertainty enters the study only through the held-out test ensemble (see Evaluation), not through any search ensemble.

| Design | Construction | Role | Representative literature |
|---|---|---|---|
| `historic` | The observed record simulated as one continuous trace, one realization of about 77 years, *K* = 1 | Prevailing-practice reference. Unmatched in ensemble size and budget, reported as a reference rather than entered into the controlled contrast | Giuliani et al. (2016); Herman et al. (2020) |
| `fixed_probabilistic` | *N* = 100 realizations of *L* = 10 years drawn i.i.d. from the stationary generator, frozen for the entire search | The discipline's random-sampling default. The exact statistical control for `hazard_filling` | Quinn et al. (2017); Zatarain Salazar et al. (2017) |
| `hazard_filling` *(proposed)* | LHS anchors in absolute, range-scaled hazard space, each snapped to the nearest unused member of an i.i.d. candidate pool; *N* = 100 realizations of *L* = 10 years | The contribution | Generalizes the one-dimensional flow stratification of Zatarain Salazar et al. (2017); moves the space-filling subsampling of Bonham et al. (2024) into the search ensemble |

The `historic` design cannot be matched in computational size and serves as a reference for prevailing practice rather than as a controlled comparison.

---

## The core contrast

| Contrast | Held fixed | Question |
|---|---|---|
| `fixed_probabilistic` → `hazard_filling` | generator, population law, *N*, *L* | Does hazard-space coverage change robustness relative to random sampling? |

The two matched designs share the same generator, population, *N*, and *L*. Only the rule that selects which realizations enter the search ensemble differs. This single controlled contrast is the whole of RQ1. Reframing the study around one contrast rather than a larger set of designs is deliberate: it makes a null result interpretable as a well-powered null rather than as a diffuse comparison across incommensurable constructions.

**The distributional-equivalence control (load-bearing).** The candidate pool that `hazard_filling` subsamples is drawn i.i.d. from the stationary generator. A uniform random size-*N* subset of an i.i.d. pool has exactly the joint law of *N* fresh i.i.d. draws. This is what makes `fixed_probabilistic` the exact statistical control for `hazard_filling`: the two designs then differ only in the selection rule, and any difference in re-evaluated robustness is attributable to that rule alone. The condition requires the pool to be sampled i.i.d. and not by Latin hypercube (a random subset of an LHS design is not i.i.d.), and it is enforced by an automated invariant in the code because no other component would fail if it were violated.

**The deliberate distribution shift.** `hazard_filling` fills the hazard space in absolute, range-scaled units, which over-represents the severe (rare) hazard corners relative to their pool frequency. This is the genuine distribution shift the study tests. `fixed_probabilistic` presents the generator's own distribution to the search, `hazard_filling` presents a distribution shifted toward severe hazard conditions, and the held-out re-evaluation is the only point at which the two are compared.

---

## Designs not in the campaign

Earlier planning considered additional constructions (a per-evaluation resampled ensemble, an input-stratified LHS design over forcing parameters, and a DU-forced hazard-filling variant) together with a "hazard coverage beats input coverage" claim. These are not part of the current campaign and are not presented as designs here. If revisited at all, they belong in future work, and the "hazard coverage beats input coverage" claim is not made. The current study isolates one mechanism (the selection rule within a single stationary population) so that the comparison rests on an exact within-population control.

---

## Controls for fair comparison

1. **Computational budget.** Both matched designs run at *N* = 100, *L* = 10 yr, which is 1,000 scenario-years per evaluation, at equal NFE. Because *N* and *L* are common, per-evaluation simulation cost, warm-up, scenario-years, and wall-clock are identical, so equal-NFE and equal-scenario-years coincide. There is one budget condition and no confound between ensemble composition and search effort. The common *(N, L)* is required rather than convenient: if *L* differed, the selection rule would be confounded with record length. See `scenario_design_methods.md` §6 for why *N* = 100 is the smallest defensible fill and why long records are not viable at a fixed per-evaluation budget.

2. **Convergence reporting.** Within each design, search progress is monitored through the algorithm's runtime dynamics and reported per seed as a diagnostic. These internal quantities are never compared across designs, because objective values computed on different search ensembles are not commensurable and reference-set metrics scored against a pooled frontier are biased in a design-dependent way. Convergence assessment is not used as a stopping rule, so no design benefits from a tuned termination criterion, and both matched designs execute the identical NFE budget. Because the runtime archive records the approximation set at intermediate NFE levels, the final comparison can be recomputed at two or three earlier budgets at re-evaluation cost only, testing whether the ranking is an artifact of the chosen budget.

3. **Single point of comparison.** Cross-design comparison occurs only once, by re-evaluating every final Pareto-approximate set on the common held-out test ensemble and recomputing nondominated sets from the re-evaluated objective values for both designs alike.

4. **Seed-stream disjointness.** Each design, each draw, and the test ensemble generate from a namespaced seed domain, so no two designs and no design and the test ensemble ever share realizations. A collision between a search-side and a test-side seed domain is a hard error, which guards against testing on data the search has seen (Bonham et al. 2024).

---

## Replication

Two sources of variability are separated by design: the random construction of the ensemble and the stochasticity of the search algorithm. A **draw** is a scenario design's entire construction re-run from scratch with a fresh seed. For `fixed_probabilistic` a draw is a fresh i.i.d. sample. For `hazard_filling` a draw is a fresh candidate pool together with a fresh anchor plan. Re-drawing the pool within each draw is essential, because generating the pool is part of the hazard-filling construction and pinning it across draws would make hazard filling appear more stable than `fixed_probabilistic` as an artifact of the replication scheme rather than as a finding. A **seed** is an independent MOEA trial on a fixed draw.

Each matched design runs *K* = 3 draws with *S* = 2 seeds per draw. These are targets, revisable from a pilot minimum-detectable-effect calculation on the primary endpoint and from the estimated seed-versus-draw variance ratio. The `historic` design has *K* = 1, because its composition variance is zero by construction, and is replicated across *S* seeds only.

The unit of analysis for the between-design comparison is the draw, and seeds within a draw are treated as pseudoreplicates. The comparison uses a mixed-effects structure in which the design is a fixed effect, the draw is a random effect nested within design, and the seed is a random effect nested within draw, so the effective sample size per design is approximately *K* rather than *K·S*. This structure separates ensemble-construction variance from search stochasticity, which is the feature that distinguishes the present comparison from Eker and Kwakkel (2018), who compared a single designed set against a single random set by counting solutions and could not separate a genuine design effect from the variance of a single draw.

---

## Evaluation

All Pareto-approximate sets are re-evaluated, with the full untrimmed model, on one large held-out test ensemble E_test, never used during any search and never the source of any search ensemble.

**E_test is the sole carrier of deep uncertainty and the measuring stick.** It is the largest ensemble in the study by a wide margin and is built to encompass a deeply uncertain climate-forcing envelope rather than to represent a probability distribution: a Latin hypercube over the full range of the CMIP6 harmonic forcing factors, with an envelope deliberately wider than any variation the search ensembles contain, crossed with many stochastic realizations per LHS point. Each LHS point is a state of the world, and its realizations sample natural variability within it. E_test is sampled by LHS, not i.i.d.; the i.i.d. rule applies only to the candidate pool, where it underwrites the distributional-equivalence control. E_test is never subsampled and is never a control, so it covers the deeply uncertain space rather than sampling it in proportion to a measure. Construction and sizing: `scenario_design_methods.md` §5.

**The re-evaluation is a generalization test.** The search ensembles are drawn from the unperturbed stationary generator, whereas E_test spans a forced climate envelope that no search ensemble contains. The re-evaluation therefore measures whether hazard-space coverage of the natural-variability manifold produces policies that generalize to conditions absent from search. This is a stronger test than re-evaluating on the search distribution, and it keeps the test instrument structurally distinct from both designs, which removes any favorable-instrument concern: neither matched design is constructed to resemble E_test.

Consequently no robustness number is an expectation. Under deep uncertainty there is no probability measure over the forcing space, so a satisficing fraction over E_test is a coverage-weighted count over a designed exploration. The cross-design comparison is commensurable because E_test is identical across both designs, not because it is probability-faithful.

The full re-evaluation matrix is persisted, every (solution × realization × objective) value in natural units plus each realization's SOW id, so any robustness metric can be scored offline without re-simulating, and both the SOW unit and the realization unit are available at no additional compute cost.

Design rankings are conditional on the test-ensemble design, since robustness is only defined relative to the conditions over which it is measured (McPhail et al. 2018; Quinn et al. 2020). The campaign uses one construction (Kirsch–Nowak over the wide DU box), so this conditioning is declared, not bounded. A structurally different second construction (a multi-site HMM annual generator, which varies interannual persistence) is registered as an optional variant; standing it up would let ranking stability across test-ensemble constructions be measured rather than assumed. That is a scope decision, not a technical blocker.

**Comparison metrics.** The primary comparison measure is the multivariate Starr satisficing fraction on E_test. The pre-specified run-level endpoint is the maximum satisficing fraction attained by any policy in the run's re-evaluated set, reported with the leave-one-out reference-set correction and the robustness-space hypervolume of the run's set as bounding co-reports (the maximum over a set carries a mild upward bias that grows with set cardinality, which those two quantities bound), and with the per-objective satisficing decomposition alongside so that a single scalar does not carry the whole comparison. Secondary metrics are univariate satisficing, the coverage-weighted mean (Laplace, risk-neutral), maximin (risk-averse), and the signed improvement over the status-quo FFMP policy on the same E_test. Ranking agreement across these metrics is summarized by Kendall's τ_b over the design rankings (`objective_definitions.md` §3). Standard errors are clustered by forcing parameter (θ), since realizations within a state of the world are not independent.

The signed improvement over the status quo is the only regret-type quantity computed. It is a fixed-reference, design-independent regret in the McPhail et al. (2018) taxonomy: the reference is the default 2017 FFMP policy simulated on the same E_test, and it does not move when a design is added or dropped. No set-relative (best-in-set) regret is computed, because it is design-coupled (removing one design changes every other design's score) and is the slowest-converging robustness family in Bonham et al. (2024). No perfect-foresight (Cohen-style baseline) regret is computed, because it requires one perfect-foresight optimization per scenario and does not scale to a candidate pool. No search-minus-test overfitting gap is reported, because under the shifted `hazard_filling` measure the in-sample term is not an expectation under E_test's measure, so the difference is an artifact of the distribution change rather than an overfitting quantity.

**The criterion sweep is retained.** Where a satisficing criterion is not fixed by a Decree or FFMP goalpost it is a convention, so each such criterion is swept over a grid and the question asked is whether the design difference is invariant rather than whether robustness values move (Quinn et al. 2020). This is the sense in which the comparison is reported as a difference that holds or fails to hold across the range of defensible criteria.

**Hazard-space coverage is method verification, not a comparison result.** Coverage statistics (centered L2-star discrepancy, minimum-spanning-tree edge statistics, snap-distance distribution on the normalized hazard coordinates) confirm that the selector administered the intervention at strength, meaning that the `hazard_filling` ensemble is compositionally shifted relative to `fixed_probabilistic`. Because the LHS + nearest-neighbour selector does not optimize a discrepancy objective, these are independent build-QC measurements, reported as method diagnostics rather than as a comparison result. Hazard-space scenario discovery (whether a design's policies fail on E_test in the hazard region the design under-covered) is demoted to an optional supporting analysis of the coverage-to-robustness mechanism, not a primary or falsification result. The primary and only comparison endpoint is re-evaluated robustness on E_test.

The two matched designs depart differently from the scenario probabilities of their population: `hazard_filling` does so deliberately, and `fixed_probabilistic` reproduces them. Objective values computed during search therefore estimate different quantities across designs. This departure is the design choice under study, not an artifact, and the held-out re-evaluation is the common basis of comparison.

---

## Open questions

1. **Test-ensemble design** — which deeply uncertain factors it spans, its size, and how realizations are allocated across them. Size matters doubly because nondominated sets are recomputed from re-evaluated values.
2. **Hazard-metric axes** — pending the redundancy screen on the production candidate pool. The retained *m* determines whether *N* = 100 fills adequately.
3. **Sizing not yet fixed** — the candidate pool size *P* and the draw and seed counts *K*, *S* against the compute budget (a pilot minimum-detectable-effect calculation precedes fixing *K*).
4. **The flood-days unit operator** (mean vs P99) and the frequency objectives' annual failure criteria, set by the ensemble objective-sensitivity experiment.
5. **NFE budget** — the 500,000-NFE target may be lowered once initial searches reveal convergence behavior at the campaign ensemble size.
6. **Figure plan** for the results — not yet drafted.
