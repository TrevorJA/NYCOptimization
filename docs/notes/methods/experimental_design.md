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
| `hazard_filling` *(proposed; registry key `hazard_filling_stationary`)* | LHS anchors in absolute, robust range-scaled hazard space (per-axis p1/p99 bounds), each snapped to the nearest unused member of an i.i.d. candidate pool; *N* = 100 realizations of *L* = 10 years | The contribution | Generalizes the one-dimensional flow stratification of Zatarain Salazar et al. (2017); moves the space-filling subsampling of Bonham et al. (2024) into the search ensemble |

The `historic` design cannot be matched in computational size and serves as a reference for prevailing practice rather than as a controlled comparison.

---

## The core contrast

| Contrast | Held fixed | Question |
|---|---|---|
| `fixed_probabilistic` → `hazard_filling` | generator, population law, *N*, *L* | Does hazard-space coverage change robustness relative to random sampling? |

The two matched designs share the same generator, population, *N*, and *L*. Only the rule that selects which realizations enter the search ensemble differs. This single controlled contrast is the whole of RQ1. Reframing the study around one contrast rather than a larger set of designs is deliberate: it makes a null result directly interpretable rather than a diffuse comparison across incommensurable constructions.

**The distributional-equivalence control (load-bearing).** The candidate pool that `hazard_filling` subsamples is drawn i.i.d. from the stationary generator. A uniform random size-*N* subset of an i.i.d. pool has exactly the joint law of *N* fresh i.i.d. draws. This is what makes `fixed_probabilistic` the exact statistical control for `hazard_filling`: the two designs then differ only in the selection rule, and any difference in re-evaluated robustness is attributable to that rule alone. The condition requires the pool to be sampled i.i.d. and not by Latin hypercube (a random subset of an LHS design is not i.i.d.), and it is enforced by an automated invariant in the code because no other component would fail if it were violated.

**The deliberate distribution shift.** `hazard_filling` fills the hazard space in absolute, robust range-scaled units (per-axis p1/p99 bounds; `scenario_design_methods.md` §4.3), which over-represents the severe (rare) hazard corners relative to their pool frequency. This is the genuine distribution shift the study tests. `fixed_probabilistic` presents the generator's own distribution to the search, `hazard_filling` presents a distribution shifted toward severe hazard conditions, and the held-out re-evaluation is the only point at which the two are compared.

---

The campaign comprises exactly the three designs above; other registered
constructions in `src/scenario_designs.py` are future-work material, and no
"hazard coverage beats input coverage" claim is made. The study isolates one
mechanism (the selection rule within a single stationary population) so that
the comparison rests on an exact within-population control.

---

## Controls for fair comparison

1. **Computational budget.** Both matched designs run at *N* = 100, *L* = 10 yr, which is 1,000 scenario-years per evaluation, at equal NFE (500,000 per search; geometry and SU accounting in `scenario_design_methods.md` §6, single-source constants in the `production` entry of `src/moea_config.py`). Because *N* and *L* are common, per-evaluation simulation cost, scenario-years, and wall-clock are identical, so equal-NFE and equal-scenario-years coincide. There is one budget condition and no confound between ensemble composition and search effort. The common *(N, L)* is required rather than convenient: if *L* differed, the selection rule would be confounded with record length. See `scenario_design_methods.md` §6 for why *N* = 100 is the smallest defensible fill and why long records are not viable at a fixed per-evaluation budget.

2. **Convergence reporting.** Within each design, search progress is monitored through the algorithm's runtime dynamics and reported per seed as a diagnostic. These internal quantities are never compared across designs, because objective values computed on different search ensembles are not commensurable and reference-set metrics scored against a pooled frontier are biased in a design-dependent way. Convergence assessment is not used as a stopping rule, so no design benefits from a tuned termination criterion, and both matched designs execute the identical NFE budget. Because the runtime archive records the approximation set at intermediate NFE levels, the final comparison can be recomputed at two or three earlier budgets at re-evaluation cost only, testing whether the ranking is an artifact of the chosen budget.

3. **Single point of comparison.** Cross-design comparison occurs only once, by re-evaluating every final Pareto-approximate set on the common held-out test ensemble and recomputing nondominated sets from the re-evaluated objective values for both designs alike.

4. **Seed-stream disjointness.** Each design, each draw, and the test ensemble generate from a namespaced seed domain, so no two designs and no design and the test ensemble ever share realizations. A collision between a search-side and a test-side seed domain is a hard error, which guards against testing on data the search has seen (Bonham et al. 2024).

---

## Replication

Two sources of variability are separated by design: the random construction of the ensemble and the stochasticity of the search algorithm. A **draw** is a scenario design's entire construction re-run from scratch with a fresh seed. For `fixed_probabilistic` a draw is a fresh i.i.d. sample. For `hazard_filling` a draw is a fresh candidate pool together with a fresh anchor plan. Re-drawing the pool within each draw is essential, because generating the pool is part of the hazard-filling construction and pinning it across draws would make hazard filling appear more stable than `fixed_probabilistic` as an artifact of the replication scheme rather than as a finding. A **seed** is an independent MOEA trial on a fixed draw.

Each matched design runs *K* = 3 draws with *S* = 2 seeds per draw, set against the compute allocation. The `historic` design has *K* = 1, because its composition variance is zero by construction, and is replicated across *S* seeds only.

The unit of analysis for the between-design comparison is the draw, and seeds within a draw are treated as pseudoreplicates. Draw- and seed-level results are reported transparently, so the design effect is read against the observed construction and search variability. Replicated draws separate ensemble-construction variance from search stochasticity, which is the feature that distinguishes the present comparison from Eker and Kwakkel (2018), who compared a single designed set against a single random set by counting solutions and could not separate a genuine design effect from the variance of a single draw.

---

## Evaluation

All Pareto-approximate sets are re-evaluated, with the trimmed model used in search (the policy-independent non-NYC releases are presimulated once per E_test realization and reused for every Pareto set), on one large held-out test ensemble E_test (N_θ = 1,000 LHS SOWs × R = 25 realizations × L_test = 50 yr; sizing derivation in `scenario_design_methods.md` §5.4), never used during any search and never the source of any search ensemble.

**E_test is the sole carrier of deep uncertainty and the measuring stick.** It is the largest ensemble in the study by a wide margin and is built to encompass a deeply uncertain climate-forcing envelope rather than to represent a probability distribution: a Latin hypercube over the full range of the CMIP6 harmonic forcing factors, with an envelope deliberately wider than any variation the search ensembles contain, crossed with many stochastic realizations per LHS point. Each LHS point is a state of the world, and its realizations sample natural variability within it. E_test is sampled by LHS, not i.i.d.; the i.i.d. rule applies only to the candidate pool, where it underwrites the distributional-equivalence control. E_test is never subsampled and is never a control, so it covers the deeply uncertain space rather than sampling it in proportion to a measure. Construction and sizing: `scenario_design_methods.md` §5.

**The re-evaluation is a generalization test.** The search ensembles are drawn from the unperturbed stationary generator, whereas E_test spans a forced climate envelope that no search ensemble contains. The re-evaluation therefore measures whether hazard-space coverage of the natural-variability manifold produces policies that generalize to conditions absent from search. This is a stronger test than re-evaluating on the search distribution, and it keeps the test instrument structurally distinct from both designs, which removes any favorable-instrument concern: neither matched design is constructed to resemble E_test.

Consequently no robustness number is an expectation. Under deep uncertainty there is no probability measure over the forcing space, so a satisficing fraction over E_test is a coverage-weighted count over a designed exploration. The cross-design comparison is commensurable because E_test is identical across both designs, not because it is probability-faithful.

The full re-evaluation matrix is persisted, every (solution × realization × objective) value in natural units plus each realization's SOW id, so any robustness metric can be scored offline without re-simulating, and both the SOW unit and the realization unit are available at no additional compute cost.

Design rankings are conditional on the test-ensemble design, since robustness is only defined relative to the conditions over which it is measured (McPhail et al. 2018; Quinn et al. 2020). The campaign uses one construction (Kirsch–Nowak over the wide DU box), so this conditioning is declared, not bounded. A structurally different second construction (a multi-site HMM annual generator, which varies interannual persistence) is registered as an optional variant; standing it up would let ranking stability across test-ensemble constructions be measured rather than assumed. That is a scope decision, not a technical blocker.

**Comparison metrics.** Two families. For RQ1 the primary measure is the multivariate Starr satisficing fraction on E_test; for RQ2 it is incumbent-relative regret (below). The pre-specified run-level endpoint is the maximum satisficing fraction attained by any policy in the run's re-evaluated set — the most robust policy the design can find — reported with the per-objective satisficing decomposition alongside so that a single scalar does not carry the whole comparison (a maximum over a set carries a mild upward bias that grows with set cardinality; this is disclosed). Secondary metrics are univariate satisficing, the coverage-weighted mean (Laplace, risk-neutral), maximin (risk-averse), and the signed improvement over the status-quo FFMP policy on the same E_test. Ranking agreement across these metrics is summarized by Kendall's τ_b over the design rankings (`objective_definitions.md` §3). Uncertainty on ensemble-level quantities is assessed at the SOW level, since realizations within a state of the world are not independent and precision is governed by N_θ.

**Incumbent-relative regret is the co-primary family, and it answers RQ2.** Its reference is the default 2017 FFMP policy evaluated *in the same state of the world* — a fixed, design-independent T1 reference that McPhail et al. (2018) §3.1 explicitly license and that no published water-resources study formalizes, and equivalently Savage regret on the action set the Decree parties actually face, `{retain the FFMP, adopt the candidate}`. Magnitudes are reported per objective in natural units and are never combined across objectives; the unit-free harm frequencies (per objective, per Decree party, and the joint no-harm frequency `Π_τ`) carry the cross-objective and cross-design summary. The tolerance `τ_i = k · ε_i` is swept rather than fixed, on the same reasoning that makes the satisficing criteria a sweep. Construction, exclusions, and the four-reference comparison table: `objective_definitions.md` §3.2b/§3.3.

Regret is not redundant with satisficing here. The satisficing criteria are fixed scalars anchored on the incumbent's *historic* attainment, whereas the regret bar is the incumbent's performance in each SOW and therefore moves with the forcing; where the fixed criterion saturates the domain criterion at zero for every policy, regret still separates them. No set-relative (best-in-set) regret, no perfect-foresight (Cohen-style) regret, no Herman R1 baseline-SOW regret, and no search-vs-test gap is computed.

**The hypothesis this pair exists to test.** Bartholomew & Kwakkel (2020) report both that search-phase robustness survives re-evaluation and that it is normally paid for elsewhere — the price of robustness (Bertsimas & Sim 2004) — and Giuliani & Castelletti (2016) predict a systematic penalty for a policy searched under one across-scenario measure and scored under another. `hazard_filling` is exactly that case. The pre-registered hypothesis is that it attains higher satisficing robustness *without* a higher incumbent-relative regret; the endpoint is read off the **same** policy on both axes, the unit of analysis remains the draw, and the non-inferiority margin (a `k` on the tolerance ladder and a tolerated drop in `Π_τ`) is declared before any result is inspected. Whether the price, if paid, falls in the benign futures is tested by decomposing both axes over terciles of the dominant forcing factor.

**The criterion sweep is retained.** Where a satisficing criterion is not fixed by a Decree or FFMP goalpost it is a convention, so each such criterion is swept over a grid and the question asked is whether the design difference is invariant rather than whether robustness values move (Quinn et al. 2020). This is the sense in which the comparison is reported as a difference that holds or fails to hold across the range of defensible criteria.

**Hazard-space coverage is method verification, not a comparison result.** Coverage statistics (centered L2-star discrepancy, minimum-spanning-tree edge statistics, snap-distance distribution on the normalized hazard coordinates) confirm that the selector administered the intervention at strength, meaning that the `hazard_filling` ensemble is compositionally shifted relative to `fixed_probabilistic`. Because the LHS + nearest-neighbour selector does not optimize a discrepancy objective, these are independent build-QC measurements, reported as method diagnostics rather than as a comparison result. Scenario discovery, where it is run, operates in the DU factor space of E_test after re-evaluation (`objective_definitions.md` §4) as an optional supporting analysis; no scenario discovery is performed in hazard space. The primary and only comparison endpoint is re-evaluated robustness on E_test.

The two matched designs depart differently from the scenario probabilities of their population: `hazard_filling` does so deliberately, and `fixed_probabilistic` reproduces them. Objective values computed during search therefore estimate different quantities across designs. This departure is the design choice under study, not an artifact, and the held-out re-evaluation is the common basis of comparison.

---

## Open questions

1. **Satisficing criterion values** — the centre of the threshold grid (Decree/FFMP anchors where they exist; elicited-convention defaults elsewhere) and the grid's span.
2. **Figure plan** for the results — not yet drafted.

Fixed sizing: E_test = 1,000 LHS SOWs × R = 25 × L_test = 50 yr; selection axes m = 6 with N = 100 and P = 10⁶ (`scenario_design_methods.md` §3.3/§5.4/§6; `hazard_selector_diagnostics.md` §5); K = 3 draws × S = 2 seeds per matched design. Framing conventions (failure-week counts, flood unit operator = MEAN, NJ delivery active, 0.99 satisfaction factor) are measured verdicts in `framing_convention_diagnostics.md` §0b.
