# Experimental Design: Comparison of Scenario Designs for Optimization

*Last updated: 2026-06-11 (rev. 5). Working draft. Terminology per `docs/notes/terminology.md`. Citations per the literature notes indexed by `docs/notes/literature/scenario_design.md`. Objective formulations, hazard-metric axes, ensemble sizes, scenario length, and the test-ensemble design are intentionally undecided and are listed as open questions at the end.*

---

## Purpose

The study tests whether the composition of the streamflow scenario set used during many-objective search affects the quality and robustness of the resulting Pareto-approximate policies. Each scenario design below (all but the historical record) constructs an evaluation ensemble from the same master ensemble, an independent optimization is performed with each, and every resulting Pareto-approximate set is re-evaluated on a single large held-out test ensemble. Re-evaluated performance is the sole basis of cross-design comparison.

Each design represents one family from the literature taxonomy, chosen to reflect the most common or most recent practice in that family.

## Scenario designs compared

| Design | Construction | Representative literature |
|---|---|---|
| Historical record | The observed record, simulated as one continuous trace | Giuliani et al. (2016) |
| Fixed probabilistic ensemble | Scenarios drawn once, at random, from the master ensemble | Kleywegt et al. (2002); Herman et al. (2014) |
| Fixed probabilistic ensemble, long records | A few multi-decadal synthetic records from the same generator parameterizations, at equal total simulated years | Quinn et al. (2017); standard stochastic practice |
| Resampled probabilistic ensemble | Scenarios redrawn at random from the master ensemble at every function evaluation | following the per-evaluation reshuffling of Trindade et al. (2017) |
| Input-stratified ensemble | Latin hypercube sample over generator parameters, one realization per parameter set | LHS over generator parameters as in re-evaluation practice (Quinn et al. 2018), transposed into search; fixed in-search LHS set per Bartholomew & Kwakkel (2020) |
| Hazard-filling ensemble (proposed) | Space-filling subsample of the master ensemble in hazard-metric space | Minasny & McBratney (2006); Morris & Mitchell (1995) |

The historical record cannot be matched in computational size and serves as a reference for prevailing practice rather than a controlled comparison.

## Rationale

**Historical record.** Optimization over a single observed record remains common in applied reservoir studies (see Table 1 of the companion summary tables) and anchors the comparison to prevailing practice.

**Fixed probabilistic ensemble.** Random sampling from an assumed distribution is the standard well-characterized approach, used throughout the regional water supply literature and formalized as the sample average approximation in stochastic programming. It is the baseline against which structured alternatives are judged.

**Fixed probabilistic ensemble, long records.** The literature predominantly evaluates policies on few multi-decadal records. Comparing this design with the short-sequence probabilistic design at equal total simulated years directly tests the study's premise that many short sequences support better coverage of hazard conditions than few long records.

**Resampled probabilistic ensemble.** Re-randomizing the scenario set at every function evaluation follows the group's prior deep-uncertainty optimization work. Comparing it with the fixed ensemble isolates the effect of freezing the scenario set, separately from the effect of which scenarios are chosen. Because objective values then vary across evaluations of the same policy, comparisons involving this design rely entirely on the final re-evaluation.

**Input-stratified ensemble.** Stratifying scenarios across deeply uncertain generator parameters is the most common recent approach in the decision-making-under-deep-uncertainty literature. Contrasting it with the hazard-filling ensemble isolates the central claim. The hypothesis, supported by Quinn et al. (2020) and Guo et al. (2018), is that distinct parameter sets produce hydrologically redundant realizations, so stratification applied in hazard space yields better coverage of the conditions that stress the system.

**Hazard-filling ensemble.** The proposed method. Scenarios are selected from the master ensemble so that their hazard-metric coordinates are approximately uniform and well separated, following established space-filling subsampling methods from the design-of-experiments literature.

Three families from the taxonomy are deliberately not represented. Adaptive scenario selection (Giudici et al. 2020) modifies the optimization algorithm rather than the ensemble. Tail-only selection is uncommon in this literature. One-dimensional stratification on a single flow statistic (Zatarain Salazar et al. 2017) is the nearest published precedent of the proposed method, and rather than appearing as a separate design it is treated as the limiting case that the multi-dimensional design generalizes. Each exclusion is acknowledged in the manuscript discussion.

## Controls for fair comparison

1. **Computational budget.** Designs differ in scenario count and realization length, so a fixed number of function evaluations would not equalize computational effort (an evaluation over many short sequences costs less than one over few long records). The budget is therefore controlled as **total simulated scenario-years** (function evaluations times ensemble size times realization length), which is equivalent to equal simulation effort and, under identical parallel configuration, approximately equal wall-clock time. This follows the budget-controlled comparison logic of Zatarain Salazar et al. (2017). A consequence is that designs with cheaper evaluations complete more function evaluations within the budget, which is part of what is being compared, since the number of policy evaluations achievable per unit of computing is one of the practical consequences of scenario design.
2. **Convergence reporting.** Within each design, search progress is monitored through the algorithm's standard runtime dynamics (archive evolution and operator behavior) and reported per seed as a diagnostic. These internal quantities are never compared across designs, since objective values computed on different search ensembles, or on resampled draws, are not commensurable. Convergence assessment does not serve as a stopping rule, so no design benefits from a tuned termination criterion.
3. **Single point of comparison.** Cross-design quality comparison occurs only once, by re-evaluating every final Pareto-approximate set on the common held-out test ensemble and recomputing nondominated sets from the re-evaluated objective values for all designs alike.

## Replication

Two sources of variability are distinguished, the random construction of the ensemble and the stochasticity of the search algorithm. The replication structure differs by design. The fixed probabilistic, long-record, and input-stratified designs require several independent ensemble draws, each optimized with several random seeds. The resampled design and the historical record have no fixed ensemble to replicate and require seeds only. The hazard-filling design is replicated through independent runs of its selection procedure (the simulated-annealing search used in conditioned Latin hypercube sampling is itself stochastic), plus seeds. This structure matters because the hazard-filling construction has far less sampling variability than a random draw, and comparing seed variance alone would attribute stability to it by definition. Replicate and seed counts will be set against the computational budget.

## Evaluation

All Pareto-approximate sets are re-evaluated on one large held-out test ensemble, never used during any search. The test ensemble is deliberately broad and is treated as containing deep uncertainty. Its design is an open decision, but it is expected to comprise many realizations drawn from many deeply uncertain generator parameterizations together with other uncertain factors in the system. Two qualifications are stated in the manuscript. First, all comparison measures are computed only after the across-scenario aggregation rule is chosen (an open decision), and the ranking of scenario designs may itself depend on that choice, so the aggregation decision precedes any results. Second, design rankings are conditional on the test-ensemble design, since robustness is only defined relative to the conditions over which it is measured (McPhail et al. 2018; Quinn et al. 2020).

Comparison measures include the quality of the re-evaluated Pareto-approximate sets (hypervolume and additive epsilon indicator against a common reference set, following Reed et al. 2013), robustness computed under several established metrics since metric choice affects rankings (Herman et al. 2015; McPhail et al. 2018), and the gap between objective values estimated on each design's own search ensemble and values under re-evaluation, which measures overfitting to the search ensemble (Brodeur et al. 2020). For the resampled design this gap additionally reflects single-draw noise in the search estimates, which is noted where reported. Following the finding of McPhail et al. (2020) that scenario composition can move robustness values more than rankings, the comparison also reports whether the designs produce materially different policies (decision-variable and simulated-behavior differences among selected compromise solutions), not only different indicator scores. Stability across replicate ensembles and seeds follows the in-sample and out-of-sample stability criteria of Kaut and Wallace (2007). Coverage statistics of each realized ensemble in hazard space (discrepancy and minimum-distance measures) are reported so that performance differences can be related to coverage.

One point requires explicit statement in the manuscript. Several designs depart from the scenario probabilities of the master ensemble. The hazard-filling design does so deliberately, and the input-stratified design and historical record also do so by construction, so objective values computed during search estimate different quantities across designs. This departure is the design choice under study, not an artifact, and the held-out re-evaluation provides the common basis of comparison.

## Decided since rev. 5

- **Comparison metrics:** satisficing robustness + regret on the held-out test
  ensemble; a coverage → re-evaluated-robustness association is reported in the
  supplement as a mechanism analysis (`objective_definitions.md` §3).
- **Search aggregation:** the two-layer annual-unit scheme — annual metrics per
  (realization × water-year) unit, per-objective unit operators over the pooled
  unit-years (`objective_definitions.md` §2). Importance reweighting during search
  is not used.
- **Scenario length:** L = 10 yr, disjoint windows, fixed 0.80 initial storage with
  a 365-day warm-up (`scenario_design_methods.md` §3.2).
- **Forcing space:** historical interannual persistence retained; claims scoped
  accordingly (`scenario_design_methods.md` §8).

## Open questions (not assumed by this design)

1. The design of the held-out test ensemble, including which deeply uncertain factors it spans (generator parameterizations and other system uncertainties), its size, and how realizations are allocated across those factors. Test-ensemble size matters doubly because nondominated sets are recomputed from re-evaluated values.
2. The hazard-metric axes, pending the index-redundancy screening on the production master.
3. Ensemble sizes, replicate counts, and seed counts, to be fixed against a computational budget estimate expressed in total simulated scenario-years (informed by the Anvil scaling experiment).
4. Whether a distribution-representative subsample (support points, Mak & Joseph 2018) is added as a supplementary scenario design. It would isolate whether any benefit of the hazard-filling ensemble stems from uniform coverage specifically or from designed subsampling generally. Both independent reviews of this document recommended its inclusion on identifiability grounds. Held for discussion.
5. The flood-days unit operator (mean vs P99) and the frequency objectives' annual failure criteria, set by the ensemble objective-sensitivity experiment.
