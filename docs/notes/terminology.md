# Project Terminology

*Last updated: 2026-06-10. Controlled vocabulary for NYCOptimization manuscripts, code, and notes. Full citations live in `docs/notes/literature/optimization_scenario_sampling_review.md`. When writing, use these terms exactly and avoid the flagged synonyms.*

---

## The three spaces

**Input space** (synonym to avoid in prose, "parametric space"). The space of factors that *define* scenario generation. Examples are stochastic generator parameters, HMM transition and emission parameters, climate-change multipliers, and demand factors. In the MORDM literature a sampled point in this space is a **state of the world (SOW)** (Kasprzyk et al. 2013, *EMS*; Trindade et al. 2017, *AWR*). Most prior scenario design samples this space, e.g., LHS over generator parameters (Quinn et al. 2018, *WRR*; Steinschneider et al. 2019, *WRR*).

**Hazard space.** This project's term for the space of hydrologic hazard metrics computed directly on each realized streamflow sequence, before any system simulation. Axes include SSI-based drought event metrics (intensity, duration, severity per run theory), low-flow indices, and high-flow metrics. Grounding citations are Yevjevich (1967, run theory), Vicente-Serrano et al. (2012, SSI), Richter et al. (1996, IHA low/high-flow indices), and Olden & Poff (2003, index redundancy and selection). "Hazard" follows the risk-triplet usage where risk is a function of hazard, exposure, and vulnerability (IPCC SREX 2012, UNDRR Sendai terminology). The hazard space is a property of the *scenario*, not of the simulated system response. The closest existing term is the scenario-neutral literature's **exposure space**, the grid of perturbed forcing attributes in stress testing (Culley et al. 2016, *WRR*; Guo et al. 2018, *J. Hydrol.*; Fowler et al. 2024, *WIREs Water*). We do not use "exposure space" for our construct because (a) exposure has a conflicting meaning in the risk triplet and (b) exposure spaces are typically attribute *targets* imposed on the generator inputs, whereas hazard space coordinates are *measured* on realized sequences.

**Outcome space** (synonym, "performance space" or "objective space"). Reserved strictly for simulation outputs, i.e., objective values and performance metrics of a candidate policy under a scenario. Never use "outcome" to describe scenario flow characteristics. The hazard-vs-outcome distinction matters because hazard coordinates exist before any policy is evaluated, which is what makes hazard-space subsampling a pre-optimization design step.

## Scenarios and ensembles

**Scenario.** One streamflow sequence (here ~5 years, all model inflow nodes) over which a candidate policy is simulated during one evaluation. Used in the stochastic-programming sense of a discrete realization supplied to the optimizer (Dupačová et al. 2003; Kaut & Wallace 2007), not the narrative-futures sense.

**Realization.** A single output sequence of a stochastic generator. Every scenario is a realization (or a window of one), but most realizations in the master ensemble never become evaluation scenarios.

**Ensemble.** A finite set of realizations or scenarios. Always qualify which ensemble is meant.

**Master ensemble** (synonym, "ensemble of ensembles"). The very large (order 1M member) collection of short realizations produced in pipeline step (ii) by running every sampled input-space parameter set through the generator. Cf. the large-ensemble architecture of Lamontagne et al. (2018, *Earth's Future*).

**Evaluation ensemble** (synonym, "search ensemble"). The small scenario set actually used inside `evaluation()` during MOEA search. The object RQ1 designs. Enumerated by `config.ENSEMBLE_PRESETS`.

**Test ensemble** (synonym, "re-evaluation ensemble"). The much larger held-out ensemble used in workflow step 07 to stress-test Pareto-approximate policies out of sample (the MORDM re-evaluation step, Kasprzyk et al. 2013; Herman et al. 2015, *JWRPM*). Deliberately broad and treated as containing deep uncertainty, expected to span many deeply uncertain generator parameterizations and other uncertain system factors. Its design is a standing methodological decision, not a default.

## Sampling and subsampling

**Probabilistic (well-characterized) sampling.** Drawing evaluation scenarios i.i.d. from the assumed distribution, the sample average approximation baseline (Kleywegt et al. 2002, *SIAM J. Optim.*). Our 3 probabilistic presets.

**Structured (space-filling) subsampling.** Selecting evaluation scenarios from the master ensemble to be approximately uniform and well-spread in hazard space, analogous to a conditioned Latin hypercube on the empirical hazard marginals (Minasny & McBratney 2006, *Comput. Geosci.*) with maximin spread (Johnson et al. 1990; Morris & Mitchell 1995). Our 3 LHS presets. Distinct from **representative-in-probability** subset selection (scenario reduction, Dupačová et al. 2003; support points, Mak & Joseph 2018; twinning, Vakayil & Joseph 2022), which preserves the parent distribution.

**Scenario redundancy.** Overlap of two or more scenarios' coordinates in hazard space, regardless of whether they came from different input-space samples. Motivated by the redundancy framing of Olden & Poff (2003) applied to scenarios rather than indices. Quantify via maximin/minimax distances (Johnson et al. 1990) or effective sample size.

**Uniformity and representativeness diagnostics.** Centered L2 discrepancy for uniformity in hazard space (Fang et al. 2000, *Technometrics*) and energy distance for distributional match to a target (Székely & Rizzo 2013, *JSPI*).

## Evaluation and robustness

**In-sample / out-of-sample stability.** A scenario set is in-sample stable if replicate sets of the same size yield the same optimized values, and out-of-sample stable if performance estimated on the set matches performance on the true distribution (Kaut & Wallace 2007, *Pac. J. Optim.*). Our hypervolume-reliability and re-evaluation-bias diagnostics operationalize these.

**Robustness.** Performance of a policy across the re-evaluation ensemble, computed with explicitly named metrics (satisficing, regret, percentile), since metric choice changes rankings (Herman et al. 2015; McPhail et al. 2018, *Earth's Future*).

**Stress test.** Systematic evaluation of a policy across a designed condition space, per the bottom-up tradition (Brown et al. 2012, decision scaling, *WRR*; Fowler et al. 2024).

**Deep uncertainty / well-characterized uncertainty.** Standard DMDU usage (Maier et al. 2016, *EMS*; Marchau et al. 2019). Input-space parameter ranges are treated as deeply uncertain. Within a single parameter set, generator output is well-characterized.

## Style rules

1. All `_pct` quantities are 0-1 fractions (repo-wide rule).
2. Say "evaluation ensemble" not "training set" in manuscripts, but the ML training/generalization analogy (Brodeur et al. 2020, *WRR*) may be invoked explicitly when discussing overfitting.
3. Sequence length is stated in years and window construction (disjoint vs overlapping, initialization of storages, handling of partial drought events at window edges) must be specified wherever 5-year scenarios are introduced (truncation caveats per Pardo et al. 2018, ICML).
4. The units of the experimental comparison are called **scenario designs** (or experiments where the optimization run is meant). Avoid clinical-trial vocabulary such as "arm", "treatment", and "ablation". For a comparison that isolates a mechanism, write a controlled or diagnostic comparison.
