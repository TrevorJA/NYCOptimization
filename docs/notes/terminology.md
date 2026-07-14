# Project Terminology

*Controlled vocabulary for NYCOptimization manuscripts, code, and notes. Full citations live in the literature notes indexed by `docs/notes/literature/scenario_design.md`. When writing, use these terms exactly and avoid the flagged synonyms.*

---

## The three spaces

**Input space** (synonym to avoid in prose, "parametric space"). The space of factors that *define* scenario generation. Examples are stochastic generator parameters, HMM transition and emission parameters, climate-change multipliers, and demand factors. In the MORDM literature a sampled point in this space is a **state of the world (SOW)** (Kasprzyk et al. 2013, *EMS*; Trindade et al. 2017, *AWR*). Most prior scenario design samples this space, e.g., LHS over generator parameters (Quinn et al. 2018, *WRR*; Steinschneider et al. 2019, *WRR*).

**Hazard space.** This project's term for the space of hydrologic hazard metrics computed directly on each realized streamflow sequence, before any system simulation. Axes include SSI-based drought event metrics (intensity, duration, severity per run theory), low-flow indices, and high-flow metrics. Grounding citations are Yevjevich (1967, run theory), Vicente-Serrano et al. (2012, SSI), Richter et al. (1996, IHA low/high-flow indices), and Olden & Poff (2003, index redundancy and selection). "Hazard" follows the risk-triplet usage where risk is a function of hazard, exposure, and vulnerability (IPCC SREX 2012, UNDRR Sendai terminology). The hazard space is a property of the *scenario*, not of the simulated system response. The closest existing term is the scenario-neutral literature's **exposure space**, the grid of perturbed forcing attributes in stress testing (Culley et al. 2016, *WRR*; Guo et al. 2018, *J. Hydrol.*; Fowler et al. 2024, *WIREs Water*). We do not use "exposure space" for our construct because (a) exposure has a conflicting meaning in the risk triplet and (b) exposure spaces are typically attribute *targets* imposed on the generator inputs, whereas hazard space coordinates are *measured* on realized sequences.

**Outcome space** (synonym, "performance space" or "objective space"). Reserved strictly for simulation outputs, i.e., objective values and performance metrics of a candidate policy under a scenario. Never use "outcome" to describe scenario flow characteristics. The hazard-vs-outcome distinction matters because hazard coordinates exist before any policy is evaluated, which is what makes hazard-space subsampling a pre-optimization design step.

## Scenarios and ensembles

**Scenario.** One streamflow sequence (here 10 years, all model inflow nodes) over which a candidate policy is simulated during one evaluation. Used in the stochastic-programming sense of a discrete realization supplied to the optimizer, not the narrative-futures sense.

**Realization.** A single output sequence of a stochastic generator. Every scenario is a realization (or a window of one).

**Ensemble.** A finite set of realizations or scenarios. Always qualify which ensemble is meant.

**Population.** The law from which a design's realizations are drawn. Two are used: the **stationary** population (Kirsch–Nowak fit to the historic record, forcing held at the historic fit) and the **DU-forced** population (forcing parameters sampled from the CMIP6 harmonic hypercube). A design's population and its selection rule are independent choices, and the comparison holds one fixed while varying the other.

**Candidate pool.** The pool of i.i.d. realizations that a hazard-filling design subsamples. It **belongs to that design**, is generated with its own seed stream, and is disjoint from the test ensemble. No other design draws from it. Hazard-filling is the only design that needs one, because hazard coordinates cannot be prescribed at generation — they are measured on a realized sequence, so a hazard-space design must *select from* a pool rather than *generate to* a target. Input-space designs face no such constraint and generate directly to their design points.

**Evaluation ensemble** (synonym, "search ensemble"). The scenario set actually used inside `evaluation()` during MOEA search. The object this study designs. Enumerated by `src/scenario_designs.py`.

**Test ensemble** ($E_{\text{test}}$; synonym, "re-evaluation ensemble"). The large held-out ensemble used in workflow step 08 to stress-test Pareto-approximate policies out of sample (the MORDM re-evaluation step, Kasprzyk et al. 2013; Herman et al. 2015, *JWRPM*). It is **never the source of any search ensemble**, and no search ensemble is a subset of it.

**The largest ensemble in the study, by a wide margin**, and built to be maximally *uncertainty-encompassing*: a Latin hypercube over the **full range** of the deeply-uncertain forcing factors, with **many realizations per LHS point**. The campaign uses one construction (Kirsch–Nowak over the wide DU box); rankings are therefore conditional on it, which is a declared limitation. A structurally different second construction is registered as an optional sensitivity.

$E_{\text{test}}$ is sampled by **LHS, not i.i.d.** The i.i.d. rule applies only to the candidate pools, where it underwrites the distributional-equivalence control. $E_{\text{test}}$ is never subsampled and is never a control — it is the measuring stick, and should *cover* the deeply-uncertain space, not sample it in proportion to a measure. It is therefore a **designed exploration, not a probability sample**: a satisficing fraction over it is a coverage-weighted count, never an expectation.

**State of the world (SOW).** One deeply-uncertain factor vector $\theta$ — one LHS point of $E_{\text{test}}$. Its $R$ realizations sample natural variability *within* that SOW. The SOW is the unit of robustness in the MORDM lineage (Herman et al. 2014; Trindade et al. 2017; Gold et al. 2022, 2023), which collapses the stochastic traces inside each SOW before applying the domain criterion across SOWs. Precision is governed by the number of SOWs ($N_\theta$), not by the total realization count.

## Sampling and subsampling

**Probabilistic sampling.** Drawing evaluation scenarios i.i.d. from the generator (Quinn et al. 2017, *WRR*; Zatarain Salazar et al. 2017, *AWR*). The reference against which designed selection is judged.

**Input stratification.** Latin hypercube sampling over the generator's forcing parameters, with realizations **generated at** each design point (Quinn et al. 2020, *Earth's Future*; Bartholomew & Kwakkel 2020, *EMS*). LHS alone — there is nothing to select from, because the parameters are a knob on the generator.

**Hazard filling** (space-filling subsampling). Selecting evaluation scenarios from a candidate pool so their hazard coordinates are approximately uniform and well-spread. Implemented as Latin hypercube anchors in hazard space snapped to the nearest unused pool member. The nearest-neighbour step is **intrinsic, not an approximation**: hazard coordinates are emergent properties of a realized sequence, so no generator can be asked to produce a realization at a prescribed hazard point. Distinct from **representative-in-probability** subset selection (scenario reduction; support points), which preserves the parent distribution rather than filling the space.

**Distributional equivalence (the control).** A uniform random size-*N* subset of an i.i.d. pool has exactly the joint law of *N* fresh i.i.d. draws. This is what makes a probabilistic design the *exact* statistical control for a hazard-filling design on the same population: only the selection rule differs. It requires the pool to be sampled **i.i.d., not LHS** — a random subset of an LHS design is not i.i.d.

**Scenario redundancy.** Overlap of two or more scenarios' coordinates in hazard space, regardless of whether they came from different input-space samples. Motivated by the redundancy framing of Olden & Poff (2003) applied to scenarios rather than indices. Quantify via maximin/minimax distances (Johnson et al. 1990) or effective sample size.

**Uniformity and representativeness diagnostics.** Centered L2 discrepancy for uniformity in hazard space (Fang et al. 2000, *Technometrics*) and energy distance for distributional match to a target (Székely & Rizzo 2013, *JSPI*).

## Evaluation and robustness

**In-sample / out-of-sample stability.** A scenario set is in-sample stable if replicate sets of the same size yield the same optimized values, and out-of-sample stable if performance estimated on the set matches performance on the true distribution (Kaut & Wallace 2007, *Pac. J. Optim.*). Our hypervolume-reliability and re-evaluation-bias diagnostics operationalize these.

**Robustness.** Performance of a policy across the re-evaluation ensemble, computed with explicitly named metrics (satisficing, regret, percentile), since metric choice changes rankings (Herman et al. 2015; McPhail et al. 2018, *Earth's Future*).

**Stress test.** Systematic evaluation of a policy across a designed condition space, per the bottom-up tradition (Brown et al. 2012, decision scaling, *WRR*; Fowler et al. 2024).

**Deep uncertainty / well-characterized uncertainty.** Standard DMDU usage (Maier et al. 2016, *EMS*; Marchau et al. 2019). Input-space parameter ranges are treated as deeply uncertain. Within a single parameter set, generator output is well-characterized.

## Style rules

1. All `_pct` quantities are 0-1 fractions (repo-wide rule).
2. **"Master ensemble" is retired.** It previously named a single pool that every design subsampled — an architecture that has been removed. Write **candidate pool** (a hazard-filling design's own pool) or **test ensemble** (the held-out re-evaluation set), never "master".
3. Say "evaluation ensemble" not "training set" in manuscripts, but the ML training/generalization analogy (Brodeur et al. 2020, *WRR*) may be invoked explicitly when discussing overfitting.
4. Sequence length is stated in years, and window construction (disjoint vs overlapping, initialization of storages, handling of partial drought events at window edges) must be specified wherever scenarios are introduced.
5. The units of the experimental comparison are called **scenario designs** (or experiments where the optimization run is meant). Avoid clinical-trial vocabulary such as "arm", "treatment", and "ablation". For a comparison that isolates a mechanism, write a controlled or diagnostic comparison.
