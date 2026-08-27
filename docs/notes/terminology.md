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

**Population.** The law from which a design's realizations are drawn. All search designs use the **stationary** population (Kirsch–Nowak fit to the historic record, forcing held at the historic fit). Deep uncertainty enters only in the test ensemble, which is built over the **DU-forced** forcing space (forcing parameters sampled from the CMIP6 harmonic hypercube); that forcing space is the construction basis of $E_{\text{test}}$, not a search population.

**Candidate pool.** The pool of i.i.d. realizations that the hazard-filling design subsamples. It **belongs to that design**, is generated with its own seed stream, and is disjoint from the test ensemble. Hazard-filling is the only design that needs one, because hazard coordinates cannot be prescribed at generation — they are measured on a realized sequence, so a hazard-space design must *select from* a pool rather than *generate to* a target. An i.i.d. probabilistic design faces no such constraint and generates its members directly.

**Evaluation ensemble** (synonym, "search ensemble"). The scenario set actually used inside `evaluation()` during MOEA search. The object this study designs. Enumerated by `src/scenario_designs.py`.

**Test ensemble** ($E_{\text{test}}$; synonym, "re-evaluation ensemble"). The large held-out ensemble used in workflow step 08 to stress-test Pareto-approximate policies out of sample (the MORDM re-evaluation step, Kasprzyk et al. 2013; Herman et al. 2015, *JWRPM*). It is **never the source of any search ensemble**, and no search ensemble is a subset of it.

**The largest ensemble in the study, by a wide margin**, and built to be maximally *uncertainty-encompassing*: a Latin hypercube over the **full range** of the deeply-uncertain forcing factors, with **many realizations per LHS point**. The campaign uses one construction (Kirsch–Nowak over the wide DU box, locked at $N_\theta = 1{,}000$ LHS SOWs × $R = 25$ × $L_{\text{test}} = 50$ yr); rankings are therefore conditional on it, which is a declared limitation. A structurally different second construction (multi-site HMM) is registered but parked outside the campaign.

$E_{\text{test}}$ is sampled by **LHS, not i.i.d.** The i.i.d. rule applies only to the candidate pool, where it underwrites the distributional-equivalence control. $E_{\text{test}}$ is never hazard-subsampled and is never a control (the campaign re-evaluates its leading 500 SOWs, a prefix of the randomly ordered LHS rows, not a selection; `campaign_design.md` §5) — it is the measuring stick, and should *cover* the deeply-uncertain space, not sample it in proportion to a measure. It is therefore a **designed exploration, not a probability sample**: a satisficing fraction over it is a coverage-weighted count, never an expectation. Because the search designs are stationary while $E_{\text{test}}$ is DU-forced, re-evaluation is a **generalization test** to conditions absent from search, and $E_{\text{test}}$ is structurally distinct from both search designs.

**State of the world (SOW).** One deeply-uncertain factor vector $\theta$ — one LHS point of $E_{\text{test}}$. Its $R$ realizations sample natural variability *within* that SOW. The SOW is the unit of robustness in the MORDM lineage (Herman et al. 2014; Trindade et al. 2017; Gold et al. 2022, 2023), which collapses the stochastic traces inside each SOW before applying the domain criterion across SOWs. Precision is governed by the number of SOWs ($N_\theta$), not by the total realization count.

## Sampling and subsampling

**Probabilistic sampling.** Drawing evaluation scenarios i.i.d. from the generator (Quinn et al. 2017, *WRR*; Zatarain Salazar et al. 2017, *AWR*). The reference against which designed selection is judged.

**Input stratification.** Latin hypercube sampling over the generator's forcing parameters, with realizations **generated at** each design point (Quinn et al. 2020, *Earth's Future*; Bartholomew & Kwakkel 2020, *EMS*). LHS alone — there is nothing to select from, because the parameters are a knob on the generator. Not a campaign design in this study; retained here as a vocabulary reference for the prevailing input-space practice.

**Hazard filling** (space-filling subsampling). Selecting evaluation scenarios from a candidate pool so their hazard coordinates cover the hazard space. Implemented as Latin hypercube anchors on the campaign selection axes (a fixed six-descriptor subset of the screened hazard descriptors, `config.HAZARD_SELECTION_AXES`) snapped to the nearest unused pool member. The campaign selector fills the space in **absolute, range-scaled** magnitude units, which deliberately over-represents the severe (rare) hazard corners relative to their pool frequency; a rank-space (empirical-CDF) variant, which preserves the pool marginals and distorts only the joint dependence, is registered only as a non-campaign sensitivity. The nearest-neighbour step is **intrinsic, not an approximation**: hazard coordinates are emergent properties of a realized sequence, so no generator can be asked to produce a realization at a prescribed hazard point. Distinct from **representative-in-probability** subset selection (scenario reduction), which preserves the parent distribution rather than filling the space.

**Distributional equivalence (the control).** A uniform random size-*N* subset of an i.i.d. pool has exactly the joint law of *N* fresh i.i.d. draws. This is what makes `fixed_probabilistic` the *exact* statistical control for `hazard_filling` on the same stationary population: only the selection rule differs. It requires the pool to be sampled **i.i.d., not LHS** — a random subset of an LHS design is not i.i.d.

**Scenario redundancy.** Overlap of two or more scenarios' coordinates in hazard space, regardless of whether they came from different input-space samples. Motivated by the redundancy framing of Olden & Poff (2003) applied to scenarios rather than indices. Quantify via maximin/minimax distances (Johnson et al. 1990) or effective sample size.

**Uniformity and representativeness diagnostics.** Centered L2 discrepancy for uniformity in hazard space (Fang et al. 2000, *Technometrics*) and energy distance for distributional match to a target (Székely & Rizzo 2013, *JSPI*).

## Evaluation and robustness

**In-sample / out-of-sample stability.** A scenario set is in-sample stable if replicate sets of the same size yield the same optimized values, and out-of-sample stable if performance estimated on the set matches performance on the true distribution (Kaut & Wallace 2007, *Pac. J. Optim.*). Our hypervolume-reliability and re-evaluation-bias diagnostics operationalize these.

**Robustness.** Performance of a policy across the re-evaluation ensemble, computed with explicitly named metrics (satisficing, regret, percentile), since metric choice changes rankings (Herman et al. 2015; McPhail et al. 2018, *Earth's Future*).

**Per-SOW objective value** ($J_i(x,\theta)$). The study's single metric currency: the annual-unit search objective recomputed per E_test state of the world, by pooling that state's realizations' unit-years through the objective's own unit operator. Every robustness and regret metric is a transformation of $J_i(x,\theta)$ — the standard construction in which the performance measure inside a robustness calculation *is* the optimization objective re-evaluated per state (Herman et al. 2014, 2015; Trindade et al. 2017; McPhail et al. 2018). There is no separate re-evaluation metric set, and no whole-trace robustness statistic.

**Regret.** Never write "regret" unqualified — it has four incompatible references in this literature, and conflating two of them is a mistake this project has already made once. Use the qualified names:

- **Incumbent regret** (the one computed). The amount by which a candidate policy is worse than the **status-quo 2017 FFMP policy** evaluated *in the same state of the world*, in the objective's own natural units, oriented so that positive advantage means better than current operations. Reference: McPhail et al. (2018) §3.1 for the licensed T1; Herman et al. (2015) R1/R2 for the per-objective → tail-quantile shape; Kwakkel et al. (2016b) for the adverse-subset restriction. Its unit-free companions are the **harm frequencies** and the **no-harm frequency** $\Pi_\tau$.
- **Best-in-set regret** (excluded). Savage regret against the best policy in the evaluated set, per SOW — set-relative and design-coupled.
- **Baseline-SOW regret** (not computed). Herman et al. (2015) R1 / Kasprzyk et al. (2013) percent deviation: the *same* policy's deviation from its own performance in a reference state of the world. A sensitivity measure, not a comparison against a policy.
- **Perfect-foresight regret** (not computed). Cohen et al. (2021): the gap between a baseline policy and a per-scenario perfect-foresight optimum. Cited as motivation only; it requires one optimization per scenario.

**No-harm frequency** ($\Pi_\tau$). The fraction of re-evaluation SOWs in which a policy degrades **no** objective by more than its tolerance $\tau_i$ relative to the incumbent. $\Pi_0$ is a weak Pareto improvement on the status quo. It is a Starr domain criterion whose threshold vector is the incumbent's own per-SOW performance, and it is the literal operationalization of the second research question. Say "no-harm frequency", not "regret rate".

**Price of robustness.** Bertsimas & Sim (2004), via Bartholomew & Kwakkel (2020): the performance given up in individual conditions in exchange for robustness across them. Use it for the trade-off itself; do not use it as a name for any metric.

**Stress test.** Systematic evaluation of a policy across a designed condition space, per the bottom-up tradition (Brown et al. 2012, decision scaling, *WRR*; Fowler et al. 2024).

**Deep uncertainty / well-characterized uncertainty.** Standard DMDU usage (Maier et al. 2016, *EMS*; Marchau et al. 2019). Input-space parameter ranges are treated as deeply uncertain. Within a single parameter set, generator output is well-characterized.

## Decision variables

**Allocation reduction.** A diversion decision variable
(`{nyc,nj}_allocation_reduction_*`): the *additional* fractional reduction of
the party's Decree allocation applied on entry to a drought stage. Stage-wise
increments, not absolute factors — the effective delivery factor at a stage is
1 minus the running sum of reductions, so monotone curtailment across stages
holds by construction.

**Delivery factor.** The absolute multiplier on the Decree allocation that the
Pywr-DRB model consumes per drought level (model parameters
`{level}_factor_delivery_{nyc,nj}`). A *decoded* quantity, never a decision
variable: the simulation wrapper converts allocation reductions to delivery
factors before handoff. Do not call the DVs "factors".

## Style rules

1. All `_pct` quantities are 0-1 fractions (repo-wide rule).
2. Never write "master ensemble". Write **candidate pool** (a hazard-filling design's own pool) or **test ensemble** (the held-out re-evaluation set).
3. Say "evaluation ensemble" not "training set" in manuscripts, but the ML training/generalization analogy (Brodeur et al. 2020, *WRR*) may be invoked explicitly when discussing overfitting.
4. Sequence length is stated in years, and window construction (disjoint vs overlapping, initialization of storages, handling of partial drought events at window edges) must be specified wherever scenarios are introduced.
5. The units of the experimental comparison are called **scenario designs** (or experiments where the optimization run is meant). Avoid clinical-trial vocabulary such as "arm", "treatment", and "ablation". For a comparison that isolates a mechanism, write a controlled or diagnostic comparison.
