# NYCOptimization Study Plan

Trevor Amestoy, PhD Candidate
Cornell University, Dept of Civil and Environmental Engineering
Reed Research Group

*Version 0.2, February 2026. This is an iterative planning document.*

**Implementation Status (updated 2026-02-27):** Phase 1 infrastructure is complete. The full simulation pipeline (build/run/extract) is implemented and verified against pywrdrb source. Presim generation script is ready. Baseline and Borg scaffolding are in place. Next step: run `00_generate_presim.sh` then `01_run_baseline.sh` to validate the pipeline end-to-end.

---

## 1. Study Motivation

The Delaware River Basin supports water supply for over 13 million people, including approximately half of NYC's daily supply through the Delaware Aqueduct. The 2017 Flexible Flow Management Program (FFMP) governs reservoir operations through storage zone-based rules that balance competing demands: NYC water supply, NJ diversions, downstream flow targets (Montague and Trenton), ecological flows, and flood mitigation.

Prior work in the Pywr-DRB ecosystem has established the simulation model (Hamilton, Amestoy, and Reed, 2024), characterized parameter sensitivity (NYCOperationExploration), and developed stochastic streamflow generation infrastructure (StochasticExploratoryExperiment). However, no study has yet performed many-objective optimization of DRB operations or compared alternative policy formulations within this framework.

This study fills that gap by applying state-of-the-art multi-objective optimization and robustness analysis to discover and evaluate alternative NYC reservoir operating policies.


## 2. Research Questions

### Primary Questions

**RQ1**: How much performance improvement is achievable by re-optimizing the FFMP operational parameters compared to the current default 2017 FFMP rules?

**RQ2**: Does increasing policy flexibility (from parameterized FFMP rules to state-aware RBF or MLP policies) yield meaningful improvements in the Pareto-approximate tradeoff set?

**RQ3**: How robust are Pareto-approximate policies under stochastic hydrological variability, and which uncertain conditions most strongly determine policy failure?

### Secondary / Exploratory Questions

**RQ4**: How do tradeoffs shift across policy formulations? Are there regions of objective space that are only accessible to more flexible policy forms?

**RQ5**: What is the "value of flexibility" in operational policy design for this system, and how does it vary across objectives and stakeholder perspectives?

**RQ6**: Can scenario discovery identify actionable "vulnerability conditions" that inform adaptive management or monitoring signpost design?


## 3. Candidate Problem Formulations

The study considers three nested policy formulations of increasing flexibility. The comparison across formulations is itself a key contribution.

### 3.1 Formulation A: Parameterized FFMP (Baseline)

**Decision variables**: The existing FFMP parameters exposed through `NYCOperationsConfig`:
- MRF baselines (Cannonsville, Pepacton, Neversink, Montague, Trenton): 5 variables
- Drought level factors for NYC and NJ diversions: ~10 variables (per-level multipliers)
- Storage zone thresholds (aggregated seasonal scaling): ~5-10 variables (depending on parameterization of the 366-day profiles)
- Flood release maximums: 3 variables
- Delivery constraints: 2-4 variables (NYC max, NJ daily/monthly)

**Estimated dimensionality**: 25-35 decision variables

**Advantages**: Preserves institutional rule structure. Results are directly interpretable and implementable within the existing FFMP framework. Most credible to decree party stakeholders.

**Limitations**: Confined to the FFMP policy space. Cannot discover fundamentally different operational strategies.

### 3.2 Formulation B: FFMP + Enhanced Flexibility

**Decision variables**: FFMP parameters (as above) plus additional degrees of freedom:
- Seasonal variation in drought factors (currently constant per level)
- Asymmetric storage zone curves (currently symmetric scaling)
- Dynamic blending weights between aggregate and individual drought levels
- Reservoir-specific delivery allocation rules

**Estimated dimensionality**: 40-60 decision variables

**Advantages**: Explores what is achievable within an extended FFMP-like framework. Still interpretable.

**Limitations**: Design of additional flexibility is somewhat ad hoc. Boundary between "extended FFMP" and "fully flexible" is subjective.

### 3.3 Formulation C: State-Aware Direct Policy Search

**Decision variables**: Parameters of nonlinear policy functions (RBFs or MLPs) that map system state to actions.

**State inputs** (potential):
- Individual reservoir storage levels (3)
- Aggregate storage fraction (1)
- Current inflow or recent inflow (3-6, possibly lagged)
- Day of year / season indicator (1-2)
- Recent downstream flow at Montague/Trenton (2)

**Actions**:
- Individual reservoir releases (3)
- NYC diversion allocation (1-3)
- NJ diversion (1)

**RBF parameterization**: For N state inputs, K basis functions, and M actions: N*K (centers) + K (radii) + K*M (weights) parameters. With ~10 inputs, 6 RBFs, 5 actions: ~130 parameters.

**MLP parameterization**: A small network (10 inputs, 1 hidden layer of 20 nodes, 5 outputs) would have ~300 parameters. Could be reduced with architecture constraints.

**Advantages**: Can discover fundamentally novel policies. State-aware and adaptive. Represents the frontier of direct policy search methods.

**Limitations**: High dimensionality. Results are "black box" and harder to interpret. May face institutional resistance. Requires careful constraint handling to ensure physical and legal feasibility.


## 4. Candidate Objectives

The objective set should capture the key competing demands in the DRB system. A working candidate set (to be refined):

| # | Objective | Direction | Metric | Stakeholder Relevance |
|---|-----------|-----------|--------|-----------------------|
| 1 | NYC water supply reliability | Maximize | Fraction of days NYC delivery meets demand | NYC |
| 2 | NYC drought severity | Minimize | Max consecutive days below delivery threshold | NYC |
| 3 | Montague flow compliance | Maximize | Fraction of days Montague flow >= target | All decree parties, River Master |
| 4 | Trenton flow compliance | Maximize | Fraction of days Trenton flow >= target (Jun-Mar) | NJ, PA, DE, salt front |
| 5 | Ecological flow quality | Maximize | Composite metric: thermal regime + flow variability | Environmental stakeholders |
| 6 | Flood risk | Minimize | Days where downstream stage exceeds action threshold | Upper Delaware communities |
| 7 | Storage resilience | Maximize | Minimum storage fraction observed (worst-case drought reserve) | All parties |

**Notes on objective selection**:
- 5-7 objectives is within the range where Borg MOEA has demonstrated effectiveness
- The exact metric formulations (reliability vs. vulnerability vs. resilience-based) need careful specification
- Some objectives may be combined or replaced after initial exploratory runs
- NJ diversion reliability could be added as a separate objective or tracked as a constraint


## 5. Methodological Workflow (MORDM-Inspired)

### Phase 1: Problem Setup and Baseline
- [x] Finalize objective functions and decision variable bounds for Formulation A (ffmp)
- [x] Implement Borg MOEA Python wrapper for Pywr-DRB simulation-optimization coupling
- [x] Implement in-memory simulation path (no HDF5 I/O per evaluation)
- [x] Create `src/simulation.py` with `run_simulation_inmemory()` and `run_simulation_to_disk()`
- [x] Create `src/objectives.py` with `Objective`/`ObjectiveSet` classes and `DEFAULT_OBJECTIVES`
- [x] Create numbered bash workflow (00-05) and supporting Python scripts
- [ ] Run `00_generate_presim.sh` to generate presim data (one-time, ~5-10 min)
- [ ] Run `01_run_baseline.sh` to establish FFMP baseline (full model, ~10-30 min)
- [ ] Verify all 6 objective values compute correctly from baseline run

### Phase 2: Optimization
- Run MOEA diagnostics (random seed analysis, hypervolume convergence) for each formulation
- Perform MM-Borg optimization on HPC (likely Cornell's computing clusters)
- Generate Pareto-approximate reference sets for Formulations A, B, and C
- Compute runtime metrics and verify convergence/reliability

### Phase 3: Re-evaluation Under Uncertainty
- Generate stochastic streamflow ensembles using Kirsch-Nowak generator (building on StochasticExploratoryExperiment code)
- Re-evaluate all Pareto-approximate policies across the ensemble
- Compute robustness metrics (satisficing, regret-based, domain criterion) for each policy
- Identify robust vs. fragile policies within each formulation

### Phase 4: Scenario Discovery and Analysis
- Apply PRIM and/or CART to identify vulnerability conditions
- Compare robustness across formulations (is the "value of flexibility" preserved under deep uncertainty?)
- Visualize tradeoffs using parallel coordinates, pairwise scatterplots, and scenario maps
- Map results to stakeholder perspectives (decree party-specific tradeoff views)

### Phase 5: Synthesis
- Characterize the Pareto tradeoff structure and its sensitivity to formulation choice
- Quantify the "value of flexibility" across the objective space
- Identify actionable policy insights for DRB stakeholders
- Write up findings for dissertation chapter / journal publication


## 6. Open Methodological Questions

These require resolution during iterative planning:

**Q1: Simulation period and inflow data**
- Optimization over full historical record vs. a representative subset?
    A: We will want the optimization record to be easily configurable to test different periods/datasets. But, use the full pub_nhmv10_BC_withObsScaled dataset (1945-2022) as the default period for optimization. 
- Which inflow type? `nhmv10_withObsScaled` is the most validated, but alternatives exist.
    A: See above answer.
- Use of `use_trimmed_model` for 50-70% speedup acceptable for optimization?
    A: Yes. The trimmed model is valid in this optimization context, since we are only modifying the NYC operations.


**Q2: Constraint handling**
- Hard constraints (1954 Decree limits) vs. penalty functions?
    A: The penalty will be reflected in the objective functions via Borg. The Pywr-DRB model will natural attempt to maintain the 1954 rules, unless it is physically unable. The objectives should measure violations of these rules.    
- How to handle the running-average diversion constraint within the optimizer?
    A: To be honest I am not sure, I have not thought about this. We should think carefully about the current code implementation in Pywr-DRB as well as the regulation.
- For Formulation C, how to ensure policies respect physical/legal bounds?
    A: There may be some heuristics, but we may also implement some constraints into the optimization to avoid infeasible solutions. 

**Q3: Computational budget**
- Estimated simulation time per function evaluation (with trimmed model)?
    A: ~30 seconds
- Target NFE for each formulation?
    A: 1 million
- Number of random seeds for diagnostic reliability?
    A: 10
- HPC resource allocation (which cluster, how many cores)?
    A: NSF Discover ACCESS allocation on Anvil CPU HPC

**Q4: Stochastic ensemble design**
- Number of realizations for re-evaluation (balance between coverage and computation)?
    A: To be determined... 
- Stationary vs. nonstationary generation? Include climate change scenarios?
    A: To be determined...
- Length of synthetic sequences (match historical? 70-year as in StochasticExploratoryExperiment)?
    A: 70 is a good starting assumption, but we may want to modify this at some point

**Q5: Ecological flow metrics**
- Which specific ecological indicators to use? Temperature-based? Flow variability? Habitat suitability?
    A: Use the flow targets as a proxy. For now, we should plan on _not_ using the temperature or salinity LSTM plugins, since they have not been fully validated. 
- Data availability for calibrating/validating ecological metrics?
    A: N/A.  Use flow targets as proxy. 

**Q6: Formulation C architecture**
- RBFs vs. MLPs vs. both?
    A: Eventually we may want to test both, but can start with Gaussian RBFs. 
- How many basis functions / hidden nodes?
    A: This needs to be tested to be determined. 
- Which state variables to include as inputs?
    A: Look at the literature for the best examples, and make informed decisions based upon the variables available in Pywr-DRB. 
- How to handle multi-reservoir coordination (one policy per reservoir vs. centralized)?
    A: Treat the NYC reservoirs like an aggregated system for now when determining system diversion and flow requirements (same way the FFMP uses combined storage) although recognize that they may have unique/specific minimum releases (also like the current FFMP)


## 7. Potential Novel Contributions

1. **First MOEA optimization of DRB/FFMP operations**: Moves beyond sensitivity analysis to discover Pareto-approximate tradeoff sets for a real, multi-stakeholder river basin.

2. **Multi-formulation comparison in a real system**: Systematic comparison of parameterized institutional rules vs. state-aware policies in an operational context, quantifying the "value of flexibility."

3. **Robustness of DRB policies under stochastic hydrology**: First application of MORDM-style re-evaluation to optimized DRB operating policies.

4. **Flood-drought tradeoff characterization**: The DRB's void management vs. drought resilience tension is underexplored in the optimization literature.

5. **Stakeholder-relevant tradeoff analysis**: Mapping optimization results to the five decree party perspectives provides a template for participatory decision support in legally-constrained river basins.

6. **Methodological contributions to direct policy search**: Insights on architecture selection (RBF vs. MLP) and dimensionality management for complex multi-reservoir systems with institutional constraints.


## 8. Key Dependencies and Risks

- **Pywr-DRB `nyc_opt` branch stability**: The parameterized operations interface must be robust. Currently supports Formulation A; Formulations B and C require additional development.
- **Borg MOEA access and licensing**: Confirm access to MM-Borg for HPC runs.
- **HPC allocation**: Sufficient compute for diagnostic runs across three formulations.
- **Simulation speed**: If full Pywr-DRB runs are too slow for large NFE budgets, the trimmed model or surrogate approaches may be necessary.
- **Ecological metrics**: Data and model limitations may constrain the ecological flow objective.


## 9. Relevant Repositories

| Repository | Role | Branch/Status |
|-----------|------|---------------|
| Pywr-DRB (nyc_opt) | Simulation model with parameterized operations | Active development |
| NYCOptimization (this repo) | Optimization framework, analysis, and results | New |
| NYCOperationExploration | Sensitivity analysis (reference) | Complete |
| StochasticExploratoryExperiment | Stochastic generation and re-evaluation (reference) | Complete |

## 10. References

- Giuliani, M., Castelletti, A., Pianosi, F., Mason, E., & Reed, P. M. (2016). Curses, tradeoffs, and scalable management. *J. Water Resources Planning and Management*, 142(2).
- Hadka, D., & Reed, P. (2013). Borg: An auto-adaptive many-objective evolutionary computing framework. *Evolutionary Computation*, 21(2), 231-259.
- Hadka, D., & Reed, P. (2015). Large-scale parallelization of the Borg MOEA. *Environmental Modelling & Software*, 69, 353-369.
- Hamilton, A. L., Amestoy, T. J., & Reed, P. M. (2024). Pywr-DRB: An open-source Python model for water availability and drought risk assessment. *Environmental Modelling & Software*, 106185.
- Herman, J. D., Reed, P. M., Zeff, H. B., & Characklis, G. W. (2015). How should robustness be defined for water systems planning under change? *J. Water Resources Planning and Management*, 141(10).
- Kasprzyk, J. R., Nataraj, S., Reed, P. M., & Lempert, R. J. (2013). Many objective robust decision making for complex environmental systems undergoing change. *Environmental Modelling & Software*, 42, 55-71.
- Kolesar, P., & Serio, J. (2011). Breaking the deadlock: Improving water-release policies on the Delaware River. *Interfaces*, 41(1), 18-34.
- Quinn, J. D., Reed, P. M., Giuliani, M., & Castelletti, A. (2019). What is controlling our control rules? *Water Resources Research*, 55(7), 5962-5984.
- Reed, P. M., Hadka, D., Herman, J. D., Kasprzyk, J. R., & Kollat, J. B. (2013). Evolutionary multiobjective optimization in water resources. *Advances in Water Resources*, 51, 438-456.
