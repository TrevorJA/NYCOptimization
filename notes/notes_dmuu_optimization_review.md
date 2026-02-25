# Literature Review: Decision Making Under Uncertainty and Multi-Objective Optimization in Water Resources

Trevor Amestoy, Reed Research Group, Cornell University
February 2026

---

## 1. Framing: Deep Uncertainty in Water Resources

Deep uncertainty arises when analysts and stakeholders cannot agree on (1) the appropriate models to describe system dynamics, (2) the probability distributions of key inputs, or (3) how to value different outcomes. In water resources, deep uncertainty is driven by climate variability and change, evolving demand patterns, institutional and political dynamics, and incomplete understanding of ecological responses to flow alteration.

Traditional approaches based on "best-guess" futures and expected-value optimization are insufficient because they can produce solutions that perform well under assumed conditions but fail catastrophically under plausible alternative futures. This motivates a shift toward *robust* decision making, where solutions are sought that perform satisfactorily across a wide range of plausible futures rather than optimally under a single assumed future.

**Key reference**: Marchau, V. A., Walker, W. E., Bloemen, P. J., & Popper, S. W. (Eds.). (2019). *Decision Making under Deep Uncertainty: From Theory to Practice*. Springer.


## 2. Many-Objective Robust Decision Making (MORDM)

### 2.1 The Framework

MORDM, as developed by Kasprzyk et al. (2013) and refined by the Reed Group, combines three components:

1. **Many-objective evolutionary optimization** to discover Pareto-approximate tradeoff sets across competing objectives
2. **Robust Decision Making (RDM)** principles from RAND Corporation to evaluate solution performance under deep uncertainty
3. **Interactive visual analytics** to support stakeholder exploration of tradeoffs

The standard MORDM workflow:

1. **Problem formulation**: Define objectives, decision variables, constraints, and deeply uncertain factors
2. **Search**: Use a many-objective evolutionary algorithm (e.g., Borg MOEA) to discover Pareto-approximate solutions under a reference scenario
3. **Re-evaluation**: Stress-test Pareto-approximate solutions across a large ensemble of deeply uncertain scenarios (e.g., alternative streamflow realizations, demand trajectories, climate futures)
4. **Scenario discovery**: Use statistical learning methods (PRIM, CART) to identify the conditions under which candidate solutions fail to meet performance thresholds

**Key references**:
- Kasprzyk, J. R., Nataraj, S., Reed, P. M., & Lempert, R. J. (2013). Many objective robust decision making for complex environmental systems undergoing change. *Environmental Modelling & Software*, 42, 55-71.
- Herman, J. D., Reed, P. M., Zeff, H. B., & Characklis, G. W. (2015). How should robustness be defined for water systems planning under change? *Journal of Water Resources Planning and Management*, 141(10).

### 2.2 Robustness Metrics

Several robustness metrics have been explored in the literature:

- **Satisficing**: Fraction of scenarios in which a solution meets all performance thresholds simultaneously (Starr, 1963; Simon, 1956)
- **Regret-based**: Minimax regret or expected regret relative to the best-performing solution in each scenario
- **Domain criterion**: Performance at a specified percentile of the scenario distribution (e.g., 90th percentile worst-case)
- **Signal-to-noise ratio**: Mean performance divided by standard deviation across scenarios

Herman et al. (2015) argue that robustness metric choice can significantly alter which solutions are preferred, and recommend evaluating solutions under multiple robustness framings.


## 3. Multi-Objective Evolutionary Algorithms (MOEAs)

### 3.1 The Borg MOEA

The Borg MOEA (Hadka and Reed, 2013) is the primary optimization engine for this study. Key features:

- **Auto-adaptive multi-operator search**: Maintains a portfolio of recombination operators (SBX, DE, PCX, SPX, UNDX, UM) and adapts selection probabilities based on recent success rates
- **Epsilon-dominance archiving**: Controls the resolution of the Pareto approximation, preventing bloat while maintaining diversity
- **Adaptive population sizing**: Dynamically restarts with injection from the archive when search stagnates
- **Efficient for many-objective problems**: Demonstrated advantages over NSGA-II, SPEA2, GDE3, MOEA/D, and epsilon-MOEA on standard test suites and real-world water problems

**Key references**:
- Hadka, D., & Reed, P. (2013). Borg: An auto-adaptive many-objective evolutionary computing framework. *Evolutionary Computation*, 21(2), 231-259.
- Hadka, D., & Reed, P. (2015). Large-scale parallelization of the Borg multiobjective evolutionary algorithm to enhance the management of complex environmental systems. *Environmental Modelling & Software*, 69, 353-369.

### 3.2 Multi-Master Borg

The Multi-Master (MM) variant enables massively parallel optimization by:
- Running multiple independent Borg instances ("masters") simultaneously
- Periodically sharing archive solutions between masters to enhance global search
- Scaling efficiently to thousands of processors on HPC systems
- Each master manages a pool of workers that evaluate candidate solutions in parallel

This is critical for computationally expensive simulation-optimization problems like Pywr-DRB, where each function evaluation requires running a full hydrologic simulation.

### 3.3 MOEA Diagnostics

Reliable use of MOEAs requires diagnostics to assess:

- **Convergence**: Is the algorithm approaching the true Pareto front? Measured via hypervolume indicator.
- **Reliability**: Are independent random seeds producing consistent results? Measured via attainment probability and hypervolume variance across seeds.
- **Controllability**: How sensitive are results to algorithm parameterization?

The standard diagnostic protocol (Reed et al., 2013) involves running multiple random seeds (typically 10-50), computing reference sets, and evaluating hypervolume convergence over NFE (number of function evaluations).

**Key references**:
- Reed, P. M., Hadka, D., Herman, J. D., Kasprzyk, J. R., & Kollat, J. B. (2013). Evolutionary multiobjective optimization in water resources: The past, present, and future. *Advances in Water Resources*, 51, 438-456.
- MOEAFramework: http://moeaframework.org/ (standard tool for MOEA diagnostics)


## 4. Direct Policy Search and Policy Representations

### 4.1 Evolutionary Multi-Objective Direct Policy Search (EMODPS)

EMODPS (Giuliani et al., 2016) couples simulation models with MOEAs to optimize the parameters of closed-loop control policies rather than open-loop decision sequences. This is the state-of-the-art for reservoir operations optimization because:

- Policies are **state-aware**: they condition releases on current system state (storage levels, inflows, time of year)
- They naturally handle sequential decision making over long simulation horizons
- They avoid the "curse of dimensionality" that plagues scenario-tree approaches

**Key reference**: Giuliani, M., Castelletti, A., Pianosi, F., Mason, E., & Reed, P. M. (2016). Curses, tradeoffs, and scalable management: Advancing evolutionary multiobjective direct policy search to improve water reservoir operations. *Journal of Water Resources Planning and Management*, 142(2).

### 4.2 Policy Representations

**Radial Basis Functions (RBFs)**:
- Map state variables (e.g., storage, inflow, time) to actions (release, diversion) through a weighted sum of radial basis functions
- Each RBF is characterized by a center, radius, and weight
- Provide smooth, nonlinear mappings with relatively few parameters
- Shown to be effective for multi-reservoir systems
- Quinn et al. (2019) used time-varying sensitivity analysis to diagnose which inputs control RBF policy outputs

**Key references**:
- Giuliani, M., Herman, J. D., Castelletti, A., & Reed, P. (2014). Many-objective reservoir policy identification and refinement to reduce policy inertia and myopia in water management. *Water Resources Research*, 50(4), 3355-3377.
- Quinn, J. D., Reed, P. M., Giuliani, M., & Castelletti, A. (2019). What is controlling our control rules? Opening the black box of multireservoir operating policies using time-varying sensitivity analysis. *Water Resources Research*, 55(7), 5962-5984.

**Multi-Layer Perceptrons (MLPs/ANNs)**:
- Neural network-based policies that can represent arbitrarily complex nonlinear mappings
- More parameters than RBFs, potentially harder to optimize but more flexible
- Emerging evidence of advantages in high-dimensional state spaces

**Rule-based (parameterized existing rules)**:
- Parameterize existing institutional rule structures (e.g., FFMP thresholds, factors)
- Preserve institutional interpretability and acceptability
- May miss performance gains available through more flexible policy forms
- Directly comparable to status quo operations

### 4.3 Comparison Across Formulations

A key methodological question is whether "opening up" the policy space (from parameterized rules to RBFs to MLPs) yields meaningful performance improvements. This can be framed as a "value of flexibility" analysis, comparing Pareto fronts across formulations. Prior work has shown that more flexible policies often dominate parameterized rules, but the marginal value of additional flexibility varies by system and objective set.


## 5. Scenario Generation and Robustness Analysis

### 5.1 Stochastic Streamflow Generation

For re-evaluation of optimized policies, synthetic streamflow ensembles provide plausible alternative hydrological futures. The Pywr-DRB ecosystem uses the Kirsch-Nowak (KN) generator, which:

- Preserves key statistical properties (mean, variance, skew, autocorrelation, cross-correlation)
- Generates long synthetic sequences (e.g., 70-year realizations)
- Can be conditioned on climate scenarios (stationary, drier, wetter)
- Existing infrastructure in StochasticExploratoryExperiment provides reference code

**Key reference**: Kirsch, B. R., Characklis, G. W., & Zeff, H. B. (2013). Evaluating the impact of alternative hydro-climate scenarios on transfer agreements: Practical improvement for generating synthetic streamflows. *Journal of Water Resources Planning and Management*, 139(4), 396-406.

### 5.2 Re-evaluation Protocol

The standard re-evaluation protocol:

1. Optimize policies under a reference scenario (e.g., historical or median climate)
2. Simulate each Pareto-approximate policy across the full ensemble (e.g., 100-1000 synthetic streamflow realizations)
3. Compute performance metrics for each policy-scenario pair
4. Evaluate robustness using satisficing, regret, or other metrics
5. Apply scenario discovery (PRIM/CART) to identify vulnerability conditions

### 5.3 Scenario Discovery

Patient Rule Induction Method (PRIM) and Classification and Regression Trees (CART) are used to identify regions of the uncertainty space where policies fail:

- PRIM iteratively peels away data to find "boxes" in input space that are highly predictive of policy failure
- CART provides interpretable decision trees
- Results inform stakeholders about which uncertain conditions pose the greatest risk to candidate policies

**Key reference**: Bryant, B. P., & Lempert, R. J. (2010). Thinking inside the box: A participatory, computer-assisted approach to scenario discovery. *Technological Forecasting and Social Change*, 77(1), 34-49.


## 6. Reed Group Applications in Water Resources

### 6.1 Colorado River Basin

The Bureau of Reclamation has applied Borg MOEA and MORDM-inspired approaches to explore Lake Mead operating strategies. This represents one of the highest-profile real-world applications of many-objective optimization in water resources. The analysis discovered strategies that outperform the 2007 Interim Guidelines across a wider range of hydrological scenarios than traditional planning methods explored.

### 6.2 Research Triangle (North Carolina)

Zeff et al. (2014, 2016) and Gold et al. applied MORDM to regional water supply portfolio management in the Research Triangle, examining multi-utility cooperation, infrastructure investment pathways, and transfer agreements under deep uncertainty. The "Water Pathways" simulation system emerged from this work.

**Key references**:
- Zeff, H. B., Herman, J. D., Reed, P. M., & Characklis, G. W. (2016). Cooperative drought adaptation: Integrating infrastructure development, conservation, and water transfers into adaptive policy pathways. *Water Resources Research*, 52(9).
- Gold, D. F., et al. Consequential compromises: Exploring the cooperative stability of multi-actor robustness compromises in regional infrastructure investment pathways.

### 6.3 Lake Como and Italian Systems

Giuliani, Castelletti, and Reed have extensively studied the Lake Como system in Italy using EMODPS with RBF policies, establishing many of the methodological foundations for direct policy search in multi-reservoir systems.

### 6.4 Upper Colorado River Basin

Hadjimichael et al. (now at Penn State) have examined water scarcity vulnerabilities in the Upper Colorado basin, emphasizing the importance of consistent model representations when exploring deep uncertainty.

### 6.5 Pywr-DRB and Delaware Basin

Hamilton, Amestoy, and Reed (2024) published the Pywr-DRB model for water availability and drought risk assessment in the DRB. The NYCOperationExploration repository extends this with global sensitivity analysis of FFMP parameters. The current NYCOptimization study builds directly on this foundation.


## 7. Adaptive Policy Pathways

Dynamic Adaptive Policy Pathways (DAPP) offer a complementary framework to MORDM by explicitly sequencing decisions over time:

- **Adaptation tipping points**: Conditions under which a current policy no longer meets objectives
- **Pathways maps**: Show sequences of available adaptations and their dependencies
- **Monitoring signposts**: Observable indicators that trigger pathway transitions

Integration of MORDM with DAPP has been explored by Kwakkel et al. (2016), combining the strengths of optimization-based solution discovery with adaptive sequencing. This could be relevant if the DRB study considers staged implementation of operational changes.

**Key reference**: Haasnoot, M., Kwakkel, J. H., Walker, W. E., & ter Maat, J. (2013). Dynamic adaptive policy pathways: A method for crafting robust decisions for a deeply uncertain world. *Global Environmental Change*, 23(2), 485-498.


## 8. Visualization and Decision Support

### 8.1 Parallel Coordinates

The standard visualization for many-objective tradeoff analysis. Each axis represents an objective, and each polyline represents a solution. Tradeoffs appear as crossing lines between axes, and synergies as parallel lines. The Parasol library (developed within the Reed Group) provides interactive filtering and exploration.

### 8.2 Tradeoff Visualization

Beyond parallel coordinates, effective tradeoff communication includes:
- Pairwise scatterplots of objectives
- Brushing/linking across views
- Glyph plots for high-dimensional solutions
- Scenario maps showing robustness across uncertainty dimensions

### 8.3 Stakeholder Engagement

Visual analytics are critical for translating optimization results into actionable insights for decision makers. The MORDM framework explicitly includes stakeholder interaction at multiple stages, particularly in problem formulation and tradeoff exploration.


## 9. State of the Art and Emerging Directions

### 9.1 Machine Learning Integration

Recent work explores using ML to accelerate simulation-optimization:
- Surrogate models (neural networks, Gaussian processes) to approximate expensive simulations
- Feasibility-guided evolutionary algorithms using ML classifiers to steer search toward feasible regions
- Transfer learning across problem instances

### 9.2 Multi-Formulation Comparison

Comparing alternative problem formulations (different objective sets, decision variable representations, constraint structures) is increasingly recognized as essential. Different formulations can lead to qualitatively different solution sets and policy recommendations.

### 9.3 Coupled Human-Natural Systems

HydroCNHS and similar frameworks integrate human decision-making feedback into hydrological simulation, recognizing that human responses to drought (e.g., demand reduction, policy changes) alter system dynamics.

### 9.4 Nonstationarity

Moving beyond stationary assumptions requires methods that can handle time-varying probability distributions. This includes nonstationary extreme value analysis, trend-aware stochastic generation, and dynamic policy adaptation.


## 10. Key Software and Tools

| Tool | Purpose | Reference |
|------|---------|-----------|
| Borg MOEA | Multi-objective optimization | Hadka & Reed, 2013 |
| Multi-Master Borg | Parallel Borg for HPC | Hadka & Reed, 2015 |
| MOEAFramework | MOEA diagnostics and benchmarking | moeaframework.org |
| Rhodium | MORDM workflow in Python | RAND/Reed Group |
| Parasol | Interactive parallel coordinates | Reed Group |
| Exploratory Modeling Workbench | Scenario discovery and RDM | Kwakkel, 2017 |
| Pywr-DRB | DRB simulation model | Hamilton et al., 2024 |
| SALib | Sensitivity analysis (Sobol, etc.) | Herman & Usher, 2017 |
| PRIM (sdtoolkit) | Scenario discovery | Bryant & Lempert, 2010 |


## 11. Gaps and Opportunities for NYCOptimization

Based on this review, several gaps and opportunities emerge:

1. **No prior MOEA optimization of DRB operations**: Sensitivity analysis has been performed (NYCOperationExploration), but full many-objective optimization of FFMP parameters has not been published.

2. **Policy formulation comparison**: Comparing parameterized FFMP rules vs. RBF vs. MLP policies would directly contribute to the direct policy search literature while being practically relevant for DRB stakeholders.

3. **Robustness under stochastic ensembles**: The StochasticExploratoryExperiment provides the infrastructure for re-evaluation, but applying this to optimized (rather than default) policies is novel.

4. **Multi-stakeholder objectives**: The decree party structure naturally maps to a multi-stakeholder MORDM analysis where different parties weight objectives differently.

5. **Flood-drought tradeoffs**: The tension between maintaining void for flood mitigation and maintaining storage for drought resilience is underexplored in the optimization literature for this basin.

6. **Salt front considerations**: As sea level rise intensifies, incorporating salinity objectives into the optimization could provide forward-looking policy recommendations.
