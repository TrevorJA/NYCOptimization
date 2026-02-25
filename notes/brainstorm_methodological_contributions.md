# Brainstorm: Novel Methodological Contributions

Trevor Amestoy, February 2026
*Working document for iterative refinement*

---

The basin-specific novelty (first MOEA optimization of DRB/FFMP, multi-formulation comparison in this system, robustness of DRB policies) is real but not sufficient for a strong dissertation chapter. Below are targeted ideas for methodological contributions that could be demonstrated through the DRB application but generalize beyond it.

---

## Idea 1: Formulation Diagnostics - Quantifying the Value of Policy Flexibility

**The gap**: Multiple studies compare alternative policy representations (parameterized rules vs. RBFs vs. ANNs), but there is no standardized diagnostic framework for quantifying *where* in objective space additional flexibility yields gains, *when* those gains survive re-evaluation under uncertainty, and *why* certain formulations dominate in certain objective dimensions.

**The contribution**: Develop a "formulation diagnostic" methodology that goes beyond simply overlaying Pareto fronts. Specifically:

- Hypervolume contribution decomposition by formulation, showing which regions of objective space are uniquely accessible to each policy form
- Robustness-conditional formulation dominance: does the ranking of formulations change when evaluated under deep uncertainty vs. the reference scenario?
- Dimensionality-efficiency curves: plot Pareto front quality (hypervolume) against decision variable count, identifying diminishing returns in policy complexity

**Why it matters**: Zatarain Salazar et al. (2024) showed RBF architecture choices affect outcomes, but nobody has provided a systematic diagnostic toolkit for answering "is additional policy complexity worth it?" This is directly actionable for practitioners deciding whether to propose flexible policies to institutional stakeholders.

**DRB demonstration**: The three-formulation setup (FFMP, FFMP+, RBF) provides the perfect testbed since they are nested in complexity and the institutional context creates real stakes around interpretability vs. performance.

**Feasibility**: High. Uses existing optimization outputs, no additional infrastructure needed.

---

## Idea 2: Scenario-Conditional Tradeoff Topology

**The gap**: Standard MORDM re-evaluates policies across scenarios and computes aggregate robustness metrics, but the *structure* of tradeoffs can change qualitatively across scenarios. A policy set that shows a smooth tradeoff between water supply and ecology under normal hydrology might exhibit a cliff-edge collapse under drought. Nobody systematically characterizes how Pareto front topology (shape, extent, knee points, discontinuities) varies across scenario conditions.

**The contribution**: Develop metrics and visualizations for scenario-conditional tradeoff analysis:

- Track how the Pareto front shape (convexity, extent, number of knees) varies across stochastic realizations
- Identify "tradeoff regime shifts" where the relationship between objectives fundamentally changes
- Connect tradeoff topology changes to specific hydrological conditions (drought severity, timing, sequencing) using scenario discovery methods

**Why it matters**: Decision makers care not just about "how robust is my policy" but "under what conditions do the tradeoffs I'm facing look fundamentally different?" This is a richer diagnostic than scalar robustness metrics provide.

**DRB demonstration**: The flood-drought tension is ideal. Under wet conditions, flood risk dominates and storage resilience is cheap. Under drought, the tradeoff between NYC supply and Montague compliance becomes binding while flood risk is irrelevant. Characterizing this regime-dependent tradeoff structure is novel.

**Feasibility**: Medium-high. Requires the stochastic re-evaluation to be well-designed, but the analysis is post-processing on existing outputs.

---

## Idea 3: Explainable Scenario Discovery via SHAP/LIME

**The gap**: PRIM and CART remain the standard tools for scenario discovery in MORDM. They identify "boxes" or "trees" in uncertainty space that predict policy failure, but they struggle with (1) correlated uncertainties, (2) nonlinear interaction effects, and (3) providing continuous importance measures rather than binary classification.

**The contribution**: Replace or augment PRIM/CART with gradient-based or model-agnostic explainability methods (SHAP, LIME) applied to the policy re-evaluation dataset:

- Train gradient-boosted models (XGBoost) on [scenario features, policy parameters] -> [objective outcomes]
- Use SHAP values to decompose each policy's performance into contributions from individual uncertainty dimensions
- Compare SHAP-based vulnerability drivers across formulations: do more flexible policies shift which uncertainties matter most?
- Use SHAP interaction values to identify synergistic vulnerabilities (e.g., low inflow + high demand is worse than the sum of their individual effects)

**Why it matters**: The WaterProgramming blog introduced LIME for water resources in 2024, and a recent WRR paper (2025) examined robustness of SHAP explanations. But nobody has deployed these as a *replacement* for PRIM/CART in a full MORDM workflow and compared the resulting vulnerability narratives.

**DRB demonstration**: The DRB has correlated uncertainties (wet/dry years affect multiple tributaries simultaneously, seasonal patterns matter, drought sequences are autocorrelated). SHAP should capture interaction effects that PRIM boxes miss.

**Feasibility**: High. Standard ML libraries. The key work is framing the analysis and comparing it against traditional scenario discovery.

---

## Idea 4: Multi-Stakeholder Robustness Decomposition

**The gap**: MORDM typically evaluates robustness using a single satisficing criterion applied uniformly. But in multi-stakeholder basins, different actors have different performance thresholds and risk tolerances. A policy that is "robust" for NYC may be fragile for NJ, and vice versa. No framework systematically decomposes robustness by stakeholder and analyzes the *equity* of robustness across parties.

**The contribution**: Define stakeholder-specific robustness profiles and analyze their joint distribution:

- Each decree party (NYC, NJ, PA, DE, environmental interests) defines their own satisficing thresholds on their priority objectives
- Compute per-stakeholder robustness (fraction of scenarios meeting that party's thresholds)
- Analyze the joint distribution: are there policies where all stakeholders achieve acceptable robustness? How does this "robustness equity" frontier compare to the performance Pareto front?
- Apply Gini coefficients or other inequality metrics to robustness distributions across stakeholders

**Why it matters**: Fletcher et al. (2022) and Quinn's group have advocated for equity in water resources optimization, but nobody has applied equity metrics to *robustness itself* rather than just performance outcomes. This connects the equity literature to the MORDM literature in a novel way.

**DRB demonstration**: The five decree parties with legally defined interests make this a natural application. The unanimous consent requirement for operational changes means robustness equity is not just academically interesting but institutionally necessary.

**Feasibility**: High. Requires defining stakeholder-specific thresholds (which maps naturally to the decree party structure) and computing per-party robustness from existing re-evaluation data.

---

## Idea 5: Active Formulation Selection via Surrogate-Guided Search

**The gap**: Standard practice is to optimize each formulation independently and compare results post-hoc. But with expensive simulations (~30s/eval), running full 1M NFE optimizations for three formulations is computationally demanding. Can we use early optimization results to guide computational allocation across formulations?

**The contribution**: Develop an adaptive meta-optimization protocol:

- Run initial diagnostic optimization (e.g., 100K NFE) for all three formulations
- Build surrogate models (Gaussian process or random forest) of the hypervolume improvement rate for each formulation
- Allocate remaining computational budget preferentially to formulations showing the steepest improvement, using a multi-armed bandit or Bayesian optimization framework
- Compare final results against the "brute force" approach of equal allocation

**Why it matters**: As problem dimensionality grows (especially for Formulation C with 100+ variables), computational allocation becomes a first-order concern. This contributes to the "computational efficiency of MORDM" literature, which is underdeveloped relative to the methodological complexity of the full workflow.

**DRB demonstration**: The three formulations have very different dimensionalities (25-35, 40-60, 100-300), so convergence rates will differ substantially. Early allocation decisions could save significant HPC time.

**Feasibility**: Medium. Requires implementing the meta-optimization wrapper, which adds complexity but is conceptually straightforward. Risk: the overhead of surrogate fitting may not pay off if all three formulations are run to completion anyway.

---

## Idea 6: Narrative Scenario Storylines for DRB Stakeholders

**The gap**: Hadjimichael et al. (2024) introduced the FRNSIC framework for narrative scenario storylines in the Upper Colorado. This is very new and has only been demonstrated in one basin. Applying and potentially extending it to the DRB would be novel.

**The contribution**: Generate narrative scenario storylines from the re-evaluation results that are directly tied to decree party concerns:

- Cluster stochastic realizations by their impact profiles across stakeholders
- For each cluster, construct a narrative storyline: "In a future characterized by [drought timing/severity], [NYC supply faces X], [NJ diversions face Y], [ecological flows face Z]"
- Map these storylines to the formulation comparison: which policy formulations are most/least vulnerable to which storylines?

**Why it matters**: Bridges the gap between quantitative optimization results and qualitative stakeholder deliberation. The DRBC's decision-making process is fundamentally deliberative (unanimous consent), so storylines that help parties understand shared and divergent vulnerabilities could be transformative.

**DRB demonstration**: Perfect fit. The governance structure demands this kind of communication tool.

**Feasibility**: Medium-high. The FRNSIC methodology is published but not yet widely replicated. Being an early adopter/extender adds novelty.

---

## Assessment Matrix

| Idea | Novelty | Feasibility | Contribution Scope | Fit with DRB | Standalone Paper Potential |
|------|---------|-------------|-------------------|--------------|---------------------------|
| 1. Formulation diagnostics | Medium-High | High | Methods + Application | Excellent | Medium (could be a section) |
| 2. Scenario-conditional tradeoff topology | High | Medium-High | Methods | Good | High |
| 3. SHAP/LIME scenario discovery | Medium-High | High | Methods | Good | Medium-High |
| 4. Multi-stakeholder robustness equity | High | High | Methods + Equity lit | Excellent | High |
| 5. Active formulation selection | High | Medium | Methods + Computation | Good | Medium |
| 6. Narrative scenario storylines | Medium | Medium-High | Methods + Communication | Excellent | Medium |

---

## Recommended Combinations

**Option A (Methodological depth)**: Ideas 1 + 2 + 3. A chapter focused on "diagnosing the value of policy flexibility under deep uncertainty," with formulation diagnostics as the backbone, scenario-conditional topology as the key analytical innovation, and SHAP-based discovery as the interpretability tool.

**Option B (Stakeholder relevance)**: Ideas 1 + 4 + 6. A chapter focused on "equitable and robust decision making for multi-stakeholder river basins," with formulation comparison as the optimization core, robustness equity as the analytical innovation, and storylines as the communication tool.

**Option C (Computational innovation)**: Ideas 1 + 3 + 5. A chapter focused on "efficient multi-formulation optimization under deep uncertainty," combining adaptive allocation with explainable vulnerability analysis.

**My instinct**: Option A or B are strongest. Option A is more "methods-forward" and would appeal to WRR or Environmental Modelling & Software. Option B is more "application-forward" and would appeal to JWRPM or Earth's Future while also being more directly useful for your DRB stakeholder engagement.

These are not mutually exclusive. The optimization and re-evaluation infrastructure is shared; the ideas differ primarily in post-processing analysis. You could pursue 3-4 of these within a single chapter if the analytical narrative holds together.
