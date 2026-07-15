# Supporting Information for "Hazard-Space Scenario Selection for Robust Many-Objective Reservoir Policy Search in the Delaware River Basin"

Trevor J. Amestoy¹

¹ Reed Research Group, School of Civil and Environmental Engineering, Cornell University, Ithaca, NY, USA

*Draft status (2026-07-15): full outline, with Section S8 (computational scaling) drafted from the completed Anvil packing/scaling experiment and the measured ensemble cost-surface experiment. All other sections are outlined only, because their content depends on experiments that have not yet run. Section numbering mirrors the main-text cross-references.*

---

## Contents

**S1. Trimmed-model verification.** This section will demonstrate that the trimmed Pywr-DRB configuration used during search, in which the releases of the non-NYC reservoirs are pre-simulated under default rules and supplied as boundary inflows, reproduces full-model objective values under matched inputs and decision variables. It will present objective-by-objective agreement statistics and residual plots, and it will state the structural conditions under which the approximation holds.

**S2. Forcing-space parameterization for the test ensemble.** The deeply uncertain climate forcing enters this study only in the held-out test ensemble (main-text Section 6.1), so this section documents the forcing parameterization as the basis of E_test rather than of any search ensemble. It will present the harmonic decomposition of the monthly change factor, the fits to the 54 CMIP6 future-run change profiles (the median per-profile shape R² is 0.85), the argument for holding the harmonic phases fixed at the canonical CMIP6 shape while sampling only the amplitudes, the amplitude ranges that define the sampling box, and the lognormal moment-matching transform through which the forcing is injected into the generator. Existing diagnostic figures cover the harmonic fit quality, the fitted parameter space, the Latin hypercube sampling envelope, the best and worst individual fits, and the monthly-flow envelope comparison.

**S3. Synthetic streamflow generation, seed architecture, and determinism.** This section will describe the Kirsch–Nowak generation pipeline as implemented, including monthly Cholesky-based generation, Nowak daily disaggregation, and the downstream-gauge fill, together with the two-generator arrangement required by the forcing transform used for E_test. It will present the statistical validation of generated flows against the observed record (monthly moments, lag-1 and cross-site correlation, and flow-duration curves), and it will document the deterministic seed architecture: disjoint seed domains for the candidate pool and the test ensemble, global realization indexing that is invariant to parallel partitioning, and exact single-realization regeneration. This architecture is what allows a candidate pool of 100,000 to 1,000,000 members to be stored as a hazard image alone.

**S4. Hazard metrics and the redundancy screen.** This section will define every candidate hazard axis, comprising the SSI-6 run-theory drought event metrics and the peaks-over-threshold flood metrics, document the single-fit SSI convention that makes coordinates comparable across the pool and the historic reference, and report the redundancy screen on the production candidate pool, including the correlation structure, the clustering, the retained axis set, and the per-axis distributions. It will be populated after the production pool is generated.

**S5. Search-ensemble build diagnostics and the manipulation check.** This section verifies that the hazard-filling selector administers the intended treatment at strength, which is a precondition for interpreting the main comparison. For each draw it will report the achieved hazard-space coverage (L2-star discrepancy and minimum-spanning-tree edge statistics, benchmarked against a random design of the same size and dimension), the anchor-to-member snap-distance distributions that diagnose pool density, and, most importantly, the compositional divergence between the hazard-filling and `fixed_probabilistic` draws: the exceedance frequencies of drought severity, duration, and flood magnitude above high pool quantiles in each design's realized ensemble, which quantify how far the absolute-space selector shifts the search distribution toward the severe corners relative to i.i.d. sampling. These coverage statistics are reported as method verification rather than as a comparison result; the design comparison is made only on re-evaluated robustness (main-text Section 6).

**S6. Objective formulation support.** This section will document the epsilon calibration for the single-trace objective set, the objective-sensitivity experiment that fixes the annual-unit epsilon values and the sampling variance of the worst-first-percentile operator at the campaign ensemble size, the flood-days unit operator choice (mean or 99th percentile), the annual failure criteria and their saturation screen (including whether the near-saturating Trenton reliability precision must be adjusted), and the decision on the optional eighth objective. It will also record the Decree and FFMP threshold anchors and the criterion-sweep grid used in main-text Section 6.4.

**S7. MOEA runtime diagnostics.** This section will report the runtime dynamics of the Multi-Master Borg searches for each design, draw, and seed, including archive snapshots, operator probabilities, restart behavior, and within-design hypervolume against per-design reference sets, following the diagnostic conventions of Reed et al. (2013). These diagnostics are read as within-design convergence evidence only and are never compared across designs (main-text Section 5.3). Because the runtime archive records intermediate NFE levels, this section will also report whether the design comparison is stable when recomputed at earlier budgets, which is the evidence that the 500,000-NFE target is sufficient and not merely conventional.

**S8. Computational scaling and campaign cost on Anvil.** Drafted in full below.

**S9. Reference-set analyses.** This section will compare pooled and design-leave-one-out reference sets for the re-evaluated policy pool, quantify the self-reference inflation of each design, and report cardinality diagnostics (main-text Section 6.5).

**S10. Supplemental comparison analyses.** This section will report analyses that support but do not constitute the primary comparison: the state-of-the-world unit versus the realization unit of the robustness metric, ranking-stability tables using Kendall's τ_b across metrics and across criterion-sweep levels, the attainability screen in detail, and, as an optional mechanism analysis, whether the hazard-filling policies' vulnerabilities on E_test are concentrated in the hazard regions the design under-represented. The mechanism analysis is reported as supporting evidence for any observed robustness difference, not as a primary result.

**S11. Scenario-design survey table.** This section will present the survey of scenario designs used during search in water-resources and MOEA policy-search studies, with the design space, the ensemble sizing, and per-study verification notes, situating the `fixed_probabilistic` and `hazard_filling` constructions within that survey.

---

## S8. Computational Scaling and Campaign Cost on Anvil

### S8.1 Overview

The campaign requires many independent Multi-Master (MM) Borg searches, one for each combination of design, draw, and seed, each evaluating a trimmed Pywr-DRB ensemble simulation at every function evaluation, followed by a re-evaluation phase on the full model. Sizing the campaign against the compute allocation required three measurements on the Purdue Anvil system, whose nodes carry 128 AMD EPYC cores: how densely independent evaluator ranks can be packed onto a node, how per-evaluation cost scales with the ensemble shape, and the strong-scaling behavior of the MM Borg driver. Unlike an earlier packing and strong-scaling experiment that used a single stand-in ensemble and could not price the campaign, the measurements reported here include a direct sweep of the per-evaluation cost surface across ensemble size, record length, and model variant, so the campaign is priced from measurement rather than extrapolation.

### S8.2 Node-packing density

A node-packing sweep placed *k* concurrent, fully independent evaluator ranks on one node, for *k* from 1 to 128, each repeatedly executing an ensemble evaluation with threading pinned to one thread per rank. Per-evaluation wall time is nearly constant through half-node packing and rises by at most 17 percent at full packing, while node throughput rises almost linearly with packing density, so the cost per evaluation falls by two orders of magnitude from single-rank to full-node operation. The densest packing of 128 ranks per node is therefore the production choice, subject to node memory. Memory at the campaign ensemble size is not binding: at 100 realizations of 10 years the resident set is approximately 1.2 GB per rank, which projects to roughly 150 GB per node at 128 ranks, within the node's capacity. [TABLE S8-1 and FIGURES S8-1 through S8-3 present the packing sweep; source `packing_summary.csv` and figures F1–F3 of the packing experiment.]

### S8.3 The ensemble cost surface

Per-evaluation wall time was measured at 128 ranks per node across a grid of ensemble sizes (*N* from 1 to 200) and record lengths (*L* from 5 to 30 years), for both the trimmed and the full model (Table S8-2). Three results size the campaign.

First, the campaign evaluation is inexpensive. At the campaign configuration of *N* = 100 realizations of *L* = 10 years, one trimmed-model function evaluation takes a warm median of 173.8 s, corresponding to 48.3 service units (SU) per 1,000 evaluations at full node packing.

Second, cost scales with total scenario-years, close to linearly in each factor. Power-law fits give an exponent on *N* of 0.95 to 0.96 and an exponent on *L* near 1.0 across the surface (Table S8-3), so per-evaluation cost is slightly sub-linear in the number of realizations at fixed record length. This confirms that the campaign cost cannot be obtained by scaling a smaller ensemble linearly, and it is why the surface was measured rather than extrapolated.

Third, the full model is only modestly more expensive than the trimmed model. Across the surface the full-to-trimmed time ratio is approximately 1.16, and at the campaign configuration the full-model evaluation takes 202.1 s against the trimmed model's 173.8 s (Table S8-4). Because re-evaluation uses the full model but is performed once per final policy rather than at every function evaluation, this ratio makes re-evaluation a small fraction of the campaign cost and allows E_test to be sized generously.

[TABLE S8-2: Ensemble cost surface — for each (N, L, model variant): scenario-years, warm-evaluation median and interquartile range, SU per 1,000 evaluations, resident set per rank, and node memory at 128 ranks. Source `cost_surface.csv`; figure F1.]

[TABLE S8-3: Power-law scaling fits of per-evaluation cost versus N (at fixed L) and versus L (at fixed N), with exponents and standard errors. Source `scaling_fits.csv`.]

[TABLE S8-4: Full-versus-trimmed model cost ratio across the surface. Source `model_ratio.csv`; figure F3.]

### S8.4 MM Borg strong scaling

A strong-scaling experiment ran the MM Borg driver to a fixed function-evaluation budget while growing the evaluator pool, across single-island geometries from 8 to 64 evaluator slots and two multi-island geometries at 64 slots. Parallel efficiency is 0.73 at 64 slots relative to the 8-slot baseline, and at a fixed slot count island partitioning is nearly free, so island count is treated as a search-quality choice rather than a throughput choice, consistent with the reliability arguments for multi-master search made by Hadka and Reed (2015) and Zatarain Salazar et al. (2017). [TABLE S8-5 and FIGURE S8-4 present the strong-scaling results; source `borg_summary.csv` and figure F4 of the packing experiment. The strong-scaling geometry used a lighter evaluation than the campaign ensemble, so the measured efficiency is conservative for the more compute-heavy campaign evaluation.]

### S8.5 Campaign cost and the allocation

Combining the measured campaign evaluation cost (173.8 s) with the measured strong-scaling efficiency (0.73), one search of 500,000 function evaluations costs approximately 33,300 SU. The campaign consists of two matched designs (`fixed_probabilistic` and `hazard_filling`), each run at *K* = 3 ensemble draws and *S* = 2 MOEA seeds per draw, for 12 matched searches, plus the `historic` reference run at *S* = 2 seeds. At 500,000 NFE the 12 matched searches cost approximately 400,000 SU. The `historic` reference evaluates a single long trace per function evaluation and is comparatively cheap, adding roughly 10,000 SU. Re-evaluation of every final policy on E_test uses the full model but is performed once per policy rather than per function evaluation; at the provisional E_test sizing it is a low-thousands-of-SU addition. The campaign total is therefore on the order of 415,000 SU, against a total Anvil allocation of 750,000 SU, which leaves a reserve of roughly 45 percent for the variable-resolution FFMP sweep (RQ3), any additional ensemble draws indicated by the minimum-detectable-effect calculation of main-text Section 5.2, and the optional second test-ensemble construction of main-text Section 6.1. [TABLE S8-6: campaign projection by NFE, draws, seeds, and E_test sizing, with per-run and total SU and the fraction of allocation. Source `campaign_projection.csv`; figure F4 of the cost-surface experiment.]

Two operational notes follow from these numbers. First, service units, not the allocation, are comfortably within budget, but wall time per search is the binding practical constraint: a 500,000-NFE search at four nodes runs for roughly 65 h, which exceeds typical queue limits, so production searches are distributed across 8 to 16 nodes to bring wall time under about 32 h. Because SU cost is nearly flat with node count over this range, this is a scheduling choice with no cost penalty. Second, the 500,000-NFE budget is a target rather than a fixed commitment; the runtime archive (main-text Section 5.3, Supporting Information Section S7) records intermediate NFE levels so that the attained budget can be justified against convergence behavior after the first searches, and a lower budget adopted if search has plateaued, which would proportionally reduce the campaign cost.

---

## References (Supporting Information)

- Hadka, D., & Reed, P. (2015). Large-scale parallelization of the Borg multiobjective evolutionary algorithm to enhance the management of complex environmental systems. *Environmental Modelling & Software*, 69, 353–369.
- Reed, P. M., Hadka, D., Herman, J. D., Kasprzyk, J. R., & Kollat, J. B. (2013). Evolutionary multiobjective optimization in water resources: The past, present, and future. *Advances in Water Resources*, 51, 438–456.
- Zatarain Salazar, J., Reed, P. M., Quinn, J. D., Giuliani, M., & Castelletti, A. (2017). Balancing exploration, uncertainty and computational demands in many objective reservoir optimization. *Advances in Water Resources*, 109, 196–210.
