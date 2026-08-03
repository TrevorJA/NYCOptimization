# Supporting Information S1 for "Hazard-Space Scenario Selection for Robust Many-Objective Reservoir Policy Search in the Delaware River Basin"

Trevor J. Amestoy¹

¹ Reed Research Group, School of Civil and Environmental Engineering, Cornell University, Ithaca, NY, USA

*Draft status (2026-08-03). Texts S2 through S6 and S8 carry drafted content from completed pre-campaign experiments; the remaining texts are outlined because their content depends on experiments that have not yet run. All embedded figures are working versions from pilot-scale or pre-campaign diagnostics; the selector diagnostics (Text S4) will be regenerated on the production candidate pool, and every figure is replaced by its production counterpart before submission. Text numbering follows the order of first reference from the main text.*

---

## Contents

**Text S1.** Trimmed-model verification (outlined). **Text S2.** Synthetic streamflow generation, seed architecture, and determinism (partially drafted). **Text S3.** Hazard metrics, screening, and the selection axes (drafted). **Text S4.** Search-ensemble build diagnostics (drafted at pilot scale). **Text S5.** Objective formulation support (drafted). **Text S6.** Forcing-space parameterization for the re-evaluation ensemble (drafted). **Text S7.** MOEA runtime diagnostics (outlined). **Text S8.** Computational scaling and campaign cost on Anvil (drafted). **Text S9.** Reference-set analyses (outlined). **Text S10.** Supplemental comparison analyses (outlined). **Text S11.** Scenario-design survey table (outlined).

Figures S1 through S24. Tables to be added with the production diagnostics.

---

## Text S1. Trimmed-Model Verification

[Outlined.] This text will demonstrate that the trimmed Pywr-DRB configuration used during search and re-evaluation (main-text Section 2.2), in which the releases of the non-NYC reservoirs are pre-simulated under default rules and supplied as boundary inflows, reproduces full-model objective values under matched inputs and decision variables. The planned check evaluates the default FFMP baseline and a small set of reference policies both ways, on the historical trace and on a slice of the re-evaluation ensemble, and reports objective-by-objective agreement statistics and residuals. The structural condition under which the approximation holds is that the boundary releases are policy-independent, so they are computed once per realization and reused for every policy. A separate reproducibility check under the production configuration found objective values deterministic across repeated fresh-process evaluations to a worst relative deviation of $2 \times 10^{-13}$.

## Text S2. Synthetic Streamflow Generation, Seed Architecture, and Determinism

[Partially drafted.] This text will describe the Kirsch-Nowak generation pipeline as implemented, comprising monthly Cholesky-based generation preserving cross-site and lag correlations, nearest-neighbour daily disaggregation, and the downstream-gauge fill, together with the statistical validation of generated flows against the observed record (monthly moments, lag-1 and cross-site correlation, and flow-duration curves; figures to be produced).

**Seed architecture and determinism.** This text documents the deterministic seed architecture that the study's controls rely on. Namespaced, hash-derived seed domains keep the candidate pool, the probabilistic ensembles, the selector's anchor plans, and the re-evaluation ensemble on provably disjoint random streams. Global realization indexing makes generation invariant to how work is partitioned across processes, verified by bit-identical serial-versus-sharded comparison, and any single realization is regenerable exactly from its stored seed and index alone. Two consequences carry methodological weight. First, a candidate pool of $10^6$ members is stored as its hazard image and seeds only, with selected members regenerated on demand (main-text Section 3.2.2). Second, a prefix of a generated pool is itself an exact i.i.d. pool of the smaller size, so the nested pool-size diagnostic of Text S4 is internally consistent.

**Interannual persistence.** The generator reproduces monthly and cross-site structure but, like the annual bootstrap it descends from, carries essentially no interannual memory. Water-year lag-1 autocorrelation is approximately 0.10 in generated flows against 0.26 in the fitted record. The record's own persistence is epochal (lag-1 of 0.51 before 1980 and 0.06 after, within an approximately 43-year pluvial), and the CMIP6 ensemble's apparent persistence changes are statistically indistinguishable from 39-year sampling variability. Persistence is therefore stressed in neither the search population nor the re-evaluation ensemble, and every claim is scoped accordingly (main-text Section 3.4.1). Figure S1 summarizes the characterization.

![Figure S1](../../outputs/supplemental/persistence_axis/persistence_axis.png)

**Figure S1.** Interannual persistence characterization. Water-year lag-1 autocorrelation in the fitted record and in generated flows, the epochal structure of the record's persistence, and the CMIP6 ensemble's persistence changes relative to 39-year sampling variability.

## Text S3. Hazard Metrics, Screening, and the Selection Axes

**Descriptor definitions.** All hazard coordinates are computed on the aggregate inflow to the three NYC reservoirs, with the same six-month metric exclusion applied by date as in the objectives (main-text Section 3.2.3). The five drought descriptors derive from run theory on the six-month Standardized Streamflow Index (SSI-6), with the gamma SSI fit made once on the observed record per calendar month. An event is a run of SSI-6 below zero that reaches minus one or lower, terminated by three consecutive non-negative months, and the controlling event, the one with the largest cumulative deficit, is scored. Deficit volume is the absolute accumulated SSI over the run, duration its length in months, peak depth the absolute minimum SSI, onset rate the peak depth divided by months from onset to peak, and recovery rate the peak depth divided by the recovery period. All five are zero for a window with no qualifying event (0.7 percent of pool windows). The three flood descriptors derive from a peaks-over-threshold analysis of daily flows at the 95th percentile of the observed record, scoring the critical pulse, the exceedance run containing the window's maximum daily flow, so exactly one pulse is scored per window with no declustering parameter. Peak magnitude is the window maximum normalized by the observed mean daily flow, pulse duration the days above threshold within the run, and rise rate the largest one-day rise on the rising limb, likewise normalized. The threshold and normalizing mean are fixed once on the observed record, so coordinates are comparable across the pool, the realized ensembles, and the re-evaluation ensemble.

**The redundancy screen.** Before selection, descriptors are screened on the pool's hazard image. Degenerate descriptors (near-zero spread) are dropped, and near-duplicate groups at absolute Spearman rank correlation of at least 0.95 are pruned to one canonical member. On this population the screen retains all eight candidates. The largest pairwise rank correlation is 0.88, between deficit volume and duration; the next largest between-concept correlations are 0.61 (deficit volume and peak depth) and 0.60 (peak magnitude and rise rate). The retained set is unchanged if the pruning cut is tightened to 0.90 and is stable across pool sizes from $2 \times 10^3$ to $10^6$ members. The full rank-correlation structure, characterized following Olden and Poff (2003), is shown in Figure S2.

![Figure S2](../../outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/figures/F6_axis_screen.png)

**Figure S2.** The descriptor screen and rank-correlation structure of the eight candidate hazard descriptors on the pool's hazard image.

**The selection axes.** Six of the eight descriptors enter the selection distance (main-text Section 3.2.3): deficit volume, peak depth, onset rate, and recovery rate on the drought side, and peak magnitude and pulse duration on the flood side. Drought duration and flood rise rate are excluded from the distance, duration as the most redundant retained pair member and rise rate as entangled with the retained flood pair, and both remain computed for every pool member and reported post-hoc.

**Selection invariance.** A leave-one-axis-out analysis shows no single axis dominates the selection. Removing any one axis changes the selected set by a comparable amount (Jaccard similarity 0.18 to 0.27 against the full-set selection), and the per-axis shares of squared snap displacement are near-uniform (0.08 to 0.16 against 0.125 under exact uniformity). The drought and flood concept groups carry weight in proportion to their axis counts, so the selection distance encodes no hidden re-weighting beyond the disclosed axis composition (Figure S3).

![Figure S3](../../outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/figures/F10_invariance.png)

**Figure S3.** Selection-invariance diagnostics. Leave-one-axis-out overlap with the full-set selection and per-axis shares of squared snap displacement.

## Text S4. Search-Ensemble Build Diagnostics

This text verifies that the hazard-filling selector achieves the intended distribution shift, which is a precondition for interpreting the main comparison. Coverage statistics are reported as method verification only; the design comparison is made exclusively on re-evaluated robustness (main-text Section 3.5). The results below are from the diagnostic battery at pilot scale (a 2,000-member pool, $N = 100$, ten selector seeds, with a 50-seed random null); the battery is rerun on each production pool, and each production draw's realized coverage, bounds, and clipped fractions are persisted as build quality control.

**Realized coverage.** In the campaign geometry the selector attains an L2-star discrepancy of 0.023 against 0.132 for a random subset of the same size, places on average 26 percent of selected members above the pool's 90th percentile on each axis against the 10 percent that i.i.d. sampling delivers, and selects at least one member in the severe corner of 95 percent of axes against 51 percent for random selection. Every axis is individually better stratified than the null. The selector also avoids the pool's zero-event atom (windows with no qualifying drought event), whereas a maximin selector over-samples that corner. Figures S4 through S6 show the selection geometry, the coverage comparison against the random null, and the tail-enrichment and zero-event-atom behavior.

![Figure S4](../../outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/figures/F1_selection_scatter.png)

**Figure S4.** Selected members in the scaled hazard space, with Latin hypercube anchors and their snapped pool members across pairs of selection axes.

![Figure S5](../../outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/figures/F2_coverage_vs_null.png)

**Figure S5.** Realized coverage of the hazard-filling selection against a random subset of the same size, including L2-star discrepancy and per-axis stratification.

![Figure S6](../../outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/figures/F3_tail_and_atom.png)

**Figure S6.** Per-axis tail enrichment above the pool's 90th percentile and treatment of the zero-event atom, for the hazard-filling selection and the random null.

**Selector simplicity checks.** An optimal one-to-one assignment of anchors to members (Hungarian algorithm) is indistinguishable from the greedy nearest-neighbour snap on every coverage metric, so the snap's order dependence is immaterial and the simpler rule is retained. Selections built on disjoint halves of the pool achieve identical tail enrichment, evidence that the construction is stable under replication of its inputs. Figure S7 reports the snap-distance and member-separation distributions.

![Figure S7](../../outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/figures/F4_snap_and_separation.png)

**Figure S7.** Anchor-to-member snap distances and pairwise separation of selected members, for the greedy snap and the optimal-assignment variant.

**Robust bounds.** Sweeping the normalization bounds confirms the robust p1/p99 choice (main-text Section 3.2.4). Full-range scaling degrades realized coverage roughly threefold, the outlier-fixation failure mode, while coverage and enrichment vary smoothly across nearby robust-percentile settings, with no sensitivity cliff at the default (Figure S8).

![Figure S8](../../outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/figures/F5_bounds_sweep.png)

**Figure S8.** Coverage and enrichment across normalization-bound settings, from full-range scaling to interior robust-percentile bounds.

**Dimension and pool size.** Mean anchor-to-member snap distance roughly triples between four-axis and eight-axis selection on the same pool, with per-axis tail enrichment falling accordingly, the finite-pool dilution that motivates restricting the selection space to six axes (Figures S9 and S10). The nested pool-size diagnostic repeats the selection on nested prefixes of a single $10^6$-member pool, which are exact i.i.d. pools of every intermediate size (Text S2). On the six selection axes at $P = 10^6$, every axis places at least 31 percent of selected members above the pool's 90th percentile, and enlarging $N$ at fixed pool size lowers rather than raises this enrichment, supporting the campaign $N = 100$ (main-text Section 3.3.1; Figure S11).

![Figure S9](../../outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/figures/F7_per_axis_coverage.png)

**Figure S9.** Per-axis marginal coverage of the selection across the six selection axes.

![Figure S10](../../outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/figures/F8_snap_vs_dimension.png)

**Figure S10.** Snap distance as a function of the number of selection axes at fixed pool size.

![Figure S11](../../outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/figures/F9_n_sweep.png)

**Figure S11.** Per-axis tail enrichment as a function of ensemble size $N$ at fixed pool size.

**Compositional divergence.** For each production draw this text will report the exceedance frequencies of drought severity, duration, and flood magnitude above high pool quantiles in the realized HF and PS ensembles, the direct measurement of how far the selector shifts the search distribution toward the severe corners relative to i.i.d. sampling.

## Text S5. Objective Formulation Support

**Epsilon calibration.** The epsilon-dominance precisions of main-text Table 2 were calibrated on the campaign ensembles themselves. For each matched design, 512 random decision vectors drawn uniformly on the constraint-feasible region, plus the default FFMP baseline, were evaluated on that design's own first-draw search ensemble. Each objective's precision is the largest of three floors, namely one tenth of the signal interquartile range (Reed et al., 2013), a bootstrap noise floor on the annual-unit estimator, and the estimator's frequency granularity, taken over the two matched designs and rounded up to a clean value. The raw requirement differs by at most a factor of 2.5 between the two designs, so a shared vector sacrifices little. The Historical reference is measured but excluded from the maximum. Its single trace supplies only 76 annual units, and its noise floors would coarsen the shared reliability precisions well beyond what the ensemble designs need; it searches at the shared precisions instead, with its archive resolving below its own sampling noise on the reliability objectives. An archive-size sweep at the calibrated vector projects roughly 300 to 400 policies per search and 1,000 to 1,200 merged Pareto policies per design across seeds and draws, the figures used to price re-evaluation (Text S8). [Open item: the calibration is rerun on the feasible population under the current decision-variable bounds before problem-encoding generation, including the flood-severity precision.]

**Formulation convention screens.** The formulation's conventions were fixed by post-hoc reductions of the epsilon-calibration policy populations (512 constraint-feasible policies plus the default-FFMP baseline per campaign design) rather than asserted. The annual failure-week counts $k$ were screened for saturation over $k \in \{1, \ldots, 4\}$ in both ensemble compositions. No adopted value saturates (worst case 0.4 percent of the population at the band edges), rankings are stable to $k \pm 1$ ($\tau_b \ge 0.94$ for NYC delivery, Montague, and NJ), and Trenton's $k = 1$ is binding: at $k = 3$ between 24 and 44 percent of policies tie at reliability of at least 0.95, and at $k = 4$ about 97 percent. The flood unit operator was fixed to the mean. At the campaign's 900 pooled unit-years the 99th-percentile variant collapses onto a few discrete values (population interquartile range near zero in the hazard-filling composition), carries a bootstrap noise 12 to 30 times that of the mean, and induces an unstable ranking (bootstrap $\tau_b$ of 0.35 to 0.60 against 0.92 to 0.93 for the mean). The same policy populations bound the flood objective's controllability. The policy-invariant empirical floor accounts for 57 to 61 percent of the baseline's flood exposure, so at least 39 to 43 percent is controllable by the decision space. The eighth objective, NJ delivery reliability, was activated by the redundancy screen (Spearman $|\rho| \le 0.38$ against every objective and $\le 0.08$ against Trenton flow reliability), with its precision already calibrated so that activation required no recalibration.

**The flood-severity metric.** The flood objective's magnitude-weighted severity form (main-text Section 3.1.2) was selected by a dedicated diagnostic on the same policy populations. The day-count form is degenerate across policies, taking few distinct integer values that move in shelves along the flood-release ladder, whereas the stage-severity form resolves the full policy population, is strictly monotone in realized flood exposure across the ensemble (Spearman $\rho = -1.00$ with the release ladder), and tracks observed annual flood magnitude. Day-count variants at the minor, major, and action stages remain reported diagnostics.

**Remaining items.** The satisficing criterion anchors and their threshold-margin diagnostics, consisting of each objective's cumulative distribution across the re-evaluation ensemble with the criterion overlaid (following Gold et al., 2023), accompany the criterion sweep of main-text Section 3.5.3. The 0.99 weekly satisfaction factor is bounded by a dedicated factor sweep on the same policy population.

## Text S6. Forcing-Space Parameterization for the Re-Evaluation Ensemble

The deeply uncertain climate forcing enters this study only in the held-out re-evaluation ensemble (main-text Section 3.4.1), so this text documents the forcing parameterization as the basis of that ensemble rather than of any search ensemble.

**Harmonic parameterization.** Each of 54 CMIP6 future-run change profiles, the ratio of future to historical monthly mean flow computed per model run, is fit with a second-order harmonic in the log change factor. The three sampled coordinates are the annual-mean level and the two harmonic amplitude ratios, which govern annual volume, seasonal amplitude, and the snowmelt-shoulder shape; the median per-profile shape $R^2$ is 0.85. The harmonic phases are held fixed at the fit to the CMIP6 ensemble-mean profile, because individual-run phases, especially the semiannual phase, are noisy, and sampling them would add dimensions that encode fit noise rather than climate information. The re-evaluation ensemble samples the three amplitudes by Latin hypercube over their full empirical range across the 54 runs, widened by a 25 percent margin. Figures S12 through S16 show the harmonic fit quality, the fitted parameter space, and the sampled envelope.

![Figure S12](../../outputs/diagnostics/forcing_parameterization/SI_harmonic_fit.png)

**Figure S12.** Second-order harmonic fit to the CMIP6 ensemble-mean monthly change profile.

![Figure S13](../../outputs/diagnostics/forcing_parameterization/SI_harmonic_best_worst_fits.png)

**Figure S13.** Best and worst individual-run harmonic fits across the 54 CMIP6 change profiles.

![Figure S14](../../outputs/diagnostics/forcing_parameterization/SI_harmonic_param_space.png)

**Figure S14.** The fitted harmonic parameter space of the 54 CMIP6 change profiles.

![Figure S15](../../outputs/diagnostics/forcing_parameterization/SI_harmonic_lhs_sampling.png)

**Figure S15.** The Latin hypercube sampling envelope over the fitted amplitudes, widened by the 25 percent margin, relative to the CMIP6-fitted points.

![Figure S16](../../outputs/diagnostics/forcing_parameterization/SI_harmonic_monthly_flow_comparison.png)

**Figure S16.** Monthly-flow envelopes implied by the sampled forcing space, compared with the CMIP6 change profiles.

**Injection.** The forcing is applied to the fitted monthly moments of the stochastic generator through the lognormal moment-matching transform of Kirsch et al. (2013). The convention preserves each month's coefficient of variation, so a mean reduction carries a proportional reduction in absolute variability and relative drought and flood behavior is not distorted by the moment arithmetic itself.

**The variance axis, evaluated and excluded.** An independent coefficient-of-variation axis, derived from each CMIP6 run's own historical-to-future variance ratio, was implemented and evaluated with a paired hazard-footprint diagnostic using matched realizations generated with and without the axis across the forcing box. Adding the axis left the drought hazard footprint essentially unchanged (span ratios 0.88 to 1.20 across the drought axes) and contracted all three flood-axis spans (0.72 to 0.83), because the CMIP6 envelope reduces relative variability in the winter and spring flood season. Three additional dimensions would therefore dilute the Latin hypercube stratification of the forcing space without adding stress coverage, and the axis is excluded from the campaign (Figure S17).

![Figure S17](../../outputs/supplemental/cv_axis_footprint/cv_axis_footprint.png)

**Figure S17.** Paired hazard-footprint diagnostic for the coefficient-of-variation axis, comparing drought-axis and flood-axis spans with and without the axis across the forcing box.

## Text S7. MOEA Runtime Diagnostics

[Outlined.] This text will report the runtime dynamics of the Multi-Master Borg searches for each design, draw, and seed, comprising archive snapshots, operator probabilities, restart behavior, and within-design hypervolume against per-design reference sets, following the diagnostic conventions of Reed et al. (2013). These diagnostics are read as within-design convergence evidence only and are never compared across designs (main-text Section 3.3.3). Because the runtime archive records intermediate NFE levels, this text will also report whether the design comparison is stable when recomputed at earlier budgets, the evidence that the 500,000-NFE target is sufficient rather than merely conventional.

## Text S8. Computational Scaling and Campaign Cost on Anvil

### S8.1 Overview

The campaign requires many independent Multi-Master (MM) Borg searches, one for each combination of design, draw, and seed, each evaluating a trimmed Pywr-DRB ensemble simulation at every function evaluation, followed by a trimmed-model re-evaluation phase on the re-evaluation ensemble (the full model is used only for the one-time presimulation passes and the historical baseline). Sizing the campaign against the compute allocation required three measurements on the Purdue Anvil system, whose nodes carry 128 AMD EPYC cores: how densely independent evaluator ranks can be packed onto a node, how per-evaluation cost scales with the ensemble shape, and the strong-scaling behavior of the MM Borg driver. The measurements include a direct sweep of the per-evaluation cost surface across ensemble size, record length, and model variant, so the campaign is priced from measurement rather than extrapolation.

### S8.2 Node-packing density

A node-packing sweep placed $k$ concurrent, fully independent evaluator ranks on one node, for $k$ from 1 to 128, each repeatedly executing an ensemble evaluation with threading pinned to one thread per rank. Per-evaluation wall time is nearly constant through half-node packing and rises by at most 17 percent at full packing, while node throughput rises almost linearly with packing density, so the cost per evaluation falls by two orders of magnitude from single-rank to full-node operation (Figures S18 and S19). The densest packing of 128 ranks per node is therefore the production choice, subject to node memory. Memory at the campaign ensemble size is not binding. At 100 realizations of 10 years the resident set is approximately 1.2 GB per rank, which projects to roughly 150 GB per node at 128 ranks, within the node's capacity (Figure S20).

![Figure S18](../../outputs/supplemental/anvil_scaling_experiment/figures/F1_packing_eval_time.png)

**Figure S18.** Per-evaluation wall time as a function of node-packing density.

![Figure S19](../../outputs/supplemental/anvil_scaling_experiment/figures/F2_packing_throughput_cost.png)

**Figure S19.** Node throughput and cost per evaluation as functions of node-packing density.

![Figure S20](../../outputs/supplemental/anvil_scaling_experiment/figures/F3_packing_memory.png)

**Figure S20.** Per-rank resident memory across the node-packing sweep.

### S8.3 The ensemble cost surface

Per-evaluation wall time was measured at 128 ranks per node across a grid of ensemble sizes ($N$ from 1 to 200) and record lengths ($L$ from 5 to 30 years), for both the trimmed and the full model (Figures S21 through S23). Three results size the campaign.

First, the campaign evaluation is inexpensive. At the campaign configuration of $N = 100$ realizations of $L = 10$ years, one trimmed-model function evaluation takes a warm median of 173.8 s, corresponding to 48.3 service units (SU) per 1,000 evaluations at full node packing.

Second, cost scales with total scenario-years, close to linearly in each factor. Power-law fits give an exponent on $N$ of 0.95 to 0.96 and an exponent on $L$ near 1.0 across the surface, so per-evaluation cost is slightly sub-linear in the number of realizations at fixed record length. The campaign cost therefore cannot be obtained by scaling a smaller ensemble linearly, which is why the surface was measured rather than extrapolated.

Third, the full model is only modestly more expensive than the trimmed model. Across the surface the full-to-trimmed time ratio is approximately 1.16, and at the campaign configuration the full-model evaluation takes 202.1 s against the trimmed model's 173.8 s. The full model appears in the campaign only in the one-time presimulation of the policy-independent boundary releases (once per staged realization, including the re-evaluation ensemble's) and in the historical baseline, so this ratio prices those passes; re-evaluation itself runs the trimmed model at the measured trimmed rate.

![Figure S21](../../outputs/supplemental/ensemble_cost_experiment/figures/F1_cost_surface.png)

**Figure S21.** The per-evaluation cost surface across ensemble size $N$ and record length $L$ for the trimmed model.

![Figure S22](../../outputs/supplemental/ensemble_cost_experiment/figures/F2_memory.png)

**Figure S22.** Per-rank memory across the ensemble cost surface.

![Figure S23](../../outputs/supplemental/ensemble_cost_experiment/figures/F3_model_ratio.png)

**Figure S23.** The full-to-trimmed model evaluation-time ratio across the cost surface.

### S8.4 MM Borg strong scaling

A strong-scaling experiment ran the MM Borg driver to a fixed function-evaluation budget while growing the evaluator pool, across single-island geometries from 8 to 64 evaluator slots and two multi-island geometries at 64 slots (Figure S24). Parallel efficiency is 0.73 at 64 slots relative to the 8-slot baseline, and at a fixed slot count island partitioning is nearly free, so island count is treated as a search-quality choice rather than a throughput choice, consistent with the reliability arguments for multi-master search made by Hadka and Reed (2015) and Zatarain Salazar et al. (2017). The strong-scaling geometry used a lighter evaluation than the campaign ensemble, so the measured efficiency is conservative for the more compute-heavy campaign evaluation.

![Figure S24](../../outputs/supplemental/anvil_scaling_experiment/figures/F4_borg_strong_scaling.png)

**Figure S24.** MM Borg strong scaling across evaluator-pool sizes and island geometries.

### S8.5 Campaign cost and the allocation

Each production search runs the MM Borg driver on eight nodes (1,024 cores) as four islands of 254 evaluator ranks, with four island masters and one controller, for 1,021 MPI ranks in total. The per-island budget is 125,000 function evaluations (500,000 in total), with runtime-archive snapshots every 2,500 island-NFE. Combining the measured campaign evaluation cost (173.8 s) with the measured strong-scaling efficiency (0.73), one 500,000-NFE search runs for approximately 32.6 h and costs approximately 33,400 SU. Because SU cost is nearly flat with node count, the eight-node geometry is a wall-time choice with no cost penalty. It halves the roughly 65-h wall time of a four-node run while remaining well inside the queue limit. Searches are NFE-bounded, with no algorithm-level wall-time cap, because such a cap could truncate NFE unequally across designs and break the equal-NFE budget condition; the scheduler wall time is the safety net and a killed run resumes from the last runtime snapshot.

The campaign consists of the two matched designs (PS and HF), each run at $K = 3$ ensemble draws and $S = 2$ MOEA seeds per draw, for 12 matched searches costing approximately 400,000 SU, plus the HIST reference run at $S = 2$ seeds. The HIST reference evaluates a single 77-year trace per function evaluation (approximately 32 s per evaluation) and adds roughly 12,000 SU at the same NFE. Ensemble generation involves no Pywr-DRB simulation. Generating one $10^6$-member candidate pool costs approximately 600 core-hours in sharded execution, so the three re-rolled pools and the remaining staging fit comfortably within a 10,000-SU allowance. Re-evaluation of every final policy on the re-evaluation ensemble runs the trimmed model (main-text Section 3.4.1) at 1,000 SOWs, 25 realizations per SOW, and 50-year records, and costs approximately 80,000 SU at the roughly 1,000 to 1,200 merged final policies implied by the calibrated epsilon precisions, including the one-time full-model presimulation pass. The campaign total is therefore approximately 503,000 SU, 67 percent of the 750,000-SU Anvil allocation, which leaves a reserve of roughly 247,000 SU. The reserve's first call is an additional ensemble draw for both matched designs (approximately 134,000 SU); the variable-resolution FFMP sweep (main-text Section 3.1.1, approximately 200,000 SU at three values, one draw, and two seeds) runs only on whatever allocation remains at the end of the campaign.

Because the runtime archive (main-text Section 3.3.3; Text S7) records intermediate NFE levels, the attained 500,000-NFE budget is justified against observed convergence behavior after the first searches, and the design comparison is recomputable at earlier budgets at re-evaluation cost only.

## Text S9. Reference-Set Analyses

[Outlined.] This text will report dominance-based summaries over the pooled re-evaluated policy pool with their caveats stated up front, namely self-reference (a design contributes points to the frontier it is scored against), cardinality asymmetry, and noise-induced spurious dominance, together with cardinality diagnostics (main-text Section 3.5.4).

## Text S10. Supplemental Comparison Analyses

[Outlined.] This text will report analyses that support but do not constitute the primary comparison: the SOW unit versus the realization unit of the robustness metric; ranking-stability tables using Kendall's $\tau_b$ across metrics and across criterion-sweep stringencies; the attainability screen in detail; the composition-sensitivity checks, in which hazard-restricted and envelope-restricted subsets of the re-evaluation ensemble are re-scored from the persisted matrix; and, as an optional supporting analysis, scenario discovery in the deeply uncertain factor space of the re-evaluation ensemble after re-evaluation, characterizing which forcing conditions drive each design's failures. The discovery analysis is reported as supporting evidence for any observed robustness difference, not as a primary result; no scenario discovery is performed in hazard space.

## Text S11. Scenario-Design Survey Table

[Outlined.] This text will present the survey of scenario designs used during search in water-resources and MOEA policy-search studies, with the design space, the ensemble sizing, and per-study verification notes, situating the PS and HF constructions within that survey (main-text Section 3.2.1).

---

## References (Supporting Information)

- Gold, D. F., Reed, P. M., Gorelick, D. E., & Characklis, G. W. (2023). Advancing regional water supply management and infrastructure investment pathways that are equitable, robust, adaptive, and cooperatively stable. *Water Resources Research*, 59(9), e2022WR033671.
- Hadka, D., & Reed, P. (2015). Large-scale parallelization of the Borg multiobjective evolutionary algorithm to enhance the management of complex environmental systems. *Environmental Modelling & Software*, 69, 353–369.
- Kirsch, B. R., Characklis, G. W., & Zeff, H. B. (2013). Evaluating the impact of alternative hydro-climate scenarios on transfer agreements: Practical improvement for generating synthetic streamflows. *Journal of Water Resources Planning and Management*, 139(4), 396–406.
- Olden, J. D., & Poff, N. L. (2003). Redundancy and the choice of hydrologic indices for characterizing streamflow regimes. *River Research and Applications*, 19(2), 101–121.
- Reed, P. M., Hadka, D., Herman, J. D., Kasprzyk, J. R., & Kollat, J. B. (2013). Evolutionary multiobjective optimization in water resources: The past, present, and future. *Advances in Water Resources*, 51, 438–456.
- Zatarain Salazar, J., Reed, P. M., Quinn, J. D., Giuliani, M., & Castelletti, A. (2017). Balancing exploration, uncertainty and computational demands in many objective reservoir optimization. *Advances in Water Resources*, 109, 196–210.
