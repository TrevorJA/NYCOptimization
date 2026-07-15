# Scenario Design in Optimization: Summary Tables

*Manuscript-candidate tables summarizing scenario design methods used during the search phase of optimization. Terminology follows `docs/notes/terminology.md`. Companion to the hub note `scenario_design.md`. All quantities verified against full texts, code releases, or dissertation chapters as documented in the verification notes at the end.*

---

## Table 1. Scenario designs used during search in water resources and MOEA policy-search studies

Rows are ordered by design family, from single reference traces through probabilistic sampling, stratified and resampled designs, multi-scenario deep-uncertainty designs, and adaptive selection.

| Study | Optimization context | Scenario design method | Design space (axes spanned) | Scenarios, no. × length (total yrs) |
|---|---|---|---|---|
| Giuliani et al. (2016) | Multi-purpose reservoir, hydropower and flood (Red River, Vietnam) | Single historical trace, hand-picked to span wet, dry, and normal years | None (one realized sequence) | 1 × 8 yr (8) |
| Kasprzyk et al. (2013) | Urban supply portfolio (LRGV, Texas) | i.i.d. Monte Carlo from fitted distributions, plus one hand-built drought scenario | Input space (inflow, demand, lease-price distributions) | 5,000 × 10 yr per FE (50,000) |
| Herman et al. (2014) | Regional supply portfolios, 4 utilities (Research Triangle, NC) | i.i.d. Monte Carlo from stationary synthetic generator | Input space (Kirsch generator, flows, evaporation, demand) | 1,000 × 13 yr per FE (13,000) |
| Quinn et al. (2017) | Shallow-lake pollution control (DPS benchmark) | i.i.d. Monte Carlo draws from pre-generated 10,000-member archive | Input space (fixed lognormal inflow distribution) | 100 × 100 yr per FE† (10,000) |
| Quinn et al. (2018) | Multireservoir flood, hydropower, supply (Red River, Vietnam) | Single long synthetic record partitioned into annual members | Input space (stationary synthetic generator) | 1,000 × 1 yr (1,000) |
| Zatarain Salazar et al. (2017) | Hydropower and environmental flows (Conowingo, Susquehanna) | Stratified subsample of 10,000-yr synthetic pool, wettest and driest years forced in | Hazard space, 1-D (empirical CDF of annual flow) | 50 / 500 / 1,000 × 1 yr (50–1,000) |
| Brodeur et al. (2020) | Reservoir control policy trees (Folsom, California) | Paleo-informed block-bootstrap window resampling | Resampled event sequencing (drought/flood ordering) | 30 × 35 yr (1,050) |
| Watson & Kasprzyk (2017) | Urban supply portfolio (LRGV, Texas) | Multi-scenario search, one optimization per scenario from prior scenario discovery | Input space (5 fixed DU scaling-factor combinations) | 5 × (5,000 × 10 yr per FE) |
| Trindade et al. (2017) | Regional supply portfolios, 4 utilities (NC) | i.i.d. Monte Carlo flows, each paired with an LHS DU vector, reshuffled per FE | Input space (synthetic flows + 13 DU factors, LHS) | 1,000 × 13 yr per FE† (13,000) |
| Eker & Kwakkel (2018) | Shallow-lake problem (multi-scenario MORDM) | Maximally diverse subset of 2,500 LHS candidates, one optimization per scenario | Outcome-space distances over input-space (5 DU factors) candidates | 1 + 4 scenarios × 100 replications |
| Bartholomew & Kwakkel (2020) | Shallow-lake problem (MORDM vs ms-MORDM vs MORO) | Reference scenario; diverse-selected set; fixed LHS set for robust search | Input space (5 DU factors) | 1 / 1 + 4 / 50 scenarios |
| Trindade et al. (2019) | Regional supply pathways, 4 utilities (NC) | Monte Carlo realizations each paired with own DU vector, reshuffled per FE | Input space (flows, demand + DU factors) | 1,000 × 45 yr per FE† (45,000) |
| Culley et al. (2016) | Lake regulation, adaptive capacity (Como, Italy) | Regular grid over exposure space, separate re-optimization per state | Exposure space (41 ΔP × 21 ΔT levels) | 861 states × 16 yr (13,776) |
| Giudici et al. (2020) | Off-grid energy-water system design (Ustica, Italy) | Active-learning subset selection from scenario pool | Outcome-informed selection from input-space pool (3,125 climate scenarios) | 1–2 × 1 yr, from pool of 3,125 |

† Resampled or reshuffled at every function evaluation (FE), so the effective set varies during search; other designs hold a fixed set.

---

## Table 2. Scenario designs in the broader stochastic optimization literature

Rows ordered from sampling-based designs through constructive discretization, reduction of existing sets, feature-space selection, and adaptive allocation.

| Study | Optimization context | Scenario design method | Design space (axes spanned) | Scenarios, no. × length (total yrs) |
|---|---|---|---|---|
| Kleywegt et al. (2002) | Stochastic discrete programs (knapsack) | Sample average approximation, replicated i.i.d. Monte Carlo samples | Input space (probability distribution of item sizes) | N ≤ 1,000 draws, static (—) |
| Infanger (1992) | Power capacity expansion; portfolio planning | Importance sampling within Benders decomposition | Input space, cost-weighted sampling density | 100–600 sampled from >5 × 10⁶ universe (—) |
| Høyland & Wallace (2001) | Multistage asset allocation | Constructive moment matching by nonlinear programming | Moment space (4 marginal moments + correlations) | 6–150 outcomes per stage, tree (—) |
| Pflug (2001) | Multiperiod portfolio optimization | Optimal discretization of the underlying process onto a fixed tree topology | Probability-metric space (transportation distance) | Tree, case-specific (—) |
| Dupačová et al. (2003); Heitsch & Römisch (2003) | Unit commitment under load uncertainty | Scenario reduction, forward selection and backward reduction | Probability-metric space (Fortet–Mourier distance) | 729 → 15–20 weekly load scenarios (—) |
| Löhndorf (2016) | Newsvendor benchmarks with analytical optima (incl. CVaR) | Systematic comparison of generation methods, MC, QMC, moment matching, Voronoi cell sampling | Probability-distribution space (all methods distribution-representative) | N swept per method, static (—) |
| Nahmmacher et al. (2016) | Power capacity expansion (LIMES-EU) | Hierarchical clustering to representative days, cluster-size weights | Feature space (daily demand + wind/solar profiles, all regions) | ≥6 representative days per model year (—) |
| Hilbers et al. (2019) | Power-system planning (UK) | Importance subsampling of timesteps, extremes forced in, then reweighted | Importance (operating-cost) functional on joint demand-weather series | 480–8,760 h subsampled from 315,360 h (36 yr) |
| Syberfeldt et al. (2010) | Manufacturing simulation-optimization (noisy MOEA) | Confidence-based dynamic resampling (replication allocation, not set design) | Outcome space (per-solution noise estimates) | Adaptive replications per solution (—) |
| Carlsen et al. (2016) | Climate-resilient hydropower infrastructure (Volta, Orange, Zambezi) | Combined vulnerability and diversity subset selection | Outcome-space distances among candidate scenarios | Small set from large climate-runoff ensemble (—) |

---

## Table 3. Design ↔ precedent ↔ declared deviation

The three campaign designs, each paired with the published study whose construction it reproduces, that study's exact search-phase construction, and any way in which ours departs from it. Columns 3 and 4 are the reviewer-checkable record behind the claim that each design is built as its precedent built it. All three draw from a single stationary population; deep uncertainty enters only in the held-out E_test.

| Design | Precedent (2016+) | Precedent's search-phase construction | Our construction & declared deviation |
|---|---|---|---|
| 1. `historic` | Giuliani et al. (2016); Herman et al. (2020) | Giuliani: one historical trace, hand-picked to span wet/dry/normal years, 1 × 8 yr, used as the sole evaluation sequence. Herman et al. (2020) documents historical-record training as the prevailing default across the control-policy literature. | Full reconstructed observed record, 1 × ~77 yr, no hand-picking. **Deviation:** longer record, and deliberately **unmatched** in (N, L) to designs 2–3 — it is the prevailing-practice reference point, not a member of the matched selection comparison. |
| 2. `fixed_probabilistic` | Quinn et al. (2017); Zatarain Salazar et al. (2017) | Quinn: i.i.d. Monte Carlo draws from a fitted stationary inflow distribution (pre-generated 10,000-member archive), 100 × 100 yr. Zatarain Salazar: 50/500/1,000 members × 1 yr drawn once from a 10,000-trace stationary Kirsch–Nowak pool and held fixed through the search. | N = 100 × 10 yr drawn i.i.d. from a stationary Kirsch–Nowak generator, frozen across the whole search and across seeds. The i.i.d.-sampling standard of the regional water-supply lineage, and the **exact statistical control** for `hazard_filling`. **Deviation:** member length (10 yr) sits between the two precedents' 100-yr and 1-yr members. |
| 3. `hazard_filling` **(novel)** | Zatarain Salazar et al. (2017) generalized; machinery from Bonham et al. (2024) | Zatarain Salazar: **1-D** stratification of a stationary pool on the empirical CDF of annual flow (rank-space LHS, probability-preserving), wettest/driest years forced in, used as the search ensemble. Bonham: cLHS subsampling of a 500-member SOW ensemble scored by space-filling metrics (mindist, MST edge mean/SD), applied **post hoc** to robustness ranking, not in search. | LHS anchors in absolute, range-scaled *m*-dimensional hazard space + nearest-neighbor snap onto its own i.i.d. stationary candidate pool; N = 100 × 10 yr. **Deviation:** generalizes Zatarain Salazar's stratification from 1-D to *m*-D and from probability-preserving to tail-over-representing coverage; moves Bonham's space-filling subsampling from post-hoc ranking into the search ensemble, and uses Bonham's metrics as *independent diagnostics* (the selector does not optimize them). |

*Designs dropped from the campaign (`resampled_probabilistic`, `input_stratified`, `hazard_filling_du`) are noted as possible future work only; the input-space stratification precedents (Bartholomew & Kwakkel 2020; Quinn et al. 2020; Eker & Kwakkel 2018) survive in Tables 1–2 as literature context and as motivation for hazard-space selection, not as compared designs.*

---

## Reading of the tables (for manuscript framing)

1. Across Table 1, the search ensemble is designed in the input space in every study except Zatarain Salazar et al. (2017), whose stratification is hazard-based but one-dimensional (annual flow magnitude only, with forced extremes). That study is therefore the nearest published precedent and the proposed multi-dimensional hazard-space design is its direct generalization.
2. Diversity-based selection exists (Eker & Kwakkel 2018, Carlsen et al. 2016) but measures diversity in outcome space after simulation, requiring policy evaluations before scenario selection. Hazard-space selection requires none.
3. Scenario length splits the literature. The probabilistic family uses 10-100 yr realizations whose hazard statistics converge toward climatology, while Quinn et al. (2018) and Zatarain Salazar et al. (2017) use 1-yr members and Trindade-lineage ROF triggers use 52-week sequences, demonstrating short-sequence evaluation is already accepted practice where event-scale interpretation matters.
4. In Table 2, every design family defines scenario-set quality relative to the input probability distribution (proximity, moments, density-weighted clusters) except Hilbers et al. (2019), which deliberately over-represents the hazardous tail and is the closest non-water analogue to uniform hazard-space coverage.
5. Löhndorf (2016) is the only systematic head-to-head comparison of scenario-design methods judged by optimization error, and Bartholomew & Kwakkel (2020) the only head-to-head of search-phase scenario treatments with common re-evaluation. Neither runs a controlled contrast of hazard-space coverage against i.i.d. probabilistic sampling at matched N, L, and NFE on a real system under a held-out deeply uncertain re-evaluation, which is the experiment this study contributes.

## Candidate rows to trim if space requires

Table 1 can drop Herman et al. (2014) (design identical to the well-characterized variant of Trindade et al. 2017) and Quinn et al. (2017) (i.i.d. family already represented by Kasprzyk et al. 2013). Table 2 can drop Pflug (2001) (family represented by Dupačová et al. 2003) and Syberfeldt et al. (2010) (allocation rather than set design, included for completeness of the noisy-MOEA thread).

## Verification notes

Numbers verified against published full texts or open postprints for Giuliani 2016, Kasprzyk 2013, Herman 2014, Kirsch 2013 (local PDFs), Quinn 2017 (authors' code release), Quinn 2018 (Polimi repository), Eker & Kwakkel 2018 (IIASA postprint), Bartholomew & Kwakkel 2020 (TU Delft, CC-BY), Culley 2016 (Polimi), Brodeur 2020 (NSF PAR), Hilbers 2019 (arXiv), Kleywegt 2002, Høyland & Wallace 2001, and Löhndorf 2016 (author PDFs; Löhndorf scenario counts per method not extracted, marked "N swept"). Verified via dissertation chapters identical in content to the paywalled articles: Zatarain Salazar 2017 (Cornell eCommons), Trindade 2019 (Cornell eCommons), Watson & Kasprzyk 2017 (CU Boulder thesis), Giudici 2020 (POLITesi). Remaining unverified details, do not quote without checking the published PDFs: Eker & Kwakkel lake horizon T, Bartholomew & Kwakkel replications per scenario, Nahmmacher input-data years, Pflug case-study tree sizes, Carlsen exact candidate and selected counts, Infanger AOR-version tables (verified via companion Stanford report SOL 91-4), Dupačová exact experiment N, Syberfeldt replication settings.

Additional detail worth retaining for the methods discussion. Trindade et al. (2017) DU optimization re-pairs the 1,000 streamflow realizations with 1,000 LHS DU vectors at every function evaluation, and DU re-evaluation used 1,000 flows × 10,000 LHS factor samples (10⁷ SOWs). Herman et al. (2014) and Kasprzyk et al. (2013) re-evaluated over 10,000 LHS SOWs. Quinn et al. (2017, 2018) re-evaluated over 1,000 LHS SOWs. Bartholomew & Kwakkel (2020) re-evaluated solutions from all three frameworks over 10,000 LHS scenarios. Giudici et al. (2020) re-scored selected solutions on the full 3,125-scenario pool.
