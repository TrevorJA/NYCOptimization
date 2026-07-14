# Scenario design for MOEA evaluation ensembles — overview & gap

Trevor Amestoy, Reed Research Group, Cornell University
*Hub note; per-subtopic summaries live in the linked notes below.*

This is the overview/hub for the scenario-design literature supporting **RQ1**: how the streamflow scenario set used *during* MOEA evaluation should be designed. It holds the scope, the gap argument, and the framing distinctions; the annotated summaries of the Zotero collection "Paper 3 NYC Reoptimization" (`ISYGLK35`) are split across the topic notes indexed at the bottom.

---

## Contribution

The proposed contribution is an MOEA evaluation ensemble whose members are **selected for space-filling coverage of a multi-dimensional hazard space** — drought-event metrics and low/high-flow indices computed on each realized 10-year sequence — in contrast to the prevailing designs, which sample either i.i.d. from a fitted generator or on a Latin hypercube over the generator's **input parameters**. The selection is deterministic: LHS anchors in the hazard box, then a nearest-neighbor snap onto the nearest available member of a candidate pool. Hypotheses: (1) computational efficiency, (2) more reliable Pareto-front convergence than i.i.d. probabilistic sampling, (3) more robust solutions under held-out re-evaluation.

**The generate-to / select-from asymmetry is the structural fact behind the design.** Generator parameters θ are a *knob*: a design can prescribe a θ and generate realizations at it, which is exactly what input-stratified designs do (LHS over θ, R realizations generated per θ). Hazard coordinates are *emergent* from a realized flow sequence and cannot be dialed in — so a hazard-space design must **select from** a finite candidate pool rather than **generate to** a target. This asymmetry is intrinsic, not an implementation convenience, and it is why only the hazard-filling designs need a pool.

**Construction.** Each design is built to its own recipe, independently of the others; there is no shared pool from which all designs are drawn. The two hazard-filling designs each own a **candidate pool** — one stationary, one DU-forced — from which their N members are selected; the comparison designs are generated directly. All pools are disjoint from the held-out test ensemble E_test, which is where the large ensemble now lives: a large ensemble is used **only for post-optimization re-evaluation**, never as a source for the search ensembles.

**Two populations, six designs.** Every design is run inside one of two populations, so that every claim has an exact within-population control.

*Stationary population* (Kirsch–Nowak fit to the historic record, θ fixed):
1. `historic` — the observed record (N = 1, ~77 yr); unmatched reference (Giuliani et al. 2016; Herman et al. 2020).
2. `fixed_probabilistic` — N × L drawn i.i.d. from the stationary generator, frozen across the search (Quinn et al. 2017; Zatarain Salazar et al. 2017).
3. `resampled_probabilistic` — a stationary pool from which N members are redrawn at every function evaluation (Brodeur et al. 2020 is the primary anchor; the Trindade/Gold lineage is cited only for the principle of re-randomizing the search ensemble across evaluations).
4. `hazard_filling_stationary` **(novel)** — LHS anchors + nearest-neighbor snap in hazard space over a stationary candidate pool.

*DU-forced population* (θ ~ CMIP6-informed harmonic forcing hypercube):
5. `input_stratified` — LHS over the harmonic forcing parameters, R realizations generated per θ (N = N_θ × R); LHS alone, no snap (Quinn et al. 2020; Bartholomew & Kwakkel 2020; Eker & Kwakkel 2018; Watson & Kasprzyk 2017).
6. `hazard_filling_du` **(novel)** — LHS anchors + nearest-neighbor snap in hazard space over a DU-forced candidate pool (motivation: Cohen et al. 2021; Zaniolo et al. 2023; machinery: Bonham et al. 2024).

Designs 2–6 all use N = 100 and L = 10 yr (1,000 scenario-years per evaluation, equal NFE); `historic` is deliberately unmatched. A common (N, L) is required or the selection rule is confounded with record length, and N is bounded below by the fill requirement (at m = 4 hazard axes, N = 100 gives ~3.2 points per dimension). Per-design precedents and the declared deviations from them are tabulated in [scenario_design_tables.md](scenario_design_tables.md), Table 3.

**The controlled contrasts.** 2 → 4: same generator, same population law, same N, same L; only the *selection rule* changes ("does hazard coverage beat random sampling?"). 5 → 6: same forcing space, same N, same L; only the *selection space* changes, θ vs h ("does hazard coverage beat input coverage?" — the central novelty claim). 4 → 6: same selection rule, different population ("what does the DU forcing space add?"). 1 and 3 anchor prevailing practice and probe overfitting.

## Terminology

Three spaces, per [terminology.md](../terminology.md): the **input space** (generator parameters, climate/forcing multipliers — the "states of the world"); the **hazard space** (hydrologic hazard metrics computed on each realized sequence — SSI drought intensity/duration/severity, low/high-flow indices); the **outcome space** (simulation outputs — objective/performance values). The scenario-neutral literature's "exposure space" sits between input and hazard space. The novelty is designing coverage of the *hazard space* directly rather than the input space, since distinct input-space samples yield realizations with overlapping, redundant hazard characteristics.

## The gap

Five adjacent literatures each stop short of the contribution:

1. **Multi-scenario MORDM** showed scenario choice during search matters but uses small, hand-selected or discovered scenario sets parameterized in the input space ([scenario choice during search](scenario_choice_in_search.md)). Its LHS-over-DU-factors construction is precisely the `input_stratified` comparison design here — the published input-space method, adopted as its authors wrote it.
2. **Cohen et al. (2021)** — the nearest search-phase antecedent — showed that *training-scenario properties* drive out-of-sample reservoir-policy robustness, but its winning property (baseline regret) is **problem-driven** (one perfect-foresight optimization per scenario; formulation-specific; unscalable to a large pool), its selection is a contrast of cluster-membership unions rather than a coverage design, its pool is 97 deterministic GCM traces, and its test sets are complementary halves of the same ensemble ([scenario choice during search](scenario_choice_in_search.md); full note `notes/Cohen et al. (2021).md`).
3. **Bonham et al. (2024)** brought space-filling subsampling of state-of-the-world ensembles into water resources but applied it to *post-hoc robustness ranking*, not the search ensemble ([scenario subset selection](scenario_subset_selection.md)).
4. **Zatarain Salazar et al. (2017)** swept search-ensemble *size* with only 1-D flow-magnitude stratification of a stationary Kirsch–Nowak pool; neither it nor related work designs multi-dimensional hazard-space coverage or varies sequence length ([sampling noise & overfitting](sampling_noise_and_overfitting.md)).
5. The **bottom-up / scenario-neutral** tradition samples condition spaces uniformly, but in climate-attribute *exposure* space via generator inversion (Guo et al. 2018), or controls hazard properties at *generation* time — the strongest exemplars being **Herman et al. (2016)** (weighted bootstrap amplifying drought severity) and **Zaniolo et al. (2023)**, which trains portfolio policies on ensembles generated to four discrete SRI-based drought persistence/intensity/frequency types (FIND, Zaniolo et al. 2024, is the generator descendant) — not by selecting realized sequences toward continuous coverage of a hazard space ([bottom-up & scenario-neutral design](bottom_up_scenario_neutral.md); full note `notes/Zaniolo et al. (2023).md`).

**Hazard-space design does not require a DU input space, and the published work mostly does not use one.** Zatarain Salazar et al. (2017) stratifies a stationary 10,000-trace pool by a realized-flow metric, in search; Herman et al. (2016) amplifies drought severity with a stationary weighted bootstrap; Zaniolo et al. (2023) and FIND (2024) control drought frequency/intensity/duration in SSI space with no climate input space at all. Input-stratification, by contrast, *cannot exist* without a DU input space (Quinn et al. 2020; Bartholomew & Kwakkel 2020; Eker & Kwakkel 2018). Running the hazard-filling design in **both** populations is therefore not redundancy — it is what supplies the exact within-population control for each claim, and it separates "hazard coverage vs random sampling" from "hazard coverage vs input coverage."

The gap statement is therefore **not** "no one has asked whether search-ensemble composition matters" — Cohen et al. (2021) asked and answered it affirmatively, and is cited as *motivation* (their high-regret enrichment is mechanistic support for over-representing severe hazard corners). The contribution is the scalable, **simulation-free**, coverage-designed *selection* of the search ensemble, and its head-to-head comparison against the published input-space and probabilistic designs at matched N and L on a real system, with cross-design comparison only under a genuinely held-out deep-uncertainty re-evaluation. The differentiations to state explicitly: search-phase vs evaluation-phase selection (vs Bonham), generation-control vs realized-sequence selection (vs FIND/Zaniolo/Herman 2016), simulation-free hazard coordinates vs problem-driven regret (vs Cohen), and multi-dimensional coverage vs 1-D probability-preserving stratification (vs Zatarain Salazar).

## Key framing distinctions

- **Input-space vs hazard-space stratification.** LHS over generator parameters (or over input random variables, as in SAA) does not control where realizations land in hazard space; selecting realized members by their hazard coordinates does, and eliminates cross-parameterization redundancy. The 5 → 6 contrast isolates exactly this.
- **Generate-to vs select-from.** θ can be prescribed and generated to; hazard coordinates cannot. Input-space designs are therefore pure LHS with no pool; hazard-space designs are LHS anchors snapped onto the nearest member of a candidate pool. The pool exists only because hazard is emergent.
- **Representative-in-probability vs uniform-in-hazard-space.** Classical scenario reduction targets the former; the proposed design targets the latter and therefore distorts scenario probabilities, changing the meaning of expectation/reliability objectives during search. The held-out re-evaluation is the unbiased corrective. (The stochastic-programming reduction school is *background*, not a methods precedent — see [scenario reduction](scenario_reduction_stochastic_programming.md).)
- **The redundancy argument.** Distinct input-space samples produce realizations with overlapping hazard characteristics, so input-space coverage neither guarantees nor efficiently achieves hazard-space coverage. No published paper measures redundancy among ensemble members in hazard-metric space — a quantity this study contributes.
- **The sequence-length argument.** Hazard metrics on 10-year windows vary widely between windows, while long-record aggregates converge toward climatology, so diverse hazard-space coverage requires many short sequences. L = 10 yr is also the largest length compatible with the fill requirement: at a fixed per-evaluation budget, L = 50 forces N ≈ 20, and space-filling in 4-D with 20 points is noise.
- **Coverage as a diagnostic, not an objective.** The selector is deterministic (LHS + nearest-neighbor snap) and does not optimize a discrepancy criterion, so L2-star discrepancy and the Bonham space-filling metrics (mindist, MST edge statistics) remain **independent diagnostics** of the achieved coverage rather than the quantity the selector maximized.
- **Simulation-free vs problem-driven selection.** Hazard coordinates are computable on the sequences alone, before any system simulation — reusable across objective formulations and scalable to large pools. The problem-driven alternative (select on baseline regret, Cohen et al. 2021) costs one perfect-foresight optimization per candidate scenario and is formulation-specific. Anticipated reviewer question: "why not select on (proxy) regret?" — answer: unscalable at pool size; whether simulation-free hazard coverage recovers the same benefit is part of what RQ1 tests; a regret- or vulnerability-informed design is future work, not a missing comparison design.

## Foundational references not yet in the collection

Cited in the argument but not yet imported into `ISYGLK35`: index/metric definitions (Vicente-Serrano et al. 2012 SSI; Yevjevich 1967 run theory; Richter et al. 1996 IHA), record-length statistics (Vogel & Stedinger 1988; terminating-simulation logic per Law 2015), and design-of-experiments primitives (Johnson et al. 1990 maximin; Morris & Mitchell 1995; Minasny & McBratney 2006 cLHS; L2-star discrepancy). The DoE primitives are cited **for the algorithm only** — the water-resources framing of the hazard-filling design is carried by Bonham (2024), Cohen (2021), Zaniolo (2023) + FIND (2024), Herman (2016), Culley et al. (2016), Guo et al. (2018) and Quinn et al. (2020).

---

## Index of scenario-design notes

**Core (Paper 3 methodology)**
- [Scenario choice during search](scenario_choice_in_search.md) — multi-scenario MORDM, robustness-in-search framing
- [Scenario subset selection](scenario_subset_selection.md) — informative/representative/space-filling subset selection and ensemble size
- [Scenario reduction (stochastic programming)](scenario_reduction_stochastic_programming.md) — classical reduction/generation + energy analogues (background)
- [Stochastic streamflow generation](stochastic_streamflow_generation.md) — synthetic generators for the candidate pools and test ensemble
- [Bottom-up & scenario-neutral design](bottom_up_scenario_neutral.md) — exposure-space sampling, decision scaling, stress testing
- [Hydrologic hazard metrics](hydrologic_hazard_metrics.md) — hazard-axis definitions and index redundancy
- [Sampling noise & overfitting](sampling_noise_and_overfitting.md) — finite-ensemble effects on optimizer reliability and generalization

**Supporting (cross-referenced)**
- [Objective & robustness formulations](objective_and_robustness_formulations.md) — problem formulation, robustness metrics, across-scenario aggregation
- [DU optimization workflows](du_optimization_workflows.md) — the end-to-end DU search → re-evaluate → adapt lineage
- [MOEA methods](moea_methods.md) — Borg/MM-Borg, diagnostics, EMODPS

**Related conceptual notes:** [scenario_design_taxonomy.md](scenario_design_taxonomy.md) · [scenario_design_tables.md](scenario_design_tables.md) · [synthesis.md](synthesis.md) · [../methods/experimental_design.md](../methods/experimental_design.md)

*Citations across these notes are verified against the Zotero collection `ISYGLK35`.*
