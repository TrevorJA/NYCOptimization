# Scenario design for MOEA evaluation ensembles — overview & gap

Trevor Amestoy, Reed Research Group, Cornell University
*Last updated: 2026-06-16. Hub note; per-subtopic summaries live in the linked notes below.*

This is the overview/hub for the scenario-design literature supporting **RQ1**: how the streamflow scenario set used *during* MOEA evaluation should be designed. It holds the scope, the gap argument, and the framing distinctions; the annotated summaries of the Zotero collection "Paper 3 NYC Reoptimization" (`ISYGLK35`) are split across the topic notes indexed at the bottom.

---

## Contribution

The proposed contribution is an evaluation ensemble built by **subsampling a very large stochastic ensemble of many short (~5-year) streamflow sequences** so that the retained members are space-filling and approximately uniform across a multi-dimensional **hazard space** (drought-event metrics, low/high-flow indices computed on each sequence), in contrast to prevailing input-space designs. Hypotheses: (1) computational efficiency, (2) more reliable Pareto-front convergence than i.i.d. probabilistic sampling, (3) more robust solutions under large-ensemble re-evaluation.

Pipeline: (i) sample a wide range of plausible generator parameters (standard input-space practice, retains parametric/deep uncertainty); (ii) generate a ~1M-member master ensemble of short sequences; (iii) subsample that master ensemble in hazard space (the novel step).

## Terminology

Three spaces, per [terminology.md](../terminology.md): the **input space** (generator/HMM parameters, climate multipliers — the "states of the world"); the **hazard space** (hydrologic hazard metrics computed on each realized sequence — SSI drought intensity/duration/severity, low/high-flow indices); the **outcome space** (simulation outputs — objective/performance values). The scenario-neutral literature's "exposure space" sits between input and hazard space. The method's novelty is sampling the *hazard space* directly rather than the input space, since distinct input-space samples yield realizations with overlapping, redundant hazard characteristics.

## The gap

Five adjacent literatures each stop short of the contribution:

1. **Multi-scenario MORDM** showed scenario choice during search matters but uses small, hand-selected or discovered scenario sets parameterized in the input space ([scenario choice during search](scenario_choice_in_search.md)).
2. **Cohen et al. (2021)** — the nearest search-phase antecedent — showed that *training-scenario properties* drive out-of-sample reservoir-policy robustness, but its winning property (baseline regret) is **problem-driven** (one perfect-foresight optimization per scenario; formulation-specific; unscalable to a large master), its selection is a contrast of cluster-membership unions rather than a coverage design, its pool is 97 deterministic GCM traces, and its test sets are complementary halves of the same ensemble ([scenario choice during search](scenario_choice_in_search.md); full note `notes/Cohen et al. (2021).md`).
3. **Bonham et al. (2024)** brought space-filling subsampling of state-of-the-world ensembles into water resources but applied it to *post-hoc robustness ranking*, not the search ensemble ([scenario subset selection](scenario_subset_selection.md)).
4. **Zatarain Salazar et al. (2017)** swept search-ensemble *size* with only 1-D flow-magnitude stratification; neither it nor related work designs multi-dimensional hazard-space coverage or varies sequence length ([sampling noise & overfitting](sampling_noise_and_overfitting.md)).
5. The **bottom-up / scenario-neutral** tradition samples condition spaces uniformly, but in climate-attribute *exposure* space via generator inversion (Guo et al. 2018), or controls hazard properties at *generation* time — the strongest exemplar being **Zaniolo et al. (2023)**, which trains portfolio policies on ensembles generated to four discrete SRI-based drought persistence/intensity/frequency types and quantifies the risk/regret of planning for the wrong type (FIND, Zaniolo et al. 2024, is the generator descendant) — not by subsampling realized streamflow toward continuous coverage of a hazard space ([bottom-up & scenario-neutral design](bottom_up_scenario_neutral.md); full note `notes/Zaniolo et al. (2023).md`).

The gap statement is therefore **not** "no one has asked whether search-ensemble composition matters" — Cohen et al. (2021) asked and answered it affirmatively, and is cited as *motivation* (their high-regret enrichment is mechanistic support for over-representing severe hazard corners). The contribution is the scalable, **simulation-free**, coverage-designed construction: we are aware of no published study that constructs the MOEA evaluation ensemble as a space-filling sample of hazard space from a very large master of many short sequences and tests its effect under a genuinely held-out deep-uncertainty re-evaluation. The differentiations to state explicitly: search-phase vs evaluation-phase selection (vs Bonham), generation-control vs realized-sequence subsampling (vs FIND/Zaniolo), and simulation-free hazard coordinates vs problem-driven regret (vs Cohen).

## Key framing distinctions

- **Input-space vs hazard-space stratification.** LHS over generator parameters or over input random variables (SAA) does not control where realizations land in hazard space; subsampling realized members by their hazard coordinates does, and eliminates cross-parameterization redundancy.
- **Representative-in-probability vs uniform-in-hazard-space.** Classical scenario reduction ([scenario reduction (stochastic programming)](scenario_reduction_stochastic_programming.md)) targets the former; the proposed design targets the latter and therefore distorts scenario probabilities, changing the meaning of expectation/reliability objectives during search. Hilbers et al. (2019) is precedent for deliberate distortion; the re-evaluation step is the unbiased corrective.
- **The redundancy argument.** Distinct input-space samples produce realizations with overlapping hazard characteristics, so input-space coverage neither guarantees nor efficiently achieves hazard-space coverage. No published paper measures redundancy among ensemble members in hazard-metric space — a quantity this study can contribute (e.g., effective sample size of probabilistic vs structured presets).
- **The sequence-length argument.** Hazard metrics on ~5-year windows vary widely between windows, while long-record aggregates converge toward climatology, so diverse hazard-space coverage requires many short sequences. (The classical record-length statistics behind this — Vogel & Stedinger, Whitt — are not yet in the collection; see below.)
- **Simulation-free vs problem-driven selection.** Hazard coordinates are computable on the sequences alone, before any system simulation — reusable across objective formulations and scalable to a 10⁵–10⁶ master. The problem-driven alternative (select on baseline regret, Cohen et al. 2021) costs one perfect-foresight optimization per candidate scenario and is formulation-specific. Anticipated reviewer question: "why not select on (proxy) regret?" — answer: unscalable at master size; whether simulation-free hazard coverage recovers the same benefit is part of what RQ1 tests; a regret- or vulnerability-informed design (e.g., one baseline-policy simulation per scenario) is future work, not a missing comparison design.

## Foundational references not yet in the collection

Cited in the argument but not yet imported into `ISYGLK35`: index/metric definitions (Vicente-Serrano et al. 2012 SSI; Yevjevich 1967 run theory; Richter et al. 1996 IHA), record-length statistics (Vogel & Stedinger 1988; Whitt 1991; terminating-simulation logic per Law 2015), and space-filling design machinery (Minasny & McBratney 2006 cLHS; Johnson et al. 1990 maximin; Mak & Joseph 2018 support points; Székely & Rizzo 2013 energy distance). Add as the methods sections firm up.

## Pending revisions (from independent reviews, not yet applied)

1. Frame the many-short-sequences argument on terminating-simulation logic (Law 2015), not Whitt (1991), which favors one long run for steady-state estimation.
2. Keep Zatarain Salazar et al. (2017) classified as rank-space (probability-preserving) 1-D stratification with forced extremes — a precedent for realized-metric coordinates, not for uniform-in-magnitude coverage.
3. Add the noisy-fitness elitist-archive literature (Fieldsend & Everson; Branke selection-under-noise) to support the resampled probabilistic design. Verify before adding.

---

## Index of scenario-design notes

**Core (Paper 3 methodology)**
- [Scenario choice during search](scenario_choice_in_search.md) — multi-scenario MORDM, robustness-in-search framing
- [Scenario subset selection](scenario_subset_selection.md) — informative/representative/space-filling subset selection and ensemble size
- [Scenario reduction (stochastic programming)](scenario_reduction_stochastic_programming.md) — classical reduction/generation + energy analogues
- [Stochastic streamflow generation](stochastic_streamflow_generation.md) — synthetic generators for the master ensemble
- [Bottom-up & scenario-neutral design](bottom_up_scenario_neutral.md) — exposure-space sampling, decision scaling, stress testing
- [Hydrologic hazard metrics](hydrologic_hazard_metrics.md) — hazard-axis definitions and index redundancy
- [Sampling noise & overfitting](sampling_noise_and_overfitting.md) — finite-ensemble effects on optimizer reliability and generalization

**Supporting (cross-referenced)**
- [Objective & robustness formulations](objective_and_robustness_formulations.md) — problem formulation, robustness metrics, across-scenario aggregation
- [DU optimization workflows](du_optimization_workflows.md) — the end-to-end DU search → re-evaluate → adapt lineage
- [MOEA methods](moea_methods.md) — Borg/MM-Borg, diagnostics, EMODPS

**Related conceptual notes:** [scenario_design_taxonomy.md](scenario_design_taxonomy.md) · [scenario_design_tables.md](scenario_design_tables.md) · [synthesis.md](synthesis.md) · [../methods/experimental_design.md](../methods/experimental_design.md)

*Citations across these notes were verified against the Zotero collection `ISYGLK35` on 2026-06-16.*
