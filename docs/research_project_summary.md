# NYC Reservoir Re-Optimization: Project Summary

*Entry point for new readers. Last updated 2026-07-09. Details live in `docs/notes/`;
this page states what the study is, what is decided, and what is still open. The notes
are working drafts — where a note and the code disagree, the code is the record of what
exists and the note is a proposal.*

---

## The study in one paragraph

We re-optimize the operating rules of the four NYC reservoirs in the Delaware River
Basin (the FFMP rule structure, 24 decision variables) with the multi-master Borg MOEA
coupled to the Pywr-DRB simulation model. The methodological contribution is not the
re-optimization itself but a controlled comparison of **scenario designs**: how the
streamflow scenario ensemble used to evaluate candidate policies *during search* is
constructed. The proposed design — **hazard-filling** — subsamples a very large master
ensemble of short synthetic streamflow sequences so the retained scenarios uniformly
cover a multi-dimensional **hazard space** (drought, low-flow, and high-flow metrics
computed on each sequence), in contrast to prevailing practice that samples the *input
space* (generator/climate parameters) or uses the historical record. All designs'
Pareto-approximate policies are re-evaluated on a common held-out, deeply uncertain test
ensemble; re-evaluated robustness and regret are the sole basis of comparison.

## Research questions

1. **RQ1 (core).** Does the composition of the search-time scenario ensemble change the
   robustness of the resulting Pareto-approximate policies under held-out re-evaluation?
2. **RQ2.** Can re-optimizing the FFMP parameters improve NYC/basin outcomes (supply
   reliability, Montague/Trenton flow targets, downstream flooding, storage resilience)?
3. **RQ3.** Does a variable-resolution FFMP structure (more storage zones, `ffmp_N`)
   improve performance or robustness?

See `notes/research_questions.md`, `notes/research_contributions.md`.

## Scenario designs compared (RQ1)

All non-historical designs draw from one shared master ensemble, so differences are
attributable to *selection*. Registry: `src/scenario_designs.py`.

| Design | Construction | Mimics |
|---|---|---|
| `historic` | Observed record, one continuous trace | prevailing applied practice |
| `fixed_probabilistic_short` | N short sequences drawn once at random | SAA baseline (Kleywegt 2002; Herman 2014) |
| `fixed_probabilistic_long` | Few multi-decadal records, equal total years | standard stochastic practice (Quinn 2017) |
| `resampled_probabilistic` | N sequences redrawn every function evaluation | Trindade et al. 2017 |
| `input_stratified` | LHS over generator forcing parameters, one realization each | DMDU SOW practice (Quinn 2018; Bartholomew & Kwakkel 2020) |
| `hazard_filling` | Space-filling subsample of the master in hazard space | **proposed method** |
| `support_points` *(suppl.)* | Distribution-representative designed subsample | Mak & Joseph 2018; isolates coverage vs designed-selection effects |

Full construction recipes: `notes/methods/scenario_design_methods.md`. Taxonomy and gap
statement: `notes/literature/scenario_design_taxonomy.md`, `notes/literature/scenario_design.md`.
Nearest antecedents, each differentiated there: Cohen et al. 2021 (training-scenario
properties → robustness, but problem-driven regret selection over 97 GCM traces),
Bonham et al. 2024 (space-filling subsampling, but post-hoc ranking), Zaniolo et al.
2024 (hazard control at generation, not selection from realized sequences).

## Pipeline

1. **Forcing space** — CMIP6-anchored, fixed-phase harmonic hypercube over monthly
   change-factor amplitudes (volume, seasonal amplitude, shoulder shape, optional CV
   axis), LHS-sampled (`notes/methods/forcing_parameterization.md`).
2. **Master ensemble** — Kirsch–Nowak generation across the forcing hypercube; ~10⁵–10⁶
   short sequences; only the hazard image + seeds are stored, realizations regenerate
   deterministically on demand (`src/ensemble_generation.py`; chunked storage for large
   masters).
3. **Hazard metrics + screening** — drought/low-flow/high-flow indices per sequence;
   redundancy screen selects 3–6 low-collinearity axes.
4. **Per-design selection** — each design builds its search ensemble from the master
   (random draw, LHS-over-θ, hazard-filling selector, support points).
5. **Search** — MM Borg over FFMP decision variables; objectives evaluated on the
   design's ensemble (workflow steps 00–06).
6. **Re-evaluation** — every final Pareto set re-simulated on the common held-out test
   ensemble; robustness (satisficing) + regret computed there (steps 08–09).

Operational how-to: `workflow/README.md` and the step scripts `workflow/00–09_*.sh`.

## Objectives

Seven active objectives (NYC delivery reliability + CVaR₉₀ deficit, Montague reliability
+ CVaR₉₀ deficit, Trenton reliability, downstream minor-flood days, NYC storage 5th
percentile), defined in `src/objectives.py` and documented in
`notes/methods/objective_definitions.md`. For ensemble designs, each objective's
per-realization temporal metric is collapsed across realizations by a **satisficing
fraction** (threshold θᵢ per objective) — currently the only implemented aggregation
(`src/objectives_ensemble.py`); thresholds are placeholders pending the sensitivity
experiments (`notes/methods/objective_sensitivity_experiment.md`,
`notes/methods/ensemble_objective_sensitivity_experiment.md`).

## Comparison controls (decided in principle)

- **Budget**: equal total simulated scenario-years across designs (deployment view) plus
  an equal-NFE companion (mechanism view); ≥2 budget levels to expose interaction.
- **Replication**: K independent ensemble draws × S MOEA seeds per design; the draw is
  the unit of analysis (mixed-effects framing).
- **Single comparison point**: cross-design metrics computed only on held-out
  re-evaluation, with pooled and leave-one-out reference sets.

See `notes/methods/experimental_design.md` (rev. 5) — note some of its evaluation
details are superseded by `objective_definitions.md` §3 and remain under discussion.

## Status (2026-07-09, verified against code)

**Working now:** end-to-end smoke runs; `historic` production env (`mm_full`: 4 islands ×
40 workers, 50k NFE, 10 seeds) launchable on Anvil; hazard-filling resolves from a staged
64×5yr test-scale ensemble; deterministic master generation + single-realization
regeneration verified; chunked-master machinery implemented; Anvil scaling experiment
built and currently running (informs production geometry/SU).

**Not yet in place (gates the RQ1 campaign):** production-scale master ensemble;
`input_stratified` staging; multi-draw (K>1) replication; per-design NFE derivation from
the scenario-year budget; final satisficing thresholds and epsilons; the held-out test
ensemble; the support-points selector. Current ensemble sizes everywhere are provisional
test values.

**Decided (2026-07):** comparison metrics = satisficing robustness + regret on the
held-out ensemble (a coverage→robustness association is reported in the supplement);
search aggregation = **two-layer annual-unit scheme**
(annual metrics per realization × water-year unit; per-objective unit operators over
the pooled unit-years — failure-year frequency / worst-1st-percentile / mean; every
operator citation-anchored, see `notes/methods/objective_definitions.md` §2);
scenario length L = 10 yr (editable); forcing space retains historical persistence
(claims scoped accordingly) — master generation is unblocked.

**Open decisions** (priority order): campaign scope vs the SU budget (pending Anvil
scaling results); test-ensemble design; hazard-axis set; flood unit operator (mean vs
P99) and failure-criterion k values (set by the sensitivity experiment). See
`notes/methods/experimental_design.md` for the open-questions list.

## Document index

| Topic | Doc |
|---|---|
| Research questions / contributions | `notes/research_questions.md`, `notes/research_contributions.md` |
| Experimental design (controls, replication) | `notes/methods/experimental_design.md` |
| Ensemble construction recipes | `notes/methods/scenario_design_methods.md` |
| Forcing-space parameterization | `notes/methods/forcing_parameterization.md` |
| Objective definitions | `notes/methods/objective_definitions.md` |
| Objective sensitivity experiments | `notes/methods/objective_sensitivity_experiment.md`, `notes/methods/ensemble_objective_sensitivity_experiment.md` |
| Terminology (controlled vocabulary) | `notes/terminology.md` |
| Literature hub + topic notes | `notes/literature/README.md`, `notes/literature/scenario_design.md` |
| Workflow / HPC operation | `../workflow/README.md`, `../workflow/envs/README.md` |
