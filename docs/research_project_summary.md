# NYC Reservoir Re-Optimization: Project Summary

*Entry point for new readers. Last updated 2026-07-13. Details live in `docs/notes/`;
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
constructed. The proposed design — **hazard-filling** — subsamples a large candidate pool
of short synthetic streamflow sequences so the retained scenarios uniformly cover a
multi-dimensional **hazard space** (drought, low-flow, and high-flow metrics computed on
each sequence), in contrast to prevailing practice that samples the *input space*
(generator/climate parameters) or uses the historical record. All designs'
Pareto-approximate policies are re-evaluated on a common held-out, deeply uncertain test
ensemble; re-evaluated robustness is the sole basis of comparison.

## Research questions

1. **RQ1 (core).** Does the composition of the search-time scenario ensemble change the
   robustness of the resulting Pareto-approximate policies under held-out re-evaluation?
2. **RQ2.** Can re-optimizing the FFMP parameters improve NYC/basin outcomes (supply
   reliability, Montague/Trenton flow targets, downstream flooding, storage resilience)?
3. **RQ3.** Does a variable-resolution FFMP structure (more storage zones, `ffmp_N`)
   improve performance or robustness?

See `notes/research_questions.md`, `notes/research_contributions.md`.

## Scenario designs compared (RQ1)

**Each design is constructed by its own published recipe, with its own seed stream.** No
design is subsampled from a shared pool — no published study builds its search ensemble
that way. Registry: `src/scenario_designs.py`.

Six designs across two **populations**. Population (the law realizations are drawn from)
and selection rule are independent choices; the comparison holds one fixed while varying
the other. All matched designs run at N = 100, L = 10 yr.

*Stationary population — Kirsch–Nowak fit to the historic record:*

| Design | Construction | Mimics |
|---|---|---|
| `historic` | Observed record, one continuous trace | prevailing applied practice (Giuliani 2016) |
| `fixed_probabilistic` | N × L realizations generated i.i.d.; frozen across search | Quinn 2017; Zatarain Salazar 2017 |
| `resampled_probabilistic` | Own pool; N redrawn every function evaluation | Brodeur 2020; Trindade 2017 for the principle only |
| `hazard_filling_stationary` | Space-filling subsample, in hazard space, of its own pool | **proposed method** |

*DU-forced population — CMIP6 harmonic forcing hypercube:*

| Design | Construction | Mimics |
|---|---|---|
| `input_stratified` | LHS over the harmonic forcing parameters; R realizations **generated** per point | Quinn 2020; Bartholomew & Kwakkel 2020 |
| `hazard_filling_du` | Space-filling subsample, in hazard space, of its own pool | **proposed method** |

**The controlled contrasts.** `fixed_probabilistic` → `hazard_filling_stationary` holds
the generator, population law, N and L fixed and varies *only the selection rule*:
does hazard coverage beat random sampling? `input_stratified` → `hazard_filling_du` holds
the forcing space, N and L fixed and varies *only the selection space* (θ vs hazard):
does hazard coverage beat **input** coverage — the central claim. `hazard_filling_stationary`
→ `hazard_filling_du` holds the selection rule fixed and asks what the DU forcing space adds.

Running hazard-filling in **both** populations is what makes every claim exactly
controlled: a stationary-only pool would leave `input_stratified` with no input space to
stratify, and a DU-only pool would leave hazard-filling with no exact random-selection
control. The literature supports both — hazard-space design does not require a DU input
space (Zatarain Salazar 2017 subsamples a *stationary* pool; Zaniolo 2023/2024 control
drought properties in SSI space with no climate input space at all).

Full construction recipes: `notes/methods/scenario_design_methods.md`. Gap statement:
`notes/literature/scenario_design.md`. Nearest antecedents, each differentiated there:
Cohen et al. 2021 (training-scenario properties → robustness, but problem-driven regret
selection over 97 GCM traces), Bonham et al. 2024 (space-filling subsampling, but post-hoc
ranking), Zaniolo et al. 2024 (hazard control at generation, not selection from realized
sequences).

## Pipeline

1. **Forcing space** — CMIP6-anchored, fixed-phase harmonic hypercube over monthly
   change-factor amplitudes (volume, seasonal amplitude, shoulder shape, optional CV
   axis). Three intrinsic coordinates, `[m, r1, r2]`
   (`notes/methods/forcing_parameterization.md`).
2. **Per-design generation** — every design generates its own realizations from its own
   seed domain (`src/ensemble_generation.py`). The two hazard-filling designs generate a
   large **candidate pool** each (one stationary, one DU-forced), sampled **i.i.d.**; only
   the hazard image + seeds are stored, and realizations regenerate deterministically on
   demand (chunked storage for large pools).
3. **Hazard metrics + screening** — drought/low-flow/high-flow indices per sequence;
   redundancy screen selects 3–4 low-collinearity axes.
4. **Selection (hazard-filling only)** — Latin hypercube anchors in hazard space snapped
   to the nearest unused pool member. The snap is intrinsic: hazard coordinates are
   emergent properties of a realized sequence, so a hazard-space design must *select from*
   a pool, whereas an input-space design *generates to* its design points.
5. **Search** — MM Borg over FFMP decision variables; objectives evaluated on the
   design's ensemble (workflow steps 00–06).
6. **Re-evaluation** — every final Pareto set re-simulated on the common held-out test
   ensemble, which is never the source of any search ensemble. The full (solution ×
   realization × objective) matrix is persisted in natural units, and robustness metrics
   are scored offline from it (steps 08–09), so a new metric never requires re-simulating.

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

## Comparison controls

- **Budget**: all matched designs run at N = 100, L = 10 yr — 1,000 scenario-years per
  evaluation — at **equal NFE**. Because N and L are common, per-evaluation cost, warm-up,
  scenario-years and wall-clock are *identical*, so equal-NFE and equal-scenario-years
  coincide. One budget condition, not two arms, and no confound between ensemble
  composition and search effort. The common (N, L) is required, not convenient: if L
  differed across designs, the selection rule would be confounded with record length.
- **Pools are sampled i.i.d., not LHS.** A uniform random size-N subset of an i.i.d. pool
  has exactly the law of N fresh i.i.d. draws — which is what makes `fixed_probabilistic`
  the *exact* statistical control for `hazard_filling_stationary`. A random subset of an
  LHS design is not i.i.d., so an LHS pool would silently void the control. Enforced by an
  invariant test.
- **Seed-stream disjointness**: each design, each draw, and the test ensemble generate from
  a namespaced seed domain, so no two ever share realizations.
- **Replication**: K independent ensemble draws × S MOEA seeds per design; a draw is the
  design's construction re-run from scratch with a fresh seed, and is the unit of analysis
  (mixed-effects framing). `historic` has K = 1.
- **Single comparison point**: cross-design metrics computed only on held-out
  re-evaluation, with pooled and leave-one-out reference sets.

See `notes/methods/experimental_design.md`.

## Status

**Working now:** end-to-end smoke runs; `historic` production env (`mm_full`: 4 islands ×
40 workers, 50k NFE, 10 seeds) launchable on Anvil; deterministic generation +
single-realization regeneration verified; chunked-pool machinery implemented; Anvil
scaling experiment built and running (informs production geometry/SU).

**Not yet in place (gates the RQ1 campaign):** production-scale candidate pools
(stationary and DU); **the held-out test ensemble E_test** — nothing can be compared until
it exists; multi-draw (K>1) replication; final satisficing thresholds and epsilons. Current
ensemble sizes everywhere are provisional test values, and E_test's sizing is being set
against the SU allocation.

**The test ensemble (E_test).** The **largest ensemble in the study by a wide margin**, and
the only basis of cross-design comparison. A **Latin hypercube over the full range of the
deeply-uncertain forcing factors**, with **many realizations per LHS point** — each LHS
point is a state of the world, and its realizations sample natural variability within it.
LHS, not i.i.d.: the i.i.d. rule applies only to the candidate pools, where it underwrites
the exact search-side control; E_test is never subsampled and is never a control, so it
should *cover* the uncertainty space rather than sample it in proportion to a measure. It
follows that **no robustness number is an expectation** — there is no probability measure
under deep uncertainty, so a satisficing fraction over E_test is a coverage-weighted count
over a designed exploration, and the comparison is commensurable because E_test is
*identical across designs*, not because it is probability-faithful. Built in ≥2
construction (Kirsch–Nowak over the wide DU box); rankings are therefore conditional on it,
which is a declared limitation, and a structurally different second construction is registered
as an optional sensitivity. **The full (solution × realization × objective) matrix is
persisted** in natural units with each realization's SOW id, so any robustness metric — at
either the SOW unit or the realization unit — is scored offline without re-simulating.

**Decided:** the six designs and the two-population architecture above; N = 100, L = 10 yr
at equal NFE; comparison metrics = multivariate Starr satisficing (primary) with Laplace,
maximin and signed improvement-over-status-quo as anchors, **no regret**; search aggregation =
**two-layer annual-unit scheme** (annual metrics per realization × water-year unit;
per-objective unit operators over the pooled unit-years — failure-year frequency /
worst-1st-percentile / mean; see `notes/methods/objective_definitions.md` §2); forcing space
retains historical persistence (claims scoped accordingly).

**Open decisions**: E_test sizing (N_theta x R x L_test, against the SU allocation) and
whether to stand up the optional second test-ensemble construction; hazard-axis set (from the redundancy screen on the production pools);
the N_θ/R split, pool sizes P, and K/S against the SU budget; flood unit operator (mean vs
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
