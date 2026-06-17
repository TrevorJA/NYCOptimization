# Ensemble Objective-Sensitivity Experiment

*Last updated: 2026-06-17. Ensemble counterpart of
`docs/notes/methods/objective_sensitivity_experiment.md` (the single-trace
random-DV diagnostic) and companion to
`docs/notes/methods/objective_definitions.md`. Answers the two §4 questions of
the historic doc that require a realization axis. Scope is intentionally limited
to **one fixed probabilistic ensemble**; cross-scenario-design comparison and
resampled-draw (redraw-per-evaluation) noise are deferred.*

## Purpose

The MOEA's ensemble objectives are **satisficing fractions over K realizations**
(`docs/notes/methods/objective_definitions.md` §2). Before committing compute we
confirm, on one probabilistic ensemble:

1. **Estimator noise / ensemble-size (K) convergence.** The minimum number of
   realizations K at which policy *rankings* stabilize. We sub-sample the full
   ensemble at increasing K, rank the DV population by each objective's
   satisficing fraction, and report Kendall τ_b(K) against the full-ensemble
   ranking as proxy-truth (Bonham et al. 2024). Rankings, not values, are the
   relevant target — only ranking stability matters for selection.
2. **Across-realization operator agreement.** For the same DVs on the full
   ensemble, recompute each objective under four across-realization operators —
   **satisficing fraction / mean / p90 / CVaR₉₀** — and report the pairwise
   Kendall-τ_b matrix of the operator-induced DV rankings, per objective
   (McPhail et al. 2020). High agreement validates the single-satisficing-family
   choice; low agreement flags objectives where the operator materially changes
   the search target.

Secondary (reported, not gating):

- **Redundancy.** Spearman screen over the ensemble objectives (full-ensemble
  satisficing fractions across DVs), to be compared against the historic
  single-trace redundancy figure (Olden & Poff 2003). The historic panel is that
  experiment's deliverable and is cross-referenced, not re-rendered here.
- **Threshold sensitivity.** τ_b of satisficing rankings vs the default-threshold
  ranking across a multiplier grid on each θ_i.

## The single fixed ensemble

A fixed probabilistic **Kirsch–Nowak** ensemble fit on the historic reference
`pub_nhmv10_BC_withObsScaled` (`src/ensemble_generation.py`), staged via the
dynamic `kn_{Y}yr_n{N}` inflow-type slug (`src/ensembles.py::get_ensemble_spec`)
so it is exactly what the Step-1 generator produces — no edits to the scenario-
design taxonomy or the stubbed method designs. The trimmed model is used (after
staging the ensemble presim once); salinity/temperature LSTMs are **off** (the
active objective set uses neither — a large speedup over the ensemble).

**Provisional sizes** (manuscript ensemble-design sizes and realization length
are open decisions, `docs/notes/methods/experimental_design.md`; marked TODO in
`supplemental_config.py`):

| | Full (HPC) | Smoke (laptop) |
|---|---|---|
| Realizations N | 256 | 5 |
| Realization length | 20 yr | 20 yr |
| Random DVs | 200 (+ FFMP baseline) | 3 (+ baseline) |
| K sub-sample grid | 10, 25, 50, 100, 200, 256 | 2, 3, 5 |
| K sub-sample repeats | 20 | 3 |
| Operators | satisficing, mean, p90, CVaR₉₀ | same |

## Efficiency architecture: simulate once, subsample outputs

The expensive step runs **exactly once per DV**, never per K or per operator.
For each DV the run script simulates the full ensemble (in realization batches to
bound memory) and stores the **per-realization base metric** of each objective —
a matrix

```
metrics[N_DV, N_realizations, N_objectives]
```

of raw per-realization values (`OBJECTIVES[name].compute(data_per_real[r])`),
**not** the collapsed satisficing fraction. Every diagnostic is then a post-hoc
reduction of this stored matrix in the figure script:

- K-convergence and threshold-sensitivity re-collapse subsets/thresholds with
  `SatisficingAgg`;
- operator agreement re-collapses with mean / p90 / CVaR₉₀ (CVaR via
  `src/objectives.py::_cvar_worst_mean`), each oriented higher-is-better so τ_b
  reads as agreement;
- redundancy uses the full-ensemble satisficing fractions.

This keeps run and figure stages separate: figures regenerate without
re-simulating.

### Stored matrix schema (`outputs/supplemental/ensemble_objective_sensitivity/matrix/*.h5`)

| dataset / attr | shape | meaning |
|---|---|---|
| `metrics` | (N_DV, N_real, N_obj) | per-realization base-metric values (NaN where a realization or metric failed) |
| `dv_vectors` | (N_DV, N_vars) | the decision-variable vectors |
| `sample_ids` | (N_DV,) | DV id; **-1** = FFMP baseline row |
| `objective_names` | (N_obj,) | base-objective registry names, matrix-column order |
| `realization_ids` | (N_real,) | integer realization indices |
| attrs | — | `inflow_type`, `n_realizations`, `realization_years`, `formulation`, `kn_seed`, `dv_seed` |

## Pipeline

| Stage | Entry point | Notes |
|---|---|---|
| Generate + stage ensemble | `scripts/supplemental/ensemble_objective_sensitivity_prep.py` | Step 1 (serial) + Step 3 staging via the shared `src/ensemble_prep.py::stage_pywrdrb_ensemble_inputs` (flood → STARFIT-release → predicted-inflow preprocessors; MPI). Run once. |
| DV sweep | `scripts/supplemental/ensemble_objective_sensitivity_run.py` | MPI across DVs; writes the matrix HDF5. |
| Diagnostics | `scripts/supplemental/ensemble_objective_sensitivity_figures.py` | Post-hoc reductions → figures + tables. |

SLURM: `slurm/supplemental/ensemble_objective_sensitivity_prep.sh` then
`…/ensemble_objective_sensitivity.sh`. All settings live in
`supplemental_config.py` (ensemble section) — no CLI value flags.

### Shared Step-3 prep

`src/ensemble_prep.py::stage_pywrdrb_ensemble_inputs(spec, …)` stages the files a
trimmed-model ensemble simulation with NYC flood operations requires under
`STAGED_ENSEMBLE_DIR/{inflow_type}/`:

| file | preprocessor | consumer |
|---|---|---|
| `catchment_inflow_mgd.hdf5` | Step-1 generator (pre-existing) | `FlowEnsemble` base inflows |
| `catchment_inflow_with_flood_nodes_mgd.hdf5` | `FloodNodeInflowEnsemblePreprocessor` | `FlowEnsemble` (flood ops) |
| `presimulated_releases_mgd.hdf5` (+ `_metadata.json`) | `STARFITReleaseEnsemblePreprocessor` | `PresimulatedReleaseEnsemble` (trimmed) |
| `predicted_inflows_mgd.hdf5` | `PredictedInflowEnsemblePreprocessor` | `PredictionEnsemble` (Montague/Trenton) |

STARFIT runs before predicted inflows because the `perfect_foresight` prediction
mode (the model default) reads the presimulated-release HDF5. NJ-diversion
predictions (`predicted_diversions_mgd.hdf5`) are **not** staged: the
optimization model uses `nyc_nj_demand_source="constant_max"`, under which the
model builder wires constant NJ-demand parameters instead. The same
`stage_pywrdrb_ensemble_inputs` is the main workflow's Step 3
(`scripts/main/prep_pywrdrb_inputs.py`). Realization IDs are integers `0..N-1`,
matching the generator's HDF5 keys and the model's `inflow_ensemble_indices`.

## Figures (the fixed, minimal set)

- **(a) τ_b-vs-K convergence** — one line per objective, full-ensemble ranking as
  truth, min–max band over sub-sample repeats.
- **(b) operator-agreement τ_b** — small-multiple 4×4 heatmaps (one panel per
  objective) for satisficing / mean / p90 / CVaR₉₀.
- **(c) ensemble-objective redundancy** — single Spearman heatmap; compare
  against the historic experiment's redundancy figure.
- **θ-sensitivity table** — τ_b of rankings vs the default θ across the
  multiplier grid (CSV).

Figures are written as **PNG only** (vector copies dropped; centralized in
`src/plotting/style.py::FIGURE_FORMATS`).

## How results feed objective selection

- Set the search ensemble size from the K at which τ_b(K) plateaus near 1 for the
  active objectives (use the slowest-converging objective as the binding one).
- If the operator-agreement τ_b among satisficing / mean / p90 / CVaR₉₀ is high
  for an objective, the single satisficing family is safe; a low value flags an
  objective whose ranking is operator-sensitive and warrants discussion.
- Confirm the ensemble redundancy structure matches the historic-trace screen
  (the satisficing wrapper should not introduce new collinearity).
- Confirm rankings are insensitive to θ within a reasonable band, or record the
  sensitivity for the threshold-setting decision.

## Citations

| Diagnostic | Citation(s) |
|---|---|
| Satisficing converges fastest; size K from ranking stability | Bonham et al. 2024 |
| Operator choice moves robustness values more than rankings | McPhail et al. 2020; McPhail et al. 2018 |
| CVaR₉₀ tail operator | Quinn et al. 2017; Rockafellar & Uryasev 2000 |
| Redundancy screen (rank-correlation) | Olden & Poff 2003 |
| Rank-stability via Kendall τ_b | Herman et al. 2015 |

## Deferred (out of scope here)

Cross-scenario-design comparison and resampled-draw (redraw-per-evaluation)
Monte-Carlo noise (Trindade et al. 2017; Brodeur et al. 2020) — both belong to
the main scenario-design campaign (`docs/notes/methods/experimental_design.md`),
not this single-ensemble diagnostic.
