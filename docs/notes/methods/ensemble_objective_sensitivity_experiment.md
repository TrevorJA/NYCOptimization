# Ensemble Objective-Sensitivity Experiment

*Last updated: 2026-07-09. Ensemble counterpart of
`docs/notes/methods/objective_sensitivity_experiment.md` (the single-trace
random-DV diagnostic). The search aggregation is the two-layer annual-unit
scheme — annual metrics per (realization × water-year) unit, then per-objective
unit operators over the pooled NL unit-years (failure-year frequency /
worst-1st-percentile / mean; `objective_definitions.md` §2). This experiment
calibrates and stress-tests that scheme. Scope: **two ensemble arms** (fixed
probabilistic + hazard-filled) drawn from the same staged forcing master at
L = 10 yr, plus a small long-record set for unit-choice validation.
Cross-scenario-design comparison and resampled-draw noise remain deferred.*

## Purpose

For the frequency objectives, **all search gradient comes from the annual
failure criteria**: a criterion that nearly all unit-years pass (or fail) gives
a flat objective and stalls Borg. Saturation risk **differs by design
composition** — a benign probabilistic draw can saturate where a
stress-enriched hazard-filled ensemble still discriminates — so it must be
screened under both compositions before the campaign. Gating outputs:

1. **Failure-criterion saturation screen (#1/3/5/8).** The Decree-anchored
   annual failure criteria are computed on the **forcing master at L = 10**
   (CMIP6-forced Kirsch–Nowak, `src/ensemble_generation.py`) — **not** the
   77-yr historic trace, whose metric distributions differ materially from
   10-yr windows. Screen per objective × arm;
   if a criterion saturates, adjust its failure definition (e.g., ≥k failing
   weeks per year) and re-screen.
2. **Epsilons.** Frequency objectives: ε ≥ max(1/NL, Monte-Carlo noise floor of
   the failure frequency). Mean/percentile objectives: **native-unit ε** from
   the random-DV spread (IQR/10 heuristic, Reed et al. 2013) checked against
   the unit-operator noise floor.
3. **Unit-count floor + P99 stability.** τ_b(K) ranking convergence of the
   unit-operator objectives in both arms, sub-sampling unit-years; the
   slowest-converging objective is binding. Confirm worst-1st-percentile
   stability at the campaign NL (precedent floor: WP1 used ~1000 units;
   Quinn et al. 2017/2018).
4. **Unit-choice validation.** D2(a) is decided (annual units, Quinn 2018);
   the long-record set validates it empirically: compare annual-unit vs
   realization-level rankings (τ_b), matched-unit noise, and within-record
   unit correlation (the dependence penalty). Also pick the flood-days unit
   operator (mean vs P99).

Secondary (SI diagnostics, not gating): operator-agreement panel (robustness
check of the decided choice, below), Spearman redundancy screen, θ-multiplier
sensitivity.

## Ensembles: two arms + long-record set

All sets derive from **one staged forcing master** at L = 10 yr (CMIP6-forced
Kirsch–Nowak, `src/ensemble_generation.py`; shared `master_{L}yr_n{N_M}` slug).
Trimmed model; salinity/temperature LSTMs **off** (the active objective set
uses neither).

| Set | Construction | Role |
|---|---|---|
| **Arm P** — fixed probabilistic | uniform random index-draw of N from the master | benign-composition saturation + convergence |
| **Arm H** — hazard-filled | space-filling selection of N in hazard space from the same master (`scripts/main/subsample_hazard_filling.py`, scengen selector) | stress-enriched composition saturation + convergence |
| **Long set** — few long records | N′ records of L′ = 50 yr (L′ unresolved, 25–50) | D2(a): realization- vs block- vs pooled-year satisficing |

**Provisional sizes** (TODO — final numbers await the Anvil scaling results;
marked TODO in `supplemental_config.py`):

| | Full (HPC) | Smoke (laptop) |
|---|---|---|
| Master size N_M | few thousand (~2000–4000, TODO) | 50 |
| Arm size N (each of P, H) | 100–256 (TODO) | 5 |
| Realization length L | 10 yr | 10 yr |
| Long set N′ × L′ | 10–20 × 50 yr (TODO) | 2 × 50 yr |
| Random DVs | 200 (+ FFMP baseline) | 3 (+ baseline) |
| K sub-sample grid (per arm) | 10, 25, 50, 100, …, N | 2, 3, 5 |
| K sub-sample repeats | 20 | 3 |
| θ candidates per objective | percentile-anchored grid (see below) | same, coarse |
| SI operators | satisficing, mean, p90, CVaR₉₀ | same |

## Efficiency architecture: simulate once, subsample outputs

The expensive step runs **exactly once per DV per set**, never per K, θ, or
operator. For each DV the run script simulates the ensemble (in realization
batches to bound memory) and stores the **per-block base metric** of each
objective. Every diagnostic — threshold screen, ε noise floor, K-convergence,
D2(a) comparison, SI operators — is a post-hoc reduction of the stored matrix
in the figure script. Figures regenerate without re-simulating.

**Annual-unit resolution (the schema's grain).** The matrix holds per-**unit**
annual metrics: each objective's annual metric (`objective_definitions.md` §2)
is computed on every (realization × water-year) unit, L0 = 1 yr. An L = 10
realization yields 9 metric-bearing unit-years; an L′ = 50 record yields 49.
All unit operators (frequency / P99 / mean), the realization-level comparison,
and the SI operator panel are post-hoc reductions of this one matrix.

Unit-to-realization mapping and warm-up:

- Units are consecutive water-years of each realization.
- The first 365 days of each **realization** are warm-up and excluded from
  metric computation. No re-warm-up between units.
- Units after the first **inherit state** — the simulation is continuous; unit
  boundaries are scoring windows, not restarts (Quinn et al. 2018: consecutive
  units give representative initial-condition distributions). Within-record
  units are therefore **dependent** (shared reservoir state and hydrologic
  persistence); the long set quantifies this dependence penalty (batch-means
  framing, Law 2015).

### Stored matrix schema (`outputs/supplemental/ensemble_objective_sensitivity/matrix/*.h5`, one file per set)

| dataset / attr | shape | meaning |
|---|---|---|
| `metrics` | (N_DV, N_real, N_blocks, N_obj) | per-block base-metric values; N_blocks = 1 for L = 10 sets; NaN where a block/metric failed or is absent |
| `dv_vectors` | (N_DV, N_vars) | the decision-variable vectors |
| `sample_ids` | (N_DV,) | DV id; **-1** = FFMP baseline row |
| `objective_names` | (N_obj,) | base-objective registry names, matrix-column order |
| `realization_ids` | (N_real,) | integer realization indices into the master |
| `block_years` | (N_blocks, 2) | start/end simulation-year of each block |
| attrs | — | `set_name` (armP/armH/long), `inflow_type`, `master_slug`, `n_realizations`, `realization_years`, `block_years_L0`, `warmup_days=365`, `formulation`, `dv_seed`, selector provenance for arm H |

## Failure-criterion saturation screen (frequency objectives)

1. **Candidates.** For each frequency objective (#1/3/5/8), a small grid of
   annual failure definitions: ≥k failing weeks per unit-year, k ∈ {1, 2, 4}
   (k = 1 is the Zeff/Gold any-failure form and the default), against the fixed
   Decree goalposts (§0 of `objective_definitions.md` — the goalposts themselves
   are not tuned).
2. **Screen.** For each objective × arm × candidate k: compute every DV's
   failure-year frequency over the pooled unit-years; report its distribution
   across the random-DV population; flag **saturation** = fraction of DV
   vectors with frequency ≤ 0.05 or ≥ 0.95 (all `_pct`/fraction quantities
   0–1).
3. **Output.** Recommended k per objective (unsaturated and discriminating in
   **both** arms) + a saturation table per arm. A criterion that is fine in
   arm P but saturates in arm H (or vice versa) is the operator-composition
   interaction this experiment exists to catch.

## Epsilon determination

Estimate the Monte-Carlo noise floor of the satisficing fraction at each
candidate N by sub-sampling repeats (std of the fraction across random
size-N subsets of the arm, per objective, at the recommended θ_i). Recommend
one **common fraction-unit ε** for the Borg archive across all budget-matched
designs: ε ≥ max(1/N_floor, noise floor), so the archive never resolves
sampling noise.

## N floor / K-convergence

Sub-sample each arm at increasing K, rank the DV population by each
objective's **satisficing fraction** (recommended θ_i), and report Kendall
τ_b(K) against the full-arm ranking as proxy-truth (Bonham et al. 2024).
Rankings, not values, are the target. Run in **both arms** — convergence speed
can differ by composition — and take the binding (slowest) objective × arm as
the N floor, checked against the ≥ ~100 prior.

## Unit-choice validation (long set) + flood operator

Using the long set + arm P, compute each objective two ways from the same
stored matrix: annual-unit (decided form) vs realization-level. Report: τ_b
agreement of the induced DV rankings; sub-sampling noise at matched unit
counts; and the within-record unit correlation (the dependence penalty of the
decided form — confirms the D2(a) decision empirically). Additionally compare
mean vs P99 across unit-years for `downstream_flood_days_annual` (ranking
agreement + noise) to pick the flood unit operator (Quinn et al. 2017 caution
on expectation-masking).

## SI diagnostics (reported, not gating)

- **Operator-agreement panel** — satisficing vs mean vs p90 vs CVaR₉₀
  (CVaR via `src/objectives.py::_cvar_worst_mean`), pairwise τ_b of the
  operator-induced DV rankings per objective (McPhail et al. 2020). Reframed:
  a supplemental robustness check of the **decided** satisficing choice, not
  operator selection. Mean/percentile/CVaR are SI diagnostics only.
- **Redundancy** — Spearman screen over full-arm satisficing fractions across
  DVs (Olden & Poff 2003), compared against the historic single-trace figure.
- **θ sensitivity** — τ_b of rankings vs the recommended-θ ranking across a
  multiplier grid (subsumed largely by the calibration section; retained as a
  table).

## Pipeline

| Stage | Entry point | Notes |
|---|---|---|
| Generate master + build arms + stage | `scripts/supplemental/ensemble_objective_sensitivity_prep.py` | Forcing master at L = 10 (`src/ensemble_generation.py`), arm-P draw, arm-H selection, long set; Step-3 staging via the shared `src/ensemble_prep.py::stage_pywrdrb_ensemble_inputs` (flood → STARFIT-release → predicted-inflow preprocessors; MPI). Run once per set. |
| DV sweep | `scripts/supplemental/ensemble_objective_sensitivity_run.py` | MPI across DVs; one matrix HDF5 per set (arm P, arm H, long). |
| Diagnostics | `scripts/supplemental/ensemble_objective_sensitivity_figures.py` | Post-hoc reductions → figures + tables. |

SLURM: `workflow/supplemental/ensemble_objective_sensitivity_prep.sh` then
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

STARFIT runs before predicted inflows because the `perfect_foresight`
prediction mode (the model default) reads the presimulated-release HDF5.
NJ-diversion predictions are **not** staged (`nyc_nj_demand_source=
"constant_max"` wires constant NJ-demand parameters). The same
`stage_pywrdrb_ensemble_inputs` is the main workflow's Step 3
(`scripts/main/prep_pywrdrb_inputs.py`). Realization IDs are integers `0..N-1`,
matching the staged HDF5 keys and the model's `inflow_ensemble_indices`.

## Figures (the fixed, minimal set)

- **(a) saturation panel** — per objective × arm, distribution of DV
  satisficing fractions at each candidate θ_i, with the ≤ 0.05 / ≥ 0.95
  saturation bands; accompanying saturation table (CSV).
- **(b) τ_b-vs-K convergence** — one line per objective, per arm (P vs H
  overlaid), full-arm ranking as truth, min–max band over sub-sample repeats.
- **(c) ε noise floor** — satisficing-fraction std vs N, per objective, with
  the 1/N line and the recommended common ε.
- **(d) D2(a) panel** — τ_b agreement + noise of realization- vs block- vs
  pooled-year satisficing on the long set.
- **(e) SI: operator-agreement τ_b** — small-multiple 4×4 heatmaps per
  objective (satisficing / mean / p90 / CVaR₉₀).
- **(f) SI: redundancy** — single Spearman heatmap vs the historic screen.

Figures are written as **PNG only** (vector copies dropped; centralized in
`src/plotting/style.py::FIGURE_FORMATS`).

## How results feed the campaign

- **Final annual failure criteria (k)** for the frequency objectives —
  unsaturated and discriminating in both arms; written into
  `objective_definitions.md` §2 / `objectives_ensemble.py`.
- **Epsilons**: 1/NL-scaled for frequency objectives; native-unit for the
  mean/percentile objectives.
- **Unit-count floor + P99 stability** from the binding objective × arm of the
  τ_b(K) analysis (checks the campaign NL against the WP1 precedent floor).
- **Unit-choice confirmation** (annual vs realization rankings on the long
  set) and the **flood unit operator** (mean vs P99).
- **Saturation verdicts per design arm** — whether any frequency objective
  collapses under benign (P) or stress-enriched (H) composition; a per-arm
  failure is a campaign-blocking finding.
- SI: operator-agreement and redundancy confirmations of the decided choice.

## Citations

| Diagnostic | Citation(s) |
|---|---|
| Satisficing converges fastest; size K from ranking stability | Bonham et al. 2024 |
| Threshold/count metrics stabler than worst-case; threshold-metric stability | Quinn et al. 2017 |
| Batch means / terminating-simulation block scoring (D2(a) option 3) | Law 2015 |
| Operator choice moves robustness values more than rankings (SI panel) | McPhail et al. 2020; McPhail et al. 2018 |
| CVaR₉₀ tail operator (SI) | Quinn et al. 2017; Rockafellar & Uryasev 2000 |
| Redundancy screen (rank-correlation) | Olden & Poff 2003 |
| Rank-stability via Kendall τ_b | Herman et al. 2015 |

## Deferred (out of scope here)

Full cross-scenario-design comparison (all three campaign designs) belongs to
the main scenario-design campaign
(`docs/notes/methods/experimental_design.md`). The two compositions screened
here (a probabilistic sample and a hazard-filled sample) test the
operator-composition interaction; they are not the campaign's design
comparison.
