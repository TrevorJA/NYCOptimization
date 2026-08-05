# Epsilon-Calibration Experiment

Calibrates the annual-unit (§2) search epsilons — the values Borg's
ε-dominance archive uses (`config.get_epsilons()` →
`src/objectives_ensemble._ANNUAL_REGISTRY_SPEC`) — on the campaign search
measures. An epsilon must (i) resolve *meaningful* policy differences in
interpretable native units, (ii) not resolve *sampling noise* of the ensemble
estimator, and (iii) keep the archive (and therefore the reported Pareto set)
at a tractable cardinality. The published precedent gives the signal-scale
rule (archive resolution ≈ signal scale, Reed et al. 2013) and the
noise-exclusion requirement (epsilon precision set to suppress simulation
noise, Kasprzyk et al. 2013); this experiment measures both floors directly,
per design, and sweeps the archive-size consequence.

Scripts: `scripts/supplemental/epsilon_calibration_{run,figures}.py`;
launcher `workflow/supplemental/epsilon_calibration.sh` (one sbatch job per
design, design selected by `NYCOPT_ENV_FILE` — `workflow/envs/eps_calib_*.env`);
settings in `supplemental_config.py` (`EPS_*`). Outputs under
`outputs/supplemental/epsilon_calibration/`.

## Setup

- **Designs**: each campaign design is calibrated on its own search ensemble —
  `historic` (N = 1 over the trace's consecutive water-year units),
  `fixed_probabilistic` and `hazard_filling_stationary` (N = 100 × L = 10,
  draw 0) — because the search objectives are computed under different
  measures across designs and the archive lives inside each search.
- **Policies**: `EPS_N_POLICIES` = 512 random policies drawn **uniform on the
  constraint-feasible region** (rejection against the two DV-space formal Borg
  constraints — pure DV arithmetic; realized acceptance rate persisted as QC),
  plus the FFMP baseline (id −1). Feasible-only sampling matches the archive's
  population: constraint-dominance keeps infeasible vectors out of the
  archive, so infeasible spread must not inflate the signal scale.
- **Evaluation**: each policy runs once per design through the same batched
  path Borg workers use (`run_simulation_ensemble_batched`, campaign
  realization batch), storing the full stage-(i) cube
  `(n_dv × n_real × n_obj × n_units)` for the **entire annual-unit registry**
  (the 8 active objectives + the flood-P99 diagnostic; the framing-convention
  screens read from this same artifact —
  `framing_convention_diagnostics.md` §0b). Pooling the cube and applying the
  unit operator reproduces the search scalar exactly.

## Per-objective quantities (figures script; post-hoc, no re-simulation)

1. **Signal scale** — IQR of the natural-unit search scalar across the
   feasible random policies; `eps_signal = IQR / 10`.
2. **Noise floor** — bootstrap SD of the unit-operator estimator under
   resampling of the realization axis with replacement
   (`EPS_BOOTSTRAP_B` = 1000, one shared index draw per design); summarized
   as the median (and p90) across policies. The single-trace historic design
   resamples the unit-year axis instead — an i.i.d. approximation that
   understates noise under serial dependence (disclosed, not corrected, per
   the project convention).
3. **Granularity floor** — the failure-frequency objectives take values on a
   1/(N·units) lattice; an epsilon below that step cannot merge adjacent
   attainable values.
4. **Recommendation** — `eps_rec = ceil_to_clean_step(max(1, 2, 3))` in native
   units, with the binding floor and a plain-language interpretation recorded
   per objective.
5. **Archive-size sweep** — ε-nondominated archive size (Borg box convention,
   `src.sensitivity_common.epsilon_nondominated`) of the evaluated policies
   for the recommended vector × `EPS_SCALE_GRID` and for the current registry
   vector, over the active objective subset. Random feasible policies
   under-fill a converged front, so the absolute sizes are a proxy; the
   scaling trend is the decision signal.

## Campaign vector

The Borg problem and the MOEAFramework JARs carry ONE epsilon set for all
designs, so the campaign value per objective is the clean-rounded maximum of
the raw requirement across the **ensemble campaign designs**
(`EPS_CAMPAIGN_DESIGNS` = fixed_probabilistic + hazard_filling_stationary; no
campaign design's archive may resolve below its own noise floor); the binding
design is recorded, and a >4× spread in the raw requirement across the
campaign designs triggers a review warning. The historic single-trace arm is
analyzed and reported alongside but EXCLUDED from the max: it is a
reference for prevailing practice, outside the matched
contrast, and its 76-unit estimator's noise floor would coarsen the shared
vector ~3-4× beyond what the ensemble measures need (reliability ε 0.10
instead of 0.02 on the 0-1 scale). Consequence, disclosed: the historic arm's
archive resolves below its own noise floor on those axes. Adopt the combined
table (`tables/epsilon_recommendation_*.csv`) into `_ANNUAL_REGISTRY_SPEC`
**before** JAR generation (step 00), updating the provenance comment there.

## Outputs

- `cube/unit_cube_{formulation}_{design}_seed{S}_n{N}.h5` — per-unit metric
  cube + DV vectors + acceptance-rate QC.
- `tables/epsilon_diagnostics_{…}.csv` (per design), `archive_sweep_{…}.csv`
  (per design; the campaign vector × `EPS_SCALE_GRID` plus the previous
  provisional vector), `epsilon_recommendation_{…}.csv` (combined).
- `figures/eps_calibration_ladder` (F1, combined: per-design floors vs the
  previous and adopted epsilons, log axis; designs color-keyed, historic in
  reference gray), `scalar_distributions_{design}` (F3: signal spread with
  the adopted and previous epsilon widths), `archive_size_vs_scale` (F2,
  combined: archive cardinality vs scaling of the adopted vector),
  `parallel_axes_{design}` (F4: the evaluated policies on the active
  objectives — shared parallel-coordinates renderer — with the members
  retained by ε-box nondominance under the adopted vector highlighted and
  the FFMP baseline bold; shows the adopted resolution thins the set
  without collapsing any tradeoff axis's span).

## Post-shakeout revision diagnostics (2026-08-05)

The firs## Converged-front re-filter sweep and the adopted vector

A complementary diagnostic re-filters CONVERGED search archives under
candidate epsilon vectors: `scripts/supplemental/epsilon_refilter_sweep.py`
(launcher `workflow/supplemental/epsilon_refilter_sweep.sh`; outputs under
`outputs/supplemental/epsilon_refilter/`), using
`sensitivity_common.epsilon_nondominated`. Random feasible policies
under-fill a converged front, so this sweep prices archive cardinality on
the population the archive actually holds; validation: under the registry
vector the box filter reproduces Borg's C archive membership exactly.

**The adopted campaign vector** is the registry in
`src/objectives_ensemble._ANNUAL_REGISTRY_SPEC`: reliability epsilons paired
at 0.02 (NYC/Montague) with Trenton 0.015 and NJ 0.025, the NYC/Montague
deficit-P99 pair PAIRED at 5.0, flood exceedance 0.3, storage-P01 5.0 (a
5%-of-capacity distinction is significant). Rationale recorded per axis:
the reliability axes sit on the historic trace's 1/76 unit lattice (visible
axis gaps are the lattice, not epsilon coarseness); archive cardinality is
driven by the deficit-P99 pair and the reliability axes; the re-filter
parallel-axes figures confirm every tradeoff axis span is preserved at the
adopted resolution. Two coarser fallback vectors (moderate / at-measured-
floor) remain registered in `epsilon_refilter_sweep.py` if campaign archives
run too large.

**Merger behavior (constraint).** MOEAFramework v5 `ResultFileMerger` merges
by PLAIN Pareto dominance regardless of `--epsilon`, so a bare merge
overstates the front at archive resolution. Step 07 (`run_full_diagnostics`)
therefore ε-box-filters the cross-seed `{slug}_merged.set` in place under
the campaign vector (`src/diagnostics.py::epsilon_box_filter_set`; the plain
union is kept as `*_raw.set`) before computing metrics — and that file is
the first-choice reference set of the step 08/09 re-evaluation
(`src/reevaluate{,_mpi}.py`), so the merge → ε-filter → re-evaluate ordering
holds for every optimization configuration by construction.

Still open: one confirmatory cheap search under the adopted vector
(re-filtering approximates the archive but ε also steers Borg
selection/restarts; a 2-seed historic shakeout costs ~2.4k SU).
