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
  (the 8 active objectives + the flood-P99 diagnostic; NJ was activated
  2026-07-30 by the framing-convention screens, which read from this same
  artifact — `framing_convention_diagnostics.md` §0b). Pooling the
  cube and applying the unit operator reproduces the search scalar exactly.

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
analyzed and reported alongside but EXCLUDED from the max (decision
2026-07-30): it is a reference for prevailing practice, outside the matched
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

The first search under the adopted vector (2-seed historic `mm_full`
shakeout, job 19677667) produced ~2,000-member per-seed archives — too
dense to report or affordably re-evaluate. New diagnostic:
`scripts/supplemental/epsilon_refilter_sweep.py` (launcher
`workflow/supplemental/epsilon_refilter_sweep.sh`; outputs under
`outputs/supplemental/epsilon_refilter/historic_ffmp_obj8_mm_full/`)
re-filters the CONVERGED archives under candidate vectors
(`sensitivity_common.epsilon_nondominated`) — the converged-front complement
to this experiment's random-policy sweep. Validation: under the current
vector the box filter reproduces the Borg C archive membership of both seed
`.set` files EXACTLY (2,051/2,051 and 1,990/1,990).

Measured (axis_structure + archive_size_sweep tables):

- All four reliability axes sit on the historic trace's **1/76 unit lattice**
  (0.0131579 exactly; the metrics pool 76 unit-years). Trenton's visible
  axis gaps are that lattice — only 10 attainable values span its archive
  range — NOT epsilon coarseness: refiltering at ε 0.01 changes nothing
  (2,051/1,990 → identical), and 0.01 would resolve below the ensemble
  noise floors (0.0145/0.0147). Trenton ε stays 0.015.
- Flood exceedance and storage-P01 occupy only 3 and 7 one-dimensional
  ε-boxes across the whole front — their apparent solution density is
  whole-archive density. Cardinality is driven by the deficit-P99 pair and
  the reliability axes (21-26 boxes each).
- One-at-a-time (per-seed effect): NYC-def ε 2→5 cuts 25-29% (2→10 cuts
  41-48%); Mont-def 2→4 cuts 11-14%; storage 5→7.5 cuts 24% (saturates by
  10); flood 0.2→0.3 cuts 10-12%.
- Combined **C1_adopted** {NYC-def 5.0, Mont-def 5.0, flood 0.3; storage
  kept at 5.0} — Trevor 2026-08-05: the two sites' deficit-P99 epsilons are
  PAIRED at 5.0 (matching the paired 0.02 reliability epsilons), and a
  5%-of-capacity storage distinction is significant: seed archives →
  1,159/1,036 (−43%/−48%), cross-seed ε-front 1,599; the refilter
  parallel-axes figures (single-panel overlay + two-panel full-vs-filtered)
  show every axis span preserved.
  **C1_moderate** (Mont-def 4.0, storage 7.5): → 979/926 (−53%), cross-seed
  1,410. **C2_measured** {10.0, 5.0, 0.5, 10.0} (each at its measured floor
  / historic rec): → 544/500 (−74%), cross-seed 640.
- Pipeline finding: MOEAFramework v5 `ResultFileMerger` merges by PLAIN
  Pareto dominance regardless of `--epsilon` (identical output under two
  vectors), so a bare merge overstates the front at archive resolution
  (7,544 rows vs 2,678 under the then-current ε box filter). **FIXED
  2026-08-05**: step 07 (`run_full_diagnostics`) now ε-box-filters the
  cross-seed `{slug}_merged.set` in place under the campaign vector
  (`src/diagnostics.py::epsilon_box_filter_set`; plain union kept as
  `*_raw.set`) before computing metrics against it — and that file is the
  first-choice reference set of the step 08/09 re-evaluation
  (`src/reevaluate{,_mpi}.py`), so the merge → ε-filter → re-evaluate
  ordering holds for every optimization configuration by construction.

**ADOPTED 2026-08-05**: C1_adopted — NYC-def 5.0 (softens the standing 2.0
override to 2× below the measured hazfill floor of 10.0, and equals the
historic-design requirement), Mont-def 5.0 (paired with NYC per site
symmetry; both sites' reliability epsilons are likewise paired at 0.02),
flood 0.3; storage and all reliability axes unchanged. C1_moderate /
C2_measured remain the fallbacks if campaign archives run too large.
Executed same day: `_ANNUAL_REGISTRY_SPEC` updated (+ provenance comments),
JARs rebuilt (step 00), affected suites green (90/90, job 19688123), the
pre-adoption shakeout outputs (old-ε archives, metrics, figures) deleted as
stale. Still open: one confirmatory cheap search under the new vector
(re-filtering approximates the archive but ε also steers Borg
selection/restarts; the 2-seed historic shakeout costs ~2.4k SU to repeat).
