# Epsilon Calibration

Calibrates the annual-unit search epsilons, the values Borg's ε-dominance
archive uses. `config.get_epsilons()` reads
`src/objectives_ensemble._ANNUAL_REGISTRY_SPEC` at runtime and the MOEAFramework
CLIs take `--epsilon` strings, so an epsilon change needs no JAR rebuild (the
problem JARs carry only the DV and objective counts). An epsilon must resolve
meaningful policy differences in native units, must not resolve the sampling
noise of the ensemble estimator, and must keep the archive, and therefore the
reported Pareto-approximate set, at a tractable cardinality. The signal-scale
rule follows Reed et al. (2013) and the noise-exclusion requirement Kasprzyk et
al. (2013). Manuscript statement in Section 3.2.2 and SI Text S5.

## Adopted vector

`[0.05, 10.0, 0.05, 10.0, 10.0, 0.3, 5.0, 0.05]` in registry order, one shared
precision per objective family (reliabilities 0.05, deficit-P99s 10.0 % of
target, flood exceedance 0.3 ft·d/yr, storage P01 5.0 % of capacity). Derived
in two stages.

### Stage 1, per-objective floors

Scripts `scripts/supplemental/epsilon_calibration_{run,figures}.py`, launcher
`workflow/supplemental/epsilon_calibration.sh` (one job per design, selected by
`NYCOPT_ENV_FILE=workflow/envs/eps_calib_*.env`), settings `EPS_*` in
`supplemental_config.py`, outputs `outputs/supplemental/epsilon_calibration/`.

- **Policies.** `EPS_N_POLICIES` = 512 random decision vectors drawn uniform on
  the constraint-feasible region (rejection against the two DV-space Borg
  constraints, acceptance rate persisted as QC) plus the FFMP baseline (id −1).
  Feasible-only sampling matches the archive's own population.
- **Evaluation.** Each policy runs once per design through the batched worker
  path (`run_simulation_ensemble_batched`, `EPS_REALIZATION_BATCH` = 150) on
  that design's own search ensemble, storing the stage-(i) cube
  `(n_dv × n_real × n_obj × n_units)` for the whole annual-unit registry
  (`cube/unit_cube_{formulation}_{design}_seed{S}_n{N}.h5`). Pooling the cube
  through the unit operator reproduces the search scalar exactly, and the
  framing-convention screens read the same cubes.
- **Floor.** Per objective, the clean-rounded maximum of (a) the signal scale,
  IQR of the search scalar across feasible policies divided by 10, (b) the
  noise floor, bootstrap SD of the unit-operator estimator under resampling of
  the realization axis (`EPS_BOOTSTRAP_B` = 1000, median across policies; the
  historic design resamples unit-years, an i.i.d. approximation that
  understates noise under serial dependence, disclosed), and (c) the
  granularity, the 1/(N·units) lattice of the frequency objectives. The floor
  is taken over `EPS_CAMPAIGN_DESIGNS` (monte_carlo,
  hazard_filling_stationary) with the binding design recorded; the raw
  requirement differs by at most 2.5× between them. The historic reference is
  measured and reported but excluded from the maximum, because its 77-unit
  estimator would coarsen the reliability precisions well beyond what the
  ensemble designs need. It searches at the shared vector, and its archive
  resolves below its own noise on the reliability axes (disclosed).
- **Re-verification (dropped 2026-11-03).** Floors were measured on the N = 100
  draw-0 ensembles. A re-measurement on the N = 300 campaign ensembles was
  planned and is not run: the binding floors are noise floors, the bootstrap SD
  of the unit-operator estimator falls with the realization count, and the
  adopted entries sit 2.5 to 5 times above their N = 100 floors, so the N = 100
  floors bound the N = 300 floors from above. The Trenton deficit axis keeps the
  deficit-family value; its narrow policy spread (one to two boxes at 10) is a
  disclosed loss of resolution on that axis, not a calibration failure. The
  search vector is therefore fixed; cardinality is handled after the search
  (Stage 2), and Borg's archive behaviour is insensitive to ε changes of this
  size (ε-progress and the archive-parent draw shift at the margin; the
  per-evaluation cost does not).

### Stage 2, family coarsening by archive cardinality

Ensemble-averaged objectives are far smoother than the single record, so
floor-level precisions over-resolve converged ensemble fronts.
`scripts/supplemental/epsilon_ensemble_refilter.py` (launcher
`workflow/supplemental/epsilon_ensemble_refilter.sh`, outputs
`outputs/supplemental/epsilon_refilter/`) re-filters converged search archives
(each seed's `.set` and the step-07 plain union) under grouped candidate
vectors, one ε per family, with an exact ε-box filter cross-checked against
`sensitivity_common.epsilon_nondominated` and Borg's own archive membership.
The adopted vector is the candidate whose largest merged front stays inside the
re-evaluation sizing in force at calibration (1,000 to 1,200 policies; the
campaign caps re-evaluation at 2,000) while every trade-off axis keeps its span
(`grouped_axis_coverage` tables and two-panel parallel-axes figures). The
reliability family is the dominant cardinality lever and the deficit family the
secondary one. Flood stays at its floor value 0.3 because production fronts
span only a few flood ε-boxes, so flood-ε cardinality effects are box-boundary
placement artifacts rather than resolution. Storage stays at 5.0 (a
5 %-of-capacity distinction is significant). The diagnostic is repeated on the
first archives searched at N = 300.

Since 2026-11-03 the script's selection rule is a cap rather than a band: the
finest candidate whose largest-design retained union, inflated by the
conservative 1.35 factor, is at or below `MAX_FRONT_SIZE` = 1,000 (`--max-size`
overrides it). The candidate grid also carries a reliability ladder at the
deficit-family value 10 (0.07, 0.08, 0.10) and deficit 12.5 and 15 at
reliability 0.05 and 0.06. On the August N = 100 archives the adopted vector
prices at 1,338 on Monte Carlo under that rule and reliability 0.06 at 1,104,
so neither qualifies; the rule is to be applied to the first N = 300 archives.

**Disclosures.** Static re-filter counts are lower bounds on live archives,
because the ε-archive steers selection and restarts during a run; the measured
live inflation is 10 to 35 %. `scripts/supplemental/epsilon_refilter_sweep.py`
(launcher `epsilon_refilter_sweep.sh`) prices cardinality under scaled variants
of one vector and keeps two coarser fallback vectors registered, and
`scripts/supplemental/write_refiltered_sets.py` writes archives re-filtered
under a new vector to new filenames, never over production artifacts.

## Merger behaviour

MOEAFramework v5 `ResultFileMerger` merges by plain Pareto dominance regardless
of `--epsilon`, so step 07 ε-box-filters the cross-seed `{slug}_merged.set` in
place under the campaign vector (`src/diagnostics.py::epsilon_box_filter_set` at the time; since 2026-10-23 step 07 builds its reference set directly with the same filter,
plain union kept as `*_raw.set`) before computing metrics, and that filtered
file is the reference set the step 08/09 re-evaluation consumes. The merge,
ε-filter, re-evaluate ordering therefore holds for every configuration by
construction.

## Stage-1 outputs

`tables/epsilon_diagnostics_{…}.csv` (per design), `archive_sweep_{…}.csv`
(per design, the vector × `EPS_SCALE_GRID`), `epsilon_recommendation_{…}.csv`
(combined). Figures `eps_calibration_ladder` (per-design floors against the
adopted epsilons, log axis, historic in reference gray),
`scalar_distributions_{design}`, `archive_size_vs_scale`, and
`parallel_axes_{design}` (ε-box-retained members highlighted, baseline bold).
