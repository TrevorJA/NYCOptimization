# TODO

Open action items only, in execution order. Each item is a pointer, not a spec.
The method and its justification live in `docs/notes/methods/` (campaign at
scale: `campaign_design.md`). Delete items as they land. Venue: **[HPC]**
cluster, **[local]** laptop.

## Rerun chain (2026-11-04)

State at the end of the 2026-11-04 preparation session. Done: git review and output
inventory; every pre-2026-11-04 simulation output moved to `archive/` (repo-root symlink
into projects space, `archive/README.md`; `outputs/` now holds only streamflow data,
presim, the cost experiments and `run_manifests/`); fresh projects-space homes and
`outputs/reference/demand_utilization/` (k(u) table, S12) written; the 0.80
annual-volume extension REVERTED (decision 2026-11-04; the box is the plain widened
CMIP6 range, e^m = 0.866 to 1.32, `src/etest.py`), so the E_test staged 2026-08-19 is
reused: moved back from the archive, its retired sub-window hazard image set aside,
the utilization column appended (job 21947556: u in [0.450, 0.999], k in [0.653, 6.0],
4 SOWs clipped at k = 6);
presim verified current (`start_date` 1945-12-01, `end_date` 2023-11-30: no step-01
rerun); the problem JARs need no rebuild (only DV and objective COUNTS enter the JAR,
still 8; `00_setup_borg_jars.sh` header); Pywr-DRB `nyc_opt` at `9619a90` and scengen
`main` at `c61fef8`, clean and in sync, both editable in `venv`; `squeue` empty; balance
591,765 SU. Every step below is an sbatch wrapper under `workflow/`; nothing runs on
the login node and no bare `srun` one-liners (the metadata-only E_test steps got
wrappers on 2026-11-04: `staging/etest_demand_utilization.sh`, `staging/etest_cut.sh`).
E_test is 500 SOWs only (decision 2026-11-04): `etest_kn_50yr_n12500`, cut as the
leading 25 chunks of the staged 1,000-SOW design (`etest_kn_50yr_n25000`, kept on disk
as that parent only); every re-evaluation, diagnostic and figure reads the E_test slug,
the baseline cube is computed per design on E_test itself, and the manuscript states
500 SOWs without the parent.

Submit from the repo root in this order (`H=workflow/envs/ffmp_obj8_historic_production.env`):

- [ ] **[local] 0. Tests** run on the local machine as usual (never on Anvil unless
  Trevor asks); confirm the new demand-source, utilization, etest and compare_designs
  tests are green there before item 1.
- [ ] **[HPC] 1. HIST production, seed 1** (single trace, needs only the regenerated
  historic CSV; ~6.3k SU at 8.35 SU per 1,000 NFE; the step-07 duplicate guard is clear
  because `outputs/historic/` is archived):
  `bash workflow/submit_production_search.sh historic 1`
  (the helper holds the 12 x 128 geometry and the 12 h wall; `--dry-run` prints the line).
  Optional first: `bash workflow/submit_smoke.sh ffmp` (debug queue, 2 x 40, verifies the
  model build with the new demand series before the 12-node queue wait). Seed 2
  (`bash workflow/submit_production_search.sh historic 2 --dependency=afterok:<seed-1 jobid>`,
  8 h) after seed 1 lands.
- [ ] **[HPC] 2. E_test completion** (no regeneration; the staged flows and presim are reused).
  The staging contract lists the demand series as a required chunk file, so the chunk
  staging precedes the subset build (the reverse order was refused on 2026-11-04):
  1. [x] `staging/etest_stage_nyc_demand.sh`, array 21950437: 50 of 50 tasks COMPLETED
     2026-11-05 00:14 UTC (each chunk carries its k(u)-scaled
     `diversion_nyc_extrapolated_mgd.hdf5` and `demand_utilization_realized.csv`)
  2. [x] `staging/etest_cut.sh`, job 21951643, COMPLETED 2026-11-05 00:25 UTC (job
     21950524 ran the pre-merge wrapper and failed on step order): `etest_kn_50yr_n12500`
     cut from the parent (25 of 50 chunks, 500 SOWs x R = 25, `subset_of` recorded),
     `assert_staged_etest_contract` passed with the utilization keys on the reused
     2026-08-19 meta; realized minus nominal u: mean -0.027, range [-0.066, +0.008],
     24 SOWs above nominal.
  3. (folded into 2)
  4. `J5=$(sbatch --parsable --export=ALL,NYCOPT_ETEST_VARIANT=kn workflow/supplemental/staging/etest_hazard_image_shards.sh)`;
     `sbatch --dependency=afterok:$J5 --export=ALL,NYCOPT_ETEST_VARIANT=kn workflow/supplemental/staging/etest_hazard_image_merge.sh`
     (after 2.2: 25 tasks over the E_test chunks, ~10 SU, written into the E_test dir;
     the parent's retired-scoring image was moved to the archive)
  5. later: `si_figures_design.sh` (Figure 3, S12 to S16 from the staged
     `forcing_profiles.npz`) and the `dry_envelope.sh` production leg (Section 2).
  The realization-level `hazard_image.npz` E_test inherits from the parent at the cut
  stays on the retired 8-axis scoring; only the exploratory scenario discovery reads it (rescore
  before any scenario-discovery run).
- [ ] **[HPC] 3. Current-FFMP baseline on E_test** (after 2.2; once per design env,
  ~33 SU each at the measured unit cost; geometry per the step-10 vet item in Section 2):
  `sbatch --partition=shared --nodes=1 --ntasks=16 --cpus-per-task=8 --time=12:00:00 --export=ALL,NYCOPT_ENV_FILE=<env>,NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500,NYCOPT_CHUNK_POLICIES=baseline,NYCOPT_SEARCH_REALIZATION_BATCH=50 workflow/10_simulate_test_chunks.sh`
  for `$H` and both matched-design envs (in-job merge; a baseline-policy run lands in
  `outputs/{design}/ffmp_obj8/reeval/etest_kn_50yr_n12500/baseline/`, the leaf the regret
  family of that design's step-10 policy run reads; no symlinks, no cross-design
  staging). `RTD_REEVAL_BASELINE_DIR` already names the historic copy.
- [ ] **[HPC] 4. Historic step-06 baseline** (objective anchor; the cube above replaces
  the former `--reeval` leg, now opt-in via `NYCOPT_BASELINE_REEVAL=1`):
  `sbatch --export=ALL,NYCOPT_ENV_FILE=$H workflow/06_run_baseline.sh`
  -> `outputs/baseline/historic/ffmp_baseline_objectives.csv` (now the Trenton deficit column).
- [ ] **[HPC] 5. After HIST seed 1 (and seed 2)**: `extract_runtime_archive.py --seed 1`,
  the ε re-filter, seed 2, `--merge --install`, step 09, steps 10 + 11 on
  `etest_kn_50yr_n12500`, then 10/13/14 (Section 2 items).

Day target: item 1 submitted, items 2.1 to 2.4 done or submitted, item 3 chained as 2.1 lands. Matched-design staging at N = 300 (Section 2) is independent and can
queue alongside.

## 0. NYC demand model (2026-11-03)

NYC demand is the time-dynamic extrapolated diversion series everywhere
(`config.nyc_demand_source`; per realization for ensembles, staged by step 04 as
`diversion_nyc_extrapolated_mgd.hdf5`; pywrdrb's regenerated 1945-2023
historic CSV on the inflow reconstruction for the single trace). It is the only
NYC demand source and no env file selects one (`tests/test_demand_source.py`).
Search runs the unscaled series (k = 1); E_test scales it per SOW by k(u)
(`forcing_parameterization.md`, "Demand-utilization axis"). NJ stays at its
100 MGD allocation. Every simulation output, staged demand series and historic
CSV produced before 2026-11-04 predates the current series (the regressions are
trained and applied on the reconstruction's reservoir inflow,
`demand_covariate_diagnostics.md`) and is superseded; the inventory below is
complete as of that date.

**Unaffected (do not redo).** Everything that is streamflow-only: the P = 10⁶
pools and smoke pools, the hazard images and step-03 selections, the HF design
metrics, the hazard-axis selection, the persistence and CV-axis diagnostics, the
dry-envelope diagnostic, the forcing-space figures, and the step-01 single-trace presim
(the non-NYC STARFIT releases do not depend on NYC demand; verified current 2026-11-04).
E_test IS reused: the 0.80 annual-volume extension of `3ddc64b` was reverted on
2026-11-04 and the 2026-08-19 staging over the 0.866 box is the campaign E_test (rerun
chain, item 2). The measured search
cost basis (21,850 SU per N = 100 / 500k) also stands: demand is a parameter
lookup; confirm from seed 1 rather than re-measure. The search ε vector is fixed
by decision (Anvil, 2026-11-03; `epsilon_calibration_experiment.md`).

**A. Staging (no simulation), before anything else**

- [x] Repos pulled (Pywr-DRB `9619a90`, scengen `c61fef8`); k(u) table and utilization column
  written 2026-11-04 (`etest_demand_utilization.sh` is the reproducible path; rerun with
  `NYCOPT_FORCE=1` only if `src/demand_utilization.py` constants change).
- [ ] **[HPC]** E_test demand-utilization axis on a FRESH build (the reused staging is
  handled by the rerun chain, item 2): `append_demand_utilization.py --variant kn` after
  the merge, then `staging/etest_prep_chunks.sh` stages the k(u)-scaled series itself (no
  separate `staging/etest_stage_nyc_demand.sh` pass), then
  `merge_demand_utilization_realized.py --variant kn`; `make_etest_subset.py` only when
  E_test is cut from a larger generated design. A chunk without the file fails
  `staged_ensemble_missing` and the pywrdrb build.
- [ ] **[HPC]** Every already-staged ensemble that will be simulated again (none carries
  `diversion_nyc_extrapolated_mgd.hdf5` today):
  `workflow/supplemental/staging/ensemble_stage_nyc_demand.sh` (array index = draw, or `--preset SLUG`).
  Known reusers: the N = 100 d0 ensembles of both matched designs
  (`supplemental_config.TEV_TARGETS`; the ε-calibration cubes), the ensemble-size ladder
  and library slugs (`esd_pool_slug`, `esd_library_slug`, `ESD_N_LADDER` 50 to 500), the
  memory-smoke ensemble (`envs/smoke_search_batch.env`, `monte_carlo`), `kn_50yr_n5`, and
  the scaling/cost ensembles if ever re-benchmarked. `fixprob_10yr_n300_d0` has only its
  step-02 files and needs full step 04. New N = 300 stagings (§2) get the series from
  step 04. Local fixtures get it from `ensure_local_test_ensemble` on first use.

**B. Simulation outputs superseded (regenerate, in this order)**

- [ ] **[HPC]** Step 06 baselines: the `historic` single-trace baseline (the
  anchor of the satisficing criteria and of the §3 tolerance rules) and each
  matched design's d0 baseline at N = 300 (with the §2 staging). Every pre-2026-11-04
  baseline is in `archive/2026-11-04_pre_demand_update/outputs/baseline/`.
- [ ] **[HPC]** Current-FFMP E_test baseline cube: step 10 with
  `NYCOPT_CHUNK_POLICIES=baseline` on E_test under each design env (~33 SU each;
  rerun chain item 3), after the §A demand staging so the cube carries the
  per-SOW utilization scaling. The old cubes are archived
  (`.../outputs/historic/ffmp_obj8_mm_full/reeval/`).
- [ ] **[HPC]** Searches: every stored Pareto set (the August HIST set, the
  N = 100 production and moderate MC/HF sets, the `ffmp_vr` sets) predates the
  current demand series and is superseded by the §2 campaign searches (HIST included: the
  rerun must use the regenerated historic CSV). Never evaluate an old set
  against new baselines or cubes; the §3 `.set` caveat applies doubly.
- [ ] **[HPC]** E_test re-evaluation cubes of those sets
  (`outputs/*/ffmp_obj8/reeval/etest_*`), the step-12 comparison tables under
  `outputs/comparison/`, and every step-13/14 figure: regenerate from the new
  sets; the old ones are in `archive/2026-11-04_pre_demand_update/` (2026-11-04).
- [ ] **[HPC + local]** Supplemental diagnostics that simulate, each superseded
  together with its tables and figures:
  - `etest_realization_precision.py` (per-SOW noise floors; `etr_per_sow_noise.csv`,
    `etr_options.csv`, SI Table S1): rerun on the new baseline cube, then refresh
    the quoted floors in manuscript §3.4.1, `campaign_design.md` §5 and
    `research_project_summary.md`; the regret tolerance floors read the same file.
  - `regret_tolerance_diagnostics.sh` pass A on the new incumbent cube
    (`rtol_noise_floor.csv`), pass B after the re-evaluation (§3).
  - `robustness_threshold_diagnostics.sh`, `robustness_threshold_anchor.py`,
    `criteria_reanchoring.py` on the new historic step-06 attainment and the new
    baseline cube (§3); the placements in `robustness_threshold_diagnostics.md`
    §3 and `src/satisficing_criteria.py` are provisional until then.
  - ε-calibration cubes (`epsilon_calibration.sh` with the three
    `eps_calib_*.env`; `outputs/supplemental/epsilon_calibration/`, SI Text S5):
    the ε vector does not change, but the cubes are the substrate of the
    framing-convention screens below, so they are regenerated if those screens
    are re-run; otherwise nothing changes.
  - Framing-convention screens (`framing_convention_analysis.py`,
    `satisfaction_factor.sh`; `framing_convention_diagnostics.md` "Measured
    verdicts": k = 3 / k = 1 failing weeks, the 0.99 factor): NYC delivery
    reliability is the axis most sensitive to the demand level (the curtailed
    caps bind only when demand exceeds them), so re-screen k for saturation on the
    regenerated cubes before the campaign reports it; refresh the verdict
    sentences in `objective_definitions.md` and manuscript §3.2.2.
  - Ensemble-size diagnostics (`ensemble_size_library_stage.sh`,
    `ensemble_size_library_eval.sh`, `ensemble_size_analysis.sh`;
    `ensemble_size_diagnostics.md`, SI Texts S4 and S5; the claim that N = 300
    resolves every objective's paired difference within ε/2): the NYC objectives'
    sampling variance depends on the demand series; re-run Layer B on demand-staged
    ladder ensembles, or restate the claim for the NYC axes from the new library.
  - Transfer-evaluation matrix (`transfer_evaluation_run.py`,
    `envs/transfer_evaluation.env`, N = 100 cells; `tev_path_consistency.csv`
    cited in §3): superseded with the sets; re-run only once new sets exist (it
    is also the §3 draw-sensitivity driver).
  - Flood-objective diagnostics (`flood_objective_run.py`,
    `flood_exceedance_baseline_figures.py`; `flood_objective_diagnostics.md`):
    the local pass (34 policies on the historic record and `kn_50yr_n5`) predates
    the current demand series; storage and therefore spill and exceedance move
    with demand, so re-run before quoting any of its numbers.
  - Objective-dynamics figures (`objective_dynamics.py`,
    `ensemble_objective_dynamics.py`; `objective_dynamics_diagnostics.md`):
    re-render from the new baseline (they annotate delivery against demand).
  - `objective_sensitivity_run.py` (random-DV single-trace discrimination;
    `outputs/supplemental/objective_sensitivity/`) and the single-record
    epsilons carried by `src/objectives.py` (IQR/10 on the historic record,
    diagnostics only): re-run if used; low priority.
  - `check_objective_determinism.py`
    (`outputs/supplemental/objective_determinism/summary.csv`, the SI Text S1
    numbers in §4): cheap local re-run under the current demand series before
    quoting.
  - `sim_selected_policies.sh` / `simulate_selected_policies.py` (full-model
    traces behind figures 05/06 and the historic dynamics figures, §4): re-run
    from the new sets.
- [ ] **[local]** Numbers to refresh once the above exist: the per-SOW noise
  floors (three places), the framing-convention verdicts, the criteria
  placements, and the HIST baseline attainment wherever the manuscript quotes it.

## 1. Anvil HPC rename for the Monte Carlo design

Path renames landed on the cluster 2026-08-30; run-manifest snapshots under
`outputs/run_manifests/` deliberately keep the old string as frozen per-run provenance.
Every file that still carried `fixed_probabilistic` internally (re-eval meta, comparison
tables, ε-calibration cube and tables, refilter recommendation, framing and
satisfaction-factor tables, ensemble-size Layer-B tables) was archived 2026-11-04 and
will be regenerated from the renamed inputs; nothing is left to rewrite in place.

## 2. Campaign at N = 300 (`campaign_design.md` §4–6)

- [ ] **[HPC]** Pull all four repos on Anvil; check the SU balance (`mybalance`)
  against the ~600k the budget assumes. The main repo calls the
  `NYCOptimization_scenario_generation` interface of `6790a02` from `71b3478` on, so
  pull both together.
- [ ] **[HPC]** June 1 hazard-image recompute. The P = 10⁶ pools
  (`statpool_10yr_n1000000_d{0,1}`) and the P = 2,000 smoke pools were regenerated
  2026-10-01 (3,769 SU; costs and acceptance in `ANVIL_NOTES.md`), with the pre-June-1
  artifacts under `archive/2026-10-01_pre_june1_hazard_recompute/`. Frozen by that generation (a change
  to any forces another): the scengen `hazard_metrics` scoring constants
  (`_DRY_SCORING_RULE`, `_WET_SCORING_RULE`, `_SUPPLEMENT_SCORING_RULE`,
  `_DRY_CUT_MONTHS = 6`, `_SCENARIO_STAMP_START`, `_REFERENCE_START`), the generator
  fit (flowtype `pub_nhmv10_BC_withObsScaled`, start 1945-12-01), the `stat_pool` seed
  domain and the synhydro version; the selection axes, N and ε live downstream of the
  image. Still open on the June 1 images: the step-03 selections (restage item below),
  both E_test images (regeneration item below), the cached
  `outputs/supplemental/historic_hazard_windows/` (recomputes itself on the provenance
  mismatch), and the readers: `compute_staged_hazard_image.py` for the Monte Carlo
  ensembles, the hazard-examples figure, `ensemble_size_hazard.sh`,
  `hazard_support_decomposition.sh`, the E_test overlay, and `hf_design_metrics.sh`
  (Section 7). The selector diagnostic ran on the regenerated pools 2026-10-01 and is
  repeated only on the adopted axes (selection-axis item below).
- [x] E_test regeneration: not needed. The extension to 0.80 was reverted 2026-11-04
  (`src/etest.py`, `forcing_parameterization.md` "Box width", `campaign_design.md` §5,
  manuscript §3 and SI Text S6 restated on the 0.866 box). The staged E_test of
  2026-08-19 is completed in place by the rerun chain at the top (item 2); Figure 3 and
  SI Figures S12 to S16 re-render from it.
- [ ] **[HPC]** Production leg of the dry-envelope diagnostic on the regenerated images:
  `sbatch --export=ALL,NYCOPT_DRYENV_PRODUCTION=1 workflow/supplemental/design_diagnostics/dry_envelope.sh`
  (reads `etest_kn_50yr_n12500/hazard_image_subwindows.npz` + `forcing_profiles.npz` and
  `statpool_10yr_n1000000_d0/hazard_image.npz`; writes `dryenv_production_{bins,reference}.csv`
  and figure `F2_dry_envelope_production`, the production confirmation of SI Figure S18 and
  the numbers quoted in SI Text S6). Minutes on `shared`.
- [ ] **[local → HPC]** Fix the selection-axis set from the hazard-axis selection
  (`hazard_axis_selection.md` §7, run locally on the production images): set
  `config.HAZARD_SELECTION_AXES` to the adopted set; if it holds a metric outside the
  eight event metrics, make step 03 select on the full 21-metric image
  (`scripts/main/select_hazard_filling.py` passes only `H`; `select_from_candidate_image`
  raises for an unknown axis), update `tests/test_hazard_selection_axes.py` (asserts six
  event metrics) and the readers that index the image by the selection axes
  (`hf_design_metrics_run.py`, `ensemble_size_hazard.py`, `hazard_support_run.py`,
  `hazard_examples_figures.py`, `plot_etest_hazard_overlay.py`,
  `src/plotting/{ensemble_composition,hazard_examples}.py`), and verify on the 2,000-row
  prefix that step 03 reproduces the diagnostic's member indices. Then run
  `workflow/supplemental/design_diagnostics/hazard_axis_selection.sh` (Anvil confirmation) and the selector
  diagnostic on the adopted axes (`diagnose_hazard_selectors.py`,
  `hazard_selector_diagnostics.md` §5 placeholders), and update
  `scenario_design_methods.md` §3.3 and §6, `terminology.md`, the manuscript and SI.
- [ ] **[HPC]** Restage search ensembles at N = 300, draws 0–1: step 02
  (`monte_carlo`, `--array=0-1`), step 03 (`hazard_filling_stationary`,
  `NYCOPT_CANDIDATE_POOL_N=1000000`; confirm the log line
  `pool='statpool_10yr_n1000000_d{k}'`), step 04 both (`--array=0-1`; it also
  stages the NYC demand series). Staged so far: `fixprob_10yr_n300_d0` (step 02,
  2026-09-11; its image is rescored by `compute_staged_hazard_image.py`, item above).
  The 2026-09-11 `hazfill_stat_abs_10yr_n300_d0` selection was made on the pre-June-1
  pool and sits in `archive/2026-10-01_pre_june1_hazard_recompute/`, so step 03 is open for both draws,
  as are MC draw 1 and step 04 for every draw.
- [ ] **[HPC]** Build QC on each restaged ensemble: `validate_staged_seasonality.py`
  and the per-axis tail-share record per hazfill draw; then step 06 baselines for
  both matched designs scenario-matched to d0 (`--search-ensemble`).
- [ ] **[HPC]** Batched-search memory smoke: `bash workflow/submit_search_memory_smoke.sh`.
  Go if peak node memory ≤ ~217,000 MB and warm per-evaluation time ≤ 540 s + 20 %;
  otherwise batch 100 (memory miss) or re-price `campaign_design.md` §6 (time miss).
- [x] HIST pre-2026-11-04 outputs archived (`archive/2026-11-04_pre_demand_update/outputs/historic/`,
  2026-11-04); the step-07 duplicate guard no longer trips for seed 1. Never relaunch a
  completed cell in place with `NYCOPT_OVERWRITE=1`: Borg opens the island runtime files
  with `fopen(..., "w")` and would truncate them.
- [ ] **[HPC]** Seed 1 per design (750k NFE; submit lines in the `*_production.env`
  headers). After ~12 h read SU/NFE and the seed's coordinator-archive HV at
  500k total NFE (step 09 on the partial runtime files); hold seed 2 if SU/NFE
  exceeds the model basis or HV still rises > 5 % per 100k total NFE.
- [ ] **[HPC]** Per design: `extract_runtime_archive.py --seed 1` (writes
  `seed_01_ffmp_obj8_nfe125000.set`), then the ε cardinality re-filter
  (`epsilon_ensemble_refilter.py`, selection rule: finest grouped vector whose
  largest-design union x1.35 is <= 1,000, `--max-size`); apply the §5 cap rule
  if the merged union exceeds ~2,000.
- [ ] **[HPC]** Seed 2 per design (500k NFE; `submit_production_search.sh <design> 2`), then
  `extract_runtime_archive.py --merge --install` (installs `ffmp_obj8_merged.set`,
  the step-10 reference) and step 09 (coordinator-archive diagnostics; writes
  only `ffmp_obj8_diagnostics_reference.set`, either order).
- [ ] **[HPC]** Vet the step-10 re-evaluation geometry before any step-10 submission
  (desk assessment 2026-11-04, not yet verified on the cluster; implement or reject).
  The documented layout (`shared`, 16 ranks × 8 cpus, batch 50: 09 header comment,
  `workflow/envs/README.md` campaign line, `campaign_design.md` §5, the `ANVIL_NOTES.md`
  rule of thumb) bills 8 cores per single-threaded rank (`nycopt_pin_threads` sets
  OMP = 1; the pywr solve is serial), so 7 of every 8 billed cores idle. One
  (policy, chunk) unit is 500 realizations × 50 yr = 25,000 scenario-years; the
  trimmed-model cost surface
  (`outputs/supplemental/ensemble_cost_experiment/tables/cost_surface.csv`, ~0.17 s per
  scenario-year at N = 50, × 1.01–1.17 packing slowdown from
  `anvil_scaling_experiment/tables/packing_summary.csv`) prices it at 62–72 core-min,
  so 16 × 8 costs ~8.3 SU per unit (~415k SU at the 2,000-policy cap), not the
  1.33 SU per unit / 33 SU per policy that `campaign_design.md` §6, `ANVIL_NOTES.md`
  and the next item record. That measured figure is consistent only with dense
  packing (the `--nodes=8 --ntasks-per-node=128` submission the 09 header comment still
  describes); no per-unit wall-time log from the 16 measured jobs is on hand, so
  confirm from one. The dense-packing OOM in `ANVIL_NOTES.md` is the batch, not the
  rank count: batch 50 × 50 yr = 2,500 scenario-years per rank ≈ 1.5–1.8 GB
  (`cost_surface.csv` neighbors 1.3–2.0 GB; `config.search_rank_rss_mb` 1.83),
  × 128 ranks = 190–235 GB against the 217 GB line. Per-rank memory is bounded by
  the batch, not the chunk (pywrdrb `FlowEnsemble` reads only the requested
  realization columns; `run_simulation_ensemble_batched` frees each batch before the
  next), and the per-model-run overhead (~5 s, `batch_summary.csv`) is 1–2 % at
  batch 20–25. Proposed: 128 ranks per node on `wholenode` (8 nodes, 48 h) with
  `NYCOPT_SEARCH_REALIZATION_BATCH=20–25` (~1.0–1.2 GB per rank, 140–155 GB per node;
  ~1.25 SU per unit, ~62k SU at the cap, ~2.5 days per 8-node pass) and
  `NYCOPT_CHUNK_UNIT_SECONDS=4500` so the wall guard (`NYCOPT_CHUNK_STOP_EPOCH`,
  inert while `CHUNK_UNIT_SECONDS` is 0) stops a rank before a 60–75 min unit is
  cut. Fallbacks: 96 × 1 batch 25 (~1.2–1.6 SU per unit), 64 × 2 batch 50
  (~2.4). Vet by a one-node smoke on the subset preset: 128 ranks, batch 25, ~12
  policies (`NYCOPT_CHUNK_POLICIES`), read `elapsed_s` and `rss_gb` from the
  `chunk_reeval` per-unit lines; go if peak rank RSS ≤ ~1.5 GB and node RSS ≤ 217 GB,
  otherwise 96 ranks or batch 20. Note `nycopt_check_memory` is not called by 09 and
  sizes from the search spec anyway. Whichever geometry is adopted: re-derive the
  SU-per-policy line from the smoke, fix the four records above, and re-check the
  ~66 SU current-policy cube lines (§0 B, §2) that carry the same per-unit basis.
  Then write `workflow/submit_reeval_chunks.sh <design>` (steps 10 + 11 at the
  adopted geometry, the pattern of `submit_production_search.sh`) so no step-10 line
  is typed at the prompt.
- [ ] **[HPC]** E_test re-evaluation: steps 10 + 11 at the geometry settled by the
  item above (documented: `shared`, 16 ranks × 8 cpus, batch 50),
  `NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n12500` on every
  10/11/12 line (deliberately not in the env files). ~66k SU at the
  2,000-policy cap on the 33 SU-per-policy basis.
- [ ] **[HPC]** Post-processing on the 500-SOW cube: steps 12, 13 and 14, the criteria
  re-anchoring audit (`criteria_reanchoring.py`), and the θ-subsample stability
  check (250 vs 500 SOWs).
- [ ] **[HPC]** Re-render the ensemble-size figures with the campaign marker
  (`workflow/supplemental/design_diagnostics/ensemble_size_analysis.sh`, `ESD_N_CAMPAIGN = 300`).

## 3. Diagnostics on the step-10 cube

- [ ] **[HPC + local]** The search-time `.set` data stays untouched (the HIST copy was
  moved, not deleted, to `archive/2026-11-04_pre_demand_update/outputs/historic/ffmp_obj8/` on 2026-11-04) and
  is deleted only once the campaign has generated its replacement (Section 2, after
  `extract_runtime_archive.py --merge --install`). Its objective columns were computed
  on the metric window of `dc7e70b` (`START_DATE` 1945-10-01, `END_DATE` 2022-09-30),
  before `a1e88bd` aligned the metrics on June 1, and sit 1 to 4 ε from what the current
  code computes (`outputs/supplemental/transfer_evaluation/tables/tev_path_consistency.csv`).
  Until the replacement exists: never compare stored `.set` objective columns with
  freshly computed values; the adopted ε vector and the re-filtered cardinalities
  (335/991/784 on disk, 1,040/833/335 in `campaign_design.md`) carry the old window
  and are superseded by the post-search re-filter of Section 2 (the search ε vector
  is fixed as adopted; decision 2026-11-03, `epsilon_calibration_experiment.md`).
  The E_test re-evaluation and the transfer evaluation simulate and are unaffected.

- [ ] **[local]** Regret tolerance: re-run pass A on the regenerated incumbent cube
  (`rtol_noise_floor.csv` carries stale ε), then pass B on the production cube with
  `NYCOPT_REGRET_TAU` unset (k-sweep, seed nulls, paired bootstrap, assay control),
  which yields the discrimination band δ.
- [ ] **[local]** Satisficing thresholds: re-run `robustness_threshold_diagnostics.sh`
  and adopt final criterion values and the sweep-grid centre (open decision;
  placements are provisional until then).
- [ ] **[local]** Satisficing criterion for `trenton_flow_deficit_p99_pct`: the 10 %
  placement (FFMP drought-stage target) in `src/satisficing_criteria.py`,
  `_DEFAULT_THRESHOLDS` and `RTD_RECOMMENDED_THRESHOLDS` has not been placed against
  the incumbent's E_test cube; run `criteria_reanchoring.py` on the regenerated cube
  and settle it with the other criteria.
- [ ] **[local]** Framing diagnostic 3: OAT stringency + threshold-margin CDFs on the
  persisted cube (`framing_convention_diagnostics.md`).
- [ ] **[HPC]** SI draw-sensitivity re-evaluation: a thinned subset of each matched
  design's merged set on its own d1, sized to 250 SU (`campaign_design.md` §5). Fix
  the thinning rule and the policy count before the run. The driver is
  `src/transfer_eval.py` + `scripts/supplemental/transfer_evaluation_run.py`, which
  evaluate a `.set` on any staged search-ensemble slug via `evaluate_annual_units`
  and persist per-realization units. The d1 cells are declared in
  `supplemental_config.TEV_DRAW_TARGETS`; run with `NYCOPT_TEV_INCLUDE_DRAWS=1` on
  `workflow/supplemental/results_diagnostics/transfer_evaluation_eval_array.sh`. Report paired per-policy
  shifts against ε with no variance estimate.
- [ ] **[HPC]** Optional: nested-P saturation record (`nestedp_ladder.sh`) under the
  renamed hazard axes; hazard-support no-harm arm re-read with
  on the E_test tag (the default `HSD_REEVAL_TAG`; no simulation).

## 4. Figures and manuscript

- [ ] **[local]** Historic dynamics figures (`outputs/figures/historic/ffmp_obj8/dynamics/`,
  1960s and 2002 droughts): decide which periods/panels go to the SI and whether to
  add panels (Trenton, NJ diversion, drought level) via the `PANELS` registry; render
  them from the campaign HIST set (Section 0: pre-2026-11-04 results are superseded).
- [ ] **[local]** Figure 9: keep one of the three `number=9` registry entries
  (`regret_exposure` recommended; the two `regret_surfaces*` variants degenerate at
  the adopted τ) and delete the other two from `src/figures/registry.py`.
- [ ] **[local]** Figure 4: settle the scatter geometry (`NYCOPT_COMPOSITION_SCATTER`
  3d vs 2d), the flood axis (peak discharge vs pulse duration), and whether the
  floor shadow stays.
- [ ] **[local]** Layout at the manuscript-final pass: figs 06/08 annotations sit below
  the 12 pt floor (full-page or landscape); mark figs 07/08 `per_focal` or pin the
  focal criterion.
- [ ] **[local]** Manuscript Results / Discussion / Conclusions; SI Texts S9–S11 are
  outline-only.
- [ ] **[local]** SI gaps: Text S12 (tolerance rules) is cited in §3.4.3 but does not
  exist; Text S7 runtime diagnostics finalize after the campaign; Text S3
  downstream-stress correlation and event-seasonality checks are still marked
  planned; Text S1 needs the determinism statement (frequency objectives exact,
  tail objectives jitter-limited at ≤ 5e-3 relative, three to four orders below ε;
  `outputs/supplemental/objective_determinism/summary.csv`, 2026-09-30).
- [ ] **[local]** Decide whether the variable-resolution `ffmp_N` sweep runs (leftover
  SU only) and under which design; the manuscript states two research questions,
  so either add it as an SI extension or drop RQ3 from the notes.

## Parked (scope, not blockers)

- [ ] Benchmark env files `anvil_scaling_borg.env`, `anvil_scaling_packing.env`,
  `ensemble_cost.env` still list the 7-objective set with `trenton_flow_reliability_weekly`;
  harmless for cost reruns (the diagnostic objective stays registered) but they no longer
  match the campaign formulation. Update if any is ever resubmitted.
- [ ] Forward hook only: the deferred DU-factor presets (`src/ensembles.py`,
  `du_factors`; the one live DU factor outside the forcing box is the E_test
  utilization axis, `src/demand_utilization.py`). The per-realization
  diversion staging (`src/ensemble_prep.py::stage_nyc_demand`) is live.

## 5. Proposal revision follow-ups (2026-09-02)

- [ ] **[local]** Zotero: add Hogarty (1970), Van Loon (2015), Tijdeman et al. (2020), AghaKouchak et al. (2021), Brunner (2023), Brunner & Gilleland (2020) to collection `ISYGLK35` (currently only in the Paper 2 collection); OCR the Hogarty PDF and confirm its year (1969 vs 1970) and ICP case number.
- [ ] **[local]** Decide capitalization across the two papers: proposal uses "Decree Parties" / "1954 Decree" (FFMP document form); the stochastic DRB manuscript uses "decree parties" / "1954 decree".

## 6. Proposal v2 review follow-ups (2026-09-28; Reed and Lau reviews)

Dispositions of every comment are in `docs/experimental_proposal/reviewer_feedback_synthesis_2026-09.md`.

- [ ] **[local]** Climate-change uncertainty inside the search ensembles (Reed): deferred by
  decision, not dropped. Arguments against are in the project presentation supplemental slides
  79–96 (clean search/re-evaluation split; HF as a complement to DU optimization; most
  re-evaluation hazard conditions lie inside the stationary candidate ensemble's range). Revisit
  only if the campaign shows re-evaluation failures concentrated in forcing regions whose hazards
  the stationary candidate ensemble cannot reach.
- [ ] **[local]** Lau's regional minimax (worst of the Montague and Trenton flow
  objectives): decide whether to evaluate it as an SI formulation variant.
- [ ] **[HPC]** Downstream-stress correlation of the hazard axes (asserted in proposal v1,
  never computed): the SI Text S3 diagnostic in item 4; the proposal now rests on the
  storage-conditioning and directed-release rationale until this exists.
- [ ] **[local]** Figures embedded in proposal v2 need re-rendering with current labels and
  sizes before circulation: `fig04_ensemble_composition` at N = 300 (legend "PS" → "MC",
  "SSI-months" → "deficit-months"; needs the P = 10⁶ pool hazard image on Anvil);
  `hazard_examples` on the campaign HF ensemble; the methods diagram
  (`figures/manuscript/methods_diagram/methods_diagram_draft.pptx`: N = 300, MC, 500 SOWs ×
  25 × 50 yr, volume axis 0.866 to 1.32); and a planned FFMP operating-rules figure with the decision-variable groups
  marked (baseline only, from `src/plotting/policy_rules.py`).
- [ ] **[local]** Port the v2 literature-review structure (overfitting and policy structure;
  the MORDM and DU-optimization lineage; the three ensemble-construction families) and the
  deterministic-versus-stochastic search framing into the manuscript Introduction outline P2–P5.
- [ ] **[local]** Reconcile the satisficing-criteria variant (`DEFAULT_CRITERIA_VARIANT =
  "v2_20260821"` vs the "adopted" values in `robustness_threshold_diagnostics.md`).

## 7. HF design metrics (2026-09-28)

- [ ] **[HPC]** After the June 1 recompute (Section 2 item; the pre-fix images lack
  the current provenance legs and every reader refuses them), run
  `workflow/supplemental/design_diagnostics/hf_design_metrics.sh` on the June 1 window images
  (`statpool`/`hazfill_stat_abs` d0–d1 at P = 10⁶, N = 300; `fixprob` d0–d1; the historic
  windows cache), or copy `hazfill_stat_abs_10yr_n300_d{k}/{hazard_image.npz,_meta.json}`,
  `fixprob_10yr_n300_d{k}/hazard_image.npz`, and the historic windows cache here (about
  80 MB per draw) and run the driver locally with `NYCOPT_CANDIDATE_POOL_N=1000000
  NYCOPT_SEARCH_N=300`; then fill the bracketed placeholders in SI Text S4, manuscript §3.1.3,
  the proposal §3.2.1–3.2.2, and `hf_design_metrics.md` §8 from `hfm_summary.csv`,
  `hfm_axes.csv`, `hfm_bound_stability.csv`; place `F1_marginals_range`,
  `F2_coverage_diversity`, `F3_target_displacement`, `F4_measure_weights` as Figures S4–S7.
- [ ] **[local]** Later decision on the selection rule, argued from the production metrics
  (not part of the current method): replace the sequential rule (Eq. 7) by the certified
  exact assignment if the measured gap in total displacement or the Jaccard overlap is
  material across d0–d1; optimize the target design (maximin or centered-discrepancy
  Latin hypercube) if the targets' own MST edge statistics fall below the random
  reference; a minimax-optimal (k-center) selection on the candidate ensemble if the HF
  minimax distance is not below the random reference.

## 8. Hazard scoring review (2026-09-29)

- [ ] **[local]** Manuscript §3.1.3: the selection-diagnostic numbers (target
  displacement across axis counts, the tail shares at $P = 10^6$) were measured under
  the retired scoring; replace them from the production selector diagnostic.
- [ ] **[local]** SI Text S3: the numbers in the redundancy-screen, selection-axes and
  selection-invariance paragraphs were measured under the retired scoring and the
  previous axis set; replace the selection-axes paragraph with the method and result of
  `hazard_axis_selection.md` (continuity requirement, pairwise rank-correlation rule,
  dry/wet minimum, the measured statistic and the adopted set) and the other numbers from
  the production selector diagnostic on the adopted axes.
- [ ] **[local]** After the production selector diagnostic, add to the SI:
  - the truncation table (`truncation_summary.csv`: pool, selected-member and
    top-decile fractions of truncated onsets and terminations);
  - the metric redundancy figure on the 21 metrics with the participation ratio
    (`descriptor_redundancy.csv`, figure F6);
  - the selection-set profile with tail shares on every metric for the adopted set
    (`selection_set_profile.csv`, figure F11) and the hazard-axis selection's ranking
    and enrichment-profile figures (`hazard_axis_selection.md` §6);
  - one sentence on the single-event description: among windows with two or more
    qualifying events the controlling event is also the deepest in 65%, the fastest
    developing in 21% and the fastest terminating in 21% (local sample of 4,000
    realizations). No script writes these shares. Compute them from staged flows with
    `scengen.hazard_metrics.drought_events` if they are reported.
- [ ] **[local]** Regenerate the proposal's Figure 3 under the current scoring
  (`workflow/supplemental/design_diagnostics/hazard_examples.sh`). The embedded image carries the retired
  rate labels, and in two of its four examples the shaded event was not the largest
  because a larger drought open at the window end had been dropped. State in the
  caption that an arrowhead marks an event cut by the window edge.
