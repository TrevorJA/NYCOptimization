# Pre-campaign TODO

Concise action items between now and the HPC optimization campaign, in rough
priority order. Each item is a pointer, not a spec — investigate and plan the
details when picked up. Delete items as they land; decision records live in
the methods notes, not here.

Venue tags: **[local]** laptop-only, **[HPC]** needs the cluster,
**[local→HPC]** decide/dry-run locally, execute at scale on the cluster.

## 1. Remaining method closures

- [ ] **[HPC]** UNIFIED METRIC CURRENCY re-runs (2026-08-07 substrate change:
  robustness/regret now score the per-SOW annual-unit search objectives; the
  whole-trace re-eval metrics are retired). Items 1 and 3 DONE 2026-08-08
  (incumbent cube regenerated at
  `outputs/historic/ffmp_obj8_mm_full/reeval/etest_kn_50yr_n25000/baseline/`
  — jobs 19733672 + 19738752, old cube archived alongside as
  `baseline_oldsubstrate_20260803/`; threshold vector ADOPTED into
  `objectives_ensemble._DEFAULT_THRESHOLDS`, values unchanged from
  provisional, storage arbitration documented in the status comment).
  Remaining:
  1. steps 08/09(+09b) re-evals — wait for FRESH campaign Pareto sets; the
     prior `.set`/`.ref` archives use the retired factor DV encoding and
     `reeval_core` has no encoding-version handling, so re-simulating them
     under the new decoder would silently misdecode DVs. The 2026-08-08 50k
     runs (both designs, draw 0 / seed 1) ARE on the current encoding and
     could be re-evaluated, but they will be superseded by the scaled-NFE
     campaign — re-evaluate the campaign sets, not these,
  2. regenerate steps 10/11/13 artifacts (needs the step-08 policy cubes).
  INTERIM SUBSET RE-EVAL staged 2026-08-12 (Trevor approved): the first-round
  500k-NFE fronts will be re-evaluated on a 200-SOW chunk-prefix subset of
  E_test (`etest_kn_50yr_n25000_first10ch`, 10/50 chunks, R=25 untouched —
  per-SOW values stay in the final metric currency; worst-case Starr-fraction
  SE +/-3.5 pp) to shape the robustness message ahead of the converged
  production fronts. Metadata-only: no data regenerated/copied/re-prepped
  (`scripts/supplemental/make_etest_subset.py`; subset dir points at the
  existing staged chunk dirs). Incumbent baseline symlinked under the subset
  tag for all three designs
  (`scripts/supplemental/stage_etest_subset_baseline.py`; label-based join
  makes the full cube a valid superset baseline). Acceptance gate PASSED
  2026-08-12: job 19845743 (`workflow/supplemental/etest_subset_reeval_check.sh`)
  — subset rows bit-identical to the full mini-fixture run's SOW-0 rows in the
  same job, subset meta records exactly its own SOWs. Step-09 SUBMITTED
  2026-08-12 (Trevor approved) on the eps20260812 re-filtered sets
  (`outputs/{design}/ffmp_obj8/sets/ffmp_obj8_merged_eps20260812.set`;
  335 / 991 / 784 policies, loader-verified): jobs 19845823 (fixedprob, 18 h)
  + 19845825 (hazfill, 16 h) RUNNING healthily 2026-08-13 (unit wall ~1.2 h,
  rank RSS 1.65 GB, ~62%/78% done at t+8 h); 19845822 (historic) died
  OUT_OF_MEMORY at 11 min during rank startup (one-node spike, sacct step .0,
  ~195 GB peak; siblings at identical geometry are fine — same signature as
  the 2026-08-10 search OOM); resubmit 19859148 (8x128) was cancelled while
  still pending and re-shaped as job 19859233: 14 wholenode x 128 = 1,792
  ranks, so 3,350 units run in exactly 2 claim rounds (~93% parallel
  efficiency, ~2.5-3 h wall, --time=05:00) instead of 4 quantized rounds at
  1,024 ranks (~82%, ~5.4 h) — slightly LESS SU than the 8-node shape. If it
  OOMs again, drop to 96 ranks/node. Ensemble jobs: 8 wholenode x 128 ranks.
  All jobs: batch=50, claim scheduling, CHUNK_MERGE=off, wall guard at
  4,900 s/unit; est. ~28.5k SU total (~4.5k/13.4k/10.6k). After completion run
  09b per design with the SAME env identity (env file + subset preset + same
  .set path, no SEED). Prefix-only subsets (rows keyed by global SOW id);
  interim cubes live under `reeval/etest_kn_50yr_n25000_first10ch/` and must
  never be mixed with full-cube numbers in the manuscript.
  GEOMETRY NOTE for step-09 chunk re-evals on E_test: submit on `shared`
  with ~8 cpus per rank and an explicit realization batch (job 19738752:
  1 node x 16 ranks x 8 cpus, `NYCOPT_SEARCH_REALIZATION_BATCH=50`, 4h14m,
  ~1.7 GB/rank RSS). Denser packings (64x2, 32x4) OOM the job cgroup —
  each rank streams a DIFFERENT ~7.3 GB chunk-HDF5 set and the page cache
  is charged to the job (jobs 19733674/19733773; single-rank code peak is
  only ~1.7 GB, so it is concurrency, not a leak).
- [x] **[local→HPC]** RE-ASSESS EPSILON VALUES under ensemble evaluation —
  DONE 2026-08-12, new vector ADOPTED (Trevor accepted `keepf_a`):
  **[0.05, 10.0, 0.05, 10.0, 0.05, 0.3, 5.0, 0.05]** — one ε per objective
  family (reliabilities 0.05, deficit-P99s 10.0 = the measured hazfill
  eps_rec, flood 0.3 and storage 5.0 UNCHANGED). Full record in
  `epsilon_calibration_experiment.md` (three sweep rounds, jobs
  19839023/19840678/19844656/19845321; diagnostic:
  `scripts/supplemental/epsilon_ensemble_refilter.py`, outputs under
  `outputs/supplemental/epsilon_refilter/`). Adoption landed (job 19845739,
  tests 173 passed / 2 skipped): registry edited
  (`objectives_ensemble._ANNUAL_REGISTRY_SPEC`), regret-τ deficit entries
  re-pinned 5.0→10.0 in all env files (reliability/flood/storage τ stay
  floor-bound), and the re-filtered production reference sets written as
  `outputs/{design}/ffmp_obj8/sets/ffmp_obj8_merged_eps20260812.set`
  (historic 335 / fixedprob 991 / hazfill 784 — these are the coarser-eps
  re-filtered refs the step-09 item waits on; expect live-search archives
  ~10-35% larger). ALL 500k-NFE originals kept verbatim. NOTE: NO JAR
  rebuild is required for ε-only changes — only DV/obj COUNTS reach the
  JARs (per `workflow/00_setup_borg_jars.sh` header; ε enters Borg at
  runtime via `config.get_epsilons()` in `src/mmborg.py`); step 00 was
  re-run anyway (idempotent, JARs byte-identical). Campaign fan-out
  draws/seeds will search at the new resolution automatically. Point step
  08/09 at the `_eps20260812.set` refs (or swap names keeping the
  2026-08-05 file under a preserved name) — decide at submit time.
- [ ] **[local→HPC]** Satisficing-criterion OAT stringency + threshold-margin
  CDFs (framing diagnostic 3) — waits on the persisted re-evaluation cube
  (post E_test re-evaluation of the Pareto sets).
- [ ] **[local]** Regret-tolerance pass B after step 08: discrimination band,
  seed/draw empirical nulls, paired SOW bootstrap, and the assay-sensitivity
  control against `historic`. It derives the non-inferiority margin `delta`.
  Pass A ADOPTED 2026-08-08 (max(eps, floor) shape, headline rung k = 1;
  record in `regret_tolerance_diagnostics.md` §8 and
  `supplemental_config.RTOL_ADOPTED_K`; tau vector in the run env files as
  `NYCOPT_REGRET_TAU`). NOTE: run the pass-B k-sweep with `NYCOPT_REGRET_TAU`
  unset so the ladder does not degenerate to the single adopted rung.
- [ ] **[local]** SI estimator-stability + convergence diagnostics:
  block-bootstrap effective-sample-size analysis of the annual-unit
  aggregation (Text S5) and the MOEA runtime convergence content (Text S7).
- [ ] **[local→HPC]** Flood-axis validity diagnostics for Text S3:
  downstream-stress correlation as a required build diagnostic plus the
  selected-ensemble event-seasonality span check.


## 2. Production gates

- [ ] **[HPC]** Launch campaign searches. IN PROGRESS (2026-08-10): go/no-go
  cell (draw 0 / seed 1, 500k NFE `production`, 8 nodes x 128, 1,021 ranks)
  submitted for all three designs — jobs 19770937 (historic, 10 h wall),
  19770938 (hazard_filling_stationary, 40 h wall), 19770939
  (fixed_probabilistic, 40 h wall). All three started 2026-08-10 20:24.
  STATUS: 19770938 (hazfill) FAILED OUT_OF_MEMORY at 56 min (~19 evals/worker;
  rank 741 killed on node a466; typical per-node RSS ~139 GB of ~240 GB
  limit, so the OOM was a spike on one node, not steady-state pressure).
  No .set written, so the relaunch guard did not block the resubmit:
  RESUBMITTED as-is (8x128) 2026-08-11 as job 19782745 per Trevor's call:
  COMPLETED 2026-08-12 16:25 (21 h 37 m, ~22,100 SU, 5,761 solutions, RSS
  flat ~140 GB throughout — the 08-10 OOM was a one-off spike). ALL THREE
  go/no-go cells are now in; total spend ~48k SU incl. the OOM loss.
  Step-07 diagnostics + full figure suites DONE for all three (2026-08-12);
  hazfill ref set 5,544. Like fixedprob, ZERO hazfill solutions dominate
  the scenario-matched incumbent on all 8 axes (historic: 188/2,604). historic COMPLETED 2026-08-11 00:29
  (4 h 04 m, ~4,200 SU, 2,685 solutions in
  outputs/historic/ffmp_obj8/sets/seed_01_ffmp_obj8.set; step-07
  diagnostics + explore_results figures done, ref set 2,604 solutions).
  fixedprob COMPLETED 2026-08-11 17:29 (21 h 04 m, ~21,600 SU, 6,353
  solutions in outputs/fixed_probabilistic/ffmp_obj8/sets/); per-node RSS
  flat at ~140 GB throughout — no memory creep at 128/node.
  Verify surviving runs before fanning out to the remaining draws x seeds. Production inputs (pools d0–d2,
  search ensembles, E_test + presim) are staged, verified, and adequacy-gated
  (campaign 6-axis min tail share 0.311 / 0.306 / 0.303 across draws). The
  baseline-on-E_test matrix is regenerated on the unified substrate
  (2026-08-08) and the threshold + regret-tolerance parameters are adopted —
  no metric-side blockers remain. The Anvil shakeout is closed (see Done),
  so nothing gates the scale-up. Remaining fan-out: the 2026-08-08 runs cover
  draw 0 / seed 1 only, so the campaign still needs the draw (d0–d2) x seed
  replication for both designs, at the scaled NFE.
  SIZING for the scale-up (measured, not projected): 50k NFE = 3 h 17 m and
  ~1,680 SU per seed at 511 ranks, i.e. ~0.034 SU per NFE per seed. So
  2 designs x 3 draws x 2 seeds is ~20k SU at 50k NFE and ~81k SU at 200k
  NFE, against 679k SU remaining (2026-08-10). Budget is not the binding
  constraint; the SLURM `--time` wall is. Runs are NFE-bounded
  (`max_time_hours=None`), so `--time` must scale with NFE or it silently
  truncates the search: ~13 h at 200k NFE. Convergence evidence that the
  scale-up is warranted, from the 25k NFE/island endpoint: hypervolume still
  rising monotonically (island 0: 0.0642 -> 0.0673 -> 0.0705), archive still
  growing (1003 -> 1029 over the last 1k NFE), Improvements still accruing
  (3382 -> 3498), `Restarts=0`.

## 3. Post-campaign deliverables

- [ ] **[local]** Results figure plan + the scripts to build them. First
  tranche landed: `src/solution_selection.py` (dominance / scaling /
  compromise / diverse selection), `src/plotting/front_overview.py`,
  `src/plotting/historic_timeseries.py`, driven by
  `scripts/main/explore_results.py` (+ `workflow/supplemental/sim_selected_policies.sh`
  for the simulation-dependent panels). Still to come: re-eval / robustness
  result figures and the manuscript-final styling pass.
- [ ] **[local]** Manuscript Results / Discussion / Conclusions; SI sections
  beyond S8 are outline-only.
- [ ] **[local]** Import the manuscript's † references into Zotero before
  submission, plus the five persistence-literature DOIs listed in
  `docs/notes/literature/persistence_and_low_frequency_variability.md`, plus the
  regret-methodology references absent from the library: Savage 1951
  (10.1080/01621459.1951.10500768), Bertsimas & Sim 2004
  (10.1287/opre.1030.0065), Kwakkel/Eker/Pruyt 2016 (the undesirable-deviations
  chapter — verify the DOI), McPhail et al. 2021
  (10.1016/j.envsoft.2021.105059), Starr (resolve the 1962-vs-1963 year
  discrepancy between Herman 2015 and McPhail 2018), Schneller & Sphicas 1983,
  Popper et al. 2009. Also fetch McPhail et al. (2018) Supporting Information
  S1, which holds the metric equations the article body omits.

## Parked (scope decisions, not blockers)

- [ ] Remaining manuscript scoping sentence: demand is Decree-capped and held
  stationary (no demand-growth DU axis — consider other justifiable
  socio-economic parameterizations).
- [ ] Per-realization diversion staging (`src/ensemble_prep.py` supports only
  `constant_max`) and the deferred DU-factor presets (`src/ensembles.py`) —
  both forward hooks, not campaign blockers.


## Done (can be deleted when info is no longer needed)
- [x] **[HPC]** Anvil shakeout / `pilot` go/no-go — CLOSED 2026-08-08, answered
  by the first full `mm_moderate` runs rather than a separate pilot. Jobs
  19733754 (`hazard_filling_stationary`) and 19733755 (`fixed_probabilistic`),
  both COMPLETED on `wholenode`: 4 nodes / 511 ranks (2 islands x 254 workers),
  50k total NFE, 3 h 17 m, ~1,680 SU per seed — HALF the ~3,340 SU upper bound
  in `src/moea_config.py`, which is what the feasibility-rate check existed to
  establish (the cheap DV-space `flood_zone_ordering` rejections are real and
  roughly halve the cost; the rejected-NFE fraction itself is not logged
  directly, only its cost effect). Archives: 1466 solutions (hazfill) and 1657
  (fixedprob) at draw 0 / seed 1. Use the measured 0.034 SU/NFE/seed for all
  further sizing.
- [x] **[HPC]** Baseline-on-E_test re-eval under the unified substrate and the
  first 50k-NFE searches under both `hazard_filling_stationary` and
  `fixed_probabilistic` — DONE 2026-08-08/09 (commit 16599bc; sets, runtime
  archives, `.metrics`, and the `explore_results` / search-diagnostic figure
  tranche are committed under `outputs/`).
- [x] **[HPC]** End-to-end smoke of `hazard_filling_stationary` — DONE
  2026-08-07 (job 19731839, `shared`, 41 ranks, 3m10s, ~2 SU;
  `workflow/envs/smoke_hazfill.env`). Exercised MPI fan-out, MM-Borg, the
  staged N=100/L=10 ensemble over the correct 1945-10-01 -> 1955-09-30 window,
  the 2026-08-06 allocation-reduction DV decode, both constraints, and
  `.set`/runtime writing. The single archived solution is feasible (NYC
  reliability 0.5078 >= 0.5 floor). Full pytest suite also green the same day
  (366 passed / 2 skipped, job 19727560) — the DV decode's first validation.
  Step 04's MPI fan-out was already verified (33 ranks on Anvil, 2026-08-04,
  both campaign designs — `logs/prep_pywrdrb_inputs_19663199_1.out`).
  OBSERVATION for search sizing: runtime snapshots show ~150 of the first 200
  NFE/island returning in <0.1 s, i.e. rejected by the DV-space
  `flood_zone_ordering` constraint BEFORE simulation, and the archive never
  grew past 1. Expected at a 400-NFE plumbing budget, but it means (a) random
  initial DVs are mostly flood-zone-infeasible, so early NFE are nearly free,
  and (b) the 173.8 s x NFE cost projections are an UPPER bound, not an
  estimate. Worth a feasibility-rate check at pilot scale before sizing the
  campaign wall clock.
