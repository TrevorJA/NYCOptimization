# Pre-campaign TODO

Concise action items between now and the HPC optimization campaign, in rough
priority order. Each item is a pointer, not a spec — investigate and plan the
details when picked up. Move finished items to **DONE** at the bottom.

Venue tags: **[local]** laptop-only, **[HPC]** needs the cluster,
**[local→HPC]** decide/dry-run locally, execute at scale on the cluster.

## 1. Artifact regeneration

The objective set is FINAL (8 objectives, NJ activated 2026-07-30; flood
exceedance `downstream_flood_exceedance_minor` ADOPTED 2026-08-03 in place of the
day count) and the delivery-factor bounds moved to the symmetric FFMP ± 0.15
rule (2026-07-31). Sequence: pywrdrb sync + input re-staging (first two items)
→ the epsilon-confirmation rerun → step-00 JARs — the objective set, epsilons,
and DV bounds are all baked into the problem definition, and every simulation
must run on the rebased pywrdrb + re-staged inputs first.

- [x] **DONE 2026-08-03 (both machines)** — Pywr-DRB `nyc_opt` was REBUILT on
  v2.2.0 master (curated rebase; history rewritten and force-pushed; old
  branch head `0bca357` survives only in reflog). Laptop and Anvil both
  hard-synced at `v2.2.0-10-ge293825`. Numerically
  breaking deltas for this project, all landing at once: (a) **new STARFIT
  default params** in `istarf_conus.csv` (2004–2023 refits for blueMarsh,
  fewalter, prompton; beltzvilleCombined capacity 13,500 → 17,750 MG);
  (b) **corrected STARFIT physics** — unconditional `max(target, R_min)`
  clamp (was below-NOR only) and the offline presimulator now matches the
  model's consumption timing (`CU·withdrawal_{t-1}`, inflow-limited) and
  bounds releases by net available water; (c) packaged
  `predicted_inflows_mgd.csv` for `pub_nhmv10_BC_withObsScaled` regenerated
  under (a)+(b); (d) observations refreshed through 2026-08-02 (DRBC NYC
  storage splice, pre-1990 prompton storage dropped, the three flood gauges
  now in obs `gage_flow_mgd.csv` → the sim/obs exceedance ratios quoted in
  the flood items shift when recomputed). Non-events, verified: pywrdrb's
  `flow_prediction_mode` default flipped to `regression_disagg`, but
  `config.py` pins `perfect_foresight` explicitly (the guard comment did its
  job); `use_individual_storage` and all flood/N-zone machinery are
  unchanged from the pre-rebase branch.
- [x] **DONE 2026-08-03 (both machines)** Re-stage ALL staged pywrdrb inputs
  with `force=True` and discard any objective values simulated before the sync:
  `presimulated_releases_mgd.hdf5` (STARFIT-dependent → stale under (a)+(b))
  and `predicted_inflows_mgd.hdf5` (its perfect-foresight columns read the
  presim artifact → stale) for EVERY staged ensemble, local and Anvil. The
  historic-trace trimmed path is stale too: its presim CSV is generated
  locally (`workflow/01_generate_presim.sh` / `generate_presim.py`) — rerun
  on both machines; its predicted-inflows CSV is packaged and arrives with
  the sync. `ensemble_prep` skips silently when outputs exist — force is
  mandatory,
  same trap as the 2026-07-31 flood re-staging. NOT stale (no pywrdrb
  simulation involved): the flood-node inflow HDF5s (pure streamflow
  redistribution), the P=1e6 pools + hazard images, hazard selection, and
  E_test generation — the ~1.9k SU pool investment is untouched.
  Step-05 baseline: stale a THIRD time (physics/params now, on top of the
  flood-exceedance swap) — regenerates with the §1 chain as already
  sequenced. The ε-calibration populations are stale too, priced into the
  rerun below. The DONE determinism/agreement records (2026-07-29/31) are
  convention-level verdicts — no rerun needed, but their measured numbers
  are pre-rebase.
  **Local half DONE 2026-08-03** (post-sync at `e293825`): historic presim
  CSV rerun (step 01); `kn_50yr_n5` + `hazfill_stat_abs_10yr_n50_d0`
  presim/predicted HDF5s deleted and re-staged (flood-node inflow HDF5s
  untouched, as scoped); the pre-sync step-05 baseline vector and the
  orphaned `outputs/presim/full_model_baseline.*` +
  `outputs/diagnostics/random_sample_objectives.npz` deleted; full test
  suite passes (261). (Anvil's surviving copies of the
  `full_model_baseline.*` orphans deleted 2026-08-04 night.) **Anvil half DONE 2026-08-03 evening**: historic presim
  regenerated; both staged pywrdrb-input ensembles (`fixprob_10yr_n100_d0`,
  `hazfill_stat_abs_10yr_n100_d0`) force-re-staged all four inputs (jobs
  19644341/42) — this also closes the Anvil half of the flood-inflow
  re-staging item below; step-05 baseline regenerated (0.789 Montague vector
  below). `prep_pywrdrb_inputs.py` now ALWAYS force-regenerates (committed) —
  the silent-skip trap is retired. E_test staging is first-time generation,
  in flight (§5 pools item).

- [x] **JAR half DONE 2026-08-03** — problem JARs rebuilt on the Anvil login
  node under the adopted ε vector and 36-DV/8-obj scheme (ffmp 36 DV, ffmp_8
  45, ffmp_10 55, ffmp_12 64). Post-reset ε re-verification (19644709-11)
  COMPLETED 2026-08-04 01:38: vector UNCHANGED on every axis (all
  previous/campaign ratios 1.0; NYC-deficit measured 10.0 vs adopted 2.0 is
  the standing deliberate override) → **no rebuild needed; JARs are final**.
  Combined tables/figures + framing analysis regenerated on the all-fresh
  cubes (job 19650686) after a mid-batch stale-cube race (19644712 read a
  pre-reset hazfill cube; final writers were fresh; refreshed anyway). `.set` artifacts
  (Borg search outputs) regenerate with the next smoke/campaign runs (§4).
  The stale local `.set` trees (`outputs/optimization/`,
  `outputs/historic/ffmp_obj7_smoke/`) predated the current DV scheme, locked
  epsilons, and NJ activation — DELETED 2026-07-31.
- [x] **CLOSED 2026-08-03** Regenerate any `du_forced` ensemble staged before 2026-07-28 with the variance axis off: they carry the fixed `c = 1` convention (absolute-SD) instead of the CV-preserving `c = a` (bug fixed in `src/ensemble_generation.py`). Verified 2026-07-30: no `du_forced` ensemble is staged locally. Verified 2026-08-03: none staged on Anvil either (all stages are stationary pools/ensembles + the kn E_test) — nothing to regenerate.
  Historic single trace: DONE 2026-07-31 — the step-05 anchor is regenerated on the
  corrected `pub_nhmv10_BC_withObsScaled` flood-node inflows (restaged 15:09).
  Measured effect on the baseline policy, corrected vs pre-fix: `downstream_flood_days_annual`
  0.1447 → 0.1842 (+27 %); the other seven objectives are unchanged except
  `montague_flow_deficit_p99_pct` at 3e-6 relative, consistent with the
  redistribution being mass-conserving. Any baseline vector persisted before
  2026-07-31 15:09 is on the pre-fix inflows.
- [x] **DONE 2026-08-03 (both machines)** Re-stage any flood-augmented ensemble inflows (`catchment_inflow_with_flood_nodes_mgd.hdf5`) generated before 2026-07-31, and discard flood objective values computed from them. Local half DONE 2026-08-03: both locally staged ensembles are now post-fix by content check (HE/FE ratio 0.3374) — `kn_50yr_n5` re-staged by the flood-objective run's audit, `hazfill_stat_abs_10yr_n50_d0` re-staged directly (its 07-31 12:19 staging predated the 15:54 fix commit; no flood objective values from it were ever persisted). Anvil half DONE 2026-08-03 evening via the force re-staging above (jobs 19644341/42 regenerate all four inputs, flood inflows included). Pywr-DRB's flood-node inflow preprocessor carried a double subtraction of already-marginal upstream inflows plus three wrong USGS drainage areas, which left the three tail-gauge local catchments at ~2 % of physical magnitude (fixed in `../Pywr-DRB/src/pywrdrb/pre/flood_node_inflows.py`; evidence in `../Pywr-DRB/experiments/nyc_flood_gauge_diagnostics/`). `ensemble_prep` SKIPS staging when the file already exists unless `force=True`, so stale files are reused **silently** — re-stage with `force`. The fix is a strict mass-conserving redistribution, so net basin flow and the Montague/Trenton totals are unchanged and only the `downstream_flood_days_*` objectives move — but they move a lot: on the historic trace the aggregate sim/obs exceedance ratio went 0.44 → 0.56 at minor stage and 0.46 → 0.96 at action stage.
- [ ] **[local→HPC]** Regenerate everything downstream of the 6-month metric window and the flood days/yr normalization; discard any objective values computed under the old 365-day warm-up or the whole-trace flood count. 2026-07-31: the step-05 baseline is regenerated locally under the current method (obj8, skip-reeval) and now scores on the annual-unit set via `config.get_objective_set()` — `run_baseline.py` had been building the §1 whole-trace set, so every pre-2026-07-31 baseline vector is on a different objective function and must be discarded, not compared. The baseline re-eval matrix half stays open pending E_test (run step 05 with `--reeval` once E_test is staged). 2026-08-03: the step-05 baseline is REGENERATED locally on the rebased branch under the tolerance-fixed objective function (`_FLOW_TARGET_TOL_MGD`, commit 0468b31 — the rebase's ulp-level changes had flipped 190 exactly-on-target Montague weeks to failures and halved the reliability). Final vector: `downstream_flood_exceedance_annual` = 0.3467 ft·days/yr, `montague_flow_reliability_annual` = 0.789 (vs 0.763 pre-rebase, which itself lost a few zero-deficit weeks to the strict comparison); the remaining shifts vs the discarded pre-sync vector are the expected small STARFIT/forecast deltas.

## 2. Sizing decisions (evidence-based, before generation)

- [x] **CONFIRMED 2026-08-03** K=3 / S=2 against the allocation: the total is
  750k SU (Anvil registers 300k tranches, so `mybalance` under-reports);
  campaign ≈ 503k + reserve ≈ 247k per the §6 ledger fits. Decision stands.

## 3. Remaining method closures

Evidence base: the framing-convention diagnostics
(`docs/notes/methods/framing_convention_diagnostics.md`) — cube reductions of
the epsilon-calibration policy populations
(`scripts/supplemental/framing_convention_analysis.py`, run locally
2026-07-30: tables + SI figures under `outputs/supplemental/framing_convention/`)
plus the satisfaction-factor sweep (`satisfaction_factor.sh`, one Anvil job per
ensemble design, ~26 SU each).

- [x] **DONE 2026-08-04** Satisfaction-factor sweep re-verified post-reset
  (jobs 19644712/13; fresh cubes 01:36): **0.99 factor ADOPTED** — on both
  ensemble designs τ-vs-shipped ≥ 0.92 at 0.98/0.99 while the strict 1.00
  collapses rankings (τ 0.59–0.65) and 0.95 drifts (τ 0.70). Framing
  verdicts re-confirmed on the same fresh cubes (19650686 refresh): k counts
  unsaturated (τ = 1.0, zero boundary mass, all designs), flood operator
  MEAN (boot τ-vs-full 0.95–0.96 vs 0.53–0.56 for P99), NJ screen clean
  (max |ρ| 0.36).
- [x] **ADOPTED 2026-08-05 — Epsilon revision (C1_adopted)**: registry
  updated (`_ANNUAL_REGISTRY_SPEC`: NYC deficit-P99 2.0→5.0, Montague
  deficit-P99 2.0→5.0 PAIRED per site symmetry, flood exceedance 0.2→0.3;
  storage 5.0 + all reliability ε unchanged), JARs rebuilt (step 00, login,
  2026-08-05 12:19), affected suites 90/90 (job 19688123, mpirun-wrapped
  pytest on shared). Step 07 now ε-BOX-FILTERS the cross-seed
  `{slug}_merged.set` under the campaign vector before metrics
  (`src/diagnostics.py::epsilon_box_filter_set`; plain union kept as
  `*_raw.set`; verified end-to-end job 19688141) — and that file is the
  first-choice re-eval reference set (`src/reevaluate{,_mpi}.py`), so
  merge → ε-filter → re-evaluate holds for every optimization
  configuration. Old-ε shakeout search outputs (sets/runtime/metrics +
  search figures) DELETED as stale; the E_test baseline reeval subdir and
  the epsilon_refilter decision-record outputs are KEPT. **REMAINING: one
  confirmatory 2-seed historic search under the new vector (~2.4k SU)** —
  measured expectation ~1,100-1,200/seed archives.
  Measurement record (jobs 19684079-19684597): the shakeout archives
  (~2k members/seed) re-filtered under candidate vectors
  (`scripts/supplemental/epsilon_refilter_sweep.py`; tables + acceptance
  figure under `outputs/supplemental/epsilon_refilter/`; dated section in
  `epsilon_calibration_experiment.md`). Recommended **C1_adopted** (Trevor
  2026-08-05, two revisions of C1_moderate: storage-P01 ε stays 5.0 — a
  5%-of-capacity distinction is significant — and the NYC/Montague
  deficit-P99 epsilons are PAIRED, matching the sites' paired 0.02
  reliability epsilons): NYC deficit-P99 2.0→5.0, Montague deficit-P99
  2.0→5.0, flood exceedance 0.2→0.3; storage + reliability axes unchanged
  (Trenton's visible gaps are the historic 1/76 unit lattice, not ε — 0.01
  changes nothing and resolves below the ensemble noise floors). Effect:
  seed archives −43%/−48% (1,159/1,036), cross-seed ε-front 1,599;
  fallbacks C1_moderate (Mont-def 4.0 + storage 7.5: −53%) and C2_measured
  (10/5/0.5/10, each at its measured floor: −74%). Box filter validated:
  reproduces Borg's archive membership exactly under the current vector.
  Pipeline finding: step-07 `{slug}_merged.set` is a PLAIN nondominated
  union (ResultFileMerger ignores --epsilon for archiving) — ε-box-filter
  it before step-08/09 sizing. On adoption: `_ANNUAL_REGISTRY_SPEC` →
  step-00 JAR rebuild → tests → one confirmatory 2-seed historic search
  (~2.4k SU).
- [ ] **[local→HPC]** Satisficing-criterion OAT stringency + threshold-margin
  CDFs (framing diagnostic 3) — waits on the persisted re-evaluation cube
  (post E_test). The threshold vector it sweeps around is now the measured
  recommendation below.
- [ ] **[HPC]** Satisficing criterion values + sweep grid (`_DEFAULT_THRESHOLDS`).
  **RECOMMENDATION MEASURED 2026-08-05** (jobs 19675365/19675551, zero
  simulation; `docs/notes/methods/robustness_threshold_diagnostics.md`,
  tables+figures under `outputs/supplemental/robustness_threshold_diagnostics/`;
  code `scripts/supplemental/robustness_threshold_{anchor,figures}.py`,
  `RTD_*` config section, `tests/test_robustness_threshold_diagnostics.py`
  13/13 pass). Verdicts from the baseline-on-E_test cube + a recomputed
  base-metric historic anchor: the three delivery criteria the HISTORIC
  status quo itself fails are re-anchored at historic attainment rounded
  stricter (NYC rel 0.95→**0.87**, NYC CVaR90 10→**29**, NJ rel
  0.95→**0.92**); flood adopts the observed 2000-2023 burden anchor
  1.0→**1.17** ft·d/yr; Montague rel 0.85 kept (discriminating, frac 0.856);
  Montague CVaR 25 / Trenton 0.85 / storage 25 kept as non-binding
  guardrails. Headline preserved and sharpened: baseline still fails NYC
  rel/CVaR in 84%/67% of E_test SOWs under the merely-maintain-status-quo
  criteria (the old 0.0005/0.001 fractions were support-outside artifacts).
  θ-attribution: annual-mean factor m dominates all 8 objectives
  (|ρ|=0.91-0.98), r1 secondary, r2 inert. REMAINING to close: adopt the
  vector into `objectives_ensemble._DEFAULT_THRESHOLDS` + `__satNN` labels
  (checklist §5 of the note), rerun tests, then one wrapper rerun so the SI
  figures' "current" markers show the adopted values.
- [x] **DONE 2026-08-04 (all three draws)** E_test hazard-space overlay (simulation-free; **candidate main-text figure**): d0 overlay + containment clean (job 19663001); d1/d2 refreshed post-staging (19673821/22) with identical verdicts — E_test inside the pool hull on all 6 campaign axes for every draw (details on the §5 pools item). Original spec: compute E_test's hazard image and overlay it on the candidate pool AND each realized search ensemble (E_d per design/draw) in the retained axes / p1–p99 box. Answers whether E_test's severe droughts (forced mean reduction) occupy the same hazard coordinates as the pool's natural-variability corners the selector enriches; also feeds the registered hazard-restricted E_test composition-sensitivity checks and the generalization claim. IMPLEMENTED 2026-07-30 (smoke-tested on synthetic data; blocked only on E_test staging): `scripts/main/compute_etest_hazard_image.py` (disjoint 10-yr sub-window image, shard-resumable → `hazard_image_subwindows.npz`) + `scripts/main/plot_etest_hazard_overlay.py` / `src/plotting/etest_hazard_overlay.py` (corner overlay + per-axis containment `overlap_stats.json`).

## 4. Pipeline shakeout on Anvil (before production submissions)

- [x] **DONE 2026-08-05 — 2-seed historic mm_full shakeout (first search under
  the `nyc_reliability_floor` constraint)**: job 19677667 (array 1-2,
  `ffmp_obj8_historic.env`, 5 nodes x 33 ranks, --time=12:00:00). Both seeds
  COMPLETED clean: seed 1 wall 1:51:15, seed 2 1:50:32 (~29 s/eval, matches
  the calibration; far under the ~4 h estimate) ≈ 1,187 + 1,179 ≈ **2,366 SU
  total** (640 cores x wall). Archives
  (`outputs/historic/ffmp_obj8_mm_full/sets/`): seed 1 = 2,051 members
  (1.20 MB), seed 2 = 1,990 members (1.16 MB), every row exactly 44 columns
  (36 DV + 8 obj, no constraint columns). **Constraint wiring VERIFIED**:
  min NYC weekly reliability across all members = 0.5000 on the natural
  scale in both archives, zero violations — the front presses against the
  floor exactly as an active constraint should. Zero `eval exception`
  warnings, zero fatal signatures in either .err log. Step-07 diagnostics
  DONE (job 19682073, 3:16 on shared): per-seed merged reference sets
  (`ffmp_obj8_mm_full_seed{01,02}_merged.set`) + 4 per-island runtime
  metrics files per seed under `metrics/`; final-row indicators finite and
  cross-seed consistent (island-0 hypervolume 0.0704 / 0.0672). Step 08/09
  re-eval NOT run — Trevor inspects the Pareto-approximate sets first.
  **Standard post-search diagnostics ADDED 2026-08-05 (uncommitted, for
  review)**: step 07 now (a) merges the CROSS-SEED reference set
  (`sets/{slug}_merged.set` — the artifact step 08/09 re-evaluates; 7,544
  members here) before per-seed work, (b) scores every island's runtime
  metrics against it so indicators are comparable across seeds (Reed et al.
  2013 convention; the 19682073 metrics were per-seed-referenced and are
  overwritten), and (c) renders a failure-isolated figure suite
  (`src/plotting/search_diagnostics.py` → `figures/{scenario}/{slug}/`):
  seed-overlay parallel axes + FFMP baseline, hypervolume convergence, and
  the six-indicator runtime panel. Exercised on this run (job 19683608,
  1:03). Read: seeds consistent on every indicator (HV bands overlap,
  spacing settled ~1.0, GD → ~0.002), but **hypervolume is still climbing
  at 12,500 NFE/island on all 8 islands** — 50k NFE/seed has not plateaued
  on the historic problem; revisit NFE sizing (or rely on multi-seed
  merging) before campaign submissions. (2026-08-05 addendum: step 07 also
  ε-box-filters the merged reference set now — §3 epsilon-adoption record —
  and this run's search outputs were deleted as stale after the ε revision
  they motivated; the numbers above remain the shakeout's record.)
- [ ] **[HPC]** End-to-end smoke of `hazard_filling_stationary`: step 06 only (`submit_smoke.sh` = 79 MPI ranks). Local half DONE 2026-07-31 — steps 02→03→04 at P=1000/N=50 exercised wet-exclusion, the robust p1/p99 bounds and the dedupe-only screen; selection ran on the m6 campaign axes (6/6 retained), `axis_screen` + per-axis bounds/clipped fractions + coverage-vs-null persisted in the staged meta, all four pywrdrb inputs staged, and one trimmed-model evaluation on the result returned 8 finite objectives. Untested locally: the MPI fan-out in step 04 (no `mpirun` on the laptop; ran 1 rank).
- [ ] **[HPC]** `pilot` MOEA config go/no-go run.

## 4b. Pre-campaign parallelization (2026-08-04, all working trees UNCOMMITTED for review)

Wallclock/SU reduction pass, output-identical by construction and gated on
bit-compare acceptance tests (plan:
`~/.claude/plans/prompt-task-reduce-total-silly-spark.md`). Measured profile
first: step-04's ~52-59 min/chunk was 98.9% the perfect-foresight
`PredictedInflowEnsemblePreprocessor` (flood 19-24 s, presim 10-14 s — the §5
"presim ~3 min/realization" note below was a MISATTRIBUTION; the presim
vectorization bought ~no wall time); pool shards are 12.1-12.4 h median (not
~9 h) with the fit ≈ 30 s (the log-inferred "75-min fit" is unattributed
generation-loop overhead — instrumentation added, next production shard log
answers it); the E_test sub-window hazard job paces 0.386 s/realization.

- [x] **DONE — pywrdrb predicted-inflow vectorization** (`../Pywr-DRB`
  `pre/predict_inflows.py`): perfect-foresight kernel vectorized per
  realization (get_indexer shift + hoisted wd/cu + np.minimum elementwise
  twin, 99fd7d6 discipline; scalar path retained as reference via
  `_vectorize_perfect_foresight=False`); narrowed HDF5 reads to the 18
  travel-time nodes; datetime strings encoded once in save(). Gates:
  `../Pywr-DRB/tests/test_predicted_inflow_vectorized.py` 3/3 (float32
  fixtures, both-direction .iloc[-1] fills, interior windows);
  `tests/test_predicted_inflow_vectorization.py` @slow PASSED (kn_50yr_n5,
  every dataset bit-equal); **production bitcheck PASSED job 19662429 —
  chunk000 regenerated on 33 ranks and EXACT bit-identical to the staged
  scalar artifact, whole job 31 s** (stage was ~51 min → step-04 chunk now
  ~1-2 min, I/O-bound; retained gate:
  `workflow/supplemental/predicted_inflow_bitcheck.sh`). Future staging
  passes drop ~1,500 core-h each; the already-staged E_test is untouched.
- [x] **DONE — hazard-kernel efficiency** (scengen sibling tree +
  NYCOptimization): reference-SSI fit + POT threshold now cached
  content-keyed in `scengen/hazard_metrics.py::get_reference_fits` (was
  refit per call: 1,250×/E_test image run; same-fitted-object reuse ⇒ exact;
  scengen tests 10/10 incl. 2 new cache tests);
  `compute_etest_hazard_image.py` array-parallelized via
  `NYCOPT_ETEST_HAZARD_SHARD_INDEX`/`_MERGE` + new
  `workflow/supplemental/etest_hazard_image_{shards,merge}.sh` (array 0-49;
  wall ~2h42m → ~5-10 min; serial loop unchanged as reference; merge proven
  row-order-invariant in `tests/test_etest_hazard_shards.py`). Row-level
  gate vs the production `hazard_image_subwindows.npz` (landed 2026-08-04
  ~15:00, job 19660004): `scripts/supplemental/verify_etest_hazard_rows.py`
  — run in flight. Phase instrumentation added to
  `src/ensemble_generation.py` (fit/generate/disagg/hazard/write totals) and
  `_score_chunk` (read vs score). SSI-transform fast path DEFERRED and
  generator-state persistence DROPPED (Trevor 2026-08-04); zero-code note:
  stream-only pool shards have no alignment constraint —
  `NYCOPT_ENSEMBLE_SHARD_COUNT=100` halves a future draw's 12.2 h wall.
- [x] **DONE — step 08/09 re-eval hardening** (`src/chunk_reeval.py`
  rewrite): incremental per-unit atomic persistence + resume
  (`partial/units/chunk{j}/sol{sid}.parquet`, `.failed` sidecars keep the
  NaN semantics; resubmitting the same sbatch resumes), claim-file dynamic
  scheduling over a chunk-major list (`NYCOPT_CHUNK_SCHEDULE`, legacy
  contiguous retained), wall guard (`NYCOPT_CHUNK_STOP_EPOCH` exported by
  the sbatch script), merge split out (`NYCOPT_CHUNK_MERGE=off` removes the
  await_all_done barrier whose 1800 s deadline would have discarded every
  campaign merge; new `scripts/main/merge_test_chunks.py` +
  `workflow/09b_merge_test_chunks.sh`; direct-placement merge, no 240M-row
  pivots), `[unit]`/`[phase]` telemetry, model-dict cache FIFO-bounded
  (`NYCOPT_MODEL_DICT_CACHE_MAX=8`). Gates: `tests/test_chunk_reeval.py`
  5/5 (incremental==legacy, schedule invariance, resume==oneshot with
  skip-idempotency call counts, standalone==in-job merge + partial-NaN) —
  NOTE: mpi-touching pytest suites must run as `mpirun -np 1 python -m
  pytest ...` on compute nodes (bare `srun python -m pytest` dies silently
  in MPI_Init); real-sim gate `workflow/supplemental/
  mini_etest_reeval_bitcompare.sh` (mini etest_kn_10yr_n4, legacy serial vs
  4-rank claim path, bit-identical reeval_raw) submitted 19663187.
- [x] **CANCELLED 2026-08-04 (Trevor)** — dedicated re-eval unit profiling
  P1-P4 (jobs 19662118-21) scancelled before running: Anvil batch-scaling
  experiments already exist (the §6 ledger's 128-ranks/node 0.729
  strong-scaling figure), and the rewrite makes precise pre-measurement
  unnecessary — resume means a mis-sized job loses at most one unit/rank,
  and the always-on `[unit] ... rss_gb` telemetry makes the FIRST campaign
  job its own profile. Campaign runbook: start step 09 at a conservative
  geometry (e.g. 64 ranks/node shared, batch=100), read the telemetry after
  ~1 h, then scale the resume-chained resubmissions (sketch: 8 wholenode ×
  128 ranks × 24 h ≈ 69k SU vs the ledger's ~80k; NEVER 64 ranks/node on
  exclusive wholenode). Submission line documented in
  `workflow/09_simulate_test_chunks.sh` header.

## 5. Production gates

- [x] **DONE 2026-08-04 night — every production input is staged, verified,
  and gated.** Pools: d0/d1/d2 all merged, boundary+prefix verified, and
  adequacy-gated (campaign 6-axis min tail share 0.311 / 0.306 / 0.303).
  Search ensembles: `fixed_probabilistic` d0-d2 and
  `hazard_filling_stationary` d0-d2 staged with all four pywrdrb inputs
  (d0 restoration bit-verified after the step-03 overwrite incident, below).
  E_test: generated (sharded, bit-identical to serial), all 50 chunks
  staged, hazard image + per-draw overlays done, baseline-on-E_test matrix
  in the reeval tag's `baseline/` subdir. Details in the dated records
  below. Original item:
  Generate production pools (K draws) + `fixed_probabilistic` draws + E_test (step 12 at the locked sizing, then step 04 incl. the one-time full-model presim pass over its 25,000 realizations). P=1e6 draw-0 image already staged (`statpool_10yr_n1000000_d0`; a prefix is an honest pool of any smaller P); draws 1..K−1 still to generate (sharded path: `workflow/supplemental/gen_pool_shards.sh` + `gen_pool_merge.sh`, ~600 core-hours each). **Per-draw gate**: re-confirm the per-axis tail-share adequacy gate (min ≥ ~0.30 on the campaign 6-axis set, N=100) on EACH production draw's hazard image — the P=1e6 draw-0 margin is thin (0.311; per-seed min 0.28–0.35), per the §2 (m, N, P) decision and the battery rerun (both in DONE).
  **IN FLIGHT 2026-08-03 evening**: draw-1/2 shard arrays running (19640254 /
  19640289, 100 tasks) with merges + 2k-prefix determinism verification
  chained (19640288 / 19640292); run the per-draw adequacy gate on each merged
  image when they land.
  **Draw-2 DONE 2026-08-04 morning**: merged + verified (all 8 boundary rows
  within the 1%-range tolerance; 2k prefix bit-identical to the standalone
  pool) and the **adequacy gate PASSES: campaign 6-axis min tail share 0.303**
  (worst axis drought_deficit_volume; mean 0.414; vs draw-0's 0.311 — thin
  margin holds), saturation-mode battery output under
  `outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n1000000_d2/`.
  **Draw-1 repair in flight**: shard 43 (rows 860k-880k) OOM-killed at the 4G
  limit after 8h50 (other 49 shards peaked ~1.2G — anomalous spike); rerun
  submitted at 8G (19653925) with the merge rechained on it (19653926; the
  original 19640288 was cancelled — its afterok dependency was permanently
  failed). Gate draw-1 when the new merge lands.
  E_test generation running (19644593,
  `etest_kn_50yr_n25000`, 50 × 500-realization chunks) with step-04 staging +
  the one-time full-model presim pass chained (19644594).
  **E_test TIMEOUT + resubmit 2026-08-04**: 19644593 hit its 12h wall limit at
  ~7/50 chunks (measured ~1.65 h/chunk → ~84 h serial; the 12h sizing was
  wrong). Chunks can't be resumed without code changes (the streamed hazard
  image accumulates in memory across chunks and sharded generation explicitly
  rejects store_daily ensembles), and generation is deterministic per profile,
  so the clean fix is a full rerun: resubmitted verbatim at
  `--time=96:00:00` (19654203; shared MaxTime is UNLIMITED) with step-04
  staging rechained (19654207, same submit line incl. `--preset
  etest_kn_50yr_n25000`). Completed chunk dirs 0-6 get overwritten with
  bit-identical content. ETA ~3.5 days — E_test-dependent closures (step-05
  --reeval, hazard overlay, §3 OAT diagnostic) slip accordingly. The dead
  19644594 (DependencyNeverSatisfied) needs a manual `scancel` — the
  assistant's scancel was permission-blocked.
  **SUPERSEDED same morning — sharded rewrite (Trevor-directed)**: the 96h
  serial rerun (19654203) + its staging chain were cancelled and E_test
  generation was PARALLELIZED: `generate_forcing_ensemble`'s shard/merge
  machinery extended from stream-only pools to daily-CHUNKED ensembles
  (`src/ensemble_generation.py`: chunk-aligned shard validation, GLOBAL chunk
  numbering, merge-side chunk_index reconstruction + deterministic generator
  refit for forcing_profiles.npz), the same NYCOPT_ENSEMBLE_SHARD_* env
  contract wired into `scripts/main/generate_test_ensemble.py`, and new
  submit scripts `workflow/supplemental/gen_etest_shards.sh` (array 0-49,
  1 chunk/shard, ~2.3h each vs ~84h serial) + `gen_etest_merge.sh`.
  Validated: new `test_sharded_chunked_generation_matches_serial` (every
  artifact bit-identical serial vs shard⊎merge) + misalignment-rejection
  test; full determinism/shard/etest suites 31/31 pass (job 19654507).
  **GENERATION DONE 2026-08-04 ~11:00**: all 50 shards completed in ~2h
  each (19654577), merge 19654610 verified 50 chunks tile [0, 25000), wrote
  the canonical artifacts, and passed assert_staged_etest_contract. The
  production-scale cross-partition check came back BIT-IDENTICAL (serial vs
  shard-0 chunk000, worst diff 0.0 on both HDF5s; snapshot deleted after the
  check). Wall time ~3h vs ~84h serial.
  **Staging gap found + fixed 2026-08-04**: step-04 against the parent slug
  (19654612) FAILED immediately — a LATENT gap, not a shard regression:
  `ensemble_prep` expects a monolithic catchment_inflow_mgd.hdf5 in the slug
  dir, but E_test daily data lives only in the 50 chunk dirs (the original
  chained staging job 19644594 would have failed identically; this path had
  never run end-to-end). The chunked re-eval (step 09, src/chunk_reeval.py)
  simulates PER CHUNK — each chunk is a standalone staged ensemble — so the
  step-04 inputs belong per chunk dir: new
  `workflow/supplemental/prep_etest_chunks.sh` (array 0-49, one
  `--preset {slug}__chunk{JJJ}` prep per task through the ordinary prep
  path). Chunk-0 validation (19659496) PASSED 2026-08-04 12:05 — all four
  inputs staged in 52m41s on 33 ranks. Measured presim pace ~3 min per 50-yr
  realization (pure-Python day loop in pywrdrb's STARFITOfflineSimulator) →
  the full array would cost ~1,300 SU, not the ~70 SU the §6 ledger assumed.
  **Presim VECTORIZED + verified 2026-08-04 afternoon**: pywrdrb commit
  `99fd7d6` (nyc_opt, on top of the pinned e293825) adds
  `simulate_reservoir_ensemble` — the same sequential day loop with each
  iteration vectorized across realizations — and rewires the ensemble
  preprocessor to one vectorized run per reservoir (plus reservoir-only
  input reads). Reviewed line-by-line (elementwise twin, identical op
  order, physics verbatim; scalar path untouched as reference). ACCEPTANCE
  TEST PASSED at production scale: the vectorized path reproduces chunk-0's
  scalar-produced staged presim BIT-IDENTICALLY across all 500 realizations
  x 14 reservoirs, in **10.4 s** vs ~25 core-hours scalar (job 19660330).
  No downstream diagnostics re-run needed (bit-identity), no
  NYCOptimization API changes needed. Chunks 1-49 staging array launched on
  the fast path (19660403). Laptop needs a `git pull` in Pywr-DRB to pick
  up 99fd7d6. NYCOptimization code
  changes UNCOMMITTED for Trevor's review.
  **STAGING COMPLETE 2026-08-04 evening**: all 50 chunks carry the four
  pywrdrb inputs. Main array 19660403 (47 ok, ~52 min/chunk); chunks 3-4
  OOM'd via node co-location and passed on resubmit at 80G (19661166). With
  the vectorized presim the per-chunk cost is flood/predicted-inflow
  dominated (candidates for the same treatment, flagged to the
  parallelization effort).
  **E_test hazard image + overlay DONE 2026-08-04 evening**:
  `hazard_image_subwindows.npz` (125,000 sub-window rows x 8 axes, job
  19660004); overlay + containment vs pool d0 + staged draw-0 search
  ensembles (job 19663001 -> `outputs/supplemental/etest_hazard_overlay/`):
  E_test is FULLY INSIDE the pool hull on every campaign axis (outside-hull
  0.000 x6); beyond-robust-box tails modest (max 0.064 above p-hi
  flood_peak_magnitude; 0.049 below p-lo deficit_volume). Supports the
  generalization claim; rerun the overlay per draw once d1/d2 search
  ensembles are staged.
  **Baseline-on-E_test matrix (the step-05 --reeval half): path corrected**
  — `run_baseline --reeval` cannot consume a chunked reeval preset
  (evaluate_raw needs monolithic daily HDF5s; latent gap, never exercised).
  Running the baseline through step 09's chunked path instead
  (NYCOPT_CHUNK_POLICIES=baseline, same persist_reeval_raw artifact; job
  19663131 on shared, 32 ranks x 4 cpus, 50 units, incremental+resumable).
  When done: move reeval_raw.{parquet,json} into the reeval tag's
  `baseline/` subdir (the location step-08/09 auto-detects for
  improvement_vs_baseline) and give each campaign design's reeval dir a
  copy/symlink of it at campaign time.
  **OOM + resubmit 2026-08-04 evening**: 19663131 started at 15:34 and died
  OUT_OF_MEMORY 3 min in (MaxRSS 214G/237G) — root cause `batch=0` in its log
  header: the pre-hardening copy never set NYCOPT_SEARCH_REALIZATION_BATCH,
  so each of 32 ranks held a full 500-realization chunk model (the hardened
  script's submit-time guard only fires above 64 ranks/node). Its job-scoped
  claim files were pruned (no completed units — resume state clean).
  RESUBMITTED through the hardened `workflow/09_simulate_test_chunks.sh` as
  **19667723** (shared, 1 node, 64 ranks x 2 cpus, batch=100,
  NYCOPT_CHUNK_MERGE=off, NYCOPT_CHUNK_UNIT_SECONDS=9000) with the 09b merge
  chained as **19667724** — this also exercised the campaign split-merge
  path once at small scale.
  **DONE 2026-08-04 19:12**: step 09 completed in 1:07 (50/50 units, ~67
  min/unit, rss ~2.8 GB/rank at batch=100 — campaign-relevant unit
  telemetry: T_unit ≈ 4,010 s/policy-chunk on 2 cpus), 09b merge wrote
  reeval_raw + objectives_summary + the robustness family. The full baseline
  output (reeval_raw.parquet, reeval_raw_meta.json, summaries, partial/) now
  lives in the tag's `baseline/` subdir
  (`outputs/historic/ffmp_obj8_mm_full/reeval/etest_kn_50yr_n25000/baseline/`)
  — the step-08/09 auto-detect location; the tag top level is left empty for
  campaign-time artifacts (leftover top-level partial/ would have collided
  with campaign sol-id resume keys). Matrix verified: 200,000 rows = 1
  solution x 25,000 realizations x 8 objectives, zero NaN cells. At campaign
  time give each campaign design's reeval dir a copy/symlink of baseline/. The 09 script's
  `#SBATCH --exclusive` line was REMOVED (uncommitted) — it blocked every
  shared-partition submission ("node configuration is not available") and is
  redundant on wholenode, which is exclusive by partition policy. The
  hardened path's real-sim gate passed earlier the same day: mini bitcompare
  19663187, legacy serial vs 4-rank claim path, 32 cells bit-identical.
  **Search-ensemble draws 1-2, status 2026-08-04 evening**:
  `fixed_probabilistic` d1/d2 GENERATED + STAGED (19663196/19663199, all
  four inputs; ~5 min/draw with the vectorized preprocessors).
  **hazard_filling d1/d2 NOT staged, and d0 needs restoration**: a
  mis-configured step-03 submission (missing NYCOPT_CANDIDATE_POOL_N=1000000
  → the script defaulted to the P=2000 smoke pools, and it stages ALL draws
  in one loop) OVERWROTE `hazfill_stat_abs_10yr_n100_d0`'s selection files
  (gage/catchment HDF5s, hazard_image.npz, _meta.json) at 15:01 with a
  2k-pool selection; forensic check (job 19666041) proved the original was a
  DIFFERENT selection (the P=1e6 production one — its Aug-3 staged pywrdrb
  inputs are intact and encode the original inflows). The bad d1 dir was
  deleted; the chained staging was cancelled before running; nothing has
  consumed the bad d0 files. **RESTORATION (deterministic)**: after the d1
  pool merge lands, run step 03 ONCE with
  `--export=ALL,NYCOPT_SCENARIO_DESIGN=hazard_filling_stationary,NYCOPT_CANDIDATE_POOL_N=1000000`
  (stages d0/d1/d2 together from the P=1e6 images; same selector seeds →
  d0 restores bit-exactly), VERIFY by regenerating presim from the restored
  d0 inflows and bit-comparing to the Aug-3 staged
  `presimulated_releases_mgd.hdf5` (the 19666041 forensic script pattern),
  then step-04 stage d1/d2 (d0's Aug-3 inputs stay valid once selection
  bit-restores), re-run the per-draw adequacy gates' overlay refresh.
  **Restoration chain RUNNING 2026-08-04 night** (first submission
  19667742-45 was dependency-killed by the merge's false FAIL below and
  resubmitted): **19672618** gate_d1 (saturation-mode battery on
  statpool_10yr_n1000000_d1; the first resubmit 19672561 died instantly —
  `sbatch --wrap` runs /bin/sh and `_common.sh` needs bash, so wrap commands
  must be `bash -c '...'`), **19672562** step-03 all-draws selection at
  NYCOPT_CANDIDATE_POOL_N=1000000, **19672619** verify_d0 (new
  `scripts/supplemental/verify_hazfill_d0_restoration.py`: regenerates d0
  presim in place from the restored inflows and bit-compares every dataset
  vs the Aug-3 snapshot at
  `outputs/supplemental/hazfill_d0_restore_check/presimulated_releases_mgd.aug3.hdf5`;
  on mismatch it restores the snapshot and exits 1, afterok select),
  **19672620** step-04 array 1-2 staging hazfill d1/d2 (afterok verify).
  Overlay refresh follows staging.
  **RESTORATION VERIFIED 2026-08-04 ~19:4x**: step 03 (19672562) staged all
  three draws from the P=1e6 pools on the campaign 6-axis set (d0's
  hazard_image.npz back at 72 MB vs the bad run's 147 KB); verify_d0
  (19672619) **PASS — regenerated presim bit-identical to the Aug-3
  artifact across all datasets**, proving the d0 selection restored
  bit-exactly and the Aug-3 staged inputs remain valid. Incident closed.
  **hazfill d1/d2 STAGED 2026-08-04 19:56** (19672620_1/2, 15 s each on 33
  ranks — vectorized preprocessors): all four pywrdrb inputs present in both
  dirs, sizes matching d0's same-shape artifacts.
  **Per-draw overlays DONE 2026-08-04 ~20:15** (19673821 d1 / 19673822 d2,
  pool + hazfill + fixprob layers per draw): containment matches d0 on both
  — E_test fully inside the pool hull on all 6 campaign axes (outside-hull
  0.000), max 0.064 above p-hi (flood_peak_magnitude), 0.049/0.048 below
  p-lo (drought_deficit_volume). Generalization claim holds uniformly
  across draws; outputs under
  `outputs/supplemental/etest_hazard_overlay/etest_kn_50yr_n25000__statpool_10yr_n1000000_d{1,2}/`.
  **Draw-1 pool MERGED + VERIFIED 2026-08-04 ~22:30**: shard-43 rerun
  completed at 8G in 12:35 (19653925_43); merge 19653926 wrote the canonical
  image and passed the boundary check (5/8 rows exact, 3 within 22-26% of
  the 1%-range tolerance — same cross-era FP profile as d2) but exited
  FAILED on the prefix check because the resubmission dropped
  NYCOPT_NESTEDP_SMOKE_SLUG, so it compared d1 against the d0 smoke pool
  (different seed domain). Rerun with the correct comparator
  (statpool_10yr_n2000_d1, job 19672548): **2k prefix bit-identical — d1
  pool verified**, no re-merge needed.
  **Draw-1 adequacy gate PASSES 2026-08-04 night (job 19672618): campaign
  6-axis min tail share 0.306** (worst axis drought_deficit_volume; mean
  0.414; screen retains 8/8) — all three production draws now clear the
  gate: d0 0.311 / d1 0.306 / d2 0.303, thin margin consistent across
  draws. Battery output under
  `outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n1000000_d1/`. E_test lives on
  project space (`/anvil/projects/x-ees260021/NYCOptimization/`, 5 TB, no
  purge) via symlinks under `outputs/synthetic_ensembles/` — the 25 GB home
  quota killed the first generation attempt; keep big artifacts OUT of home.
- [x] **CLOSED 2026-08-03 (decision)** Trimmed-vs-full agreement check: the historic-trace validation is SUFFICIENT — measured from the 2026-07-29 determinism data, all 28 trimmed-vs-full policy × objective pairs (4 policies × 7 objectives) agree to ≤ 2.5e-13 relative, and the structural argument (boundary releases are policy-independent) is input-independent. The planned E_test-slice half is DROPPED, not deferred; SI S1 reports the historic check only (`scenario_design_methods.md` §5.4 and the S1 outline updated to match).
- [x] **DONE 2026-08-04 (uncommitted, for review)** NYC reliability
  stakeholder floor (0.5) promoted from post-hoc Pareto screening
  (`src/pareto_filter.py`, retained for pre-change archives) to the formal
  post-simulation Borg constraint `nyc_reliability_floor`: violation =
  max(0, floor − weekly reliability) on the natural 0-1 scale, floor via
  `config.NYC_RELIABILITY_FLOOR` (`NYCOPT_NYC_RELIABILITY_FLOOR`, default
  0.5), objective resolved by NAME in the active set. `get_n_constrs()`
  2 → 3 (DV-space pair + post-sim floor; composed in
  `src/mmborg.py::make_borg_objective`); failed evals stay
  feasible-with-penalty (zero violations). JARs UNAFFECTED — .set/runtime
  files are feasible-only with no constraint columns, no rebuild. Tests:
  `tests/test_reliability_floor_constraint.py` + updated
  `tests/test_constraints.py`; docs: `decision_variables.md`.
  **REVIEWED + FIXED 2026-08-04 night**: the original implementation
  resolved the objective by its BASE name (`nyc_delivery_reliability_weekly`)
  against `get_obj_names()`, which reports the ANNUAL-unit registry names in
  every wired-design context — the constraint would have raised on first
  eval. Fixed with `reliability_floor_objective_index()` (accepts the base
  spelling or its `_BASE_TO_ENSEMBLE` annual form; production + tests both
  use it). Suites green post-fix: reliability-floor + constraints +
  sensitivity_common 27/27, objectives_ensemble 35/35 (jobs 19675862,
  19675*).
- [ ] **[HPC]** Launch campaign searches.

## 6. Post-campaign deliverables

- [ ] **[local]** Results figure plan + the scripts to build them (none exist yet).
- [ ] **[local]** Manuscript Results / Discussion / Conclusions; SI sections beyond S8 are outline-only.
- [ ] **[local]** Import the manuscript's † references into Zotero before submission, plus the five persistence-literature DOIs listed in `docs/notes/literature/persistence_and_low_frequency_variability.md` (MCP was read-only when they were annotated).

## Parked (scope decisions, not blockers)

- [ ] Optional HMM E_test variant — deprioritized (it introduces a second DU factor set, and the DRB-fitted HMM is near-memoryless so it is not a persistence stress). Persistence stressing was decided against 2026-07-29 — disclosure only (`persistence_axis_diagnostics.md`); a persistence-stressed test ensemble (Mechanism A, prototyped in `scengen.persistence`) is future-work material.
- [ ] Hazard descriptors beyond the controlling event (event multiplicity / frequency within a window) — independent follow-up investigation, own session.
- [ ] Manuscript scoping sentences (hazard-axis NYC-only + single-event scope LANDED in §4.3/§8 2026-07-28; remaining): demand is Decree-capped and held stationary (no demand-growth DU axis — consider other justifiable socio-economic parameterizations).
- [ ] Per-realization diversion staging (`src/ensemble_prep.py` supports only `constant_max`) and the deferred DU-factor presets (`src/ensembles.py`) — both forward hooks, not campaign blockers.

## DONE

Completed and decided items, chronological. Kept for the measured numbers and
decision records the open items reference.

- **(§4, 2026-07-29)** `check_objective_determinism` on the new metric window: DETERMINISTIC on all four paths (historic/ensemble × trimmed/full; 4 policies × 5 fresh-process repeats). Worst max relative deviation 2.1e-13 (Montague deficit CVaR90); LP state jitter (NYC storage differing up to 0.79 %-capacity points on 80% of days, historic_trimmed) does not propagate through the annual-unit aggregation. `scripts/supplemental/check_objective_determinism.py`; outputs under `outputs/supplemental/objective_determinism/`.
- **(§2, 2026-07-29)** Nested-P saturation diagnostic on Anvil (`outputs/supplemental/hazard_selector_diagnostics/nested_P_saturation.md`): ONE P=1e6 stream-only pool (`statpool_10yr_n1000000_d0`, sharded generation, ~1.9k SU), prefix rungs {2k, 5k, 20k, 1e5, 3e5, 1e6}. **Verdict: the (m = 8, N = 100) gate FAILS at every rung** — min per-axis tail share 0.144 → 0.256 (2k → 1e6), improvement exponent ~0.04 (far below the P^(−1/8) bound; geometry-limited), fitted crossing ~1.8e7 (unaffordable extrapolation). **Fallback (1) is quantified: m4 concept-group set PASSES from P ≈ 1e5** (0.328; 0.353 at 1e6); m6 fails everywhere (≤ 0.249). Screen retains 8/8 at every rung (max |ρ_S| 0.87–0.88); 2k rung reproduces the laptop battery exactly. (m4/m6 here = the legacy diagnostic nestings, retired from the battery 2026-07-30 — it now scores campaign vs full only.)
- **(§2, DECIDED 2026-07-30)** (m, N, P) = (campaign 6-axis selection set, 100, 1e6): campaign selection axes fixed to {deficit_volume, peak_depth, onset_rate, recovery_rate, peak_magnitude, pulse_duration} (drops duration + rise_rate; both stay computed/reported). Passes the gate at P=1e6 (min 0.311; thin margin — **re-confirm per production draw**, the driver now scores a `campaign` axis set; carried on the §5 pools item). Wired: `config.HAZARD_SELECTION_AXES` (env `NYCOPT_HAZARD_SELECTION_AXES`) → step-03 `select_from_candidate_image(selection_axes=...)`; docs synced (`scenario_design_methods.md` §3.3/§6, `hazard_selector_diagnostics.md` §5b). The staged P=1e6 draw-0 image is the production draw-0 pool; draws 1-2 still to generate.
- **(§2, DECIDED 2026-07-30)** E_test sizing: N_θ = 1,000 LHS SOWs × R = 25 × L_test = 50 yr (25,000 realizations, 1.25M scenario-years; ~80k SU re-eval at the ~1,000–1,200 merged campaign policies). Re-evaluation runs the TRIMMED model, like search (policy-independent presim computed once per realization by step 04 and reused across all Pareto sets; one full-model presim pass over E_test ≈ 70 SU). Constants in `src/etest.py`; derivation + ledger in `scenario_design_methods.md` §5.4/§6 (campaign ≈ 503k SU, reserve ≈ 247k). Staging tracked in §5; adequacy verified post hoc by θ-/R-subsample ranking stability scored offline from the persisted matrix. Hazard coordinates of E_test are computed on disjoint 10-yr sub-windows (commensurable with the pool convention).
- **(§2, DECIDED 2026-07-30)** Variable-resolution `ffmp_N` sweep DEPRIORITIZED: runs only on leftover SU at the end of the campaign (the reserve's first call is an additional draw); which scenario design it runs under is decided if/when the leftover permits.
- **(§3, ADOPTED 2026-07-30)** Framing verdicts adopted repo-wide: failure-week counts k CONFIRMED as shipped; flood unit operator = MEAN (P99 stays a registered diagnostic); NJ 8th objective ACTIVATED (clean redundancy screen) → **the active set is FINAL at 8 objectives; slugs are now `*_obj8`** (env files renamed, pinned NYCOPT_OBJECTIVES lists updated). Findings in `framing_convention_diagnostics.md` §0b.
- **(§3, DONE 2026-07-30)** Epsilons for the campaign objective set: epsilon-calibration experiment run (512 feasible policies + baseline per design; historic / fixed_probabilistic / hazard_filling_stationary, draw 0) and the combined recommendation adopted into `_ANNUAL_REGISTRY_SPEC`. The campaign vector is the max over the ENSEMBLE designs only (`EPS_CAMPAIGN_DESIGNS`); the historic reference arm is excluded from the max (its 76-unit noise floor would set reliability ε = 0.10 on the 0-1 scale) and its archive is disclosed to resolve below its own noise floor. 2026-07-31: the delivery-factor bounds change triggers the confirmation rerun in §1 (before JARs — epsilons are baked into the archive).
- **(§5, DONE 2026-07-30)** Full `diagnose_hazard_selectors.py` battery rerun on the production P=1e6 pool image (laptop, ~90 min): screen retains **8/8** axes (dropped none); adequacy gate **at the decided (campaign 6-axis set, N=100) PASSES — min per-axis tail share 0.311** (reproduces the nested-P rung exactly; the full 8-axis set fails at 0.256, the measured basis for restricting selection). Margin is thin: per-seed min ranges 0.28–0.35, so the per-production-draw re-confirmation stands (carried on the §5 pools item). Battery cleaned 2026-07-30 to score exactly two axis sets (campaign vs full; legacy m4/m6 nestings + `assess_m6_axis_sets.py` deleted) and rerun for the final SI artifacts: figures F1–F10 + 7 tables + `summary.json` under `outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n1000000_d0/`.
- **(§4, DONE 2026-07-31)** Baseline anchor vs search path: NO disagreement under the current objective function. Measured on the historic trace, baseline policy, disk path (`run_simulation_to_disk`) vs search path (`run_simulation_inmemory`), both model modes: all 8 objectives agree to ≤2e-13 relative (trimmed) and to 0.00e+00 (full; `run_baseline --test-inmemory` now passes its own 1e-6 gate on every objective). The ~5e-4 figure this item recorded was measured under the RETIRED §1 whole-trace objective set, whose fraction-of-days metrics have 1/28124 granularity and so resolve the known LP degeneracy jitter; the annual-unit set (38 water-year units) does not. Daily series still differ between any two runs (up to ~2.3e3 MGD on major flows, ~62% of days on cannonsville storage) — same magnitude for same-path repeats as for cross-path, i.e. run-to-run LP jitter, not a path effect, and it does not propagate to the objectives. No code change made; `improvement_vs_baseline` was already path-consistent (its anchor comes from `run_baseline --reeval` → `evaluate_solution_raw` → `evaluate_raw`, the search path).
- **(§1, DONE 2026-08-03)** Epsilon confirmation rerun (3 designs, new-bounds
  36-DV feasible population, post-fix flood inflows; tables refreshed in
  commit 0468b31 alongside the framing + satisfaction-factor sweep outputs):
  five axes unchanged; Montague deficit-P99 ε 1.5 → 2.0 (hazfill noise floor
  binds); flood exceedance ε FINAL at 0.2 (noise floor ~0.15–0.18 on both
  ensemble designs, replacing the provisional 0.01); NYC deficit-P99 kept at
  2.0 as a DELIBERATE exception against the measured 10.0 rec (hazfill P99
  bootstrap noise floor — the archive resolves below that design's sampling
  noise on this axis, disclosed, parallel in kind to the historic-arm
  exclusion). The k / flood-MEAN / NJ verdicts re-confirmed. The same
  rebuild exposed the strict weekly Decree-target comparison (ulp-level
  changes flipped 190 exactly-on-target Montague weeks to failures) →
  `_FLOW_TARGET_TOL_MGD = 1e-6` added to `_weekly_flow_ok`
  (`src/objectives.py`); step-05 baseline re-scored under it (Montague
  reliability 0.421 → 0.789).
- **(§3, ADOPTED 2026-08-03)** Flood-exceedance objective swap landed repo-wide: `downstream_flood_exceedance_minor` (§1, ε provisional 0.01 ft-days/yr) / `downstream_flood_exceedance_annual` (§2, PooledMean, worst 5490) are the ACTIVE flood objective; the day counts stay registered diagnostics; `config._DEFAULT_OBJECTIVES`, `_BASE_TO_ENSEMBLE`, `_REGISTRY_SPEC` + `__sat1` placeholder (1.0 ft-days/yr), style labels, the 11 pinned `NYCOPT_OBJECTIVES` env lists, framing/epsilon/determinism supplemental scripts, and tests all updated (261 tests pass; slugs stay `*_obj8` — the set is still 8 objectives). CONSEQUENCE: the step-05 baseline objective vector is stale again (flood entry is a day count) and regenerates with the §1 chain; the ε rerun prices the final exceedance ε before JARs. Evidence + checklist: `flood_objective_diagnostics.md` §0b/§7.
- **(§1/§2/§5, DONE 2026-08-03 evening)** Post-reset Anvil recovery chain:
  Anvil clone hard-synced at `e293825`; historic presim + both staged
  ensembles force-re-staged (19644341/42); step-05 baseline regenerated and
  the sentinel verified under the tolerance fix (Montague reliability
  0.7895, flood exceedance 0.3467 — matches the §1 record);
  `prep_pywrdrb_inputs.py` made always-force (committed); K=3/S=2 confirmed
  against the 750k SU total; no `du_forced` staged anywhere; SI S1's
  historic-trace trimmed-vs-full agreement extracted from the determinism
  data (28/28 pairs ≤ 2.5e-13). Storage: E_test moved to project space via
  symlinks after the home-quota incident; logs/ pruned (keep doing this).
  Left in flight: pool draws 1-2 (+merges/gates) and E_test gen → staging →
  presim. The post-reset ε (19644709-11) + satisfaction-factor (19644712/13)
  re-verification batch LANDED 2026-08-04 01:38, all verdicts unchanged (ε
  vector, 0.99 sat-factor, k/flood-MEAN/NJ) — recorded on the §1 JAR and §3
  sweep items.
- **(Parked → §3, DIAGNOSED 2026-08-03)** Flood-threshold objective reconsidered: the full diagnostic (`docs/notes/methods/flood_objective_diagnostics.md`; scripts `scripts/supplemental/flood_objective_{run,figures}.py`, outputs under `outputs/supplemental/flood_objective/`) **recommends replacing the day count with C4 = Σ over days of the max-across-gauges (stage − minor)⁺ [ft-days/yr]**. Measured: the incumbent count is degenerate on the historic trace (9 distinct values across 25 feasible policies, 14.7 % tied pairs) while exceedance resolves 25/25 with zero ties; the pre-stated monotone-response gate PASSES for exceedance (ρ_S = −1.00 on the ensemble flood-release ladder, no cliffs) while the count moves in integer shelves; rating-curve exposure is nil (0 of 3,556 flood gauge-days beyond the rated range → stage-ft basis safe, flow-basis C6 unnecessary); C4 has the best annual sim-obs correlation (Pearson 0.91) and, unlike the gauge-summed variants, is robust to the model's structural inability to flood two gauges simultaneously. Costs disclosed: ~1.6× ensemble sampling noise, top unit-year carries ~30 % of the integral. `action` stays rejected for the active set (control-rule discontinuity). Adoption checklist in the note §7 — swap joins the §1 epsilon-calibration rerun + JAR regeneration at zero extra cost.
