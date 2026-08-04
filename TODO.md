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
  suite passes (261). **Anvil half DONE 2026-08-03 evening**: historic presim
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
- [ ] **[local→HPC]** Satisficing-criterion OAT stringency + threshold-margin
  CDFs (framing diagnostic 3) — waits on the persisted re-evaluation cube
  (post E_test).
- [ ] **[HPC]** Satisficing criterion values + sweep grid (`_DEFAULT_THRESHOLDS`).
- [ ] **[local]** E_test hazard-space overlay (simulation-free; **candidate main-text figure**, run as soon as E_test is staged): compute E_test's hazard image and overlay it on the candidate pool AND each realized search ensemble (E_d per design/draw) in the retained axes / p1–p99 box. Answers whether E_test's severe droughts (forced mean reduction) occupy the same hazard coordinates as the pool's natural-variability corners the selector enriches; also feeds the registered hazard-restricted E_test composition-sensitivity checks and the generalization claim. IMPLEMENTED 2026-07-30 (smoke-tested on synthetic data; blocked only on E_test staging): `scripts/main/compute_etest_hazard_image.py` (disjoint 10-yr sub-window image, shard-resumable → `hazard_image_subwindows.npz`) + `scripts/main/plot_etest_hazard_overlay.py` / `src/plotting/etest_hazard_overlay.py` (corner overlay + per-axis containment `overlap_stats.json`).

## 4. Pipeline shakeout on Anvil (before production submissions)

- [ ] **[HPC]** End-to-end smoke of `hazard_filling_stationary`: step 06 only (`submit_smoke.sh` = 79 MPI ranks). Local half DONE 2026-07-31 — steps 02→03→04 at P=1000/N=50 exercised wet-exclusion, the robust p1/p99 bounds and the dedupe-only screen; selection ran on the m6 campaign axes (6/6 retained), `axis_screen` + per-axis bounds/clipped fractions + coverage-vs-null persisted in the staged meta, all four pywrdrb inputs staged, and one trimmed-model evaluation on the result returned 8 finite objectives. Untested locally: the MPI fan-out in step 04 (no `mpirun` on the laptop; ran 1 rank).
- [ ] **[HPC]** `pilot` MOEA config go/no-go run.

## 5. Production gates

- [ ] **[HPC]** Generate production pools (K draws) + `fixed_probabilistic` draws + E_test (step 12 at the locked sizing, then step 04 incl. the one-time full-model presim pass over its 25,000 realizations). P=1e6 draw-0 image already staged (`statpool_10yr_n1000000_d0`; a prefix is an honest pool of any smaller P); draws 1..K−1 still to generate (sharded path: `workflow/supplemental/gen_pool_shards.sh` + `gen_pool_merge.sh`, ~600 core-hours each). **Per-draw gate**: re-confirm the per-axis tail-share adequacy gate (min ≥ ~0.30 on the campaign 6-axis set, N=100) on EACH production draw's hazard image — the P=1e6 draw-0 margin is thin (0.311; per-seed min 0.28–0.35), per the §2 (m, N, P) decision and the battery rerun (both in DONE).
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
  IN FLIGHT: shard array 19654577 → merge 19654610 → step-04 staging+presim
  19654612. Serial chunk000 snapshotted to `_serialref_etest_chunk000` on
  project space for a production-scale cross-partition check (compare at the
  1%-range tolerance per the cross-job FP convention, then delete). Code
  changes are UNCOMMITTED for Trevor's review. E_test lives on
  project space (`/anvil/projects/x-ees260021/NYCOptimization/`, 5 TB, no
  purge) via symlinks under `outputs/synthetic_ensembles/` — the 25 GB home
  quota killed the first generation attempt; keep big artifacts OUT of home.
- [x] **CLOSED 2026-08-03 (decision)** Trimmed-vs-full agreement check: the historic-trace validation is SUFFICIENT — measured from the 2026-07-29 determinism data, all 28 trimmed-vs-full policy × objective pairs (4 policies × 7 objectives) agree to ≤ 2.5e-13 relative, and the structural argument (boundary releases are policy-independent) is input-independent. The planned E_test-slice half is DROPPED, not deferred; SI S1 reports the historic check only (`scenario_design_methods.md` §5.4 and the S1 outline updated to match).
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
