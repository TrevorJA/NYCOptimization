# Pre-campaign TODO

Concise action items between now and the HPC optimization campaign, in rough
priority order. Each item is a pointer, not a spec — investigate and plan the
details when picked up. Delete items as they land.

Venue tags: **[local]** laptop-only, **[HPC]** needs the cluster,
**[local→HPC]** decide/dry-run locally, execute at scale on the cluster.

## 1. Artifact regeneration

The objective set is FINAL (8 objectives, NJ activated 2026-07-30) and the
delivery-factor bounds moved to the symmetric FFMP ± 0.15 rule (2026-07-31).
Sequence: the epsilon-confirmation rerun (first item) comes BEFORE step-00
JARs — epsilons and DV bounds are both baked into the problem definition.

- [ ] **[HPC]** Re-run the epsilon calibration (3 designs, ~57 SU) + framing
  figures on the NEW-bounds feasible population and adopt any ε delta into
  `_ANNUAL_REGISTRY_SPEC` (expect NYC-axis signal IQRs to shrink and NJ's to
  grow; the k / flood-operator / NJ verdicts are convention-level and are
  re-confirmed by the same figures for free). Batch with the
  satisfaction-factor sweep, which samples its population under the current
  bounds automatically.
- [ ] **[HPC]** Regenerate everything downstream of the 39-DV scheme, the 8-objective set, AND the ±0.15 factor bounds: problem JARs (step 00; Anvil login node — no JDK/MOEAFramework on the laptop; nobjs is now 8) + `.set` artifacts (Borg search outputs — regenerate with the next smoke/campaign runs). The stale local `.set` trees (`outputs/optimization/`, `outputs/historic/ffmp_obj7_smoke/`) predated the 39-DV scheme, locked epsilons, and NJ activation — DELETED 2026-07-31.
- [ ] **[HPC]** Regenerate any `du_forced` ensemble staged before 2026-07-28 with the variance axis off: they carry the fixed `c = 1` convention (absolute-SD) instead of the CV-preserving `c = a` (bug fixed in `src/ensemble_generation.py`). Verified 2026-07-30: no `du_forced` ensemble is staged locally (all local stages are stationary), so this is Anvil-side only.
  Historic single trace: DONE 2026-07-31 — the step-05 anchor is regenerated on the
  corrected `pub_nhmv10_BC_withObsScaled` flood-node inflows (restaged 15:09).
  Measured effect on the baseline policy, corrected vs pre-fix: `downstream_flood_days_annual`
  0.1447 → 0.1842 (+27 %); the other seven objectives are unchanged except
  `montague_flow_deficit_p99_pct` at 3e-6 relative, consistent with the
  redistribution being mass-conserving. Any baseline vector persisted before
  2026-07-31 15:09 is on the pre-fix inflows.
- [ ] **[local→HPC]** Re-stage any flood-augmented ensemble inflows (`catchment_inflow_with_flood_nodes_mgd.hdf5`) generated before 2026-07-31, and discard flood objective values computed from them. Pywr-DRB's flood-node inflow preprocessor carried a double subtraction of already-marginal upstream inflows plus three wrong USGS drainage areas, which left the three tail-gauge local catchments at ~2 % of physical magnitude (fixed in `../Pywr-DRB/src/pywrdrb/pre/flood_node_inflows.py`; evidence in `../Pywr-DRB/experiments/nyc_flood_gauge_diagnostics/`). `ensemble_prep` SKIPS staging when the file already exists unless `force=True`, so stale files are reused **silently** — re-stage with `force`. The fix is a strict mass-conserving redistribution, so net basin flow and the Montague/Trenton totals are unchanged and only the `downstream_flood_days_*` objectives move — but they move a lot: on the historic trace the aggregate sim/obs exceedance ratio went 0.44 → 0.56 at minor stage and 0.46 → 0.96 at action stage.
- [ ] **[local→HPC]** Regenerate everything downstream of the 6-month metric window and the flood days/yr normalization; discard any objective values computed under the old 365-day warm-up or the whole-trace flood count. 2026-07-31: the step-05 baseline is regenerated locally under the current method (obj8, skip-reeval) and now scores on the annual-unit set via `config.get_objective_set()` — `run_baseline.py` had been building the §1 whole-trace set, so every pre-2026-07-31 baseline vector is on a different objective function and must be discarded, not compared. The baseline re-eval matrix half stays open pending E_test (run step 05 with `--reeval` once E_test is staged).

## 2. Sizing decisions (evidence-based, before generation)

- [x] **[HPC — DONE 2026-07-29]** Nested-P saturation diagnostic on Anvil (`outputs/supplemental/hazard_selector_diagnostics/nested_P_saturation.md`): ONE P=1e6 stream-only pool (`statpool_10yr_n1000000_d0`, sharded generation, ~1.9k SU), prefix rungs {2k, 5k, 20k, 1e5, 3e5, 1e6}. **Verdict: the (m = 8, N = 100) gate FAILS at every rung** — min per-axis tail share 0.144 → 0.256 (2k → 1e6), improvement exponent ~0.04 (far below the P^(−1/8) bound; geometry-limited), fitted crossing ~1.8e7 (unaffordable extrapolation). **Fallback (1) is quantified: m4 concept-group set PASSES from P ≈ 1e5** (0.328; 0.353 at 1e6); m6 fails everywhere (≤ 0.249). Screen retains 8/8 at every rung (max |ρ_S| 0.87–0.88); 2k rung reproduces the laptop battery exactly. (m4/m6 here = the legacy diagnostic nestings, retired from the battery 2026-07-30 — it now scores campaign vs full only.)
- [x] **[DECIDED 2026-07-30]** (m, N, P) = (campaign 6-axis selection set, 100, 1e6): campaign selection axes fixed to {deficit_volume, peak_depth, onset_rate, recovery_rate, peak_magnitude, pulse_duration} (drops duration + rise_rate; both stay computed/reported). Passes the gate at P=1e6 (min 0.311; thin margin — **re-confirm per production draw**, the driver now scores a `campaign` axis set). Wired: `config.HAZARD_SELECTION_AXES` (env `NYCOPT_HAZARD_SELECTION_AXES`) → step-03 `select_from_candidate_image(selection_axes=...)`; docs synced (`scenario_design_methods.md` §3.3/§6, `hazard_selector_diagnostics.md` §5b). The staged P=1e6 draw-0 image is the production draw-0 pool; draws 1-2 still to generate.
- [ ] **[HPC]** Confirm K=3 / S=2 against the allocation.
- [x] **[DECIDED 2026-07-30]** E_test sizing: N_θ = 1,000 LHS SOWs × R = 25 × L_test = 50 yr (25,000 realizations, 1.25M scenario-years; ~80k SU re-eval at the ~1,000–1,200 merged campaign policies). Re-evaluation runs the TRIMMED model, like search (policy-independent presim computed once per realization by step 04 and reused across all Pareto sets; one full-model presim pass over E_test ≈ 70 SU). Constants in `src/etest.py`; derivation + ledger in `scenario_design_methods.md` §5.4/§6 (campaign ≈ 503k SU, reserve ≈ 247k). Staging tracked in §5; adequacy verified post hoc by θ-/R-subsample ranking stability scored offline from the persisted matrix. Hazard coordinates of E_test are computed on disjoint 10-yr sub-windows (commensurable with the pool convention).
- [x] **[DECIDED 2026-07-30]** Variable-resolution `ffmp_N` sweep DEPRIORITIZED: runs only on leftover SU at the end of the campaign (the reserve's first call is an additional draw); which scenario design it runs under is decided if/when the leftover permits.

## 3. Remaining method closures

Evidence base: the framing-convention diagnostics
(`docs/notes/methods/framing_convention_diagnostics.md`) — cube reductions of
the epsilon-calibration policy populations
(`scripts/supplemental/framing_convention_analysis.py`, run locally
2026-07-30: tables + SI figures under `outputs/supplemental/framing_convention/`)
plus the satisfaction-factor sweep (`satisfaction_factor.sh`, one Anvil job per
ensemble design, ~26 SU each).

- [x] **[ADOPTED 2026-07-30]** Framing verdicts adopted repo-wide: failure-week
  counts k CONFIRMED as shipped; flood unit operator = MEAN (P99 stays a
  registered diagnostic); NJ 8th objective ACTIVATED (clean redundancy screen)
  → **the active set is FINAL at 8 objectives; slugs are now `*_obj8`** (env
  files renamed, pinned NYCOPT_OBJECTIVES lists updated). Findings in
  `framing_convention_diagnostics.md` §0b.
- [ ] **[HPC]** Run the satisfaction-factor sweep on Anvil (both ensemble
  designs) and adopt the 0.99-factor verdict; smoke validated locally
  2026-07-30 (bit-exact vs the method functions).
- [ ] **[local→HPC]** Satisficing-criterion OAT stringency + threshold-margin
  CDFs (framing diagnostic 3) — waits on the persisted re-evaluation cube
  (post E_test).
- [ ] **[HPC]** Satisficing criterion values + sweep grid (`_DEFAULT_THRESHOLDS`).
- [x] **[HPC]** Epsilons for the campaign objective set: DONE 2026-07-30 — epsilon-calibration experiment run (512 feasible policies + baseline per design; historic / fixed_probabilistic / hazard_filling_stationary, draw 0) and the combined recommendation adopted into `_ANNUAL_REGISTRY_SPEC`. The campaign vector is the max over the ENSEMBLE designs only (`EPS_CAMPAIGN_DESIGNS`); the historic reference arm is excluded from the max (its 76-unit noise floor would set reliability ε = 0.10 on the 0-1 scale) and its archive is disclosed to resolve below its own noise floor. 2026-07-31: the delivery-factor bounds change triggers the confirmation rerun in §1 (before JARs — epsilons are baked into the archive).
- [ ] **[local]** E_test hazard-space overlay (simulation-free; **candidate main-text figure**, run as soon as E_test is staged): compute E_test's hazard image and overlay it on the candidate pool AND each realized search ensemble (E_d per design/draw) in the retained axes / p1–p99 box. Answers whether E_test's severe droughts (forced mean reduction) occupy the same hazard coordinates as the pool's natural-variability corners the selector enriches; also feeds the registered hazard-restricted E_test composition-sensitivity checks and the generalization claim. IMPLEMENTED 2026-07-30 (smoke-tested on synthetic data; blocked only on E_test staging): `scripts/main/compute_etest_hazard_image.py` (disjoint 10-yr sub-window image, shard-resumable → `hazard_image_subwindows.npz`) + `scripts/main/plot_etest_hazard_overlay.py` / `src/plotting/etest_hazard_overlay.py` (corner overlay + per-axis containment `overlap_stats.json`).

## 4. Pipeline shakeout on Anvil (before production submissions)

- [ ] **[HPC]** End-to-end smoke of `hazard_filling_stationary`: step 06 only (`submit_smoke.sh` = 79 MPI ranks). Local half DONE 2026-07-31 — steps 02→03→04 at P=1000/N=50 exercised wet-exclusion, the robust p1/p99 bounds and the dedupe-only screen; selection ran on the m6 campaign axes (6/6 retained), `axis_screen` + per-axis bounds/clipped fractions + coverage-vs-null persisted in the staged meta, all four pywrdrb inputs staged, and one trimmed-model evaluation on the result returned 8 finite objectives. Untested locally: the MPI fan-out in step 04 (no `mpirun` on the laptop; ran 1 rank).
- [x] **[local]** `check_objective_determinism` on the new metric window — 2026-07-29: DETERMINISTIC on all four paths (historic/ensemble × trimmed/full; 4 policies × 5 fresh-process repeats). Worst max relative deviation 2.1e-13 (Montague deficit CVaR90); LP state jitter (NYC storage differing up to 0.79 %-capacity points on 80% of days, historic_trimmed) does not propagate through the annual-unit aggregation. `scripts/supplemental/check_objective_determinism.py`; outputs under `outputs/supplemental/objective_determinism/`.
- [x] **[local — DONE 2026-07-31]** Baseline anchor vs search path: NO disagreement under the current objective function. Measured on the historic trace, baseline policy, disk path (`run_simulation_to_disk`) vs search path (`run_simulation_inmemory`), both model modes: all 8 objectives agree to ≤2e-13 relative (trimmed) and to 0.00e+00 (full; `run_baseline --test-inmemory` now passes its own 1e-6 gate on every objective). The ~5e-4 figure this item recorded was measured under the RETIRED §1 whole-trace objective set, whose fraction-of-days metrics have 1/28124 granularity and so resolve the known LP degeneracy jitter; the annual-unit set (38 water-year units) does not. Daily series still differ between any two runs (up to ~2.3e3 MGD on major flows, ~62% of days on cannonsville storage) — same magnitude for same-path repeats as for cross-path, i.e. run-to-run LP jitter, not a path effect, and it does not propagate to the objectives. No code change made; `improvement_vs_baseline` was already path-consistent (its anchor comes from `run_baseline --reeval` → `evaluate_solution_raw` → `evaluate_raw`, the search path).
- [ ] **[HPC]** `pilot` MOEA config go/no-go run.

## 5. Production gates

- [ ] **[HPC]** Generate production pools (K draws) + `fixed_probabilistic` draws + E_test (step 12 at the locked sizing, then step 04 incl. the one-time full-model presim pass over its 25,000 realizations). P=1e6 draw-0 image already staged (`statpool_10yr_n1000000_d0`; a prefix is an honest pool of any smaller P); draws 1..K−1 still to generate (sharded path: `workflow/supplemental/gen_pool_shards.sh` + `gen_pool_merge.sh`, ~600 core-hours each).
- [ ] **[HPC]** Trimmed-vs-full re-eval agreement check (SI S1's E_test half): evaluate the baseline + the 4 determinism-check policies both ways on a slice of E_test; report objective-by-objective agreement. A few SU; converts the trimmed-re-eval choice into a measured statement.
- [x] **[local — DONE 2026-07-30]** Full `diagnose_hazard_selectors.py` battery rerun on the production P=1e6 pool image (laptop, ~90 min): screen retains **8/8** axes (dropped none); adequacy gate **at the decided (campaign 6-axis set, N=100) PASSES — min per-axis tail share 0.311** (reproduces the nested-P rung exactly; the full 8-axis set fails at 0.256, the measured basis for restricting selection). Margin is thin: per-seed min ranges 0.28–0.35, so the per-production-draw re-confirmation (§2) stands. Battery cleaned 2026-07-30 to score exactly two axis sets (campaign vs full; legacy m4/m6 nestings + `assess_m6_axis_sets.py` deleted) and rerun for the final SI artifacts: figures F1–F10 + 7 tables + `summary.json` under `outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n1000000_d0/`.
- [ ] **[HPC]** Launch campaign searches.

## 6. Post-campaign deliverables

- [ ] **[local]** Results figure plan + the scripts to build them (none exist yet).
- [ ] **[local]** Manuscript Results / Discussion / Conclusions; SI sections beyond S8 are outline-only.
- [ ] **[local]** Import the manuscript's † references into Zotero before submission, plus the five persistence-literature DOIs listed in `docs/notes/literature/persistence_and_low_frequency_variability.md` (MCP was read-only when they were annotated).

## Parked (scope decisions, not blockers)

- [ ] Optional HMM E_test variant — deprioritized (it introduces a second DU factor set, and the DRB-fitted HMM is near-memoryless so it is not a persistence stress). Persistence stressing was decided against 2026-07-29 — disclosure only (`persistence_axis_diagnostics.md`); a persistence-stressed test ensemble (Mechanism A, prototyped in `scengen.persistence`) is future-work material.
- [ ] **Reconsider the flood-threshold objective** (`downstream_flood_days_minor`) — deferred 2026-07-31, revisit before the objective set is re-frozen. Evidence: `../Pywr-DRB/experiments/nyc_flood_gauge_diagnostics/` (README § "Objective selection", figures F2/F7/F9). Two findings pull opposite ways. (a) `minor` is the regulatorily correct definition — FFMP2017 Appendix A §6.iv-vi names 11 / 13 / 13 ft at Hale Eddy / Fishs Eddy / Bridgeville as the **NWS flood stage** — but even after the inflow fix the model reproduces only ~56 % of observed minor-stage gauge-days, and at ~10 simulated exceedance days per 24 yr the binary count is near-degenerate (many candidate policies score the identical integer → weak MOEA gradient). The shortfall is an input-data ceiling, not a bug: the `delLordville` donor catchment carries ~half the observed unit-area peak yield, and a mass-conserving redistribution inherits its donor's yield by construction. (b) `action` fits far better (sim/obs 0.96) but is the FFMP **"cautionary stage"** (9 / 11 / 12 ft) — a control-rule trigger set 1-2 ft *below* flood stage, explicitly changeable by the Decree Parties, and simultaneously the switching boundary of `NYCFloodRelease` / `NYCCombinedReleaseFactor`, so an objective there puts its own discontinuity on the control rule's. It measures "days discharge mitigation is curtailed", not flood damage. Candidate resolution: severity-weighted exceedance above flood stage, Σ(stage − minor)⁺ pooled across gauges — keeps the regulatory definition while recovering resolution; **not yet checked for monotone response to NYC operations**, which is the gate before adopting it. Swapping is cheap either way (`downstream_flood_days_action` / `_major` are already registered diagnostics in `src/objectives.py`), but any change reopens §1 (epsilons + problem JARs) and the `__sat1` threshold in `src/objectives_ensemble.py`.
- [ ] Hazard descriptors beyond the controlling event (event multiplicity / frequency within a window) — independent follow-up investigation, own session.
- [ ] Manuscript scoping sentences (hazard-axis NYC-only + single-event scope LANDED in §4.3/§8 2026-07-28; remaining): demand is Decree-capped and held stationary (no demand-growth DU axis — consider other justifiable socio-economic parameterizations).
- [ ] Per-realization diversion staging (`src/ensemble_prep.py` supports only `constant_max`) and the deferred DU-factor presets (`src/ensembles.py`) — both forward hooks, not campaign blockers.
