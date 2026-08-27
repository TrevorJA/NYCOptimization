# Pre-campaign TODO

Concise action items between now and the HPC optimization campaign, in rough
priority order. Each item is a pointer, not a spec — investigate and plan the
details when picked up. Delete items as they land; decision records live in
the methods notes, not here.

Venue tags: **[local]** laptop-only, **[HPC]** needs the cluster,
**[local→HPC]** decide/dry-run locally, execute at scale on the cluster.

## 1. Remaining method closures

- [ ] **[HPC]** REGENERATE everything invalidated by the SEASONAL-ROTATION FIX
  plus the 2026-08-18 pre-campaign audit closures (all local halves LANDED
  across all four repos):
  * **December epoch + June-1 FFMP-year alignment** — realizations now span
    Dec 1 – Nov 30 (`ENSEMBLE_START_DATE = 1945-12-01`; generation produces
    L+1 Jan-anchored calendar years and trims the monthly frames to the epoch,
    so stamps stay true), the annual unit moved from the water year to the
    FFMP operating year (`ffmp_year_unit_slices`, Jun 1 – May 31), and the
    hazard image trims the trailing partial year, so hazard metrics and
    objectives score the IDENTICAL [Jun 1 y1, May 31 yL] window (the 6-month
    SSI-6 exclusion ends exactly on the FFMP June-1 reset). The historic
    window moved to `START_DATE/END_DATE = 1945-12-01/2023-11-30` — so
    **step-01 presim and the step-05 historic baseline are now invalidated
    too** (both cheap). Scenario SSI stamp is December
    (`scengen.hazard_metrics._SCENARIO_STAMP_START = 1999-12-01`); hazard
    images carry `scenario_stamp_start` provenance and January-convention
    `.npz` refuse to load.
  * **SynHydro Nowak fixes** — the leap-February length fix wrote through a
    view into the fitted proportion pool (broke partition invariance +
    February mass balance; `.copy()` fix + pool-immutability and
    batch-vs-isolated regression tests), and `renormalize_truncated` is now
    True for monthly→daily (volume conserved in every February). The
    determinism gate runs at L=5 spanning leap 1948 (was L=2, vacuous).
  * Earlier landed half (kept for the record): truthful stamping, no writer
    re-stamp, `_ensemble_window` from the staged `_meta.json` (stale stamps
    FAIL FAST; `NYCOPT_ALLOW_STALE_STAMP=1` to inspect), full-calendar fit
    record `("1945-01-01","2023-12-31")`, historic hazard-window layers
    anchored at the scenario epoch month in both
    `compute_historic_hazard_windows.py` and
    `scenario_discovery._historic_hazard_points`,
    `validate_staged_seasonality.py` build QC.
  * Also landed from the audit: re-eval cubes now persist per-(solution, SOW)
    `n_survivors` (a crashed batch no longer silently shrinks a SOW's pool
    unrecorded), `_aligned_baseline` hard-errors on incumbent SOW
    under-coverage, and all-NaN flood-stage days propagate to the worst-value
    sentinel instead of scoring as "no flooding".
  REGENERATION COMPLETE 2026-08-20 (pre-optimization scope). All four repos on
  the fixed commits (Pywr-DRB unchanged at 4a65787, nyc_opt). Test gate job
  20033470: 415 passed / 2 skipped. Stale data cleared first: 320 GB of E_test
  deleted from /anvil (CONTENTS only — the 51 symlinks and target dirs kept, so
  regeneration wrote back to projects space, not the 25 GB home quota) + ~1.7 GB
  in $HOME. Optimization sets/runtime/metrics and the three first10ch reeval
  leaves PRESERVED; the old incumbent cube archived to
  .../reeval/etest_kn_50yr_n25000/baseline_pre20260818 with all three first10ch
  `baseline` symlinks repointed at it.
  REGENERATED + VERIFIED:
  * step 01 presim — 1945-12-01..2023-11-30, 28,489 rows (job 20033517).
  * P=1e6 pools d0-d2 (shards 20033519/20/21, 150 tasks, ~13 h) — merges wrote
    canonical artifacts; post-merge verify re-run standalone (20045602/03/04):
    shard boundaries EXACT at all 8 probe indices under a THIRD partition, and
    the P=2,000 smoke pools BIT-IDENTICAL to the leading 2,000 rows. Zero FP
    drift; tolerance never engaged.
  * fixprob d0-d2 gen (20033559) + step-04 prep (20034846) — Dec epoch, n=3652.
  * hazfill d0-d2 selection (20045700) + prep (20045701) — P=1e6, 6-axis m6,
    L2* abs 0.0123/0.0147/0.0169 vs random null ~0.166 (pctl 0).
  * E_test etest_kn_50yr_n25000 — 50 shards -> merge (tiling [0,25000) verified)
    -> sub-window hazard image (125,000 x 8) -> 50/50 chunks prepped, 290 GB.
    first10ch subset re-staged (200 SOWs x R=25).
  * Historic hazard windows — 7 Dec-anchored windows, provenance matches the
    pool and E_test images (reference_start 1945-01-01, stamp 1999-12-01).
  * Seasonality QC (validate_staged_seasonality): fixprob shift 0 @ 0.999
    (Feb ratio 1.009, i.e. the Nowak renormalize fix is live); E_test shift 0 @
    0.905 with the wider ratio spread expected of the du_forced population.
  * Baselines — historic (job 20037453): **77 FFMP-year units** (was 76 water
    years); reliabilities are exact n/77 and the storage P01 = 0.0 reproduces
    from the raw trace (2 of 77 years draw aggregate NYC storage to empty).
    fixprob (20037454) and hazfill (20045702) scenario-matched via
    --search-ensemble: reliabilities exact n/900 (100 real x 9 units).
  TWO DEFECTS FOUND AND FIXED EN ROUTE:
  (a) `_hazard_block()` gained a required `n_years` kwarg in a1e88bd but THREE
      supplemental callers were left stale — verify_shard_boundaries.py,
      diagnose_partition_mismatch.py, diagnose_cv_axis_footprint.py. This
      killed all three pool merges AFTER they wrote their artifacts. Fixed
      (one line each, `n_years=cfg.realization_years`); the tests do not cover
      these scripts. New `workflow/supplemental/pool_verify.sh` runs the two
      checks standalone, which is now REQUIRED because gen_pool_merge deletes
      its shards before verifying.
  (b) step 03 selected from the P=2,000 SMOKE pools, silently: env files do not
      set NYCOPT_CANDIDATE_POOL_N (default 2000) and 03_subsample_ensemble.sh
      does not export it the way gen_pool_shards/merge do. Caught only by the
      log line `pool='statpool_10yr_n2000_d0' P=2000`. ALWAYS pass
      NYCOPT_CANDIDATE_POOL_N=1000000 to step 03 and check that line.
  * Incumbent cube on full E_test — DONE (job 20037455, 4 h 13 m 52 s, 1 node
    x 16 ranks x 8 cpus, batch=50; ffmp_obj8_mm_full slug =
    stage_etest_subset_baseline's DEFAULT_BASELINE_SRC). 8,000 rows
    (1 solution x 1,000 SOWs x 8 objectives), ZERO NaN, and n_survivors = 25/25
    on every SOW — the a1e88bd provenance column confirms no batch silently
    lost realizations. Wall matched the 2026-08-08 run (4 h 14 m) almost
    exactly.
  * hazfill seasonality QC (job 20046983) — shift 0 @ 0.997/0.998/0.998 on
    d0/d1/d2.
  ALL GENERATION + PREPROCESSING IS COMPLETE. Nothing blocks the campaign
  searches except the epsilon question below.
  DELIBERATELY NOT RUN: stage_etest_subset_baseline.py — it recreates the three
  first10ch `baseline` symlinks pointing at `baseline`, which would aim the
  PRESERVED old scorecards at the new cube. Run it only when a new subset
  re-eval is actually wanted.
  STILL OPEN: the adopted epsilon vector was calibrated 2026-08-12 on the
  RETIRED water-year substrate; it is no longer measured on the substrate the
  campaign will search. Re-running epsilon_calibration.sh per design is now
  cheap (staged ensembles exist) and would re-anchor it before ~48k SU of
  search. Also not yet re-run: the nested-P saturation record (nestedp) under
  the renamed axes (Layer A's A3 ladder covers pool d0).

  INVALIDATED — regenerate on HPC, in step order (pull ALL FOUR repos first:
  NYCOptimization, SynHydro, NYCOptimization_scenario_generation, Pywr-DRB):
  1. step 01: HISTORIC presim (December window) and step 00 unchanged;
  2. step 02: all candidate pools (`statpool_*` d0–d2) + hazard images;
  3. step 03: `hazfill_*` / `fixprob_*` selections;
  4. step 12: E_test `etest_kn_50yr_n25000` + chunks + hazard images, then
     re-stage the `first10ch` subset + symlinked incumbent baselines
     (`make_etest_subset.py`, `stage_etest_subset_baseline.py`);
  5. step 04: pywrdrb inputs + ensemble presim for every staged ensemble;
  6. step 05: historic baseline + baseline-on-E_test matrix (incumbent cube);
  7. go/no-go searches (all three designs; the 500k-NFE sets, their
     diagnostics, and the `_eps20260812` re-filtered refs are all
     pre-convention) and every step-07–13 artifact derived from them,
     including the interim first10ch chain and figs 03–08 data.
  Run `validate_staged_seasonality.py` on each regenerated ensemble as build
  QC before anything simulates on it.
  ALSO INVALIDATED BY THE 2026-08-18 HAZARD-AXIS RENAME (drought_magnitude /
  drought_severity / flood_peak_discharge, landed across both repos with
  `test_terminology.py` enforcement): every staged `hazard_image.npz` and
  hazard-derived artifact carries the OLD axis names and will mismatch
  `config.HAZARD_SELECTION_AXES` (empty intersection, missing-axis KeyErrors).
  The regeneration above already covers this — just ensure it runs on
  post-rename code on the cluster (pull both repos before step 02).

- [ ] **[HPC]** ENSEMBLE-SIZE DIAGNOSTICS (minimum N; SI Texts S4/S5) — STARTED
  2026-08-25. Pre-registered design + criteria in
  `docs/notes/methods/ensemble_size_diagnostics.md` (§§2-5 frozen before any
  result). Code: `scripts/supplemental/ensemble_size_{hazard,library_run,figures}.py`,
  `src/ensemble_size_stats.py`, `supplemental_config.ESD_*`, four wrappers
  `workflow/supplemental/ensemble_size_*.sh`, env
  `workflow/envs/ensemble_size_diagnostics.env`, tests
  `tests/test_ensemble_size_diagnostics.py` (16 pass). Block D of
  `diagnose_hazard_selectors.py` extended (env N ladder, P99/MST/min-sep
  records, saturation-mode N sweep). NFE-asymptote reading DONE from the
  go/no-go archives (note §7.4): no island reaches 95 % of final HV before
  ~75 % of budget, last-fifth HV gain 3-8 % — NFE cannot be traded for N.
  Chain: smoke (P=2,000 pool) -> Layer A -> stage chunks (array) -> library
  eval (1 wholenode, ~50 SU) -> analysis; budget < ~100 SU total.
  LAYER A DONE 2026-08-25 (job 20142729): HF tail enrichment is FLAT in N at
  P=1e6 (min per-axis P90 share 0.28-0.29 from N=50 to 500 on all three
  pools; declines with N only at P' <= 2e4), so the selection-level argument
  against larger N does not hold at the production pool. BUILD-QC RECORD:
  on the REGENERATED pools the campaign 6-axis min per-axis P90 share is
  0.27-0.29 (job 20143066 on d0: 0.283, drought_magnitude 0.284 binding;
  Layer A seed means 0.271/0.276/0.271 on d0/d1/d2 at N=100), ~3x the 0.10
  i.i.d. share, saturated in pool size and flat in N; reported as a property
  of the selector, no threshold applied.
  LAYER B DONE 2026-08-26 (library job 20143847, 49 SU; total ≈ 125 SU incl.
  a quarantined first evaluation — model-dict cache keyed on preset name
  re-simulated blocks; fixed + two fatal QC guards added). DECISION TABLE
  (note §7.3): PS passes every pre-registered criterion from N = 300
  (binding: worst-pair paired SE ≤ ε/2 on NYC/Montague deficit P99 and
  storage P01; 0.7-0.8 ε at N = 100); HF passes at NO N ≤ 500 — NYC deficit
  P99 construction SD 2-4 ε (shelf-valued operator decided by the selected
  extremes). Reliability + flood objectives pass from N = 75 in both designs.
  n_eff/N(L-1) = 0.68-1.0. NFE cannot be traded for N. N = 300 costs 2.8x the
  campaign searches (1,139k SU) — a budget fact, NOT part of the statistical
  recommendation. STATISTICAL VERDICT: N_common >= 300, upper end NOT
  established — the HF ladder (3 constructions per N, N <= 500) cannot
  separate slow decay (fit N^-0.24 +/- 0.25) from a plateau on NYC deficit
  P99. RE-ANALYSIS 2026-08-26 (`outputs/supplemental/ensemble_size_diagnostics/
  REANALYSIS_2026-08-26.md`, RESULTS_SUMMARY.md rewritten): values at N = 250/256
  recomputed from the unit library straddle the ½ε line on the binding
  Montague-deficit pair (fitted crossing 283 [249-321]; P(N* <= 250) = 0.20 vs
  0.65 at 300), so 300 is the defensible minimum; the HF NYC-deficit gap is a
  3-plans-per-rung + shelf-valued-operator problem, not an N problem; A2
  coverage metrics are not size-corrected (only ratios to random carry
  information); B7 cross-check is void (finite-population deflation). The
  0.30 tail-share threshold is RETIRED (reported property; code/docs/manuscript
  purged 2026-08-26, `test_terminology.py` enforces). HF ladder extension
  (>= 10 anchor plans per N, ~150 SU) PARKED — not needed for the sizing call.
  ADOPTED 2026-08-26 as N_common = 300 across code, envs, notes and manuscript
  (`docs/notes/methods/campaign_design.md` states the campaign at scale; the
  run items it leaves are in §2). Remaining here: re-render the ESD figures
  with the campaign marker at 300 (`ESD_N_CAMPAIGN`; analysis wrapper only,
  minutes).

- [x] **[local]** REGRET TOLERANCE tau RE-ADOPTED 2026-08-14 on ROUND values
  (reliabilities 0.02, deficit-P99 2 pp, flood 0.25 ft-d/yr, storage 5 pp;
  k = 1 unchanged) after pass B ran (job 19910387) and the paired near-tie
  floor came back 5.7-21.8x SMALLER than the unpaired pass-A bound on the five
  axes where the floor set tau. At the old vector the RQ1 comparison was inert:
  every design at no_harm_freq_tau = 1.000, all pairwise diffs exactly 0.000
  with paired bootstrap SE 0.0000, and ASSAY SENSITIVITY FAILED. Written to all
  10 `workflow/envs/*.env`, recorded in `supplemental_config.RTOL_ADOPTED_K`
  and methods note S8; full chain re-run at the new tau (job 19910556).
  ALSO FIXED: `tau_ladder` returned the whole-vector override UNSCALED by k, so
  every rung of a k-sweep was identical and fig 07 panel (b) was a flat line by
  construction. The override is the tolerance at the ADOPTED rung, so it is now
  scaled by k/REGRET_TAU_K (identity at k = 1). Fig 07 panel (b) now sweeps and
  shows historic separating from the matched pair over k = 0-2.
  STILL OPEN from pass B ((i)/(ii) FIXED 2026-08-18: `null_differences` keeps
  its schema when no within-design pair exists and `run_pass_b` reports "null
  not estimable" at one draw per design; `regret_tolerance_sweep`,
  `paired_bootstrap_se`, and `joint_vs_independent` now pass
  `rob.adopted_floors()` — the persisted pass-A `rtol_floors.json` — so
  unset-override k-sweeps stay on the adopted max(eps, floor) basis; tests in
  `test_regret_tolerance_diagnostics.py` + an eps-only-ladder pin in
  `test_compare_designs.py`): (iii) `rtol_noise_floor.csv` carries
  STALE epsilons - pass A should be re-run to refresh it; (iv) delta is only
  `2 x paired SE` (0.102 all-8 / 0.065 compromise-3), a LOWER BOUND, until
  K > 1 draws exist; (v) the matched-design ordering on the all-8 metric flips
  with the flood tolerance alone (between 0.25 and 0.30) and must be reported
  with that sensitivity shown.

- [x] **[local]** FIG 8 OVERHAULED 2026-08-21 to the fig05 style and COMBINED
  with the regret classification: `fig08_robustness_regret_surfaces.png`
  (registry slot 8; old single-row `fig08_success_failure_surfaces.png` parked
  in `outdated_unused/`). Top row (a-d): boosted-tree P(SOW meets the
  All-Parties compromise set) for ONE policy per scenario design + the FFMP
  incumbent; bottom row (e-g): the SAME policies' per-SOW high/low-regret
  label vs the incumbent (PuOr, no incumbent panel — its regret is 0 by
  construction). Policy selection CHANGED in `factor_mapping_run.py`: from
  best-satisficing to MAX-ROBUSTNESS/MIN-REGRET on the fig07 scorecard
  frontier (selected: fixedprob #602 34%/0%, hazfill #204 49%/0%, historic
  #284 6%/0%; frontier is a single point per design). Theta surfaces are now
  PINNED to the common (e^m, r1) plane (hazfill's top-2 axes were (e^m, r2),
  silently mislabelling its y-axis on the shared grid). Artifacts regenerated
  under v2_20260821 criteria + adopted tau (jobs 20056402/20056483). FINDING:
  all three selected policies are low-regret in 200/200 E_test SOWs, so the
  bottom row is uniformly good — the same degeneracy as fig09 candidate A;
  the fig09 choice below should weigh that redundancy.

- [ ] **[local]** CHOOSE ONE of the three manuscript Figure 9 candidates
  (DU-space regret) and cut the other two. All three were built 2026-08-14
  (`src/plotting/factor_map_surfaces.py`, sharing one grid routine with fig 8;
  artifacts `regret_map_{fits,labels,surfaces}` + `regret_exposure.csv` from
  `factor_mapping_run.py`), all carry `number=9` so the numbering does not
  churn, and all render:
  * **A `fig09_regret_surfaces`** — the SAME compromise policies as fig 8.
    DEGENERATE: 200/200 low regret in all three panels, uniform blue. At
    tau = 0 (any degradation at all) it is still 0 / 0 / 17 SOWs, the 17 being
    historic #115 on Trenton alone. Keep only if the intended message is "the
    selected policies never harm the incumbent anywhere".
  * **B `fig09_regret_surfaces_worst`** — each design's MOST-regretting Pareto
    policy. The sharpest exhibit: a clean regret boundary near e^m ~ 1.1-1.2
    with an r1 interaction (fixedprob #595, 79/200 regret, CV AUC 0.99;
    hazfill #684, 78/200, AUC 0.94), and historic #143 regretting in all 200 —
    i.e. historic's front CONTAINS a policy that harms everywhere.
  * **C `fig09_regret_exposure`** — no policy selected: per SOW, the share of
    the design's whole front that stays low-regret. Medians 1.00 / 1.00 / 0.48;
    exposure rises monotonically with e^m. Immune to selection effects and
    stays a frequency (no cross-objective normalization), so it is the safest
    for a reviewer, though less striking than B.
  RE-EVALUATED 2026-08-14 AT THE NEW ROUND tau - the ranking FLIPPED:
  * A (compromise policy) is still degenerate: 200/200 low regret, all blue.
  * B (worst policy) is NOW degenerate THE OTHER WAY: at the tighter tau the
    max-regret policy of every design regrets in all 200 SOWs (0/200 low
    regret, uniform red). Its selection rule is what breaks - "the most
    regretting policy" is an extremum, so it degenerates at whichever tau. If
    B is wanted, change the rule to the MEDIAN-regret policy (front medians of
    no_harm_freq_tau__compromise are now 0.245 fixedprob / 0.705 hazfill /
    0.000 historic, so median is informative for two of three designs).
  * C (front-wide exposure) is NOW THE CLEAR WINNER and needs no selection
    rule at all: share-low-regret spans 0.432-0.859 (fixedprob), 0.453-0.811
    (hazfill), 0.063-0.146 (historic), with a clean dry-to-wet gradient and
    unambiguous design separation.
  RECOMMENDATION NOW: C as the manuscript figure. Cut A and B, or keep B with
  the median-policy rule as an SI companion.
  The underlying finding is unchanged and now visible: regret concentrates in
  WET states of the world, and hazard-filling carries the least of it.

- [ ] **[local]** DECIDE Figure 4 scatter geometry: 2-D plane vs 3-D cube.
  A 3-D variant of panels (a)/(b) is now the default
  (`NYCOPT_COMPOSITION_SCATTER=3d`, set `=2d` to revert to the original
  severity-magnitude plane; nothing else in the figure changes). The 3-D panel
  plots each ensemble in severity x peak-discharge x magnitude and shows the
  candidate pool as 50/90/99% density bands projected on the three cube walls
  instead of an in-cube point cloud, so the 100 sampled members stay the
  subject. Open: whether the flood axis is peak discharge or pulse duration,
  and whether the floor shadow of the members stays.

- [x] **[local]** RE-DESIGN manuscript Figure 4 — DONE 2026-08-17: rebuilt as
  `src/plotting/ensemble_composition.py` (`ensemble_composition`, registry
  `number=4`, §4.1), rendered to `figures/manuscript/fig04_ensemble_composition.png`
  with all three arms including the post-hoc-scored `fixed_probabilistic` (PS)
  ensemble; the seasonal-rotation finding above was measured during this build.
  Original item kept below for the record.
  The first attempt was CUT 2026-08-13
  (`src/plotting/ensemble_composition.py` deleted, spec removed from
  `src/figures/registry.py`): it was unreadable and carried no argument — a
  13-entry legend with the design label repeated once per staged draw, every
  ensemble drawn in the same color, the candidate-pool density field invisible
  underneath, and the `fixed_probabilistic` arm missing outright because its
  search ensemble is not staged under the campaign tag. Manuscript number 4 is
  RESERVED (registry has a placeholder comment; figs 05-08 keep their numbers)
  — decide whether the slot gets a redesign or is dropped and 05-08 renumbered.
  What §4.1 actually needs to show: that the three designs sample DIFFERENT
  regions of the hazard space and how each relates to E_test's support. The SI
  corner overlay (`src/plotting/etest_hazard_overlay.py`) already does the
  pairwise version and is the natural starting point. Prerequisite either way:
  stage the `fixed_probabilistic` search ensemble under the campaign tag, or
  the figure cannot be honest about all three arms.

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
  same job, subset meta records exactly its own SOWs. INTERIM WORKFLOW
  COMPLETE THROUGH STEP 13, 2026-08-13, on the eps20260812 re-filtered sets
  (`outputs/{design}/ffmp_obj8/sets/ffmp_obj8_merged_eps20260812.set`;
  335 / 991 / 784 policies). Step 09 simulate: historic job 19859233
  (14x128, 2h29m — reshaped after 19845822 died in a one-node startup OOM
  spike at 8x128; 14 nodes fits 3,350 units in exactly 2 claim rounds),
  fixedprob 19845823 (12h20m), hazfill 19845825 (9h49m); 21,100/21,100
  units, ZERO .failed, ~28k SU. 09b merges: 19861619 / 19862021 / 19860835.
  All three cubes verified (n_sow=200, R=25, zero NaN; 72-metric scorecards
  incl. regret vs the symlinked incumbent baseline). Step 10 (job 19862033):
  tables -> `outputs/comparison/ffmp_obj8/etest_kn_50yr_n25000_first10ch/`,
  figures -> `outputs/figures/comparison/ffmp_obj8/robustness/`. Step 11:
  the default satisficing label is DEGENERATE on this substrate (joint
  Starr = 0 for every policy in every design — Trenton rel/flood/storage
  bind; step-10 attainability: 100% of E_test unwinnable, and 2 saturated
  objectives flagged non-discriminating: nyc deficit p99, nj rel), so the
  mechanism test ran with NYCOPT_SD_LABEL=regret (job 19862058; job 19862034
  documents the degeneracy): historic = mechanism SUPPORTED (197/200 regret
  SOWs at HIGH coverage deficit), hazfill = null (11/200), fixedprob =
  CONTRADICTED (132/200 at LOW deficit); tables ->
  `outputs/comparison/scenario_discovery/`. Step 13 (job 19862035): fig03
  rendered (registry currently holds only fig03). Subset hazard image is
  staged (make_etest_subset now slices hazard_image.npz too).
  SUBSET-CRITERIA RE-ANCHORING + FULL FIGURE PASS DONE 2026-08-13 on the
  interim tag (job 19882748, `shared` 1x8): the audit table
  (`scripts/supplemental/criteria_reanchoring.py` ->
  `outputs/comparison/ffmp_obj8/etest_kn_50yr_n25000_first10ch/criteria_reanchoring.csv`)
  fired rule 1 on Montague reliability, and the PROVISIONAL 0.70 literal was
  replaced by the transcribed **0.50** (incumbent per-SOW median 0.482,
  stricter side at eps = 0.05; pooled stringency 0.53) in
  `src/satisficing_criteria.py`. Trenton 0.75 and storage 13.0 were confirmed
  unchanged by the audit. That kills the degeneracy for analysis purposes:
  the `compromise` set now discriminates (129-163 failures / 200 SOWs per
  design) where `reference_all8` is still 200/200 (re-run under
  `NYCOPT_SD_LABEL=criterion:reference_all8` ->
  `scenario_discovery/criterion_reference_all8/`, the documented degeneracy).
  Steps 2-4 re-run end to end: three re-scored cubes (+ the new
  `robustness_scorecard_criteria.csv` / `robustness_criterion_stability.csv`
  companions), `compare_designs`, `scenario_discovery`, `factor_mapping_run`,
  and the registry figures. Three defects in the pulled commit were FIXED the
  same day and the whole pipeline re-run on top of them (job 19883470):
  (a) the post-processing entry points called `config.derive_slug()`, which
  builds the slug from the ACTIVE run identity and so silently resolved to
  `ffmp_obj8_smoke` with no env file — replaced by `config.results_slug()`,
  which takes `NYCOPT_RESULTS_SLUG`, else the derived slug IF it carries the
  tag, else the unique on-disk slug that does (reported on stderr), else
  raises; (b) `registry.legacy()` handed builders `out_stub.parent`, so all
  11 `adapt=True` specs saved under a bare stem while `figures.py` printed
  the numbered path and the contact sheet silently dropped fig06 — builders
  now save to the stub they are given, `FigureSpec.per_focal` keeps the
  criterion key on the three focal-parameterized figures, and `figures.py`
  verifies the file exists before reporting it; (c) `tau_ladder` fell back to
  the eps-only ladder in silence when `NYCOPT_REGRET_TAU` was unset — it now
  warns that this is not the adopted basis (still non-fatal, since the
  pass-B k-sweep unsets it deliberately). NOTE figs 07/08 are focal-
  parameterized but NOT marked `per_focal`, so re-rendering under a different
  `NYCOPT_FOCAL_CRITERION` overwrites them; left as-is deliberately (it
  matches the pulled commit's naming) — decide at the manuscript-final pass.
  FIGURE-DESIGN REVISION 2026-08-14 (job 19884369) after an independent
  design review of the manuscript tier. One real DATA bug found and fixed:
  `regret_headline` treated `pareto_frontier`'s return as a boolean mask when
  it returns INDICES, so fig07 drew the wrong frontier rows and `zip()`
  truncated its companion CSV to the frontier's length (the plane looked like
  it topped out at robustness 0.09; the true max is 0.355). Layout defects
  fixed across figs 05-08, all of one species — text anchored per-panel on
  panels too narrow to hold it: per-panel axis labels replaced by shared
  `supxlabel`/short labels, panel letters folded into titles (the corners hold
  data), the 9 parallel axes moved to the project's ABBREVIATION convention
  (`short_label_for`, wrapped) and the figure rebuilt at column-true width
  (it was 13.5 in, printing at 7.48 in, so every annotation shrank ~1.8x —
  fig06 is now 5 MB, not 15 MB), and `shared_legend(y=...)` added so
  legend-above-footer stacking is anchored in the same coordinates instead of
  colliding. Also: masked the definitional diagonal in fig05's Kendall panel
  (it saturated the scale and hid the -0.22..0.45 findings) and gave it human
  tick labels; `overlap_style()` makes exactly-coincident series visible
  without ever offsetting data; degenerate panels now name the degeneracy in
  their TITLE (fig05e, fig07b) rather than reading as broken plots; fig08's
  failure marker is a white-outlined filled X (a plain black cross vanished
  into the dark-red field where most crosses fall); one `ETEST` constant now
  typesets $E_test$ everywhere. STILL OPEN from the review: figs 06 and 08
  carry annotations below the 12 pt MANUSCRIPT_MIN_FONTSIZE floor (9 pt) —
  9 parallel axes / 4 map panels do not fit a double-column width at 12 pt,
  so these two likely need to be full-page or landscape; that is a layout
  decision, not a code one.
  REMAINING for the FINAL (converged, full-E_test) pass: re-run 08/09-13 on
  the production fan-out fronts, then re-run the re-anchoring audit against
  the full cube (the placements above are transcribed from the 200-SOW
  interim table and must be re-confirmed).
  Prefix-only subsets (rows keyed by global SOW id);
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
- [x] **[HPC]** Hazard-support decomposition — DECIDED 2026-08-25: the HF
  candidate pool STAYS STATIONARY for the campaign (no climate-forced
  sub-ensembles). Evidence + decision record in
  `docs/notes/methods/hazard_support_decomposition.md` §6–7: 94.3% of E_test
  sub-windows lie within the P=1e6 stationary pool's q0.99 NN fringe (611
  in-support / 389 boundary / 0 out-of-support SOWs; the excursion is
  wet/flood-side), and on the interim go/no-go cubes the HF−PS satisficing
  advantage is larger BEYOND support (+0.23 [0.14, 0.33]) than inside (+0.09).
  Optional SI follow-up after the campaign: the same wrapper with
  `NYCOPT_HSD_REEVAL_TAG=etest_kn_50yr_n25000` (zero simulation) to re-read
  the no-harm arm on a regenerated incumbent cube.


## 2. Production gates

- [ ] **[HPC]** CAMPAIGN AT N = 300 — run items, in order (design and budget:
  `docs/notes/methods/campaign_design.md`; every step below is scripted, none has
  run). The N = 100 go/no-go cells (jobs 19782745 / 19770939 / 19770938, ~48k
  SU incl. one OOM loss) stay as the measured cost basis (21.3 h, ~21,850 SU per
  500k search on 8x128) and as the N = 100 comparison arm of the sizing note's
  Layer C reading; their sets are pre-N=300 and are NOT campaign results.
  0. Verify the SU balance (`mybalance`); the notes assume ~590-600k remaining.
     Pull all four repos on Anvil first.
  1. RESTAGE at N = 300 (pools d0-d2 and E_test are unchanged): step 02 for
     fixed_probabilistic (`--array=0-2`), step 03 for hazard_filling_stationary
     with `NYCOPT_CANDIDATE_POOL_N=1000000` (check the log line
     `pool='statpool_10yr_n1000000_d{k}' P=1000000`; the smoke pool is the
     silent default), step 04 for both (`--array=0-2`), then
     `validate_staged_seasonality.py` per ensemble and the per-axis tail-share
     record for each hazfill draw at N = 300. Step 05 baselines for both matched
     designs scenario-matched to the new d0 ensembles (`--search-ensemble`;
     reliabilities exact n/2700). ~1-2k SU.
  2. EPSILON RE-VERIFICATION at N = 300: `epsilon_calibration.sh` per design
     (`EPS_REALIZATION_BATCH=150` now; ~1 h per wholenode node each). Go if
     every adopted entry [0.05, 10, 0.05, 10, 0.05, 0.3, 5.0, 0.05] lies above
     its N = 300 floor (expected: floors fall with N); otherwise raise the
     entry, re-pin tau in every env file, and record it in
     `epsilon_calibration_experiment.md`. The archive-cardinality re-filter is
     repeated on the first N = 300 archives after seed 1 (item 5).
  3. BATCHED-SEARCH MEMORY SMOKE: `bash workflow/submit_search_memory_smoke.sh`
     (1 wholenode node, 127 ranks, N = 300, batch 150, ~400 NFE, ~256 SU). Go if
     peak used memory in `logs/mem_<jobid>_*.log` <= ~217,000 MB and the warm
     per-evaluation time in the runtime files is within +20 % of 540 s. A miss
     on memory means batch 100 (3 model runs) in both matched env files; a
     miss on time means re-pricing `campaign_design.md` §6 before seed 1.
  4. SEED 1 (750k NFE, 12 nodes x 128, `--time=96:00:00` matched /
     `12:00:00` historic): the three `*_production.env` headers carry the
     exact sbatch lines. This is the cost check AND the 8 -> 12 node scaling
     measurement (unmeasured; carried as x1.00-1.17). Read from the runtime
     files after ~12 h: island NFE rate -> projected wall; the 125,000/island
     snapshot must land inside 96 h (it does under every cost basis). Abort
     criteria before seed 2: SU per NFE above the model basis (0.0669 SU/NFE at
     N = 300) or runtime HV at 125k/island still rising > 5 %/25k on any island
     (then extend seed 2 to 750k via `max_evaluations_by_seed=(187_500,
     187_500)` only if the balance covers it).
  5. `python scripts/main/extract_runtime_archive.py --seed 1` per design (the
     equal-NFE set `seed_01_ffmp_obj8_nfe125000.set`), then the ε cardinality
     re-filter on it (`epsilon_ensemble_refilter.py`); adopt the cap rule of
     `campaign_design.md` §5 if the projected merged union exceeds ~2,000.
  6. SEED 2 (500k NFE, `--time=72:00:00` matched / `08:00:00` historic), then
     `extract_runtime_archive.py --merge --install` per design (writes
     `ffmp_obj8_merged_nfe125000.set` and installs it as `ffmp_obj8_merged.set`,
     the step-08/09 first-choice reference; step 07's own merge includes the
     750k tail and is diagnostics-only). Step 07 diagnostics per seed.
  7. E_TEST RE-EVALUATION (steps 09 + 09b on `shared`, 16 ranks x 8 cpus,
     batch 50) of the installed merged sets: ~66 SU per policy, ~132k at the
     2,000-policy cap. Then steps 10-13 and the re-anchoring audit on the full
     cube (the interim first10ch placements must be re-confirmed).
  8. Re-render the ESD figures with `ESD_N_CAMPAIGN = 300` (analysis wrapper).
  Runs are NFE-bounded (`max_time_hours=None`); `--time` must cover the NFE or
  the search is silently truncated (no resume exists). Memory: the step-06
  pre-flight (`nycopt_check_memory`) refuses N = 300 at 128 ranks/node without
  the batch; the sampler (`NYCOPT_MEM_SAMPLE_S`) logs the first node only; use
  `sstat -j <jobid> --format=MaxRSS,AveRSS,Nodelist` for multi-node jobs.

## 3. Post-campaign deliverables

- [ ] **[local]** Results figure plan + the scripts to build them. First
  tranche landed: `src/solution_selection.py` (dominance / scaling /
  compromise / diverse selection), `src/plotting/front_overview.py`,
  `src/plotting/historic_timeseries.py`, driven by
  `scripts/main/explore_results.py` (+ `workflow/supplemental/sim_selected_policies.sh`
  for the simulation-dependent panels). SECOND TRANCHE (ground-up re-eval
  results sequence) STARTED 2026-08-13: registry driver
  `scripts/main/results_figures.py` + `workflow/14_results_figures.sh`
  (`NYCOPT_REEVAL_TAG` selects the tag; reusable verbatim for the full-E_test
  rerun), substrate loader `src/results_data.py` (cubes + scorecards +
  label-joined incumbent; criterion vectors with ±inf axis-disabling), phase-1
  satisficing diagnostics `src/plotting/satisficing_diagnostics.py` (5 figures
  + companion CSVs -> `outputs/figures/comparison/ffmp_obj8/satisficing/`,
  rendered on the interim tag; cube-vs-scorecard consistency asserted
  in-builder). The old step-10 robustness figure set
  (`outputs/figures/comparison/ffmp_obj8/robustness/`, 5 PNGs) was RETIRED
  and deleted — compare_designs.py still writes the tables, but its figures
  are superseded by the results_figures registry. Okabe-Ito design style map
  promoted to `src/plotting/style.py::DESIGN_STYLE` (epsilon_calibration /
  framing_convention SI scripts now alias it). Phase-1 finding: the
  trenton×flood pincer is STRUCTURAL (pairwise best-policy joint 0.03/0.07/
  0.01 across designs — wet SOWs fail flood, dry SOWs fail trenton), so ANY
  defensible 8-axis conjunction leaves median-policy joint Starr at 0 on the
  DU-forced E_test; criterion sets A/B/C approved at check-in 2026-08-14.
  PHASE 2 LANDED 2026-08-14: named criterion vectors in
  `src/satisficing_criteria.py` (adopted + A NYC-supply + B downstream/Decree
  + C compromise, each deviation anchored in the threshold-response curves /
  incumbent medians), figures in `src/plotting/criteria_comparison.py`
  (criterion_robustness_matrix, criterion_collapse, drought_flood_split,
  axes_satisfied_cdf -> `outputs/figures/comparison/ffmp_obj8/criteria/`).
  Every results figure now carries the boxed provenance + explicit bulleted
  criteria footer (Trevor's standing requirement;
  `style.add_figure_footer`/`criteria_lines`). Nonzero-joint-Starr policy
  counts: A 69/32/0, B 192/116/0, C 11/12/0 (fp/hf/hist); incumbent 0 under
  every framing (Montague rel binds it regardless). FOCAL CRITERION = B
  (downstream/Decree) selected at check-in #2 (2026-08-14). PHASE 3 LANDED
  2026-08-14 in `src/plotting/robustness_comparison.py`, fully
  criterion-parameterized (env `NYCOPT_FOCAL_CRITERION`, default
  "downstream"; filenames carry the criterion key so future focal changes
  coexist): parallel_coords_downstream (9th robustness axis drives viridis
  coloring; `custom_parallel_coordinates` gained backward-compatible
  `axis_ranges` + `add_colorbar` params for identical cross-panel scales),
  robustness_cdf_downstream (joint Starr + mean-fraction exceedance),
  regret_robustness_plane_downstream (joint Starr vs no_harm_freq_tau with
  per-design frontiers) -> figures/comparison/ffmp_obj8/{parallel_coords,
  robustness_cdf,robustness}/. Headline: fp has MORE nonzero-robustness
  policies (192 vs 116) but hf owns the tail (0.245 vs 0.100); fp's frontier
  is a single dominating policy #793 (joint 0.100 AND no-harm 1.0); hf
  frontier spans #637 (0.135, 1.0) to #516 (0.245, 0.055). PHASE 4 LANDED
  2026-08-14 (theta DU space only, per check-in #3 — no hazard-space maps):
  `src/plotting/factor_maps.py` -> factor_maps_theta_downstream in
  figures/comparison/ffmp_obj8/factor_maps/. Rule-based policy selection
  (max joint Starr with no-harm tie-break + best near-never-harmful policy;
  mean-fraction fallback for all-zero designs; env override
  NYCOPT_FACTOR_POLICIES) picked fp#793 (20/200 pass), hf#516 (49/200),
  hf#637 (27/200), hist#7 (0/200), incumbent (0/200). Mechanism: every pass
  sits at e^m ~ 0.95-1.3 with LOW seasonal amplitude r1; drier (low e^m),
  much wetter (high e^m), or strongly summer-dried (high r1) worlds fail for
  every policy; r2 shows little structure (consistent with GBC importances).
  hf#516's edge over fp#793 = a wider pass region in (e^m, r1), not a
  different region. REVISIONS 2026-08-14 (Trevor review round 1): (1) a
  "uniform" round-number criterion set (reliabilities >= 0.70, deficits
  <= 30%, flood <= 1.5, storage >= 25) was trialed as the analysis default
  and REVERTED the same day — too strict on this ensemble (best joint 0.030
  fp / 0.010 hf / 0 hist; storage-25 + flood-1.5 bind); defaults are BACK to
  the adopted snapshot (phase 1) and focal criterion B "downstream" (phases
  3/4); "uniform" remains in `src/satisficing_criteria.py::CRITERION_SETS`
  as a comparison framing documenting that stringency; (2) the
  mean-fraction-of-criteria metric REMOVED everywhere (Trevor: no mean
  aggregations; factor-map fallback now maximin worst-axis; robustness_cdf
  is single-panel joint Starr); (3) companion CSVs moved out of
  outputs/figures/ (PNGs only) into
  `outputs/comparison/ffmp_obj8/{tag}/figure_tables/{kind}/`; (4) objective
  naming capped at 2 conventions (label_for + short_label_for;
  parallel-axis labels derived via style.axis_label_for). 9-figure registry
  re-rendered under the reverted defaults (phase-3/4 files keyed
  *_downstream). REMAINING: canonical sbatch re-render job 19878006
  (verify on completion), figure-quality iteration round with Trevor, then
  the manuscript-final styling pass + full-E_test rerun reuse.
- [ ] **[local]** Manuscript Results / Discussion / Conclusions; SI sections
  beyond S8 are outline-only.
- [ ] **[HPC]** SI — DRAW-SENSITIVITY RE-EVALUATION of the Pareto sets: after
  the searches, re-evaluate each matched design's equal-NFE merged set on the
  other two draws of its OWN search ensemble (d1, d2 at N = 300, staged in §2
  item 1; ~66 SU per 100 policies on a 300 x 10-yr trimmed run; ~1k SU total)
  to quantify draw-dependence of the objective values without re-running the
  search — the Zatarain Salazar et al. (2017) Fig. 12 analogue and the
  supporting check for the one-draw-per-design comparison in
  `experimental_design.md` (Replication) and `campaign_design.md` §5. Needs a
  small driver (evaluate a .set on a staged search-ensemble slug via
  `evaluate_annual_units`, persist per-realization units, paired shifts vs ε
  and vs the E_test values); none exists yet. SI material only.

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
