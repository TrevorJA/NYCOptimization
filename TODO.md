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
     under the new decoder would silently misdecode DVs,
  2. regenerate steps 10/11/13 artifacts (needs the step-08 policy cubes).
  GEOMETRY NOTE for step-09 chunk re-evals on E_test: submit on `shared`
  with ~8 cpus per rank and an explicit realization batch (job 19738752:
  1 node x 16 ranks x 8 cpus, `NYCOPT_SEARCH_REALIZATION_BATCH=50`, 4h14m,
  ~1.7 GB/rank RSS). Denser packings (64x2, 32x4) OOM the job cgroup —
  each rank streams a DIFFERENT ~7.3 GB chunk-HDF5 set and the page cache
  is charged to the job (jobs 19733674/19733773; single-rank code peak is
  only ~1.7 GB, so it is concurrency, not a leak).
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


## 2. Anvil shakeout (before production submissions)

- [ ] **[HPC]** `pilot` MOEA config go/no-go run. Measure the feasibility rate
  (fraction of NFE rejected in <0.1 s by `flood_zone_ordering`) — it sets how
  far below the upper-bound cost projections real runs land. `mm_moderate` is
  finalized for the runs after it (2026-08-07): 50k total NFE, 511 ranks =
  2 islands x 254 workers on 4 Anvil `wholenode` nodes at 128/node
  (<=6.5 h / ~3,340 SU per seed, upper bound; rationale in
  `src/moea_config.py`).

## 3. Production gates

- [ ] **[HPC]** Launch campaign searches. Production inputs (pools d0–d2,
  search ensembles, E_test + presim) are staged, verified, and adequacy-gated
  (campaign 6-axis min tail share 0.311 / 0.306 / 0.303 across draws). The
  baseline-on-E_test matrix is regenerated on the unified substrate
  (2026-08-08) and the threshold + regret-tolerance parameters are adopted —
  no metric-side blockers remain.

## 4. Post-campaign deliverables

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
