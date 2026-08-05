# Pre-campaign TODO

Concise action items between now and the HPC optimization campaign, in rough
priority order. Each item is a pointer, not a spec — investigate and plan the
details when picked up. Delete items as they land; decision records live in
the methods notes, not here.

Venue tags: **[local]** laptop-only, **[HPC]** needs the cluster,
**[local→HPC]** decide/dry-run locally, execute at scale on the cluster.

## 1. Remaining method closures

- [ ] **[HPC]** Adopt the measured satisficing-threshold vector
  (`supplemental_config.RTD_RECOMMENDED_THRESHOLDS`) into
  `objectives_ensemble._DEFAULT_THRESHOLDS` + the `__satNN` labels
  (adoption checklist in
  `docs/notes/methods/robustness_threshold_diagnostics.md` §5), rerun the
  affected tests, then one wrapper rerun so the SI figures' "current"
  markers show the adopted values. Also set the stringency-sweep grid
  centre/span around the adopted vector.
- [ ] **[local→HPC]** Satisficing-criterion OAT stringency + threshold-margin
  CDFs (framing diagnostic 3) — waits on the persisted re-evaluation cube
  (post E_test re-evaluation of the Pareto sets).
- [ ] **[HPC]** One confirmatory cheap search under the adopted epsilon
  vector (re-filtering approximates the archive but ε also steers Borg
  selection/restarts; a 2-seed historic shakeout ≈ 2.4k SU —
  `docs/notes/methods/epsilon_calibration_experiment.md`).
- [ ] **[local]** Decide the primary robustness unit (SOW-level vs
  realization-level satisficing) from a focused review of MOEA + robustness
  conventions; close the manuscript §3.4.2 open item.
- [ ] **[local]** Explore the archive-level satisficing measure (fraction of
  each draw's re-evaluated set above a robustness level across the criterion
  sweep) as a cardinality-robust secondary comparison metric.
- [ ] **[local]** SI estimator-stability + convergence diagnostics:
  block-bootstrap effective-sample-size analysis of the annual-unit
  aggregation (Text S5) and the MOEA runtime convergence content (Text S7).
- [ ] **[local→HPC]** Flood-axis validity diagnostics for Text S3:
  downstream-stress correlation as a required build diagnostic plus the
  selected-ensemble event-seasonality span check.

## 2. Anvil shakeout (before production submissions)

- [ ] **[HPC]** End-to-end smoke of `hazard_filling_stationary`: step 06 only
  (`submit_smoke.sh` = 79 MPI ranks). The local half is done; the MPI
  fan-out in step 04 is still untested (laptop ran 1 rank).
- [ ] **[HPC]** `pilot` MOEA config go/no-go run.

## 3. Production gates

- [ ] **[HPC]** Launch campaign searches. All production inputs (pools d0–d2,
  search ensembles, E_test + presim, baseline-on-E_test matrix) are staged,
  verified, and adequacy-gated (campaign 6-axis min tail share
  0.311 / 0.306 / 0.303 across draws).

## 4. Post-campaign deliverables

- [ ] **[local]** Results figure plan + the scripts to build them (none exist yet).
- [ ] **[local]** Manuscript Results / Discussion / Conclusions; SI sections
  beyond S8 are outline-only.
- [ ] **[local]** Import the manuscript's † references into Zotero before
  submission, plus the five persistence-literature DOIs listed in
  `docs/notes/literature/persistence_and_low_frequency_variability.md`.

## Parked (scope decisions, not blockers)

- [ ] Optional HMM E_test variant — deprioritized (a second DU factor set,
  and the DRB-fitted HMM is near-memoryless so it is not a persistence
  stress); a persistence-stressed test ensemble is future-work material
  (`persistence_axis_diagnostics.md`).
- [ ] Hazard descriptors beyond the controlling event (event multiplicity /
  frequency within a window) — independent follow-up investigation.
- [ ] Remaining manuscript scoping sentence: demand is Decree-capped and held
  stationary (no demand-growth DU axis — consider other justifiable
  socio-economic parameterizations).
- [ ] Per-realization diversion staging (`src/ensemble_prep.py` supports only
  `constant_max`) and the deferred DU-factor presets (`src/ensembles.py`) —
  both forward hooks, not campaign blockers.
