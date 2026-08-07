# Pre-campaign TODO

Concise action items between now and the HPC optimization campaign, in rough
priority order. Each item is a pointer, not a spec — investigate and plan the
details when picked up. Delete items as they land; decision records live in
the methods notes, not here.

Venue tags: **[local]** laptop-only, **[HPC]** needs the cluster,
**[local→HPC]** decide/dry-run locally, execute at scale on the cluster.

## 1. Remaining method closures

- [ ] **[HPC]** Run the satisficing-threshold diagnostic for real, then adopt.
  The earlier recommendation was measured against a placeholder cube and has
  been deleted (`RTD_RECOMMENDED_THRESHOLDS` is empty again); do NOT resurrect
  those numbers. Pass 1: run
  `workflow/supplemental/robustness_threshold_diagnostics.sh` against the
  genuine step-05 `--reeval` baseline cube on E_test, verifying its meta
  carries `n_sow` = 1000 / `realizations_per_sow` = 25 first. Pass 2: adopt per
  the checklist in `docs/notes/methods/robustness_threshold_diagnostics.md` §5
  (registry vector + `__satNN` label renames + tests + one wrapper rerun), and
  set the stringency-sweep grid centre/span around the adopted vector.
- [ ] **[local→HPC]** Satisficing-criterion OAT stringency + threshold-margin
  CDFs (framing diagnostic 3) — waits on the persisted re-evaluation cube
  (post E_test re-evaluation of the Pareto sets).
- [ ] **[HPC→local]** Regret-tolerance pass A, the moment the step-05 incumbent
  cube lands and **before any re-evaluated policy set is inspected**:
  `workflow/supplemental/regret_tolerance_diagnostics.sh`. It measures the
  per-objective noise floor, picks the ladder SHAPE (`max(eps, floor)` — the
  synthetic check already shows the flood epsilon sitting ~7x under its floor)
  and the headline rung `k`. Adopt both into `RTOL_ADOPTED_K` /
  `NYCOPT_REGRET_TAU_K` (or `NYCOPT_REGRET_TAU` for an explicit vector).
  Rules and the SI plan: `docs/notes/methods/regret_tolerance_diagnostics.md`.
  Also confirm the §1 base-metric epsilons in `src/objectives.py` are current —
  their *ratios* across objectives are load-bearing for the ladder in a way they
  never were for Borg archiving.
- [ ] **[local]** Regret-tolerance pass B after step 08: discrimination band,
  seed/draw empirical nulls, paired SOW bootstrap, and the assay-sensitivity
  control against `historic`. It derives the non-inferiority margin `delta` and
  refuses to run before a rung is adopted.
- [ ] **[local]** SI estimator-stability + convergence diagnostics:
  block-bootstrap effective-sample-size analysis of the annual-unit
  aggregation (Text S5) and the MOEA runtime convergence content (Text S7).
- [ ] **[local→HPC]** Flood-axis validity diagnostics for Text S3:
  downstream-stress correlation as a required build diagnostic plus the
  selected-ensemble event-seasonality span check.

- [ ] **[HPC]** DV re-parameterization ripple (allocation-reduction DVs,
  2026-08-06): rerun the step-05 baseline (same policy, new DV encoding in the
  persisted matrix; the staged one is 2026-08-03 and uses the retired factor
  encoding). Prior `.set`/`.ref` archives and re-eval matrices also use the
  retired encoding — never mix them with new runs, but don't delete them until
  the next optimization is run.
  No JAR rebuild is required: step 00 bakes only `nvars`/`nobjs` and a
  hardcoded `RealVariable(-1e6, 1e6)` into the problem class — the real DV
  bounds live in Python and never reach the JAR. Verified 2026-08-07 that the
  built JARs (36/45/55/64 vars, 8 objs) still match `get_n_vars`/`get_n_objs`.
  Rebuild only when a DV or objective is added/removed.

## 2. Anvil shakeout (before production submissions)

- [ ] **[HPC]** End-to-end smoke of `hazard_filling_stationary`: step 06 only
  (`submit_smoke.sh` = 79 MPI ranks). This is also the first exercise of the
  2026-08-06 allocation-reduction DV decode inside a real Borg search — no
  optimization output predates it. Step 04's MPI fan-out is already verified
  (33 ranks on Anvil, 2026-08-04, both campaign designs — see
  `logs/prep_pywrdrb_inputs_19663199_1.out`).
- [ ] **[HPC]** `pilot` MOEA config go/no-go run.
- [ ] **[local]** Finalize the `mm_moderate` parallel scheme (rank geometry) to
  maximize parallel efficiency, SU, and wall-clock — before any moderate-scale
  submission. `mm_moderate` inherits the Hopper-shaped 165 ranks = 5 nodes x 33
  (`NYCOPT_RANKS_PER_NODE=33` in `workflow/_common.sh`, mirrored in step 06's
  SBATCH header). Anvil's `wholenode` partition is node-EXCLUSIVE at 128
  cores/node, so that layout idles ~74% of every node it bills. The campaign
  cost surface already measures the dense packing
  (`outputs/supplemental/ensemble_cost_experiment/tables/campaign_projection.csv`:
  `search_ranks_per_node=128`, 173.8 s/eval at N=100/L=10, efficiency 0.729),
  and `production` is already sized that way — `mm_moderate` is the straggler.
  Projected at 20k NFE per design (t_eval 173.8 s, eff 0.729):

  | geometry                       | wall  | SU/run |
  |--------------------------------|-------|--------|
  | 5 nodes x 33 (165 ranks, now)  | 8.3 h | ~5,300 |
  | 4 nodes x 128 (511 ranks)      | 2.6 h | ~1,340 |

  Decide the island/worker split too (511 ranks = 1 + 2x(254+1); the scaling
  supplement found island partitioning throughput-free at fixed slot count, so
  it is a search-reliability choice, not a throughput one). Re-check the
  33/node memory-bandwidth rationale against the 128/node cost-surface cells
  before committing — the two were calibrated on different machines. Then
  update `src/moea_config.py` and step 06's `--nodes`/`--ntasks-per-node`
  together so `nycopt_check_allocation` stays consistent.

## 3. Production gates

- [ ] **[HPC]** Launch campaign searches. All production inputs (pools d0–d2,
  search ensembles, E_test + presim, baseline-on-E_test matrix) are staged,
  verified, and adequacy-gated (campaign 6-axis min tail share
  0.311 / 0.306 / 0.303 across draws).

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
