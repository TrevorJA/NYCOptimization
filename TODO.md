# Pre-campaign TODO

Concise action items between now and the HPC optimization campaign, in rough
priority order. Each item is a pointer, not a spec — investigate and plan the
details when picked up. Delete items as they land.

Venue tags: **[local]** laptop-only, **[HPC]** needs the cluster,
**[local→HPC]** decide/dry-run locally, execute at scale on the cluster.

## 1. Artifact regeneration

Sequence after §3: JARs encode the DV **and** objective counts, so regenerating
them before the objective set is final means doing it twice.

- [ ] **[local]** Regenerate everything downstream of the 39-DV scheme: problem JARs (step 00), baseline + `.set` artifacts.
- [ ] **[local→HPC]** Regenerate everything downstream of the 6-month metric window and the flood days/yr normalization; discard any objective values computed under the old 365-day warm-up or the whole-trace flood count. The step-05 baseline policy sim is cheap locally; its re-eval matrix is E_test-sized (skip with `NYCOPT_BASELINE_SKIP_REEVAL=1` until E_test exists).

## 2. Sizing decisions (evidence-based, before generation)

- [x] **[HPC — DONE 2026-07-29]** Nested-P saturation diagnostic on Anvil (`outputs/supplemental/hazard_selector_diagnostics/nested_P_saturation.md`): ONE P=1e6 stream-only pool (`statpool_10yr_n1000000_d0`, sharded generation, ~1.9k SU), prefix rungs {2k, 5k, 20k, 1e5, 3e5, 1e6}. **Verdict: the (m = 8, N = 100) gate FAILS at every rung** — min per-axis tail share 0.144 → 0.256 (2k → 1e6), improvement exponent ~0.04 (far below the P^(−1/8) bound; geometry-limited), fitted crossing ~1.8e7 (unaffordable extrapolation). **Fallback (1) is quantified: m4 concept-group set PASSES from P ≈ 1e5** (0.328; 0.353 at 1e6); m6 fails everywhere (≤ 0.249). Screen retains 8/8 at every rung (max |ρ_S| 0.87–0.88); 2k rung reproduces the laptop battery exactly.
- [x] **[DECIDED 2026-07-30]** (m, N, P) = (m6 selection set, 100, 1e6): campaign selection axes fixed to {deficit_volume, peak_depth, onset_rate, recovery_rate, peak_magnitude, pulse_duration} (drops duration + rise_rate; both stay computed/reported). Passes the gate at P=1e6 (min 0.311; thin margin — **re-confirm per production draw**, the driver now scores a `campaign` axis set). Wired: `config.HAZARD_SELECTION_AXES` (env `NYCOPT_HAZARD_SELECTION_AXES`) → step-03 `select_from_candidate_image(selection_axes=...)`; docs synced (`scenario_design_methods.md` §3.3/§6, `hazard_selector_diagnostics.md` §5b). The staged P=1e6 draw-0 image is the production draw-0 pool; draws 1-2 still to generate.
- [ ] **[HPC]** Confirm K=3 / S=2 against the allocation.
- [ ] **[local→HPC]** E_test sizing (arithmetic) + staging (generation) (`src/etest.py`): N_theta, R_test, L_test, envelope width against the SU budget.
- [ ] **[local]** Decide whether the variable-resolution `ffmp_N` sweep is in scope against the allocation; if so, which scenario design it runs under.

## 3. Remaining method closures

Most are gated by the ensemble objective-sensitivity experiment (scripts exist;
its own full-scale sizes in `supplemental_config.py` are still placeholders).

- [ ] **[HPC]** Flood unit operator (mean vs P99) + confirm the shipped failure-week counts k (`_DEFAULT_FAILURE_K`: 3 NYC/Montague, 1 Trenton/NJ) against the saturation screen.
- [ ] **[local→HPC]** Run the framing-convention diagnostics (`docs/notes/methods/framing_convention_diagnostics.md`): failure-week k sweep, weekly satisfaction factor, satisficing-criterion OAT stringency + threshold-margin CDFs, flood-days controllability. SI material; the criterion diagnostics may be promoted to main text.
- [ ] **[HPC]** Satisficing criterion values + sweep grid (`_DEFAULT_THRESHOLDS`).
- [ ] **[HPC]** Epsilons for the campaign objective set (annual-unit values are placeholders). Resolve the near-saturating Trenton reliability ε (0.0003) before JAR generation — epsilons are baked into the archive.
- [ ] **[HPC]** 8th objective (`nj_delivery_reliability_annual`): activate or drop, from the redundancy screen.
- [ ] **[local]** Decide the CV/variance forcing axis for the campaign E_test (`ENSEMBLE_FORCING_VARIANCE_AXIS` defaults OFF → 3-D box; docs imply +CV). If it stays off, the HMM E_test variant is no longer blocked.
- [ ] **[local]** Persistence in the DU space: evaluate adding a CMIP6-anchored interannual-persistence forcing axis to the harmonic set for E_test (preferred over standing up the HMM variant, which introduces a second DU factor set).
- [ ] **[local]** Flood-tail structure — potential method change: the Nowak fragment disaggregation bounds daily-extreme structure in both the candidate pool and E_test, so the wet hazard axes and the flood objective vary within a generator-limited tail. Assess whether a method change (e.g., fragment/tail perturbation) is needed or whether a scoped claim suffices.
- [ ] **[local]** E_test hazard-space overlay (simulation-free; **candidate main-text figure**, run as soon as E_test is staged): compute E_test's hazard image and overlay it on the candidate pool AND each realized search ensemble (E_d per design/draw) in the retained axes / p1–p99 box. Answers whether E_test's severe droughts (forced mean reduction) occupy the same hazard coordinates as the pool's natural-variability corners the selector enriches; also feeds the registered hazard-restricted E_test composition-sensitivity checks and the generalization claim.
- [ ] **[local]** Basin-coverage SI diagnostic: correlation of NYC-aggregate-inflow SSI-6 hazard metrics with Montague/Trenton deficits and NYC storage drawdown (strong correlation already observed in ../StochasticExploratoryExperiment) — pre-empts the aggregate-inflow scoping critique; flag in Discussion.

## 4. Pipeline shakeout on Anvil (before production submissions)

- [ ] **[local→HPC]** End-to-end smoke of `hazard_filling_stationary` with the new code (wet-exclusion, robust p1/p99 bounds, QC meta): steps 02→03→04 run locally at small P; step 06 needs the cluster (`submit_smoke.sh` = 79 MPI ranks).
- [ ] **[local]** `check_objective_determinism` on the new metric window.
- [ ] **[HPC]** `pilot` MOEA config go/no-go run.

## 5. Production gates

- [ ] **[HPC]** Generate production pools (K draws) + `fixed_probabilistic` draws + E_test. Gated on the (m, N, P) decision (§2): P=1e6 draw-0 image already staged (`statpool_10yr_n1000000_d0`; a prefix is an honest pool of any smaller P); draws 1..K−1 still to generate (sharded path: `workflow/supplemental/gen_pool_shards.sh` + `gen_pool_merge.sh`, ~600 core-hours each).
- [ ] **[local]** Rerun `scripts/supplemental/diagnose_hazard_selectors.py` on the production pool image (copy `hazard_image.npz` down — tens of MB, minutes to analyze): confirm the retained descriptor set and the adequacy gate **at the decided (m, N)** — the nested-P result (§2) shows (m = 8, N = 100) cannot pass at any affordable P, m4 passes from P ≈ 1e5; final SI figures (F1–F10).
- [ ] **[HPC]** Launch campaign searches.

## 6. Post-campaign deliverables

- [ ] **[local]** Results figure plan + the scripts to build them (none exist yet).
- [ ] **[local]** Manuscript Results / Discussion / Conclusions; SI sections beyond S8 are outline-only.
- [ ] **[local]** Import the manuscript's † references into Zotero before submission.

## Parked (scope decisions, not blockers)

- [ ] Optional HMM E_test variant — deprioritized (it introduces a second DU factor set); the preferred persistence stress is the forcing-axis extension in §3. Unblocked if the CV axis stays off; re-evaluation cost only, no new searches.
- [ ] Hazard descriptors beyond the controlling event (event multiplicity / frequency within a window) — independent follow-up investigation, own session.
- [ ] Manuscript scoping sentences (hazard-axis NYC-only + single-event scope LANDED in §4.3/§8 2026-07-28; remaining): demand is Decree-capped and held stationary (no demand-growth DU axis — consider other justifiable socio-economic parameterizations); the Nowak daily disaggregation limits daily-extreme (flood-tail) structure in both the search population and E_test.
- [ ] Per-realization diversion staging (`src/ensemble_prep.py` supports only `constant_max`) and the deferred DU-factor presets (`src/ensembles.py`) — both forward hooks, not campaign blockers.
