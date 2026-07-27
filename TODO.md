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
- [ ] **[local→HPC]** Regenerate everything downstream of the 6-month metric window; discard any objective values computed under the old 365-day warm-up. The step-05 baseline policy sim is cheap locally; its re-eval matrix is E_test-sized (skip with `NYCOPT_BASELINE_SKIP_REEVAL=1` until E_test exists).

## 2. Sizing decisions (evidence-based, before generation)

- [ ] **[local→HPC]** Pick candidate-pool size P from a saturation diagnostic (selector diagnostics at nested P; snap-distance + coverage curves), not the 1e5–1e6 placeholder. Per-draw pool re-roll (K=3) multiplies generation cost.
- [ ] **[local]** Budget → NFE derivation; fill the `production` MOEAConfig (islands, workers, NFE, runtime-freq, seeds). `tests/test_design_registries.py::test_production_numbers_are_tbd` asserts these are unset and must be rewritten with them.
- [ ] **[HPC]** Confirm K=3 / S=2 (or revise) from a pilot minimum-detectable-effect calculation.
- [ ] **[local→HPC]** E_test sizing (arithmetic) + staging (generation) (`src/etest.py`): N_theta, R_test, L_test, envelope width against the SU budget.
- [ ] **[local]** Decide whether the variable-resolution `ffmp_N` sweep is in scope against the allocation; if so, which scenario design it runs under.

## 3. Remaining method closures

Most are gated by the ensemble objective-sensitivity experiment (scripts exist;
its own full-scale sizes in `supplemental_config.py` are still placeholders).

- [ ] **[HPC]** Flood unit operator (mean vs P99) + annual failure criteria (`_DEFAULT_FAILURE_K`).
- [ ] **[HPC]** Satisficing criterion values + sweep grid (`_DEFAULT_THRESHOLDS`).
- [ ] **[HPC]** Epsilons for the campaign objective set (annual-unit values are placeholders).
- [ ] **[HPC]** 8th objective (`nj_delivery_reliability_annual`): activate or drop, from the redundancy screen.

## 4. Pipeline shakeout on Anvil (before production submissions)

- [ ] **[local→HPC]** End-to-end smoke of `hazard_filling_stationary` with the new code (wet-exclusion, robust p1/p99 bounds, QC meta): steps 02→03→04 run locally at small P; step 06 needs the cluster (`submit_smoke.sh` = 79 MPI ranks).
- [ ] **[local]** `check_objective_determinism` on the new metric window.
- [ ] **[HPC]** `pilot` MOEA config go/no-go run.

## 5. Production gates

- [ ] **[HPC]** Generate production pools (K draws) + `fixed_probabilistic` draws + E_test.
- [ ] **[local]** Rerun `scripts/supplemental/diagnose_hazard_selectors.py` on the production pool image (copy `hazard_image.npz` down — tens of MB, minutes to analyze): confirm the m=4 screen holds at scale; fix the axis set; final SI figures.
- [ ] **[HPC]** Launch campaign searches.

## 6. Post-campaign deliverables

- [ ] **[local]** Results figure plan + the scripts to build them (none exist yet).
- [ ] **[local]** Manuscript Results / Discussion / Conclusions; SI sections beyond S8 are outline-only.

## Parked (scope decisions, not blockers)

- [ ] Optional HMM E_test variant (persistence stress; ranking stability across test-ensemble constructions). Blocked: the HMM generator does not support the CV/variance forcing axis (`src/ensemble_generation.py`).
- [ ] Manuscript scoping sentence: hazard axes are defined on aggregate NYC inflow only.
- [ ] Per-realization diversion staging (`src/ensemble_prep.py` supports only `constant_max`) and the deferred DU-factor presets (`src/ensembles.py`) — both forward hooks, not campaign blockers.
