# TODO

Open action items only, in execution order. Each item is a pointer, not a spec.
The method and its justification live in `docs/notes/methods/` (campaign at
scale: `campaign_design.md`). Delete items as they land. Venue: **[HPC]**
cluster, **[local]** laptop.

## 1. Campaign at N = 300 (`campaign_design.md` §4–6)

- [ ] **[HPC]** Pull all four repos on Anvil; check the SU balance (`mybalance`)
  against the ~600k the budget assumes.
- [ ] **[HPC]** Restage search ensembles at N = 300, draws 0–2: step 02
  (`fixed_probabilistic`, `--array=0-2`), step 03 (`hazard_filling_stationary`,
  `NYCOPT_CANDIDATE_POOL_N=1000000`; confirm the log line
  `pool='statpool_10yr_n1000000_d{k}'`), step 04 both (`--array=0-2`).
- [ ] **[HPC]** Build QC on each restaged ensemble: `validate_staged_seasonality.py`
  and the per-axis tail-share record per hazfill draw; then step 05 baselines for
  both matched designs scenario-matched to d0 (`--search-ensemble`).
- [ ] **[HPC]** Stage the 500-SOW E_test subset (login node, metadata only):
  `python3 -m scripts.supplemental.make_etest_subset --pool etest_kn_50yr_n25000`,
  then `stage_etest_subset_baseline` per design env under
  `etest_kn_50yr_n25000_first25ch`.
- [ ] **[HPC]** ε re-verification at N = 300: `epsilon_calibration.sh` per design.
  Go if every adopted entry lies above its N = 300 floor; otherwise raise it,
  re-pin τ in every env file, and record it in `epsilon_calibration_experiment.md`.
- [ ] **[HPC]** Batched-search memory smoke: `bash workflow/submit_search_memory_smoke.sh`.
  Go if peak node memory ≤ ~217,000 MB and warm per-evaluation time ≤ 540 s + 20 %;
  otherwise batch 100 (memory miss) or re-price `campaign_design.md` §6 (time miss).
- [ ] **[HPC]** Before seed 1: run `extract_runtime_archive.py --seed 1` against an
  existing go/no-go runtime set (`outputs/fixed_probabilistic/ffmp_obj8/runtime/`)
  and check the row count against its `.set`; it is unit-tested on synthetic
  runtime text only.
- [ ] **[HPC]** Seed 1 per design (750k NFE; submit lines in the `*_production.env`
  headers). After ~12 h read SU/NFE and runtime HV at 125k/island; hold seed 2 if
  SU/NFE exceeds the model basis or HV still rises > 5 % per 25k on any island.
- [ ] **[HPC]** Per design: `extract_runtime_archive.py --seed 1` (writes
  `seed_01_ffmp_obj8_nfe125000.set`), then the ε cardinality re-filter
  (`epsilon_ensemble_refilter.py`); apply the §5 cap rule if the merged union
  exceeds ~2,000.
- [ ] **[HPC]** Seed 2 per design (500k NFE), then
  `extract_runtime_archive.py --merge --install` (installs `ffmp_obj8_merged.set`,
  the step-08/09 reference) and step 07 per seed.
- [ ] **[HPC]** E_test re-evaluation: steps 09 + 09b on `shared`, 16 ranks × 8 cpus,
  batch 50, `NYCOPT_REEVAL_ENSEMBLE_PRESET=etest_kn_50yr_n25000_first25ch` on every
  05/08/09/09b/10 line (deliberately not in the env files). ~66k SU at the
  2,000-policy cap.
- [ ] **[HPC]** Post-processing on the 500-SOW cube: steps 10–14, the criteria
  re-anchoring audit (`criteria_reanchoring.py`), and the θ-subsample stability
  check (250 vs 500 SOWs).
- [ ] **[HPC]** Re-render the ensemble-size figures with the campaign marker
  (`workflow/supplemental/ensemble_size_analysis.sh`, `ESD_N_CAMPAIGN = 300`).

## 2. Diagnostics on the step-08/09 cube

- [ ] **[local]** Regret tolerance: re-run pass A on the regenerated incumbent cube
  (`rtol_noise_floor.csv` carries stale ε), then pass B on the production cube with
  `NYCOPT_REGRET_TAU` unset (k-sweep, seed nulls, paired bootstrap, assay control),
  which yields the discrimination band δ.
- [ ] **[local]** Satisficing thresholds: re-run `robustness_threshold_diagnostics.sh`
  and adopt final criterion values and the sweep-grid centre (open decision;
  placements are provisional until then).
- [ ] **[local]** Framing diagnostic 3: OAT stringency + threshold-margin CDFs on the
  persisted cube (`framing_convention_diagnostics.md`).
- [ ] **[HPC]** SI draw-sensitivity re-evaluation: each matched design's merged set on
  its own d1/d2 (~1k SU). Needs a driver (evaluate a `.set` on a staged
  search-ensemble slug via `evaluate_annual_units`, persist per-realization units,
  paired shifts vs ε); none exists.
- [ ] **[HPC]** Optional: nested-P saturation record (`nestedp_ladder.sh`) under the
  renamed hazard axes; hazard-support no-harm arm re-read with
  `NYCOPT_HSD_REEVAL_TAG=etest_kn_50yr_n25000` (no simulation).

## 3. Figures and manuscript

- [ ] **[local]** Figure 9: keep one of the three `number=9` registry entries
  (`regret_exposure` recommended; the two `regret_surfaces*` variants degenerate at
  the adopted τ) and delete the other two from `src/figures/registry.py`.
- [ ] **[local]** Figure 4: settle the scatter geometry (`NYCOPT_COMPOSITION_SCATTER`
  3d vs 2d), the flood axis (peak discharge vs pulse duration), and whether the
  floor shadow stays.
- [ ] **[local]** Layout at the manuscript-final pass: figs 06/08 annotations sit below
  the 12 pt floor (full-page or landscape); mark figs 07/08 `per_focal` or pin the
  focal criterion.
- [ ] **[local]** Manuscript Results / Discussion / Conclusions; SI Texts S9–S11 are
  outline-only.
- [ ] **[local]** SI gaps: Text S12 (tolerance rules) is cited in §3.4.3 but does not
  exist; Text S7 runtime diagnostics finalize after the campaign; Text S3
  downstream-stress correlation and event-seasonality checks are still marked
  planned; SI Text S10 says no hazard-space scenario discovery is performed while
  step 11 performs one (reconcile).
- [ ] **[local]** Decide whether the variable-resolution `ffmp_N` sweep runs (leftover
  SU only) and under which design; the manuscript states two research questions,
  so either add it as an SI extension or drop RQ3 from the notes.

## Parked (scope, not blockers)

- [ ] Demand is Decree-capped and stationary (no demand-growth DU axis); one scoping
  sentence in the manuscript.
- [ ] Forward hooks only: per-realization diversion staging (`src/ensemble_prep.py`)
  and the deferred DU-factor presets (`src/ensembles.py`).
