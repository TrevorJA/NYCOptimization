# Task prompt: add the NYC demand-utilization factor to the E_test re-evaluation

You are a PhD-level water-resources engineer and Python developer working in the
`NYCOptimization` repository (Windows 11, Git Bash; Python at
`./venv/Scripts/python.exe`; Pywr-DRB is installed editable from `../Pywr-DRB/src`
on branch `nyc_opt`; the stochastic generator package `scengen` is installed from
`../NYCOptimization_scenario_generation`). Read `.claude/CLAUDE.md` first and obey
its rules: the user makes every git commit, no AI attribution anywhere, no
backward-compatibility wrappers, no value flags in workflow scripts (identifiers
only; everything else comes from env files and registries), type hints and
Google-style docstrings on new code, and `tests/test_terminology.py` must pass.
Several files use CRLF line endings; when you edit with scripts, preserve the
file's existing line endings.

## 0. Read these before writing code

| What | Where |
|---|---|
| Project state, pipeline, E_test definition | `docs/research_project_summary.md` (section "The re-evaluation ensemble (E_test)") |
| E_test construction, sizing, cost | `docs/notes/methods/campaign_design.md` §5, `src/etest.py` |
| Climate forcing space (the existing 3 DU axes) | `docs/notes/methods/forcing_parameterization.md` |
| How NYC demand is modeled today | `config.py` (block starting "NYC and NJ interbasin-diversion demands"), `src/ensemble_prep.py::stage_nyc_demand`, Pywr-DRB `src/pywrdrb/pre/extrapolate_nyc_nj_diversions.py` |
| How SOWs are keyed and persisted | `src/reeval_core.py::sow_grouping`, `reeval_raw_meta`; `src/ensemble_generation.py` (search for `forcing_profiles.npz` and `_write_chunk_meta`) |
| Where DU coordinates are consumed | `src/factor_mapping.py::theta_features`, `src/plotting/forcing_space.py::load_etest_sample` and `panel_parameter_space`, `scripts/main/factor_mapping_run.py`, `scripts/main/scenario_discovery.py` |
| Subset staging of the re-evaluated 500 SOWs | `scripts/supplemental/make_etest_subset.py` |
| Objectives (how demand enters the NYC delivery objectives) | `docs/notes/methods/objective_definitions.md` §0 and the entitlement `min(demand, allowance)` in `src/objectives.py` |
| Open items created by the demand decision | `TODO.md` §0b |

## 1. What already exists (decided and implemented on 2026-11-03)

- NYC demand is the time-dynamic extrapolated diversion series everywhere.
  `config.nyc_demand_source(is_ensemble)` returns `"custom"` for ensembles (a
  per-realization `diversion_nyc_extrapolated_mgd.hdf5` staged by step 04 through
  `src/ensemble_prep.py::stage_nyc_demand`) and `"historical"` for the single trace
  (Pywr-DRB's regenerated 1945 to 2026 historic-flow CSV). A constant 800 MGD NYC
  demand is never used; `config.assert_time_dynamic_nyc_demand` refuses it at every
  model build and `tests/test_demand_source.py` guards it. Do not reintroduce it,
  not even as a diagnostic.
- NJ demand is the constant 100 MGD Decree quantity (`config.NJ_DEMAND_SOURCE`,
  Pywr-DRB option `nj_demand_source`). It is not a study factor.
- The Pywr-DRB extrapolation model: for each quarter (DJF, MAM, JJA, SON) an OLS
  regression of monthly mean Delaware diversion on log monthly NYC inflow, fitted
  to the 2000 to 2021 River Master record. Slopes are negative in every quarter
  (dry months draw more). Each extrapolated month is the regression prediction
  plus a Gaussian residual draw, disaggregated to days by nearest-neighbour month
  matching, and finally every month whose mean exceeds 800 MGD is scaled down to
  an 800 MGD mean (`enforce_max_mean_demand`). The ensemble class
  `ExtrapolatedDiversionEnsemblePreprocessor` seeds the residual draws per
  realization from `(seed, flow_type, realization_id)`.
- Search ensembles use the unscaled series (scale 1.0), whose nominal utilization
  on the historic reference record is 0.69 to 0.70 of the 800 MGD entitlement.
  Search is not changed by this task.
- E_test as staged on Anvil: 1,000 LHS states of the world (SOWs) over the three
  climate axes `(m, r1, r2)` × 25 realizations × 50 yr, in 50 chunk directories
  `etest_kn_50yr_n25000__chunk{000..049}` of 500 realizations each, plus the
  metadata-only prefix subset `etest_kn_50yr_n25000_first25ch` (the campaign
  re-evaluates its 500 SOWs). Realization `k` belongs to SOW `p = k // 25`; chunk
  directories hold local keys `0..499` and record `global_realization_ids` in their
  `_meta.json`. The parent and the subset hold `forcing_profiles.npz` with keys
  `realization_ids`, `mean_factor_a`, `theta_params`, `theta_param_names`,
  `n_forcing_profiles`, `realizations_per_profile`, and others. The streamflow of
  E_test is final: do not regenerate it, and do not touch its presimulated
  releases.

## 2. The design to implement

Add one deeply uncertain factor to E_test: the **NYC entitlement utilization**
`u`, the average fraction of the 800 MGD allocation that NYC calls on in a state of
the world. Definitions, all of which are fixed decisions:

1. **Range.** `u ∈ [0.45, 1.0]`. The floor is just below the lowest River Master
   year on record (378 MGD in 2005-06 and 2018-19, 0.47). The ceiling is the legal
   cap; nothing above 1.0 is admissible.
2. **Sampling.** One Latin hypercube column appended to the existing 1,000-point
   design, one value per SOW in SOW order `p = 0..999`, drawn with
   `scipy.stats.qmc.LatinHypercube(d=1, scramble=True, seed=...)` scaled to the
   range. The seed comes from `scengen.seeds.design_seed(SEED_ROOT, "etest:kn:demand", 0)`
   (`SEED_ROOT` is in `src/scenario_designs.py`); check whether `scengen.seeds`
   enforces membership in its `SEED_DOMAINS` and add the domain there if it does.
   Appending an independent stratified column keeps the joint design a Latin
   hypercube, so no streamflow is regenerated. The prefix subset inherits the first
   500 values.
3. **Mechanism: scale, then cap.** `u` is implemented through a multiplicative scale
   `k(u)` applied to the whole regression-structured demand series of each
   realization (predictions and residual draws alike, so the seasonal shape, the
   flow elasticity and the coefficient of variation are preserved), and the 800 MGD
   monthly-mean cap is applied **after** scaling. Never cap first and scale second.
4. **Calibration of `k(u)`.** `u` is the nominal utilization on the reference
   record: generate the uncapped historic-flow extrapolation once
   (`ExtrapolatedDiversionPreprocessor(loc="nyc", max_mean_demand=<very large>)`;
   its draws are seeded, so it is deterministic), and for a dense grid of `k`
   (about 0.3 to 6.0) compute `u(k) = mean(cap_800(k · D_ref)) / 800`, where
   `cap_800` is the monthly-mean cap. Invert the monotone table with a shape-
   preserving interpolant (PCHIP) to get `k(u)`. Clip at `k_max = 6.0`
   (`u ≈ 0.996`) and record a clipped flag; `u = 1.0` is the flat-800 limit.
   Reference points measured on the pre-regeneration record, for sanity only:
   `k = 0.8 → u = 0.56`, `1.0 → 0.69`, `1.2 → 0.80`, `1.5 → 0.90`, `2.0 → 0.96`,
   `3.0 → 0.99`.
5. **Realized utilization differs from nominal by design.** Dry SOWs realize a few
   percentage points above nominal through the negative regression slopes, and
   stochastic draws add scatter. Keep that coupling; report realized `u` per SOW as
   a diagnostic, never force it to equal nominal.
6. **Search stays at `k = 1`.** Only E_test gets the factor. The comparison between
   scenario designs therefore tests generalization from the model's stationary
   demand level to the full utilization range in both directions.

## 3. Implementation, in order

### 3.1 Pywr-DRB (`../Pywr-DRB`, branch `nyc_opt`)

File `src/pywrdrb/pre/extrapolate_nyc_nj_diversions.py`:

- Add a `demand_scale: float = 1.0` attribute to `ExtrapolatedDiversionPreprocessor`
  and, in `process()`, multiply the diversion columns of `self.processed_data`
  (`pepacton`, `cannonsville`, `neversink`, `aggregate` for NYC; `D_R_Canal` for NJ)
  by it immediately **before** the call `self.enforce_max_mean_demand(...)` near the
  end of `process()` (search for `keep_dates`). Leave the cap itself unchanged.
- Give `ExtrapolatedDiversionEnsemblePreprocessor` an optional
  `demand_scales: dict[str, float] | None` constructor argument keyed by
  realization id (string keys, as the class stores ids). In its `process()` loop
  set `self.demand_scale = demand_scales.get(str(realization_id), 1.0)` before
  `super().process()`. Record the applied scale as a dataset `demand_scale` in each
  realization group written by `save()`, next to `aggregate`.
- Pitfall to assert against: the NYC branch of `process()` pastes the observed
  Dec 1999 to Nov 2021 diversions over any realization dates that fall inside that
  window and extrapolates only outside it. Synthetic date axes start at 1945-12-01
  and E_test's 50-yr windows end in 1995, so there is no overlap today; add an
  explicit check in the ensemble class that raises if a realization's date axis
  intersects the observed window, so a future longer window cannot silently inherit
  observed demand.
- Add a Pywr-DRB test under `tests/` that scales a tiny synthetic series by 1.5 and
  checks both the pre-cap multiplication and that no month exceeds the cap.

### 3.2 NYCOptimization: the utilization module and its calibration

- New module `src/demand_utilization.py` with: the constants
  `DEMAND_UTILIZATION_BOUNDS = (0.45, 1.0)`, `DEMAND_SCALE_MAX = 6.0`,
  `DEMAND_UTILIZATION_SEED_DOMAIN = "etest:kn:demand"`; `build_calibration_table()`
  (runs the uncapped reference extrapolation and returns a `DataFrame` of `k`, `u`);
  `k_from_u(u)` and `u_from_k(k)` (vectorized, PCHIP on the persisted table, with the
  clip and the clipped flag); `sample_utilization(n_sow, seed)` (the LHS column);
  and `realized_utilization(series)` (mean / 800 over complete FFMP years, June
  through May, matching the objectives' unit).
- Persist the table at `outputs/diagnostics/demand_utilization/demand_utilization_calibration.csv`
  through a small driver `scripts/supplemental/demand_utilization_calibration.py`
  (also writes a figure of `u(k)` and the implied monthly climatology at
  `u ∈ {0.45, 0.7, 0.9, 1.0}` to the SI tier via `src/figures/registry.py`). The
  table is regenerable, gitignored output; `k_from_u` must fail loudly if it is
  missing rather than fall back to an internal guess.

### 3.3 E_test metadata: carry `u` and `k` per SOW

- New script `scripts/supplemental/append_demand_utilization.py --variant kn`
  (identifier flag only) that, for the variant's parent slug: reads
  `forcing_profiles.npz`, draws the LHS column for `n_forcing_profiles` SOWs,
  computes `k` and the clipped flag, and writes back the npz with the new
  per-realization arrays `demand_utilization`, `demand_scale_k` and
  `demand_scale_clipped` (each value repeated `realizations_per_profile` times,
  exactly as `theta_params` is stored) plus scalar provenance
  (`demand_utilization_bounds`, `demand_utilization_seed`, `demand_scale_max`,
  `demand_calibration_sha256`). Store these as separate keys: do **not** append `u`
  to `theta_params`, because `assert_box_contains` and the forcing-space figure
  check `theta_params` against the harmonic box. Also write the same provenance
  into the parent `_meta.json` and into every chunk's `_meta.json`, and write a
  `demand_utilization_nominal.csv` (`sow`, `u`, `k`, `clipped`) in the parent dir.
  The script must be idempotent (same seed, same values) and refuse to run on a
  slug whose npz already carries the keys unless `NYCOPT_FORCE=1`.
- Re-run `scripts/supplemental/make_etest_subset.py` for the prefix subset after
  appending; its slicer copies any per-realization or per-SOW array, so verify the
  subset npz carries the three new keys with the right lengths.
- Extend `src/etest.py`: constants mirrored from `src/demand_utilization.py` (one
  source of truth; import, do not duplicate), and extend `assert_staged_etest_contract`
  so a staged E_test whose npz or meta lacks the utilization keys, or records other
  bounds, is refused with a message that names `append_demand_utilization.py`.

### 3.4 Staging the scaled demand

- Extend `src/ensemble_prep.py::stage_nyc_demand` so that, when the spec is an
  E_test chunk (its `_meta.json` has `kind == "forcing_pool_chunk"`; the parent slug
  is `pool_slug`), it maps each local realization to its global id
  (`global_realization_ids`), to its SOW (`global_id // realizations_per_profile`),
  and to that SOW's `k` from the parent's `demand_utilization_nominal.csv`, and
  passes the resulting `demand_scales` to the preprocessor. For every other
  ensemble (search ensembles, the local fixtures) pass no scales, which means
  `k = 1`. Fail loudly if a chunk's parent has no utilization table.
- After staging a chunk, compute each realization's realized utilization from the
  written HDF5 and append it to `demand_utilization_realized.csv` in the chunk dir
  (`global_id`, `sow`, `u_nominal`, `k`, `u_realized`). Add a merge helper (in the
  same supplemental script or a sibling) that concatenates the chunk tables into
  the parent dir and reports nominal versus realized `u` per SOW.
- `workflow/supplemental/stage_nyc_demand_etest_chunks.sh` already exists and
  stages one chunk per array task; it needs no interface change.

### 3.5 Analysis plumbing

- `src/plotting/forcing_space.py::load_etest_sample`: return `u` (one value per
  SOW, strided like `theta`) when the key is present.
- `src/factor_mapping.py::theta_features`: return four columns
  `["em", "r1", "r2", "u"]` when `u` is present; otherwise the three it returns
  today. Check every caller (`scripts/main/factor_mapping_run.py`,
  `scripts/main/scenario_discovery.py`, `tests/test_factor_mapping.py`) for a
  hard-coded three.
- `src/reeval_core.py::reeval_raw_meta`: add `sow_demand_utilization` (nominal `u`
  keyed by SOW label) when the re-eval ensemble carries it, so the persisted cube is
  self-describing.
- Subset re-scoring: the manuscript promises hazard-restricted and
  envelope-restricted re-scoring of the persisted matrix (Text S10). Find where
  that is implemented or planned (`TODO.md` §3, `src/robustness.py`,
  `scripts/main/compare_designs.py`, `scripts/main/plot_etest_hazard_overlay.py`);
  implement a shared SOW-mask re-scoring path if none exists, and add
  utilization bands `u < 0.6`, `0.6 ≤ u < 0.8`, `u ≥ 0.8` to it. Add to the
  regret family a per-objective regret and gain summary against the current FFMP
  policy binned by `u` (ten equal-width bins), which is the analysis that shows how
  much of any re-optimization gain is contingent on demand.
- Figures (register in `src/figures/registry.py`): an SI panel of the sampled
  `(e^m, u)` and `(r1, u)` planes with the box, an SI figure of nominal versus
  realized `u`, and the regret-versus-`u` figure. Figure 3 of the manuscript keeps
  its three climate panels; mention `u` in its caption.

### 3.6 Tests

- New `tests/test_demand_utilization.py`: `u(k)` is monotone and bounded in
  `(0, 1]`; `k_from_u(u_from_k(k)) ≈ k` inside the unclipped range; `u = 1.0` maps
  to `DEMAND_SCALE_MAX` with the clipped flag; the LHS column has one value per
  stratum and is reproducible from the seed.
- Extend `tests/test_etest.py` with a staged fixture (`_meta.json` +
  `forcing_profiles.npz`, as the existing tests do) that round-trips the three new
  keys through `make_etest_subset.py`'s slicer, `sow_grouping` and
  `theta_features`, and checks `assert_staged_etest_contract` refuses an npz
  without them.
- A staging test on the local fixture `etest_kn_10yr_n12` (12 realizations, 3
  chunks of 4, already staged under `outputs/synthetic_ensembles/`): append `u`,
  rebuild the local subset `etest_kn_10yr_n12_first2ch`, stage one chunk with
  scales, and assert that realizations of a higher-`u` SOW have higher mean demand
  and that no month exceeds 800. Mark it `slow` if it takes more than a minute.
- Retire the phrases "samples no demand factor" and "no demand-growth axis" in
  `tests/test_terminology.py` with the replacement text, and fix every hit.

### 3.7 Documentation (state the current method only, no decision trail)

- `docs/notes/methods/forcing_parameterization.md`: new section "Demand-utilization
  axis" (definition, range, sampling, scale-then-cap, calibration, realized versus
  nominal, why `u ≤ 1`, why search stays at `k = 1`).
- `docs/notes/methods/campaign_design.md` §5: the E_test table gains the factor; the
  "Why 500" row now argues four factors (the cited precedents cover 5 to 14 factors
  at 1,000 to 2,000 SOWs and 3 at 500, so the count stands).
- `docs/research_project_summary.md`: the E_test paragraph ("the full range of the
  deeply-uncertain climate-forcing factors") and the "Decided" paragraph.
- `docs/manuscript/Amestoy_NYC_reoptimization_manuscript_draft.md` §3.4.1: replace
  "the re-evaluation ensemble samples no demand factor" with the definition of `u`
  and two sentences of justification; update "three-dimensional amplitude space" to
  the four-factor design; the Discussion "Scope conditioning" bullet; the Figure 3
  caption. Justification to draw on: demand multipliers are the dominant robustness
  control in the deep-uncertainty re-evaluations of Herman et al. (2014), Trindade
  et al. (2017), Gold et al. (2022, 2023), all sampled uniformly and jointly with
  hydrology and all including decreases; Hadjimichael et al. (2020) apply the
  transbasin multiplier to a rights-capped maximum, the closest precedent to a
  utilization factor bounded above by the Decree; Kolesar and Serio (2011) argue the
  FFMP's design premise of a full 800 MGD diversion overstates drought days; the
  River Master record brackets realized utilization at 0.47 to 0.83 of the
  entitlement over 2000 to 2026; and the FFMP writes its curtailments against the
  allocation, so the drought-protection value of the curtailment variables scales
  with `u`.
- `docs/terminology.md`, row "well-characterized uncertainty (WCU)": the scope
  statement "demand growth is not included as an uncertainty" becomes: demand
  growth above the Decree cap is excluded by law; the utilization of the entitlement
  is the fourth deeply uncertain factor of the re-evaluation.
- `workflow/README.md`: one sentence in the E_test paragraph on
  `append_demand_utilization.py` and the order of operations below.
- `TODO.md`: delete the §0b item for this task when it lands; keep the
  re-verification items.

### 3.8 Order of operations on Anvil (write this into `workflow/README.md`)

1. Pull `NYCOptimization`, `Pywr-DRB` and `NYCOptimization_scenario_generation`.
2. `python scripts/supplemental/demand_utilization_calibration.py` (seconds).
3. `python scripts/supplemental/append_demand_utilization.py --variant kn`
   (metadata only, seconds), then `python scripts/supplemental/make_etest_subset.py`
   for `etest_kn_50yr_n25000_first25ch`.
4. `sbatch workflow/supplemental/stage_nyc_demand_etest_chunks.sh` (50 tasks; the
   local measurement is about 1.3 s per 10-yr realization on one rank, so expect a
   few minutes per 500-realization chunk on 33 ranks), then the realized-`u` merge.
5. Re-run the current-FFMP baseline cube (step 09 with
   `NYCOPT_CHUNK_POLICIES=baseline`, about 66 SU); the stored cube predates the
   demand change.
6. Re-run the satisficing-threshold diagnostic and the criteria re-anchoring on the
   new baseline cube (`docs/notes/methods/robustness_threshold_diagnostics.md`).

Re-evaluation cost per policy is unchanged (demand is a parameter lookup), and no
streamflow is regenerated.

## 4. Invariants and things not to do

- Never use a constant 800 MGD NYC demand anywhere, including diagnostics.
- Do not scale NJ demand, and do not add a basin consumptive-use factor (modeled
  consumptive use upstream of Trenton is about 20 MGD; non-consumptive withdrawals
  return to the node by construction).
- Do not change the search ensembles' demand (`k = 1`), the candidate pools, the
  hazard-filling selection, the objectives, or the epsilons in this task.
- Do not regenerate E_test streamflow or its presimulated releases; only metadata,
  the demand HDF5s, and the baseline cube change.
- Do not modify Borg sources, and do not commit.

## 5. Acceptance checklist

- [ ] All tests pass: `./venv/Scripts/python.exe -m pytest tests -q -m "not slow"`
      with `NYCOPT_SALINITY_ON=0 NYCOPT_TEMPERATURE_ON=0`, plus the new slow staging
      test once.
- [ ] Local end-to-end on `etest_kn_10yr_n12`: append, subset, stage one chunk with
      scales, simulate it through `src.simulation.run_simulation_ensemble_inmemory`,
      and confirm simulated `ibt_demands.demand_nyc` equals the staged series and
      orders by nominal `u`.
- [ ] `theta_features(tag)` returns four columns for an appended E_test and three
      for a legacy one; the factor-mapping and scenario-discovery scripts run on the
      local fixture.
- [ ] `assert_staged_etest_contract` refuses a staged E_test without the keys.
- [ ] Docs and manuscript updated as in 3.7; `tests/test_terminology.py` passes.
- [ ] A short summary for the user listing every file touched in both repositories
      and the Anvil order of operations.

## 6. Reference numbers

| Quantity | Value |
|---|---|
| Quarterly regression slopes (MGD per unit ln inflow), DJF / MAM / JJA / SON | −204 / −123 / −122 / −109 |
| Quarterly R² | 0.33 / 0.27 / 0.52 / 0.36 |
| Historic-flow extrapolation 1945 to 2026, capped, mean | 556 MGD, u = 0.70 |
| Observed 2000 to 2021 mean | 522 MGD, u = 0.65 |
| River Master and DRBC years 2000 to 2026 | 378 to 667 MGD, median about 505 |
| Share of days demand exceeds the L3 / L4 / L5 caps (680 / 560 / 520 MGD) at u = 0.69 | 0.35 / 0.56 / 0.60 |
| Same at u = 0.90 | 0.66 / 0.78 / 0.80 |
| NJ extrapolated series mean | 87 MGD, 0.87 of 100 |
