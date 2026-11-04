# Task prompt: put the NYC demand regressions on the reconstruction inflow covariate, then measure the impact

You are a PhD-level water-resources engineer and Python developer working in the
`NYCOptimization` repository (Windows 11, Git Bash; Python at
`./venv/Scripts/python.exe`; Pywr-DRB is installed editable from `../Pywr-DRB/src`
on branch `nyc_opt`). Read `.claude/CLAUDE.md` first and obey its rules: the user
makes every git commit, no AI attribution anywhere, no backward-compatibility
wrappers, no value flags in scripts (identifiers only; settings live in
`supplemental_config.py` or env), type hints and Google-style docstrings on new
code, `tests/test_terminology.py` must pass, and preserve each file's line
endings. Run tests with `NYCOPT_SALINITY_ON=0 NYCOPT_TEMPERATURE_ON=0`.

## 1. The defect (measured 2026-11-03; do not re-derive)

Pywr-DRB `src/pywrdrb/pre/extrapolate_nyc_nj_diversions.py` fits four quarterly
OLS regressions of monthly NYC diversion on log monthly NYC inflow (River Master
record Dec 1999 to Nov 2021) and applies them to any flow record. The covariate is
defined inconsistently:

- Training (`load_training_data`) and the historic single-trace extrapolation use
  `NYC_inflow = sum of seven USGS gauges` (`pywrdrb.pywr_drb_node_data.obs_site_matches`
  for `nyc_reservoirs`: pepacton 01415000, 01414500, 01414000, 01413500;
  cannonsville 01423000, 0142400103; neversink 01435000). Mean 822 MGD over 1945 to 2023.
- The custom / ensemble path (`load_extrapolation_data` with a `flow_type`, and
  `ExtrapolatedDiversionEnsemblePreprocessor`) uses
  `NYC_inflow = sum of the three reservoir node inflows` (cannonsville, pepacton,
  neversink) of the supplied record. On the reconstruction
  `pub_nhmv10_BC_withObsScaled` that sum averages 1,317 MGD over the same window
  (ratio 1.61, mean log difference 0.516), and every synthetic ensemble is
  generated from that reconstruction.

With slopes of −204 / −123 / −122 / −109 MGD per unit ln inflow (DJF / MAM / JJA /
SON), every synthetic realization's demand sits 56 to 105 MGD per month below the
historic-record level: realized utilization on the local E_test fixture is 0.07 to
0.13 below nominal (mean −0.093, `outputs/synthetic_ensembles/etest_kn_10yr_n12/demand_utilization_realized_by_sow.csv`),
and the search ensembles at scale k = 1 realize about 0.60 of the 800 MGD
entitlement instead of the 0.70 the historic series realizes. The three scenario
designs therefore search at different demand levels (HIST on the gauge-based
historic CSV, MC and HF on the reservoir-inflow covariate), which confounds the
design comparison and the criteria anchoring.

Context from the previous task (`docs/prompts/demand_utilization_axis_prompt.md`,
all landed and uncommitted): `src/demand_utilization.py` calibrates the E_test
utilization axis u(k) on the historic reference extrapolation
(`reference_uncapped_series`, `scripts/supplemental/demand_utilization_calibration.py`),
the Pywr-DRB preprocessors carry `demand_scale` / `demand_scales`, and
`src/ensemble_prep.py::stage_nyc_demand` stages per-chunk scaled demand.

## 2. The fix (decided): one covariate, the reconstruction's reservoir inflow

Train and extrapolate on the same quantity everywhere: `NYC_inflow` = the sum of
the reconstruction dataset's inflows at the three NYC reservoir nodes
(`../Pywr-DRB/src/pywrdrb/data/flows/pub_nhmv10_BC_withObsScaled/`; 1945-01-01 to
2023-12-31), for the historic single-trace extrapolation AND the custom/ensemble
path. In Pywr-DRB (branch `nyc_opt`):

- `load_training_data` (NYC branch): read the training flow from the
  reconstruction dataset's reservoir-node columns, not the USGS gauge CSV. First
  verify which file the ensemble path sums (`stage_nyc_demand` passes
  `catchment_inflow_mgd.hdf5`; the reconstruction dir holds both
  `gage_flow_mgd.csv` and `catchment_inflow_mgd.csv`) and confirm that for the three
  headwater reservoirs catchment inflow equals gage flow, so training and
  extrapolation sum the identical quantity. Assert it in a test.
- `load_extrapolation_data` (NYC, no `flow_type`): extrapolate the historic series
  on the reconstruction's reservoir inflow over 1945-01-01 to 2023-12-31 (the HIST
  simulation window is 1945-12-01 to 2023-11-30, which this covers). The USGS record
  is no longer read by the NYC branch; keep NJ exactly as it is (NJ demand is a
  constant in this study and is not a study factor).
- Regenerate the shipped `src/pywrdrb/data/diversions/diversion_nyc_extrapolated_mgd.csv`
  with the new regressions (the cap stays at 800 MGD monthly mean; the draw seed
  stays the preprocessor's default). Keep a copy of the OLD file (git HEAD version)
  under the diagnostics folder below as `diversion_nyc_extrapolated_mgd_usgs_covariate.csv`
  before overwriting it.
- Report the refit per quarter (slope, intercept, R², residual SD) against the old
  values (−204 / −123 / −122 / −109; R² 0.33 / 0.27 / 0.52 / 0.36).
- Update the module docstring and the class docstrings; add a Pywr-DRB test that
  the training covariate and the custom-path covariate are the same quantity on
  the reconstruction record (same mean to 1e-9 on the overlap). `tests/test_demand_scale.py`
  and `tests/test_diversion_extrapolation.py` must still pass.

Then in NYCOptimization:

- Re-run `scripts/supplemental/demand_utilization_calibration.py` (the reference
  series is now the reconstruction-covariate extrapolation) and update every
  quoted calibration number: `src/demand_utilization.py` comments,
  `docs/notes/methods/forcing_parameterization.md` ("Demand-utilization axis"),
  `docs/research_project_summary.md`, manuscript §2.3 ("about 0.7 of the
  allocation") and §3.4.1, `docs/notes/methods/campaign_design.md` §5,
  `tests/test_demand_utilization.py::test_persisted_table_reproduces_the_reference_points`.
- Re-stage the local fixtures' demand (`kn_50yr_n5`; `etest_kn_10yr_n12` chunks via
  the slow test `tests/test_demand_utilization_staging.py`, which also re-appends
  and re-merges) and confirm realized u now sits within a few points of nominal
  with dry SOWs above and wet SOWs below.
- `TODO.md` §0: state in one sentence that every staged demand series and the
  historic CSV predate the covariate change (they are already listed as superseded).

## 3. Impact diagnostics (new folder `scripts/supplemental/demand_covariate/`)

Scripts `run.py` (simulate + tables) and `figures.py` (plots), settings in a
`DCV_*` block of `supplemental_config.py`, outputs under
`outputs/supplemental/demand_covariate/{tables,figures}/`, and a results note
`docs/notes/methods/demand_covariate_diagnostics.md` (current method and measured
numbers only, no decision trail). Follow the layout of an existing driver pair such
as `scripts/supplemental/flood_objective_run.py` / `flood_objective_figures.py`.
Both demand variants must be simulated with everything else identical: build the
model once and swap only the NYC demand series (the `demand_nyc` dataframe
parameter's source; see `src/simulation.py::_patch_model_dict` and
`_demand_options`), old = the saved USGS-covariate CSV, new = the regenerated CSV.

Deliverables, old versus new on the historic record 1945-12-01 to 2023-11-30:

1. **Demand series.** Annual-mean series, monthly climatology, utilization
   (mean / 800) over complete June-May FFMP years, distribution of monthly means,
   number of months capped at 800, and the share of days above the FFMP drought
   caps L3 / L4 / L5 (680 / 560 / 520 MGD); the same utilization numbers for one
   staged synthetic fixture at k = 1 before and after, so the HIST-versus-ensemble
   consistency is demonstrated, not asserted.
2. **NYC system outcomes** from the full-model single-trace simulation
   (`src.simulation.run_simulation_inmemory`, the step-05 baseline configuration,
   default FFMP policy): combined NYC storage, annual-minimum storage, days and
   distinct events in each FFMP drought zone (`res_level` / drought level results),
   NYC delivery and shortfall, Montague and Trenton target compliance.
3. **Figures.** Combined NYC storage with both demands over the 1960s drought
   (1961-06 to 1967-12) with the drought-zone boundaries drawn; a screen of the
   whole record on the storage difference between variants that lists every window
   whose difference exceeds a stated threshold, with a panel per such window (expect
   the 1980-82, 1985, 1995 and 2001-02 droughts); the two demand series' monthly
   climatology; and the demand difference time series. Manuscript style
   (`src.plotting.style.apply_manuscript_style`), exact numbers to CSVs, not panel text.
4. **Baseline objective scores.** The eight active annual-unit objectives
   (`config.get_objective_set()`, the step-05 computation in
   `scripts/main/run_baseline.py`) under both demands, with the difference
   expressed in units of each objective's ε from `src/objectives_ensemble.py`.

Close with a written summary for the user: every file touched in both repos, the
refit statistics, the utilization before and after on the historic record and on
the fixture, the objective-score table with ε ratios, which windows show material
storage differences, and anything in the results that argues against the fix.

## 4. Do not

- Change NJ demand, the objectives, the epsilons, the search ensembles' k = 1, the
  E_test streamflow, or the utilization bounds.
- Reintroduce a constant 800 MGD NYC demand anywhere, including diagnostics.
- Commit, or add AI attribution.
