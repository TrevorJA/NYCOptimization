# Flood-Objective Definition Diagnostics (SI)

*Last updated: 2026-08-03. Specification and measured results of the diagnostics
that decide the flood objective's definition: the incumbent any-gauge day count
at NWS minor flood stage (`downstream_flood_days_minor`) against
magnitude-weighted exceedance candidates on stage and flow bases. One local
simulation pass (34 policies × the historic trace + the `kn_50yr_n5` KN
stationary fixture, trimmed model, ~34 simulation-minutes) plus zero-simulation
re-scorings of the completed Pywr-DRB flood-gauge diagnostic experiment
(`../Pywr-DRB/experiments/nyc_flood_gauge_diagnostics/`, post-fix 2000–2023
run). No simulation campaign is launched. Terminology per
`docs/notes/terminology.md`. The verdicts below are the RECOMMENDATION from
this diagnostic; the objective registry is unchanged until it is adopted, and
any adoption reopens the §1 artifact-regeneration chain (epsilons → problem
JARs) already open for the 2026-07-31 delivery-factor bounds change.*

---

## 0. Scope and cost model

Candidates, all minimized, all normalized by the metric-window length in years
(the `_flood_days_anygauge` convention; §2 annual units are within-water-year
sums), stage in ft from the model's rating-curve parameters, thresholds from
`pywrdrb.flood_thresholds`:

| # | Definition | Units |
|---|---|---|
| C1 | days/yr any tail gauge ≥ NWS minor flood stage (**incumbent**) | days/yr |
| C2 | days/yr any tail gauge ≥ FFMP cautionary ("action") stage (reference only) | days/yr |
| C3 | Σ over gauges and days of (stage − minor)⁺ | gauge-ft-days/yr |
| C4 | Σ over days of the max-across-gauges (stage − minor)⁺ | ft-days/yr |
| C5 | C3 with each gauge scaled by 1/(major − minor) | 1/yr |
| C6 | Σ over gauges and days of (Q − Q_minor)⁺, Q_minor by rating-curve inversion (7,593 / 17,090 / 4,842 MGD) | MG/yr |

| # | Diagnostic | Input artifact | New simulation | Deliverable |
|---|---|---|---|---|
| 1 | Staleness audit + re-stage | staged flood-augmented inflows (content-checked) | none (re-stage only) | manifest in the run cube |
| 2 | Sim-vs-obs / regulatory | gauge experiment's post-fix 2000–2023 output + observed flows | none | table + SI figure |
| 3 | Rating-curve exposure | flood-day records from the policy cube | none | table + SI figure |
| 4 | Resolution / discriminating power | policy cube (baseline + 24 feasible-uniform policies) | one 34-policy pass (`flood_objective_run.py`) | table + SI figure; **decisive** |
| 5 | Monotone-response gate | policy cube (9-point flood-release ladder) | same pass | table + SI figure; **the gate** |
| 6 | Ensemble noise + epsilon | policy cube unit-years, bootstrap | none | tables |

The simulation pass is `scripts/supplemental/flood_objective_run.py` (cube +
manifest under `outputs/supplemental/flood_objective/cube/`); every reduction
is `scripts/supplemental/flood_objective_figures.py` (tables/figures under
`outputs/supplemental/flood_objective/{tables,figures}/`). Configuration in
`supplemental_config.py` (`FLOODOBJ_*`).

Manuscript-SI illustrations of the recommended metric (default FFMP policy,
historic trace + stationary KN baseline; no candidate comparison):
`scripts/supplemental/flood_severity_baseline_figures.py` →
`S_flood_severity_event_anatomy` (how ft·days accumulate through a flood
event: per-gauge stage relative to its own NWS flood stage, the worst-gauge
envelope, daily increments), `S_flood_severity_annual_series` (water-year
severity across both domains with the objective value marked), and
`S_flood_severity_return_period` (unit-year severity vs return period; the
zero-inflated, episodic structure the objective integrates). Wiring anchors: the run's C1
machinery reproduces the registered objective exactly (0.196076 days/yr,
trimmed historic baseline, matched to 1e-12), and its sim-vs-obs block
reproduces the gauge experiment's `summary.csv` aggregate exactly (0.4167
days/yr).

## 0b. Measured verdicts (2026-08-03, recommended)

- **Flood objective — REPLACE the day count with C4, Σ over days of the
  max-across-gauges (stage − minor)⁺ [ft-days/yr].** C4 keeps the regulatory
  NWS flood-stage definition (FFMP2017 Appendix A §6.iv–vi: 11 / 13 / 13 ft),
  sits 2 / 2 / 1 ft above the control rule's cautionary-stage discontinuity,
  and is the only candidate class that passes every screen below. Proposed
  names `downstream_flood_severity_minor` (§1) /
  `downstream_flood_severity_annual` (§2); the incumbent count stays a
  registered diagnostic.
- **Staleness audit — one stale file found and re-staged.** Checked by content
  (the post-fix construction fixes Hale-Eddy/Fishs-Eddy local inflows at the
  same donor, so their column-mean ratio must equal 0.337379): the historic
  CSV (`pub_nhmv10_BC_withObsScaled`, mtime 2026-07-31 15:09) is post-fix
  (ratio exact); the `kn_50yr_n5` flood HDF5 (mtime 2026-07-23 20:09) was
  PRE-FIX (mean Hale Eddy inflow 13.4 MGD ≈ 2% of physical; ratio 1.013) and
  was re-staged with `force=True` on 2026-08-03 (now 143.6 MGD, ratio exact).
  Its presimulated releases and predicted inflows derive from the base inflow
  file (`src/ensemble_prep.py`) and were left in place. Every flood number in
  this note postdates the fix.
- **Resolution — the incumbent count is DEGENERATE on the historic trace;
  severity restores full resolution.** Across 25 baseline+feasible-uniform
  policies C1 takes 9 distinct values on the historic trace (14.7% of policy
  pairs tied; 9 occupied ε-boxes at its shipped ε) and 18 on the KN ensemble;
  C3–C6 take 25/25 distinct values with zero ties in both domains (16–17
  ε-boxes at the block-6 epsilons). The suspected count-degeneracy is
  confirmed on the historic trace and partially mitigated (not removed) by
  ensemble pooling.
- **Monotone-response gate — severity PASSES.** Ramping all six
  `flood_release_scale_*` DVs over a feasible 9-point ladder (common
  per-reservoir multiplier for both zones, 0.5 → the L1a upper bound):
  more-aggressive flood releases reduce downstream flooding (storage drawdown
  dominates the added release), and on the ensemble C4 responds strictly
  monotonically (ρ_S = −1.00, 0 reversals, largest single step 27% of range;
  C3/C6 −0.98, C5 −1.00). C1 responds in the same direction (ρ_S = −0.95)
  but in integer shelves — a 4-rung plateau followed by steps of 33–50% of
  its range — exactly the no-gradient pathology.
- **Rating-curve exposure — NONE measured; the flow basis solves a
  non-problem.** Across 3,556 flood gauge-days (34 policies × both domains)
  zero days exceed any gauge's rated maximum discharge; headroom is ≥ 2.8×
  in discharge and ≥ 8 ft in stage (max simulated stage 15.9 ft vs 23.9 ft
  rated at Hale Eddy). The endpoint-saturation fallback in
  `flood_stage.py` never engages at flood-relevant flows, so stage-ft is a
  safe integration basis and C6's motivation dissolves.
- **Sim-vs-obs — severity tracks observed flood magnitude at least as well as
  the count, and C4 is the honest basis.** Annual-series Pearson r vs
  observed (WY2001–2023, stage from flow through the shared curves on both
  sides): C4 0.91 (best), C5 0.89, C3 0.88, C6 0.87, C2 0.86, C1 0.83;
  Spearman 0.61–0.71; every candidate places 3 of the 5 worst observed flood
  years in its own top 5. Whole-window sim/obs ratios: C2 1.13, C4 0.82, C1
  0.77, C5 0.60, C3 0.53, C6 0.47. The model NEVER floods two gauges on the
  same day (simulated C3 ≡ C4) while observations do — the gauge-summed bases
  (C3/C5/C6) inherit that structural miss as extra low bias, the max-gauge
  basis C4 does not.
- **Ensemble noise — the severity integral is ~1.6× noisier and more
  tail-concentrated than the count; disclosed, not disqualifying.** Baseline
  policy, 245 pooled unit-years (N=5 × 50 yr): zero-unit fraction 0.89 for
  every minor-based candidate (same event set); the largest single unit-year
  carries 28–30% of the severity integral vs 14% of the count;
  bootstrap-over-realizations CV of the ensemble mean 0.29–0.31 (severity)
  vs 0.19 (count). At the campaign composition (N=100 × 10 yr, 900 pooled
  unit-years) both shrink; the K=3 draws bound draw-to-draw spread.
- **Epsilon — C4 ε ≈ 0.01 on both scales (provisional).** max(IQR/10,
  granularity) over the policy sample: 0.0085 ft-days/yr (§1 historic scale)
  and 0.0103 ft-days per unit-year (§2 pooled-annual scale) — the analogue of
  the incumbent's 0.02 / 0.05. Evaluation at fixed ensemble is deterministic
  (objective-determinism check 2026-07-29), so sampling noise across draws is
  a design property, not an ε input. Final value comes from the §1
  epsilon-calibration rerun (512-policy population, new bounds), which the
  swap joins at zero extra cost.
- **C2 (cautionary stage) — REJECTED for the active set** despite the best
  calibration ratio (1.13) and lowest noise (CV 0.06): 9 / 11 / 12 ft is the
  FFMP control trigger, not a flood definition — an objective there sits its
  own discontinuity on the switching boundary of
  `NYCFloodRelease` / `NYCCombinedReleaseFactor` and measures "days
  discharge-mitigation is curtailed", not flood harm. Stays a registered
  diagnostic.

---

## 1. Staleness audit and targeted re-stage

**Purpose.** Every flood metric here depends on
`catchment_inflow_with_flood_nodes_mgd.{csv,hdf5}`; files staged before the
2026-07-31 flood-node inflow fix carry local catchments at ~2% of physical
magnitude, and `ensemble_prep` skips staging silently when the file exists.

**Method.** Content check, not mtime: the post-fix redistribution
(`flood_node_inflow_fractions()`) makes the Hale-Eddy/Fishs-Eddy column-mean
ratio exactly 0.1824/0.5407 = **0.337379** in any post-fix file. The run
script verifies both files it consumes and re-stages the ensemble file with
`FloodNodeInflowEnsemblePreprocessor(force=True)` when the check fails; a
stale historic CSV aborts (that file is regenerated by the sibling repo's own
preprocessor). Presimulated releases and predicted inflows derive from the
base inflow file and are not invalidated.

**Gates / output.** The audit manifest (paths, mtimes, measured/expected
ratios, action) persists in `flood_run_manifest.json`; findings in §0b.

**Cost.** Seconds; the one re-stage was ~1 minute for 5 realizations.

---

## 2. Sim-vs-obs / regulatory relevance

**Purpose.** Score each candidate as a measurement of observed flood harm:
absolute bias, annual-magnitude tracking, and whether it ranks the bad flood
years correctly (the ranking property matters more than absolute calibration,
since the ~0.56 minor-day input ceiling is un-closable by construction).

**Method.** Zero simulation: the gauge experiment's post-fix default-policy
output (`output_floodgauge_2000_2023.hdf5`) and observed flows, both converted
flow → stage through the shared rating curves via the experiment's own
`diagnostics.py` helpers (curve artifacts cancel; the experiment window is
used without the 6-month exclusion to match its `summary.csv`). Reported per
candidate: whole-window sim/obs ratio, annual-series (complete water years
WY2001–2023) Pearson and Spearman, and the top-5 observed-flood-year overlap.

**Gates / output.** `A_sim_vs_obs.csv` + the annual-series figure; feeds the
§0b basis choice (max-gauge vs gauge-sum) via the simulated C3 ≡ C4 identity.

**Cost.** Zero simulation; seconds.

---

## 3. Rating-curve exposure

**Purpose.** A severity metric weights the stage tail far more than a count
does, and `StageFromDischargeParameter` saturates at the rated-range endpoint
— quantify how often and how far simulated flood-day discharges leave the
rated range, which is also the stage-vs-flow-basis evidence.

**Method.** Every flood gauge-day record in the cube (union of stage- and
flow-basis exceedances, all 34 policies, both domains) against each gauge's
rated maximum; plus the per-policy per-realization flow/stage maxima
regardless of flooding.

**Gates / output.** `B_rating_curve_exposure.csv` + the stage-vs-discharge
figure. Zero exceedances → stage-ft basis adopted; C6 (flow basis) loses its
rationale.

**Cost.** Zero simulation; seconds.

---

## 4. Resolution / discriminating power

**Purpose.** The decisive screen: an objective that assigns many policies the
identical value gives the MOEA no gradient (and its ε-box archive no
occupancy). Confirm or refute the incumbent's suspected near-degeneracy.

**Method.** Baseline + 24 constraint-feasible uniform policies
(`sample_feasible_dvs`, seed 19), each evaluated on the historic trace (§1
whole-window scale) and the KN fixture (§2 pooled-annual mean over 245
unit-years). Per candidate × domain: distinct values, tie-pair fraction,
range, IQR, and occupied ε-boxes at the block-6 epsilons; plus the
cross-candidate Spearman matrix (severity candidates are mutually ρ_S ≥ 0.99;
severity vs count 0.85–0.92 — same objective family, finer resolution).

**Gates / output.** `C_resolution.csv`, `C_candidate_spearman.csv`, the strip
figure and agreement heatmap. Verdict in §0b.

**Cost.** The 34-policy pass (~34 simulation-minutes, laptop) shared with §5.

---

## 5. Monotone-response gate

**Purpose.** The pre-stated adoption gate: a metric that does not respond
monotonically (and without cliffs) to the NYC levers that should move
downstream flooding is useless as a search objective however well it
calibrates.

**Method.** A 9-point ladder ramps all six `flood_release_scale_*` DVs with a
common per-reservoir multiplier for both flood zones from 0.5 to the L1a
upper bound (the common multiplier preserves the effective L1b ≤ L1a ordering
elementwise, so every rung is feasible under the formal flood constraint —
ramping each zone over its own bounds is not). Per candidate × domain:
Spearman ρ of value vs ladder fraction, direction reversals against the
dominant trend, and the largest single-step share of the response range.

**Gates / output.** `D_monotone_response.csv` + the response figure. The
historic trace is reported but not gating (single-trace responses are
sub-integer for the counts and ~3% relative for severity); the ensemble
column carries the verdict in §0b.

**Cost.** Shared with §4.

---

## 6. Ensemble sampling noise and epsilon

**Purpose.** A severity integral can be dominated by one extreme event;
quantify tail concentration and estimator noise rather than assuming them,
and propose a defensible ε on each candidate's scale.

**Method.** Baseline policy's 245 pooled unit-years: zero-unit fraction,
top-unit share of the integral, and bootstrap (B = 1,000, resampling the 5
realizations) SE/CV of the ensemble mean. ε = max(IQR/10 across the §4 policy
sample, metric granularity) on both scales; granularity is 1/76 (historic
units) and 1/245 (pooled units) for the counts, none for the continuous
integrals.

**Gates / output.** `E_ensemble_noise.csv`, `F_epsilon_proposal.csv`, the
unit-distribution figure. The N=5 fixture bounds these estimates from above;
they are noise disclosures, not campaign-scale predictions.

**Cost.** Zero simulation; seconds.

---

## 7. Adoption checklist (when the recommendation is ratified)

The swap is cheap in code and expensive only in artifacts already queued for
regeneration:

1. `src/objectives.py` — register the §1 metric (`_flood_severity_anygauge`
   core beside `_flood_over_stage_daily`; ε provisional 0.01 ft-days/yr);
   keep `downstream_flood_days_minor` registered as a diagnostic.
2. `src/objectives_ensemble.py` — annual severity function beside
   `_flood_days_minor_annual`; new `_ANNUAL_REGISTRY_SPEC` row (PooledMean,
   worst_value from 366 × the per-day cap-free maximum is unbounded — use the
   §2 convention of the count row scaled by a generous stage bound, e.g.
   366 × 15 ft-days), `_BASE_TO_ENSEMBLE` entry, `_REGISTRY_SPEC` sat row,
   and a `downstream_flood_severity_minor__sat1` threshold (placeholder
   pending the satisficing-criterion diagnostics; the observed 2000–2023
   value is 1.17 ft-days/yr, the simulated baseline 0.35).
3. `config.py` — swap the entry in `_DEFAULT_OBJECTIVES` (the set stays 8;
   slugs stay `*_obj8`); `src/plotting/style.py` labels.
4. Reopened by the swap (already open for the ±0.15 bounds change, so zero
   extra cost if batched): the epsilon-calibration rerun → problem JARs
   (TODO §1), and the step-05 baseline objective vector.

---

## Citations

| Diagnostic element | Citation(s) |
|---|---|
| Magnitude/severity beats binary exceedance counts for flood objectives | Quinn et al. 2017 |
| Saturated / tied criteria give the search no gradient | Bonham et al. 2024 |
| Rank-stability screening of metric variants | Herman et al. 2015; McPhail et al. 2018 |
| Redundant-metric retention by rank correlation | Olden & Poff 2003 |
