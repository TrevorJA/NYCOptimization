# Robustness Satisficing-Threshold Diagnostics (SI)

*Last updated: 2026-08-07. Procedure for placing the satisficing criterion
vector (`src/objectives_ensemble.py::_DEFAULT_THRESHOLDS`, PROVISIONAL pending
this diagnostic's re-run) against the status-quo FFMP baseline's persisted
$E_{\text{test}}$ re-evaluation cube of per-SOW ANNUAL-UNIT objective values.
Zero simulation: every deliverable reduces persisted artifacts. Code:
`scripts/supplemental/robustness_threshold_anchor.py` (historic-anchor
recompute → JSON cache) and
`scripts/supplemental/robustness_threshold_figures.py` (all tables + SI
figures), configuration in `supplemental_config.py` (`RTD_*`), wrapper
`workflow/supplemental/robustness_threshold_diagnostics.sh`, tests
`tests/test_robustness_threshold_diagnostics.py`. Outputs under
`outputs/supplemental/robustness_threshold_diagnostics/{cache,tables,figures}/`.
Citations author-year, resolved via `docs/notes/literature/` (primarily
`objective_and_robustness_formulations.md`).*

---

## 0. Scope and cost model

**Question.** For the baseline FFMP policy re-evaluated on the full test ensemble
$E_{\text{test}}$ (1,000 LHS θ-SOWs × 25 realizations × 50 yr;
`outputs/historic/ffmp_obj8_mm_full/reeval/etest_kn_50yr_n25000/baseline/`),
per objective: is the current threshold *defensible* (→ a degenerate fraction is
a motivation result) or *misplaced for $E_{\text{test}}$ severity* (→ it voids
the satisficing metric)? Zeff et al. (2014) elicited the Research-Triangle
lineage's thresholds once and every later value drifted by convention
(Trindade et al. 2017; Gold et al. 2023); the rule adopted here follows that
literature's diagnosis: anchor on an external goalpost wherever one exists
and treat everything else as a swept, reported choice, because robustness
rankings degrade in agreement as criteria tighten (Quinn et al. 2020;
Hadjimichael et al. 2020).

**Substrate.** The cube carries the per-SOW ANNUAL-UNIT objective values —
each θ-SOW's 25 realizations' unit-years pooled through the objective's own
§2 unit operator (`src/reeval_core.py::sow_objective_matrix`), i.e. the
search objectives recomputed per deeply-uncertain state. The satisficing
criteria act on those per-SOW values; one metric currency across search,
robustness, and regret.

**Inputs (all persisted; no simulation).** The raw cube
(`reeval_raw.parquet` + `reeval_raw_meta.json`, loader
`src/robustness.py::load_raw`); the $E_{\text{test}}$ DU forcing factors
(`outputs/synthetic_ensembles/etest_kn_50yr_n25000/forcing_profiles.npz`,
$\theta = [m, r_1, r_2]$, joined on the SOW label via
realization_id // R); the persisted historic baseline simulation
(`outputs/baseline/ffmp_baseline.hdf5`). Thresholds/kinds are read from the
cube's own meta snapshot, never the live registry — the moving-measuring-stick
guard (McPhail et al. 2020).

**Units.** The robustness unit is the SOW, and under the per-SOW substrate it
is the only counting unit: pooling each θ's 25 realizations' unit-years
through the unit operator IS the per-SOW value, so there is no separate
within-SOW collapse step and no realization-unit co-report. Worst-case
Monte-Carlo standard error of a SOW fraction is $0.5/\sqrt{1000}$ ≈ ±1.6 pp
(`scenario_design_methods.md` §5.4); every reported SOW fraction additionally
carries a Wilson 95% score interval (n = 1,000 independent LHS draws), which
reduces to that convention at $p = 0.5$ and stays honest at the degenerate
edges. Because $E_{\text{test}}$ is an LHS-designed DU box, a satisficing
fraction is a coverage-weighted count over the box, not a probability
(Lamontagne et al. 2018).

**One equivalence, stated openly.** The threshold-sensitivity curve *is* the
per-SOW value distribution: for a ≥-criterion the satisficing fraction at
threshold $t$ is the survival function $P(J_{\text{SOW}} \ge t)$, for a
≤-criterion it is the ECDF. The two SI figures present the same object with
different content: `S_rtd_baseline_sow_cdfs` adds the anchor lines;
`S_rtd_threshold_sensitivity` is the decision instrument — candidate
placements annotated with the fraction each one buys, stringency increasing
rightward on every panel.

**Historic-trace anchor (apples-to-apples).** The anchor script recomputes
the 8 annual-unit objective values from the baseline HDF5 through the same
loader + `obj.compute([data])` path (the single trace's unit-years pooled
through the unit operator — the N = 1 case of the cube's own formula), cached
at `cache/historic_anchor_annual_metrics.json`. The persisted
`outputs/baseline/ffmp_baseline_objectives.csv` is in the SAME metric space
and is carried in `rtd_historic_anchor_comparison.csv` as a cross-check
column on the anchor.

**Verification.** The figures script recomputes the SOW-mean of the cube per
objective and requires equality (atol 1e-9) with the ``sowmean__{objective}``
columns of the shipped `objectives_summary.csv` (derived from the same matrix
at persist time). This is a hard check inside the run, not a result to be
quoted here.

---

## 0b. Placement rules (declared BEFORE the numbers; applied in §6)

An earlier pass-1 recommendation measured against a placeholder cube was
deleted, results and all, because a threshold vector is only meaningful
against the baseline distribution it is placed on. These are the rules the §5
run applies, in this order, and they were declared here BEFORE the numbers
existed so the placement could not be fitted to them.

1. *A criterion the historically-observed status quo itself fails is not an
   "acceptable-performance" line.* Where the accepted, operating FFMP policy on
   the observed record cannot meet a criterion, that criterion classifies the
   observed system as never-acceptable and, if it also falls outside the
   $E_{\text{test}}$ support, zeroes both the univariate fraction and the
   multivariate Starr metric (Starr 1962; Herman et al. 2015) for the baseline
   and plausibly for every candidate policy. Such criteria are re-anchored at
   maintain-status-quo-performance (Kasprzyk et al. 2013; Lempert & Collins
   2007), rounded to the *stricter* side so rounding never flatters the
   baseline.
2. *An external goalpost beats a round number.* Where the basin's realized
   experience supplies a revealed-tolerated level (the flood criterion; anchors
   recorded in `flood_objective_diagnostics.md`), it is preferred to a round
   number chosen by convention.
3. *All-pass guardrails are kept, not re-tuned.* A criterion that is
   non-binding for the baseline does not distort the satisficing conjunction
   the way an all-fail marginal does, and moving it onto a feature of the
   baseline's own $E_{\text{test}}$ distribution would be baseline-tuning —
   exactly what this note must avoid. Guardrails remain active for catastrophic
   failures of *candidate* policies; the framing-diagnostic-3 OAT stringency
   sweep (open item, `framing_convention_diagnostics.md` §3) tests whether they
   are ranking-critical for the policy population.
4. *Distribution-feature candidates (SOW-mean P10/P50/P90) are reported in
   every table and figure but never adopted* — they are functions of the
   baseline's own distribution and would guarantee a chosen fraction by
   construction.

---

## 1. Distributions and anchors (`S_rtd_baseline_sow_cdfs`)

Per-objective ECDFs of the 1,000 per-SOW values; vertical references: current
threshold (black, pass side lightly shaded, SOW pass fraction in the panel
title), historic-trace anchor (dashed, its own color — never the fail color),
NYC stakeholder floor 0.5 (`src/pareto_filter.py::DEFAULT_STAKEHOLDER_FLOORS`,
translated to the annual name, dotted) and the observed flood anchor
(dash-dot). Axes are capped to the data support: a reference beyond it
appears as an edge chevron rather than stretching the panel until the
distribution is unreadable (its exact value is in the tables). The
threshold-margin-CDF presentation follows Gold et al. (2023, Fig. 5),
specified for this project in `framing_convention_diagnostics.md` §3.
Figures carry no in-panel text annotations; numeric companions:
`rtd_sow_summary.csv`, `rtd_default_stringency.csv`.

## 2. Threshold sensitivity (`S_rtd_threshold_sensitivity`)

Satisficing fraction vs threshold per objective (dense natural-unit grid,
201+ points, defaults and candidates lying exactly on grid samples), x-axis
flipped for ≤-criteria so stringency increases rightward on every panel; the
SOW curve carries its Wilson 95% band and the degenerate zones (fraction
within `RTD_DEGENERACY_LIMIT` of 0/1) are shaded. Candidate placements are
marked by class (current / historic anchor / floor & external anchor / SOW
quantile), with the fractions themselves in `rtd_candidate_placements.csv` —
per §0b rule 4 the distribution-feature quantiles stay on the figure as
report-only markers. The threshold-dependence presentation follows
Hadjimichael et al. (2020); the stringency coordinate in the tables
(`rtd_default_stringency.csv`, `rtd_threshold_sweep.csv`) uses the
`compare_designs.default_stringency` convention (fraction failing marginally:
strict inequality side), so this diagnostic and the step-10 cross-design
sweep speak the same coordinate.

## 3. θ-attribution (`S_rtd_theta_spearman`, `S_rtd_factor_maps`)

Spearman rank correlations over the 1,000 SOWs between the 8 per-SOW
objectives and the DU factors (theta joined on the SOW label, never
positional). The figure shows only the 8-objective × 3-θ block — the question
is which factors order performance, and the full (8+3)×(8+3) matrix
(objective–objective redundancy included) stays in `rtd_theta_spearman.csv`.
Pass/fail factor maps cover ALL 8 criteria in the single informative plane
(`RTD_FACTOR_MAP_PLANE`, m × r₁): the boundary is near-monotone in m for
every objective and r₂ is inert, so one plane per criterion beats three
planes for two criteria (Bryant & Lempert 2010 scenario-discovery factor
mapping; a visual, single-policy analogue of the `scenario_discovery.py`
machinery). Each panel carries its pass fraction and the m at which the local
pass rate crosses 0.5 (`rtd_critical_m.csv`) — the hydrologic reading of the
placement: "this criterion fails once the mean-flow change exceeds m*". A
panel that is uniformly "fail" is the visual form of the degeneracy the §0b
rules screen for.

## 4. Historic-anchor comparison (`rtd_historic_anchor_comparison.csv`)

Each objective's recomputed annual-unit anchor with its quantile position in
the per-SOW value distribution, an explicit `in_sow_support` flag with the
distance outside the support, and a near-historic consistency check: the
anchor's percentile within the `RTD_NEAR_HISTORIC_K` SOWs closest to θ = 0,
which guards the §0b rule-1 re-anchoring against generator bias (the historic
trace should be consistent with what the generator produces at near-zero
forcing change). The anchor and the cube live in ONE metric space (annual-unit
objective values), so the persisted
`outputs/baseline/ffmp_baseline_objectives.csv` value rides along as the
`baseline_csv_value` cross-check column. Whether the historic trace lies
inside the $E_{\text{test}}$ per-SOW support decides whether a status-quo
anchor is discriminating or degenerate, so the quantile position of each
anchor is the reported quantity.

**Flood anchor.** The external flood anchor is the observed basin experience,
WY2001–2023: 1.17 ft·days/yr
(`outputs/supplemental/flood_objective/tables/A_sim_vs_obs.csv`, row
`C4_max_ft`, column `obs`) — the revealed-tolerated level of realized
history, the §0b rule-2 goalpost. The simulated baseline is the anchor
script's runtime-recomputed historic value, never a hardcoded number.

## 4b. Conjunction diagnostics

Zero-simulation companions that answer the questions the marginal placements
cannot:

- **Starr conjunction decomposition** (`rtd_joint_satisficing.csv`,
  `rtd_failure_combinations.csv`, `rtd_failing_count_distribution.csv`,
  `S_rtd_conjunction`). The multivariate metric is a conjunction, so the
  marginals may not tell its story. The observed joint SOW fraction is
  bracketed by its two limiting benchmarks — the independence product
  ∏ᵢ(marginal fracᵢ) and the comonotone bound minᵢ(marginal fracᵢ) — plus
  the binding criterion, the most frequent failing-criteria combinations,
  and each criterion's sole- vs co-failure attribution (observed ≈ min says
  failures nest inside the binding criterion; observed ≈ product says they
  accumulate independently). Same construction as the regret-side
  `joint_vs_independent`; cites Starr (1962), Herman et al. (2015) —
  multivariate satisficing exposes stakeholder conflict — and Bonham et al.
  (2024) saturation. The joint fraction at the current (and, pass 2,
  recommended) vector is also the final row of
  `rtd_threshold_recommendation.csv`.
- **Critical-m boundaries** (`rtd_critical_m.csv`, overlaid on the factor
  maps). Licensed by the measured near-monotonicity in m; translates each
  threshold into the bottom-up, scenario-neutral coordinate.
- **Guardrail margins** (columns of `rtd_default_stringency.csv`): distance
  from each threshold to the worst per-SOW value, natural units and IQR
  multiples — the measured form of §0b rule 3's "kept, not re-tuned".

The former unit-collapse dispersion table is RETIRED with the per-SOW
substrate: pooling each SOW's unit-years through the unit operator IS the
per-SOW value, so there is no within-SOW collapse knob left to compare.

---

## 5. Run, then adopt (two-pass workflow)

**Pass 1 — measure.** Run `workflow/supplemental/robustness_threshold_diagnostics.sh`
against the REAL status-quo re-evaluation cube (step 05 `--reeval` on
$E_{\text{test}}$, at `RTD_REEVAL_BASELINE_DIR`). Confirm before trusting a
single number that the cube is that run and not a placeholder: its
`reeval_raw_meta.json` must carry `n_sow` = 1,000,
`realizations_per_sow` = 25 and `substrate` = `sow_annual_unit`, and the
wrapper's own SOW-mean equality check against `objectives_summary.csv` must
pass. Fill `RTD_RECOMMENDED_THRESHOLDS` / `RTD_RECOMMENDATION_BASIS` in
`supplemental_config.py` (ANNUAL objective names) by applying the §0b rules
to the measured placements — the rules first, then the numbers.

**Pass 2 — adopt.**

1. Paste the recommended vector into
   `src/objectives_ensemble.py::_DEFAULT_THRESHOLDS` and rename every threshold
   label whose magnitude changed — the labels carry the value in the suffix
   (`<annual name>__sat<thr>`), so a changed threshold is a changed label. Or
   set `NYCOPT_SAT_THRESHOLDS` for env-scoped runs.
2. Already-persisted `reeval_raw_meta.json` files keep their snapshotted
   thresholds BY DESIGN (the McPhail guard); rescoring existing cubes under
   the new vector must pass `thresholds=` explicitly to
   `score_robustness`, or re-run step 08 after the registry edit so new
   metas snapshot the adopted values.
3. Re-run `tests/test_objectives_ensemble.py` + `tests/test_robustness.py`
   (named files, compute node).
4. Re-run this diagnostic's wrapper once after adoption so the "current"
   markers in the SI figures show the adopted vector (anchor cache makes it
   ~1 min).
5. Set the stringency-sweep grid centre/span around the adopted vector, and
   close the TODO item; the sibling OAT-stringency / ranking-criticality item
   (framing diagnostic 3) stays open and then has its threshold vector.

## 6. Threshold vector status

The pass-1 re-run against the status-quo E_test cube on the annual-unit
substrate completed 2026-08-08 and the vector was ADOPTED into
`objectives_ensemble._DEFAULT_THRESHOLDS` with values unchanged (recorded in
TODO §1 and snapshotted per run in `reeval_raw_meta.json`). The adopted
all-axes conjunction is retained as the `reference_all8` criterion set only;
see §7 for its role after the subset-criteria redesign.

**PENDING PRODUCTION RESULTS.** The 2026-08-08 incumbent cube and every
interim artifact behind §7 predate the 2026-08-18 seasonal-rotation
regeneration (TODO §1). Pass 1 re-runs on the regenerated incumbent cube, and
the §7 re-anchoring audit re-runs on the regenerated production full cube,
before any placement in this note is treated as final.

## 7. Re-anchoring under subset criteria

*(Every number in this section is an interim 200-SOW, pre-regeneration
measurement; see the §6 pending flag.)*

The interim 200-SOW re-evaluation showed the adopted all-8 conjunction is
degenerate as an analysis criterion: joint Starr = 0.0 for every design and
for the incumbent (binding axes Trenton reliability, whose 0.87 placement
excludes ~90% of pooled per-SOW cells, and Montague reliability, whose 0.79
anchor lies outside the incumbent's own E_test support, max 0.746), with
`nyc_delivery_deficit_p99_pct`, `nj_delivery_reliability_annual`, and
`montague_flow_deficit_p99_pct` saturated non-discriminators (pass > 0.96
everywhere). The analysis criteria were therefore rebuilt as Quinn et al.
(2017)-style SUBSETS (`src/satisficing_criteria.py`): each named set
thresholds 1–3 objectives and leaves every other axis unconstrained, and
robustness is reported under several sets with a cross-set ranking-agreement
check. The §0b placement rules apply per member axis:

- **Rule 1 re-anchors** (status quo fails the criterion, rounded stricter):
  Trenton reliability 0.87 → 0.75 (incumbent median year 0.73); NYC storage
  P1 26% → 13.0% (incumbent median year 12.9%; the 26% FFMP L5 goalpost
  remains reported in the univariate decomposition, never conjoined);
  Montague reliability 0.79 → 0.50 (incumbent per-SOW median 0.482, rounded
  stricter at ε = 0.05; transcribed 2026-08-13 from the audit table).
- **Rule 2 external goalposts carried**: flood exceedance 1.17 ft·d/yr
  (observed WY2001–2023); NYC delivery reliability 0.65 (historic anchor,
  discriminating).
- The measured evidence behind every placement is the audit table
  `outputs/comparison/{slug}/{tag}/criteria_reanchoring.csv`
  (`scripts/supplemental/criteria_reanchoring.py`): incumbent per-SOW
  quantiles, incumbent pass fraction at the adopted threshold, pooled
  stringency, and the stricter-side epsilon-granularity round of the
  incumbent median. Placements are transcribed into
  `src/satisficing_criteria.py` as literals citing that CSV; where epsilon
  granularity is coarse (storage, ε = 5), the transcribed literal may use a
  finer decimal step with the rationale stated in the set's docstring entry.
- `reference_all8` (the adopted snapshot conjoined over every axis) is
  retained as a reported reference; its zero is the motivating degeneracy
  finding, never a selection criterion.

## Citations

Bryant & Lempert (2010); Gold et al. (2023); Hadjimichael et al. (2020);
Herman et al. (2014, 2015); Kasprzyk et al. (2013); Lamontagne et al.
(2018); Lempert & Collins (2007); McPhail et al. (2018, 2020); Quinn et al.
(2020); Starr (1962); Trindade et al. (2017); Zeff et al. (2014). Resolved
in `docs/notes/literature/objective_and_robustness_formulations.md`,
`scenario_subset_selection.md`, `dmuu_optimization_review.md`.
