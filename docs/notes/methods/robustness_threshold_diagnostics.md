# Robustness Satisficing-Threshold Diagnostics (SI)

*Last updated: 2026-08-06. Procedure for placing the satisficing criterion
vector (`src/objectives_ensemble.py::_DEFAULT_THRESHOLDS`, shipped as
placeholders "pending the satisficing-criterion diagnostics") against the
status-quo FFMP baseline's persisted $E_{\text{test}}$ re-evaluation cube.
Zero simulation: every deliverable reduces persisted artifacts. Code:
`scripts/supplemental/robustness_threshold_anchor.py` (historic-anchor
recompute → JSON cache) and
`scripts/supplemental/robustness_threshold_figures.py` (all tables + SI
figures), configuration in `supplemental_config.py` (`RTD_*`), wrapper
`workflow/supplemental/robustness_threshold_diagnostics.sh`, tests
`tests/test_robustness_threshold_diagnostics.py`. Outputs under
`outputs/supplemental/robustness_threshold_diagnostics/{cache,tables,figures}/`.
Citations author-year, resolved via `docs/notes/literature/` (primarily
`objective_and_robustness_formulations.md`). **The diagnostic has NOT yet been
run against a valid cube; there is no recommendation. Run it per §5.***

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

**Inputs (all persisted; no simulation).** The raw cube
(`reeval_raw.parquet` + `reeval_raw_meta.json`, loader
`src/robustness.py::load_raw`); the $E_{\text{test}}$ DU forcing factors
(`outputs/synthetic_ensembles/etest_kn_50yr_n25000/forcing_profiles.npz`,
$\theta = [m, r_1, r_2]$); the persisted historic baseline simulation
(`outputs/baseline/ffmp_baseline.hdf5`). Thresholds/kinds are read from the
cube's own meta snapshot, never the live registry — the moving-measuring-stick
guard (McPhail et al. 2020).

**Units.** The robustness unit is the SOW — the project-wide adopted unit
(`objective_definitions.md` §3.1): the 25 realizations within each θ are
collapsed with the within-SOW mean (`collapse_within_sow`, the shipped
risk-neutral default) and fractions are counted over the 1,000 SOW-mean values.
Every table co-reports the pooled realization unit (n = 25,000) as the unit
sensitivity. Worst-case Monte-Carlo standard error of a SOW fraction is
$0.5/\sqrt{1000}$ ≈ ±1.6 pp (`scenario_design_methods.md` §5.4). Because $E_{\text{test}}$ is an
LHS-designed DU box, a satisficing fraction is a coverage-weighted count over
the box, not a probability (Lamontagne et al. 2018).

**One equivalence, stated openly.** The threshold-sensitivity curve *is* the
SOW-mean distribution: for a ≥-criterion the satisficing fraction at
threshold $t$ is the survival function $P(\bar{v}_{\text{SOW}} \ge t)$, for a
≤-criterion it is the ECDF. The two SI figures present the same object with
different content: `S_rtd_baseline_sow_cdfs` adds the pooled-realization
underlay (within-SOW variability, justifying the mean collapse) and the
anchor lines; `S_rtd_threshold_sensitivity` is the decision instrument —
candidate placements annotated with the fraction each one buys, stringency
increasing rightward on every panel.

**Historic-trace anchor (apples-to-apples).** The persisted
`outputs/baseline/ffmp_baseline_objectives.csv` is in the ANNUAL-UNIT search
space and cannot anchor the base-metric thresholds. The anchor script
recomputes the 8 whole-trace base metrics from the baseline HDF5 through the
same loader + `objective.base.compute` path the re-eval cube uses
(byte-identical metric definitions), cached at
`cache/historic_anchor_base_metrics.json`. The annual CSV is carried in
`rtd_historic_anchor_comparison.csv` with an explicit `metric_space` column,
reference only.

**Verification.** The figures script recomputes the realization-unit
fraction at the current thresholds and requires equality (atol 1e-9) with the
shipped `objectives_summary.csv` (written by `SatisficingAgg` at re-eval
time). This is a hard check inside the run, not a result to be quoted here.

---

## 0b. Placement rules (declared; NOT yet applied)

No recommendation exists. A pass-1 recommendation measured against a
placeholder cube was deleted, results and all, because a threshold vector is
only meaningful against the baseline distribution it is placed on. These are
the rules the run in §5 must apply, in this order, and they are declared here
BEFORE the numbers exist so the placement cannot be fitted to them.

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

Per-objective ECDFs of the 1,000 SOW-means with the pooled 25,000-realization
ECDF as a grey underlay; vertical lines: current threshold, historic-trace
anchor (dashed), NYC stakeholder floor 0.5
(`src/pareto_filter.py::DEFAULT_STAKEHOLDER_FLOORS`, dotted) and the two
flood anchors. The threshold-margin-CDF presentation follows Gold et al.
(2023, Fig. 5), specified for this project in
`framing_convention_diagnostics.md` §3. The pooled underlay is what shows how
much the within-SOW mean collapse discards: the gap between the two curves is
within-SOW natural variability, and it is reported rather than assumed small.
Numeric companion: `rtd_sow_mean_summary.csv`.

## 2. Threshold sensitivity (`S_rtd_threshold_sensitivity`)

Satisficing fraction vs threshold per objective (dense natural-unit grid,
201+ points, defaults and candidates lying exactly on grid samples), x-axis
flipped for ≤-criteria so stringency increases rightward on every panel;
candidates annotated with their fractions; the realization unit dashed for
unit-sensitivity. The threshold-dependence presentation follows Hadjimichael
et al. (2020); the stringency coordinate in the tables
(`rtd_default_stringency.csv`, `rtd_threshold_sweep.csv`) uses the
`compare_designs.default_stringency` convention (fraction failing
marginally: strict inequality side), so this diagnostic and the step-10
cross-design sweep speak the same coordinate.

## 3. θ-attribution (`S_rtd_theta_spearman`, `S_rtd_factor_maps_nyc`)

Spearman rank correlations over the 1,000 SOWs between the 8 SOW-mean
objectives and the DU factors, plus pass/fail factor maps in the three
θ-planes for the two NYC criteria (Bryant & Lempert 2010 scenario-discovery
factor mapping; a visual, single-policy analogue of the
`scenario_discovery.py` machinery). Reported per objective from
`rtd_theta_spearman.csv`: which DU factors order the SOW-mean performance, and
where the pass/fail boundary falls in each θ-plane at the current thresholds
and at whatever the §0b rules recommend. A criterion whose factor map is
uniformly "fail" is the visual form of the degeneracy those rules screen for.

## 4. Historic-anchor comparison (`rtd_historic_anchor_comparison.csv`)

Each objective's recomputed base-metric anchor with its quantile position in
the SOW-mean distribution, alongside the annual-unit CSV rows
(`metric_space = search_annual_csv`) which are a DIFFERENT metric space and
carried for reference only — an annual-unit search objective is not comparable
to the weekly base metric the thresholds act on, and the table's
`metric_space` column exists to stop the two being read off one axis.
Whether the historic trace lies inside the $E_{\text{test}}$ SOW-mean support
decides whether a status-quo anchor is discriminating or degenerate, so the
quantile position of each anchor is the reported quantity.

---

## 5. Run, then adopt (NEITHER done)

**Pass 1 — measure.** Run `workflow/supplemental/robustness_threshold_diagnostics.sh`
against the REAL status-quo re-evaluation cube (step 05 `--reeval` on
$E_{\text{test}}$, at `RTD_REEVAL_DIR`). Confirm before trusting a single
number that the cube is that run and not a placeholder: its
`reeval_raw_meta.json` must carry `n_sow` = 1,000 and
`realizations_per_sow` = 25, and the wrapper's own equality check against
`objectives_summary.csv` must pass. Fill `RTD_RECOMMENDED_THRESHOLDS` /
`RTD_RECOMMENDATION_BASIS` in `supplemental_config.py` by applying the §0b
rules to the measured placements — the rules first, then the numbers.

**Pass 2 — adopt.**

1. Paste the recommended vector into
   `src/objectives_ensemble.py::_DEFAULT_THRESHOLDS` and rename every threshold
   label whose magnitude changed — the labels carry the value in the suffix
   (`<base>__sat<thr>`), so a changed threshold is a changed label. Or set
   `NYCOPT_SAT_THRESHOLDS` for env-scoped runs.
2. Already-persisted `reeval_raw_meta.json` files keep their snapshotted
   thresholds BY DESIGN (the McPhail guard); rescoring existing cubes under
   the new vector must pass `thresholds=`/`kinds=` explicitly to
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

## Citations

Bryant & Lempert (2010); Gold et al. (2023); Hadjimichael et al. (2020);
Herman et al. (2014, 2015); Kasprzyk et al. (2013); Lamontagne et al.
(2018); Lempert & Collins (2007); McPhail et al. (2018, 2020); Quinn et al.
(2020); Starr (1962); Trindade et al. (2017); Zeff et al. (2014). Resolved
in `docs/notes/literature/objective_and_robustness_formulations.md`,
`scenario_subset_selection.md`, `dmuu_optimization_review.md`.
