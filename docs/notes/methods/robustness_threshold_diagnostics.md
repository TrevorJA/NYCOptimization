# Robustness Satisficing-Threshold Diagnostics (SI)

*Last updated: 2026-08-07. Procedure for placing the satisficing criterion
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
`objective_and_robustness_formulations.md`). **Pass 1 was run 2026-08-07
against the genuine status-quo cube (job 19729980; meta gate + equality check
passed) and the vector in §6 was ADOPTED into
`objectives_ensemble._DEFAULT_THRESHOLDS` the same day.**

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
$0.5/\sqrt{1000}$ ≈ ±1.6 pp (`scenario_design_methods.md` §5.4); every
reported SOW fraction additionally carries a Wilson 95% score interval
(n = 1,000 independent LHS draws), which reduces to that convention at
$p = 0.5$ and stays honest at the degenerate edges. The pooled realization
unit gets NO interval: its draws are correlated within SOWs, so a binomial
interval would overstate precision ~5× (`objective_definitions.md` §3.1).
Because $E_{\text{test}}$ is an LHS-designed DU box, a satisficing fraction is
a coverage-weighted count over the box, not a probability (Lamontagne et al.
2018).

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

## 0b. Placement rules (declared BEFORE the numbers; applied in §6)

An earlier pass-1 recommendation measured against a placeholder cube was
deleted, results and all, because a threshold vector is only meaningful
against the baseline distribution it is placed on. These are the rules the §5
run applies, in this order, and they were declared here BEFORE the numbers
existed so the placement could not be fitted to them. (They were applied to
the genuine pass-1 outputs on 2026-08-07; the resulting vector and its one
disclosed rule-2/rule-3 seam are recorded in §6.)

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
ECDF as a grey underlay; vertical references: current threshold (black, pass
side lightly shaded, SOW pass fraction in the panel title), historic-trace
anchor (dashed, its own color — never the fail color), NYC stakeholder floor
0.5 (`src/pareto_filter.py::DEFAULT_STAKEHOLDER_FLOORS`, dotted) and the
observed flood anchor (dash-dot). Axes are capped to the data support: a
reference beyond it appears as an edge chevron rather than stretching the
panel until the distribution is unreadable (its exact value is in the
tables). The threshold-margin-CDF presentation follows Gold et al. (2023,
Fig. 5), specified for this project in `framing_convention_diagnostics.md`
§3. The pooled underlay is what shows how much the within-SOW mean collapse
discards: the gap between the two curves is within-SOW natural variability,
and it is reported rather than assumed small. Figures carry no in-panel text
annotations; numeric companions: `rtd_sow_mean_summary.csv`,
`rtd_default_stringency.csv`.

## 2. Threshold sensitivity (`S_rtd_threshold_sensitivity`)

Satisficing fraction vs threshold per objective (dense natural-unit grid,
201+ points, defaults and candidates lying exactly on grid samples), x-axis
flipped for ≤-criteria so stringency increases rightward on every panel; the
SOW curve carries its Wilson 95% band and the degenerate zones (fraction
within `RTD_DEGENERACY_LIMIT` of 0/1) are shaded; the realization unit dashed
for unit-sensitivity. Candidate placements are marked by class (current /
historic anchor / floor & external anchor / SOW quantile), with the fractions
themselves in `rtd_candidate_placements.csv` — per §0b rule 4 the
distribution-feature quantiles stay on the figure as report-only markers. The
threshold-dependence presentation follows Hadjimichael et al. (2020); the
stringency coordinate in the tables (`rtd_default_stringency.csv`,
`rtd_threshold_sweep.csv`) uses the `compare_designs.default_stringency`
convention (fraction failing marginally: strict inequality side), so this
diagnostic and the step-10 cross-design sweep speak the same coordinate.

## 3. θ-attribution (`S_rtd_theta_spearman`, `S_rtd_factor_maps`)

Spearman rank correlations over the 1,000 SOWs between the 8 SOW-mean
objectives and the DU factors. The figure shows only the 8-objective × 3-θ
block — the question is which factors order performance, and the full
(8+3)×(8+3) matrix (objective–objective redundancy included) stays in
`rtd_theta_spearman.csv`. Pass/fail factor maps cover ALL 8 criteria in the
single informative plane (`RTD_FACTOR_MAP_PLANE`, m × r₁): the boundary is
near-monotone in m for every objective and r₂ is inert, so one plane per
criterion beats three planes for two criteria (Bryant & Lempert 2010
scenario-discovery factor mapping; a visual, single-policy analogue of the
`scenario_discovery.py` machinery). Each panel carries its pass fraction and
the m at which the local pass rate crosses 0.5 (`rtd_critical_m.csv`) — the
hydrologic reading of the placement: "this criterion fails once the mean-flow
change exceeds m*". A panel that is uniformly "fail" is the visual form of
the degeneracy the §0b rules screen for.

## 4. Historic-anchor comparison (`rtd_historic_anchor_comparison.csv`)

Each objective's recomputed base-metric anchor with its quantile position in
the SOW-mean distribution, an explicit `in_sow_support` flag with the
distance outside the support, and a near-historic consistency check: the
anchor's percentile within the `RTD_NEAR_HISTORIC_K` SOWs closest to θ = 0,
which guards the §0b rule-1 re-anchoring against generator bias (the historic
trace should be consistent with what the generator produces at near-zero
forcing change). Alongside sit the annual-unit CSV rows
(`metric_space = search_annual_csv`), a DIFFERENT metric space carried for
reference only — an annual-unit search objective is not comparable to the
weekly base metric the thresholds act on, and the table's `metric_space`
column exists to stop the two being read off one axis. Whether the historic
trace lies inside the $E_{\text{test}}$ SOW-mean support decides whether a
status-quo anchor is discriminating or degenerate, so the quantile position
of each anchor is the reported quantity.

**Flood anchor provenance (recorded 2026-08-07).** The external flood anchor
is the observed basin experience, WY2001–2023: 1.17 ft·days/yr
(`outputs/supplemental/flood_objective/tables/A_sim_vs_obs.csv`, row
`C4_max_ft`, column `obs`) — the revealed-tolerated level of realized
history, the §0b rule-2 goalpost. A former second anchor
("simulated_baseline" = 0.35) was REMOVED: it traced to
`outputs/baseline/ffmp_baseline_objectives.csv::downstream_flood_exceedance_annual`
(0.3467), an ANNUAL-UNIT search-space value — exactly the metric-space mixing
this table exists to prevent — and the correct base-metric simulated baseline
is already the anchor script's runtime-recomputed historic value.

## 4b. Conjunction and estimator diagnostics

Zero-simulation companions that answer the two questions the marginal
placements cannot:

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
- **Unit-collapse dispersion** (`rtd_unit_collapse.csv`). Per objective: the
  median within-SOW SD, the SE of the SOW-mean (σ/√25), the between-SOW SD,
  and their ratio — is the SOW-mean estimator precise enough that placement
  is not blurred by realization noise? — plus the SOW fraction at the current
  threshold under the risk-averse `worst` collapse beside the risk-neutral
  `mean` (the declared co-reported sensitivity, `objective_definitions.md`
  §3.1).
- **Critical-m boundaries** (`rtd_critical_m.csv`, overlaid on the factor
  maps). Licensed by the measured near-monotonicity in m; translates each
  threshold into the bottom-up, scenario-neutral coordinate.
- **Guardrail margins** (columns of `rtd_default_stringency.csv`): distance
  from each threshold to the worst SOW-mean, natural units and IQR multiples
  — the measured form of §0b rule 3's "kept, not re-tuned".

---

## 5. Run, then adopt (BOTH done 2026-08-07; record in §6)

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

## 6. Adopted vector (2026-08-07)

Pass 1: job 19729980 on the genuine cube; both gates passed. §0b rules applied
in order; adopted into `_DEFAULT_THRESHOLDS` (labels renamed) and mirrored in
`supplemental_config.RTD_RECOMMENDED_THRESHOLDS` / `RTD_RECOMMENDATION_BASIS`.

| Objective (kind) | Old | Adopted | SOW frac at adopted | Rule |
|---|---|---|---|---|
| NYC delivery reliability (≥) | 0.95 | **0.87** | 0.169 | 1 (anchor 0.8692) |
| NYC delivery deficit CVaR90 (≤) | 10.0 | **29.0** | 0.325 | 1 (anchor 29.17; ≈ sustained L4 depth) |
| NJ delivery reliability (≥) | 0.95 | **0.92** | 0.221 | 1 (anchor 0.9188) |
| Flood exceedance (≤) | 1.0 | **1.17** | 0.445 | 2 (observed WY2001–23) |
| NYC storage p5 (≥) | 25.0 | **26.0** | 1.000 | 2 (FFMP L5 boundary seasonal floor) |
| Montague flow reliability (≥) | 0.85 | 0.85 | 0.856 | kept (binding, non-degenerate) |
| Montague flow deficit CVaR90 (≤) | 25.0 | 25.0 | 1.000 | 3 (guardrail; re-anchor measured 0.055) |
| Trenton flow reliability (≥) | 0.85 | 0.85 | 1.000 | 3 (guardrail; anchor rounds to ill-posed 1.00) |

Disclosures carried into the SI text:

1. *Guardrails kept on measured evidence.* Re-anchoring the all-pass criteria
   at maintain-status-quo would flip Montague deficit to 0.055 and Trenton to
   0.142 — from non-binding past binding into near-degenerate, because the
   SOW-mean distributions are steep at the anchor (E_test is harsher than the
   observed record), and no intermediate placement exists that is not a
   distribution feature (rule 4). With all objectives rank-correlated
   |ρ| ≥ 0.91 through m, tightening them would only double-count mean-flow
   decline inside the conjunction. Their ranking-criticality for the policy
   population is framing diagnostic 3's question.
2. *The storage move is rule 2 applied after pass 1.* 25 → 26 replaces a round
   number with the FFMP drought-emergency (L5) boundary's seasonal floor (26%
   of combined capacity; pywrdrb `ffmp_reservoir_operation_daily_profiles.csv`,
   level5 min) — an institutional quantity independent of E_test, stricter, and
   still all-pass (worst SOW-mean ≈ 29.1). Disclosed as adopted after the
   pass-1 numbers were seen, not pre-registered.
3. *Near-historic caveat.* The historic-trace anchor lies outside the spread of
   the 25 nearest-to-θ=0 SOWs on all 8 objectives, mostly on the
   better-performance side: the generator's no-change worlds are harsher than
   the observed record, so rule-1 re-anchors err strict — the direction §0b
   requires.
4. *Snapshot guard.* Persisted `reeval_raw_meta.json` files keep the
   pre-adoption thresholds; the adopted vector becomes "current" in new metas
   on the next step-05/08 run. The stringency sweep (`compare_designs.py`)
   reads the live registry at run time, so adoption re-centers it
   automatically.
5. *The incumbent's joint Starr fraction is 0.000 at the ADOPTED vector too —
   and that is a finding, not a placement failure.* Every adopted marginal is
   non-degenerate (0.161–1.000), but the pass-2 conjunction shows a NEGATIVE
   co-occurrence gap (joint 0.000 < independence 0.004 < comonotone 0.161):
   the supply criteria pass only in wet worlds (m ≳ 0.15) while the flood
   criterion passes only in dry-to-neutral worlds (m ≲ 0.03), and for the
   status-quo FFMP the two acceptable regions do not intersect anywhere in the
   E_test box — flood is the modal SOLE failure (0.16 of SOWs) under the
   adopted vector. The motivation reading for the manuscript: no state of the
   world lets current operations be simultaneously acceptable on supply and
   flood terms; whether any candidate policy can create such a world is
   exactly what the campaign's Starr metric and the attainability screen test
   (and where satisficing saturates at 0, the incumbent-regret family carries
   the discrimination — `src/robustness.py`).

## Citations

Bryant & Lempert (2010); Gold et al. (2023); Hadjimichael et al. (2020);
Herman et al. (2014, 2015); Kasprzyk et al. (2013); Lamontagne et al.
(2018); Lempert & Collins (2007); McPhail et al. (2018, 2020); Quinn et al.
(2020); Starr (1962); Trindade et al. (2017); Zeff et al. (2014). Resolved
in `docs/notes/literature/objective_and_robustness_formulations.md`,
`scenario_subset_selection.md`, `dmuu_optimization_review.md`.
