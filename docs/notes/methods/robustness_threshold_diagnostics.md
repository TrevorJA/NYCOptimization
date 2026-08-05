# Robustness Satisficing-Threshold Diagnostics (SI)

*Last updated: 2026-08-05. Measured placement of the satisficing criterion
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
`objective_and_robustness_formulations.md`). The recommendation (§0b) is
MEASURED but NOT yet adopted into the registry — see the §5 checklist.*

---

## 0. Scope and cost model

**Question.** The baseline FFMP policy re-evaluated on the full test ensemble
$E_{\text{test}}$ (1,000 LHS θ-SOWs × 25 realizations × 50 yr;
`outputs/historic/ffmp_obj8_mm_full/reeval/etest_kn_50yr_n25000/baseline/`)
satisfies the current univariate criteria in fractions ranging from 0.0005
(NYC delivery reliability) to 1.0 (Trenton, Montague CVaR). Per objective:
is the current threshold *defensible* (→ the degenerate fraction is a
motivation result) or *misplaced for $E_{\text{test}}$ severity* (→ it voids
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

**Units.** The robustness unit is the SOW (Herman et al. 2014 lineage;
Trindade et al. 2017): the 25 realizations within each θ are collapsed with
the within-SOW mean (`collapse_within_sow`, the shipped risk-neutral default)
and fractions are counted over the 1,000 SOW-mean values. Every table
co-reports the pooled realization unit (n = 25,000); the two units differ by
at most 0.033 anywhere in the sweep. Worst-case Monte-Carlo standard error of
a SOW fraction is $0.5/\sqrt{1000}$ ≈ ±1.6 pp
(`scenario_design_methods.md` §5.4). Because $E_{\text{test}}$ is an
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
time) — PASSED for all 8 objectives.

---

## 0b. Measured verdicts (2026-08-05, recommendation measured)

Historic-trace anchor (base metrics, default FFMP policy, WY1946–2022):
NYC reliability **0.8692**, NYC deficit CVaR90 **29.17 %**, Montague
reliability 0.9522, Montague CVaR 4.62 %, Trenton reliability 0.9945, flood
exceedance 0.345 ft·d/yr, storage-p5 33.4 %, NJ reliability **0.9188**.

| Objective (kind) | Current | Frac (SOW) | Verdict | Recommended | Frac at rec. | Basis | Headline flag |
|---|---|---|---|---|---|---|---|
| NYC delivery rel. (≥) | 0.95 | 0.000 | **misplaced** — above the entire $E_{\text{test}}$ support (max SOW-mean 0.922) *and* above historic attainment 0.8692 | **0.87** | 0.161 | historic status-quo anchor, rounded stricter | **yes** |
| NYC deficit CVaR90 (≤) | 10.0 | 0.000 | **misplaced** — below the entire support (min 16.2); historic is 29.17 | **29.0** | 0.325 | historic status-quo anchor, rounded stricter | **yes** |
| Montague rel. (≥) | 0.85 | 0.856 | defensible — discriminating; historic (0.952) passes comfortably | 0.85 (keep) | 0.856 | regulatory-style value, discriminating | no |
| Montague CVaR90 (≤) | 25.0 | 1.000 | defensible guardrail — non-binding for the baseline (max 11.5) | 25.0 (keep) | 1.000 | non-binding guardrail | no |
| Trenton rel. (≥) | 0.85 | 1.000 | defensible guardrail (min 0.954) | 0.85 (keep) | 1.000 | non-binding guardrail | no |
| Flood exceedance (≤) | 1.0 | 0.383 | discriminating but arbitrary; external anchors exist | **1.17** | 0.445 | observed 2000–2023 flood burden (`flood_objective_diagnostics.md`) | no |
| NYC storage p5 (≥) | 25.0 | 1.000 | defensible operational floor (min 29.1) | 25.0 (keep) | 1.000 | operational floor | no |
| NJ delivery rel. (≥) | 0.95 | 0.020 | **misplaced** — above historic attainment 0.9188; near-degenerate | **0.92** | 0.214 | historic status-quo anchor, rounded stricter | **yes** |

**Decision rules applied (in order).**

1. *A criterion the historically-observed status quo itself fails is not an
   "acceptable-performance" line.* The accepted, operating FFMP policy on the
   observed record attains 0.869 / 29.2 / 0.919 on the three delivery
   criteria — the current 0.95 / 10.0 / 0.95 placements classify the
   observed system as never-acceptable, and they lie (NYC) entirely outside
   the $E_{\text{test}}$ support, zeroing the univariate fractions and the
   multivariate Starr metric (Starr 1962; Herman et al. 2015) for the
   baseline and plausibly for every candidate policy. These are re-anchored
   at maintain-status-quo-performance (Kasprzyk et al. 2013; Lempert &
   Collins 2007), rounded to the *stricter* side (0.8692→0.87, 29.17→29.0,
   0.9188→0.92) so rounding never flatters the baseline.
2. *An external goalpost beats a round number.* The flood criterion adopts
   the basin's realized 2000–2023 burden (1.17 ft·d/yr, the
   revealed-tolerated level; anchors recorded in
   `flood_objective_diagnostics.md`) over the arbitrary 1.0.
3. *All-pass guardrails are kept, not re-tuned.* Montague CVaR, Trenton, and
   storage are non-binding for the baseline; unlike all-fail marginals they
   do not distort the satisficing conjunction, and moving them onto features
   of the baseline's own $E_{\text{test}}$ distribution would be
   baseline-tuning — exactly what this note must avoid. They remain active
   for catastrophic failures of *candidate* policies; the framing-diagnostic-3
   OAT stringency sweep (open item, `framing_convention_diagnostics.md` §3)
   tests whether they are ranking-critical for the policy population.
4. *Distribution-feature candidates (SOW-mean P10/P50/P90) are reported in
   every table and figure but never adopted* — they are functions of the
   baseline's own distribution and would guarantee a chosen fraction by
   construction.

**Headline impact (flagged).** The three re-anchored delivery criteria move
the baseline's univariate fractions 0.000→0.161 (NYC rel.), 0.000→0.325
(NYC CVaR), 0.020→0.214 (NJ). The qualitative headline **"the status-quo FFMP
policy fails the NYC criteria in the large majority of tested futures"
survives** (fails NYC reliability in 84 % of SOWs, the deficit criterion in
67 %), and becomes *stronger*, not weaker, as an argument: the criterion is
now "merely maintain the performance the basin already experiences", and the
baseline cannot do that under most of the CMIP6-spanning DU box. What does
NOT survive is the vacuous form "0.05 % of futures" — that number reflected
a threshold outside the attainable range, not policy performance.

---

## 1. Distributions and anchors (`S_rtd_baseline_sow_cdfs`)

Per-objective ECDFs of the 1,000 SOW-means with the pooled 25,000-realization
ECDF as a grey underlay; vertical lines: current threshold, historic-trace
anchor (dashed), NYC stakeholder floor 0.5
(`src/pareto_filter.py::DEFAULT_STAKEHOLDER_FLOORS`, dotted) and the two
flood anchors. The threshold-margin-CDF presentation follows Gold et al.
(2023, Fig. 5), specified for this project in
`framing_convention_diagnostics.md` §3. The SOW-mean and pooled curves are
nearly coincident everywhere (within-SOW natural variability is small
relative to across-θ spread), which is why the within-SOW mean collapse
loses almost nothing here. Numeric companion: `rtd_sow_mean_summary.csv`.

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
`scenario_discovery.py` machinery). Findings (`rtd_theta_spearman.csv`):

- **The annual-mean factor $m$ dominates every objective**: |ρ| = 0.91–0.98
  for all 8. Supply-side objectives improve with wetter $m$; flood
  exceedance worsens (ρ = +0.91) — the basin's supply–flood tension is a
  *single-axis* tension in this DU box.
- **Seasonal amplitude $r_1$ is secondary** (|ρ| = 0.14–0.38; drier-summer
  shapes hurt supply and add flood), **$r_2$ is inert** (|ρ| ≤ 0.03 except
  flood 0.19).
- At the recommended thresholds the NYC pass region is the wet corner
  ($m \gtrsim 0.1$–0.15 log-units) with the boundary tilted by $r_1$; at the
  current thresholds the maps are uniformly "fail", which is the visual form
  of the degeneracy argument in §0b.

## 4. Historic-anchor comparison (`rtd_historic_anchor_comparison.csv`)

Each objective's recomputed base-metric anchor with its quantile position in
the SOW-mean distribution, alongside the annual-unit CSV rows
(`metric_space = search_annual_csv`) which are a different metric space and
carried for reference only (e.g. `montague_flow_reliability_annual` 0.789 is
the annual-unit search objective, NOT comparable to the weekly base
reliability 0.952 that the thresholds act on). The historic trace lies
INSIDE the $E_{\text{test}}$ SOW-mean support for all 8 objectives (quantile
positions from P1.4 for flood to P99.3 for Montague reliability) —
$E_{\text{test}}$ brackets the observed record from both sides rather than
being uniformly harsher, which is what makes the status-quo anchor
placements discriminating (fractions 0.16–0.34) instead of degenerate.

---

## 5. Adoption checklist (NOT yet implemented)

1. Paste the §0b recommended vector into
   `src/objectives_ensemble.py::_DEFAULT_THRESHOLDS` and update the
   threshold labels whose magnitudes changed
   (`nyc_delivery_reliability_weekly__sat95` → `__sat87`,
   `nyc_delivery_deficit_cvar90_pct__sat10` → `__sat29`,
   `downstream_flood_exceedance_minor__sat1` → `__sat1p17`,
   `nj_delivery_reliability_weekly__sat95` → `__sat92`), or set
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
5. Close the TODO §3 sweep-grid item; the sibling OAT-stringency /
   ranking-criticality item (framing diagnostic 3) stays open and now has
   its threshold vector.

## Citations

Bryant & Lempert (2010); Gold et al. (2023); Hadjimichael et al. (2020);
Herman et al. (2014, 2015); Kasprzyk et al. (2013); Lamontagne et al.
(2018); Lempert & Collins (2007); McPhail et al. (2018, 2020); Quinn et al.
(2020); Starr (1962); Trindade et al. (2017); Zeff et al. (2014). Resolved
in `docs/notes/literature/objective_and_robustness_formulations.md`,
`scenario_subset_selection.md`, `dmuu_optimization_review.md`.
