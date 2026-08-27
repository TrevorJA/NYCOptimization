# Robustness Satisficing-Threshold Diagnostics (SI)

Places the satisficing criteria against the status-quo FFMP policy's persisted
$E_{\text{test}}$ re-evaluation cube of per-SOW annual-unit objective values.
Zero simulation. Code: `scripts/supplemental/robustness_threshold_anchor.py`
(historic-anchor recompute, JSON cache),
`scripts/supplemental/robustness_threshold_figures.py` (tables and SI figures),
`scripts/supplemental/criteria_reanchoring.py` (the per-axis re-anchoring
audit on the production cube), configuration `RTD_*` in
`supplemental_config.py`, wrapper
`workflow/supplemental/robustness_threshold_diagnostics.sh`, tests
`tests/test_robustness_threshold_diagnostics.py`. Outputs under
`outputs/supplemental/robustness_threshold_diagnostics/{cache,tables,figures}/`.

## 0. Question and substrate

For the incumbent re-evaluated on $E_{\text{test}}$ (1,000 LHS SOWs × 25
realizations × 50 yr, step-05 `--reeval`, `RTD_REEVAL_BASELINE_DIR`), per
objective: is a criterion defensible, so that a degenerate fraction is a
motivation result, or misplaced for $E_{\text{test}}$ severity, so that it
voids the satisficing metric? Zeff et al. (2014) elicited the Research-Triangle
thresholds once and later values drifted by convention (Trindade et al. 2017;
Gold et al. 2023), so the rule here anchors on an external goalpost wherever
one exists and treats everything else as a swept, reported choice, because
robustness rankings degrade in agreement as criteria tighten (Quinn et al.
2020; Hadjimichael et al. 2020). The campaign re-evaluates the leading 500 SOWs
(`etest_kn_50yr_n25000_first25ch`), a strict prefix of this cube, so the
anchors transfer without recomputation.

The substrate is each SOW's realizations' unit-years pooled through the
objective's own unit operator (`src/reeval_core.py::sow_objective_matrix`).
The SOW is the only counting unit and there is no separate within-SOW collapse.
Inputs are `reeval_raw.parquet` plus `reeval_raw_meta.json`
(`src/robustness.py::load_raw`), the $E_{\text{test}}$ forcing profiles
(`forcing_profiles.npz`, $\theta = [m, r_1, r_2]$ joined on the SOW label) and
the historic baseline HDF5. Thresholds and kinds come from the cube's own meta
snapshot, never the live registry (the moving-measuring-stick guard, McPhail
et al. 2020). The worst-case SE of a SOW fraction is $0.5/\sqrt{1000} \approx$
1.6 pp here and 2.2 pp on 500 SOWs, and every fraction carries a Wilson 95 %
interval. Because $E_{\text{test}}$ is an LHS box, a fraction is a
coverage-weighted count, not a probability (Lamontagne et al. 2018). The
figures script requires the cube's SOW means to equal the shipped
`objectives_summary.csv` columns (atol 1e-9).

## 1. Placement rules (declared before the numbers)

1. A criterion the operating status quo itself fails on the observed record is
   not an acceptable-performance line. It classifies the observed system as
   never acceptable and, if it also lies outside the $E_{\text{test}}$ support,
   zeroes the univariate and the multivariate Starr fraction (Starr 1962;
   Herman et al. 2015). Such criteria are re-anchored at
   maintain-status-quo performance (Kasprzyk et al. 2013; Lempert & Collins
   2007), rounded to the stricter side.
2. An external goalpost beats a round number (the flood criterion, anchors in
   `flood_objective_diagnostics.md`).
3. All-pass guardrails are kept, not re-tuned. Moving them onto the baseline's
   own $E_{\text{test}}$ distribution would be baseline tuning; the
   one-at-a-time stringency sweep (`framing_convention_diagnostics.md`) tests
   whether they are ranking-critical.
4. Distribution-feature candidates (SOW-mean P10/P50/P90,
   `RTD_CANDIDATE_QUANTILES`) are reported in every table and figure and never
   adopted, since they would guarantee a chosen fraction by construction.

## 2. Diagnostics

- **Distributions and anchors** (`S_rtd_baseline_sow_cdfs`;
  `rtd_sow_summary.csv`, `rtd_default_stringency.csv`). Per-objective ECDFs of
  the per-SOW values with the current threshold, the historic-trace anchor,
  the NYC stakeholder floor 0.5 (`src/pareto_filter.py::DEFAULT_STAKEHOLDER_FLOORS`)
  and the observed flood anchor, axes capped to the data support (Gold et al.
  2023, Fig. 5).
- **Threshold sensitivity** (`S_rtd_threshold_sensitivity`;
  `rtd_candidate_placements.csv`, `rtd_threshold_sweep.csv`). Satisficing
  fraction against threshold on a dense grid (`RTD_SWEEP_POINTS` = 201),
  stringency increasing rightward on every panel, Wilson band and degenerate
  zones (`RTD_DEGENERACY_LIMIT`) shaded, candidate placements marked by class
  (Hadjimichael et al. 2020). The stringency coordinate follows
  `compare_designs.default_stringency`, so this and the step-10 sweep speak
  the same coordinate.
- **θ-attribution** (`S_rtd_theta_spearman`, `S_rtd_factor_maps`;
  `rtd_theta_spearman.csv`, `rtd_critical_m.csv`). Spearman correlations of the
  per-SOW objectives with $\theta$, and pass/fail factor maps for every
  criterion in the $(m, r_1)$ plane (`RTD_FACTOR_MAP_PLANE`, the boundary is
  near-monotone in $m$ and $r_2$ is inert) with the $m$ at which the local pass
  rate crosses 0.5, the hydrologic reading of each placement (Bryant & Lempert
  2010).
- **Historic-anchor comparison** (`rtd_historic_anchor_comparison.csv`). Each
  objective's annual-unit anchor recomputed from the baseline HDF5 through the
  same `obj.compute` path (the N = 1 case of the cube's formula, cached at
  `cache/historic_anchor_annual_metrics.json`), its quantile position in the
  per-SOW distribution, an `in_sow_support` flag, and a near-historic check
  within the `RTD_NEAR_HISTORIC_K` SOWs closest to $\theta = 0$. The persisted
  `outputs/baseline/ffmp_baseline_objectives.csv` rides along as a cross-check
  column. The flood anchor is the observed WY2001–2023 exceedance,
  1.17 ft·d/yr (`flood_objective/tables/A_sim_vs_obs.csv`).
- **Conjunction diagnostics** (`S_rtd_conjunction`; `rtd_joint_satisficing.csv`,
  `rtd_failure_combinations.csv`, `rtd_failing_count_distribution.csv`,
  `rtd_threshold_recommendation.csv`). The joint SOW fraction bracketed by the
  independence product and the comonotone bound, the binding criterion, the
  most frequent failing combinations and sole- vs co-failure attribution
  (Starr 1962; Herman et al. 2015; Bonham et al. 2024), plus critical-$m$
  boundaries and guardrail margins.

## 3. Adopted criteria

**Search-time snapshot** (`src/objectives_ensemble.py::_DEFAULT_THRESHOLDS`,
overridable by `NYCOPT_SAT_THRESHOLDS`, snapshotted into each run's
`reeval_raw_meta.json`; basis in `supplemental_config.RTD_RECOMMENDED_THRESHOLDS`
and `RTD_RECOMMENDATION_BASIS`). Rule-1 placements on the stricter side of the
historic annual-unit anchors: NYC reliability 0.65, NYC deficit P99 48,
Montague reliability 0.79, Montague deficit P99 27, Trenton reliability 0.87,
NJ reliability 0.74. Rule-2 goalposts: flood exceedance 1.17 ft·d/yr and
storage P01 26 % (the FFMP L5 drought-emergency boundary, kept as a stringent
aspirational criterion because the maintain-status-quo re-anchor would be a
vacuous 0 % line).

**Analysis criterion sets** (`src/satisficing_criteria.py`, variant selected
by `NYCOPT_CRITERIA_VARIANT`, focal set by `NYCOPT_FOCAL_CRITERION`, default
`compromise`). Following Quinn et al. (2017), each named set thresholds one to
three objectives and leaves the others unconstrained, and robustness is
reported under every set with a cross-set ranking-agreement check. The
all-axes conjunction is retained only as `reference_all8`; it saturates at
zero because the Montague reliability anchor lies outside the incumbent's
$E_{\text{test}}$ support, a motivation finding never used for selection.
Placements apply the rules per member axis from the audit table
`outputs/comparison/{slug}/{tag}/criteria_reanchoring.csv` (incumbent per-SOW
quantiles, pass fraction at the adopted threshold, pooled stringency, and the
stricter-side round of the incumbent median at ε granularity):

| set | criteria |
|---|---|
| `nyc_supply` | NYC reliability ≥ 0.65 (historic anchor); storage P01 ≥ 13.0 % (rule 1, incumbent median year) |
| `downstream_flows` | Montague reliability ≥ 0.50, Trenton reliability ≥ 0.75 (rule 1, incumbent median year) |
| `flood` | flood exceedance ≤ 1.17 ft·d/yr (rule 2) |
| `compromise` | NYC reliability ≥ 0.65, Trenton reliability ≥ 0.75, flood ≤ 1.17 (one axis per Decree party) |

Saturated non-discriminators (NYC deficit P99, Montague deficit P99, NJ
reliability) are excluded from every named set and stay visible in the
univariate decomposition.

## Citations

Bryant & Lempert (2010); Gold et al. (2023); Hadjimichael et al. (2020);
Herman et al. (2014, 2015); Kasprzyk et al. (2013); Lamontagne et al.
(2018); Lempert & Collins (2007); McPhail et al. (2018, 2020); Quinn et al.
(2017, 2020); Starr (1962); Trindade et al. (2017); Zeff et al. (2014).
Resolved in `docs/notes/literature/objective_and_robustness_formulations.md`
and `scenario_subset_selection.md`.
