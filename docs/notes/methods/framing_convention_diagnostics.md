# Framing-Convention Diagnostics (SI)

Offline screens that fix the formulation's conventions from measured
sensitivity rather than assertion: the annual failure-week count k, the weekly
satisfaction factor, the flood unit operator and the flood objective's
controllability, the eighth objective's activation, and the re-evaluation
criterion diagnostics. Every screen is a post-hoc reduction of an artifact
produced once. The epsilon-calibration per-unit cubes
(`epsilon_calibration_experiment.md`; 512 constraint-feasible policies plus the
FFMP baseline evaluated on each matched design's own search ensemble, historic
alongside as reference), the satisfaction-factor sweep cube, and the persisted
re-evaluation cube. Both matched compositions are screened, since a convention
that discriminates under i.i.d. sampling but saturates under the
stress-enriched composition (or the reverse) is the operator-composition
interaction the screens exist to catch. Outputs under
`outputs/supplemental/framing_convention/`, settings `FRAMING_*` in
`supplemental_config.py`. Manuscript statement in SI Text S5.

## Conventions and the screens that fix them

| convention | adopted | screen | script, input |
|---|---|---|---|
| failure-week count k (`_DEFAULT_FAILURE_K`, `NYCOPT_FAILURE_K`) | three failing weeks for NYC delivery and Montague, one for Trenton and NJ | `FailureFrequencyOp(k)` over `FRAMING_K_GRID` = {1, 2, 3, 4} on the cubes' stored failing-week counts; saturation fraction (population share ≤ 0.05 or ≥ 0.95, `FRAMING_SATURATION_BAND`, Bonham et al. 2024) and Kendall τ_b against the adopted k (Herman et al. 2015), per objective × design | `framing_convention_analysis.py`, epsilon cubes |
| flood unit operator | mean (P99 stays a registered diagnostic) | pooled mean vs pooled P99 per policy: population IQR, bootstrap noise (`FRAMING_BOOTSTRAP_B` = 500, realization resampling) and bootstrap τ_b, with Olden–Poff retention of the stabler operator where the two are rank-correlated | same |
| flood controllability | objective retained | empirical exogenous floor F_min(u), the minimum over the policy population and the baseline per pooled unit-year; floor share Σ F_min / Σ F_base of the baseline's burden, complement the controllable fraction (a lower bound, since the floor is a sample minimum) | same |
| eighth objective | NJ delivery reliability active | annual-unit redundancy screen, pairs flagged at |ρ_S| ≥ `FRAMING_RHO_FLAG_THRESHOLD` = 0.8 (Olden & Poff 2003) | same |
| weekly satisfaction factor (`src/objectives.py::_weekly_delivery_ok`) | 0.99 | the factor sits upstream of the stored week counts, so a dedicated pass re-evaluates the same policy population and stores NYC and NJ failing-week counts and weekly reliability at {0.95, 0.98, 0.99, 1.00} from each realization's weekly sums; fraction and failure-year frequency vs factor, τ_b against the 0.99 ranking, baseline row (`sample_ids = −1`) | `satisfaction_factor_{run,figures}.py`, launcher `workflow/supplemental/satisfaction_factor.sh`, one job per matched design |
| re-evaluation criteria (`robustness_threshold_diagnostics.md`) | see that note | (a) one-at-a-time stringency tightening, every criterion at its default except j at stringency s on the `NYCOPT_COMPARE_STRINGENCY` grid, design-ranking τ_b against the all-default ranking, read against the lockstep sweep (Quinn et al. 2020); (b) threshold-margin CDFs of the re-evaluated per-SOW values per design with the criterion overlaid (Gold et al. 2023, Fig. 5) | `compare_designs.py` machinery and `src/robustness.py`, persisted re-evaluation cube plus the step-05 baseline |

## Measured verdicts

Measured on the N = 100 calibration cubes (900 pooled unit-years per design).

- **k.** No adopted k saturates in either composition (worst case 0.4 % of the
  population inside the ±0.05 bands), and rankings are stable to k ± 1
  (τ_b ≥ 0.94 for NYC, Montague and NJ). Trenton's k = 1 is binding, since a
  three-week count ties 24–44 % of policies at ≥ 0.95 reliability and a
  four-week count about 97 %.
- **Flood operator.** Measured on the day-count diagnostic that those cubes
  carry in the flood role (the analysis reads the active
  `downstream_flood_exceedance_annual` from cubes written under it). The P99
  operator collapses onto 2–3 integer day counts (population IQR 0.0 in the
  hazard-filling composition), is 12–30× noisier under bootstrap (median
  policy SD 0.48–1.16 vs 0.038–0.040 days) and ranking-unstable (bootstrap
  τ_b 0.35–0.60 vs 0.92–0.93). Where estimable the two operators are
  rank-correlated ρ_S = 0.85–0.89, so the stabler mean is retained.
- **Controllability.** Floor share of the baseline's flood burden 0.57
  (monte_carlo) and 0.61 (hazard_filling_stationary), so the
  controllable fraction is at least 0.43 and 0.39.
- **NJ delivery.** Max |ρ_S| against any objective 0.38 (flood mean), about
  0.15 against NYC delivery and ≤ 0.08 against Trenton. The only flagged pair
  is flood mean with flood P99, internal to the operator decision. NJ's ε is
  the shared reliability precision (0.05).
- **Satisfaction factor.** τ_b against the 0.99 ranking ≥ 0.92 at 0.98 and
  0.99, while strict 1.00 collapses rankings (τ_b 0.59–0.65) and 0.95 drifts
  (τ_b 0.70).

## Citations

| Diagnostic element | Citation(s) |
|---|---|
| Saturated criteria tie everything (saturation bands) | Bonham et al. 2024 |
| Rank stability via Kendall τ_b | Herman et al. 2015; McPhail et al. 2018 |
| Design-ranking agreement degrades with stringency | Quinn et al. 2020 |
| Threshold-margin CDF with the criterion overlaid | Gold et al. 2023 (Fig. 5) |
| Expectation can mask floods (mean vs P99) | Quinn et al. 2017 |
| Redundant-metric retention by rank correlation | Olden & Poff 2003 |
