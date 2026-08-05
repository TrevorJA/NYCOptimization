# Framing-Convention Diagnostics (SI)

*Last updated: 2026-08-05. Specification of four light diagnostics that test the
sensitivity of the study's framing conventions — the annual failure-week count
k, the weekly satisfaction factor, the re-evaluation satisficing criteria, and
the controllability of the flood-days objective. All four are offline
re-scorings of artifacts that already exist or are produced once by machinery
already specified: the epsilon-calibration per-unit annual-metric cubes
(`epsilon_calibration_experiment.md`; 512 constraint-feasible policies + the
FFMP baseline evaluated on each campaign design's own search ensemble), the
satisfaction-factor sweep cube (same policy population, one extra simulation
pass), the persisted re-evaluation cube (`reeval_raw.parquet`), and the
workflow step-05 default-FFMP baseline. None launches a new simulation
campaign. Terminology per `docs/notes/terminology.md`; the two ensemble
compositions screened are the campaign designs themselves,
`fixed_probabilistic` and `hazard_filling_stationary` (the single-trace
`historic` design is reported alongside as reference).*

---

## 0. Scope and cost model

Every diagnostic obeys the simulate-once, subsample-outputs efficiency
architecture: the expensive step (an ensemble simulation per policy) is never
repeated for a convention sweep; each diagnostic is a post-hoc reduction of a
stored cube.

| # | Diagnostic | Input artifact | New simulation | Deliverable |
|---|---|---|---|---|
| 1 | Failure-week count k | epsilon-calibration cubes (both ensemble designs + historic) | none | SI figure + CSV; gates `_DEFAULT_FAILURE_K` |
| 2 | Weekly satisfaction factor | satisfaction-factor sweep cube (factor axis stored in its one-time policy sweep) | one pass per design (`satisfaction_factor_run.py`) | SI table/figure; gates the 0.99 constant |
| 3 | Satisficing-criterion diagnostics | persisted re-evaluation cube + step-05 baseline | none | SI (candidate main-text) figures; gates `_DEFAULT_THRESHOLDS` |
| 4 | Flood-days controllability | epsilon-calibration cubes + baseline row | none | one SI figure; informs the flood objective's standing |

Diagnostics 1 and 4 (plus the flood unit-operator comparison and the
annual-unit redundancy screen for the 8th objective) are reductions
of the epsilon-calibration cube HDF5s
(`outputs/supplemental/epsilon_calibration/cube/*.h5`) by
`scripts/supplemental/framing_convention_analysis.py`; diagnostic 2 is the
dedicated sweep `scripts/supplemental/satisfaction_factor_{run,figures}.py`
(SLURM: `workflow/supplemental/satisfaction_factor.sh`, one job per ensemble
design); diagnostic 3 is a re-scoring of the re-evaluation cube via
`src/robustness.py`, extending the pooled stringency sweep of
`scripts/main/compare_designs.py`.

## 0b. Measured verdicts (adopted)

Run on the three epsilon cubes (tables + SI figures under
`outputs/supplemental/framing_convention/`):

- **Failure-week counts k — CONFIRMED as shipped.** No shipped k saturates in
  either ensemble composition (worst case 0.4% of the population inside the
  ±0.05 bands); rankings stable to k ± 1 (τ_b ≥ 0.94 for NYC/Montague/NJ).
  Trenton k = 1 is binding: k = 3 ties 24–44% of policies at ≥ 0.95
  reliability, k = 4 ties ~97%.
- **Flood unit operator — MEAN adopted.** At NL = 900 pooled unit-years the
  P99 operator collapses onto 2–3 integer day counts (population IQR 0.0 in
  the hazard-filling composition), is 12–30× noisier under bootstrap
  (median policy std 0.48–1.16 vs 0.038–0.040 days), and ranking-unstable
  (bootstrap τ_b 0.35–0.60 vs 0.92–0.93). Where estimable the two operators
  are rank-correlated ρ_S = 0.85–0.89, so Olden–Poff retention keeps the
  stabler mean. P99 stays a registered diagnostic.
- **Flood controllability — objective retained.** Empirical floor share of
  the baseline's flood days 0.57 (fixed_probabilistic) / 0.61
  (hazard_filling): controllable fraction ≥ 0.43 / 0.39 (lower bound).
- **8th objective — NJ delivery ACTIVE.** Max |ρ_S| vs any objective 0.38
  (vs flood mean); vs NYC delivery ~0.15; vs Trenton ≤ 0.08. Only flagged
  pair anywhere: flood mean ↔ flood P99 (0.85/0.89), internal to the operator
  decision. NJ's ε (0.025) was calibrated by the same experiment.
- **Weekly satisfaction factor — 0.99 ADOPTED** (diagnostic 2, both ensemble
  designs): τ-vs-shipped ≥ 0.92 at 0.98/0.99, while the strict 1.00
  collapses rankings (τ_b 0.59–0.65) and 0.95 drifts (τ_b 0.70).

Diagnostic 3 waits on the persisted re-evaluation cube (post E_test).

---

## 1. Failure-week count k

**Purpose.** Quantify how the frequency objectives' saturation and the policy
rankings they induce depend on the annual failure criterion "≥ k failing weeks
per unit-year", confirming or revising the shipped values (k = 3 for NYC
delivery and Montague, k = 1 for Trenton and NJ;
`objectives_ensemble.py::_DEFAULT_FAILURE_K`).

**Method.** The cubes' stage-(i) annual metric for every frequency objective
is the unit-year's **failing-week count**, not a binary failure indicator, so
every candidate k is a free post-hoc reduction: apply `FailureFrequencyOp(k)`
for k ∈ {1, 2, 3, 4} — a grid that contains the shipped values — to the pooled
unit-years of each policy × design. For each objective × design × k, report:

1. **Saturation** — the fraction of the feasible-policy population whose
   failure-year frequency is ≤ 0.05 or ≥ 0.95 (a saturated criterion ties
   everything, Bonham et al. 2024).
2. **Ranking sensitivity** — Kendall τ_b between the policy ranking induced at
   k and at the shipped k (Herman et al. 2015), per objective × design.

Both ensemble compositions are screened: a k that discriminates under the
i.i.d. composition but saturates under the stress-enriched one (or vice versa)
is exactly the operator-composition interaction this screen exists to catch.

**Gates / output.** Final `_DEFAULT_FAILURE_K` per objective: a shipped k that
saturates in either ensemble design, or whose ranking is unstable to k ± 1, is
revised (via `NYCOPT_FAILURE_K` and a re-screen). SI: one panel figure
(saturation fraction vs k and τ_b-vs-shipped-k, per objective × design) plus
the CSV.

**Cost.** Zero simulation; seconds of post-processing in the figure script.

---

## 2. Weekly satisfaction factor (0.99)

**Purpose.** The delivery reliability metrics count a week as satisfied when
weekly-total delivery ≥ 0.99 × the weekly-total entitlement; the 0.99 factor is
a convention and its influence on the weekly satisfaction fraction and the
induced failure-years must be bounded.

**Method.** The factor sits inside the weekly reduction
(`src/objectives.py::_weekly_delivery_ok`), upstream of the epsilon cubes'
stored failing-week counts, so it cannot be recovered from those counts alone.
The dedicated sweep (`scripts/supplemental/satisfaction_factor_run.py`)
therefore re-evaluates the same feasible-policy population (identical
seed/count, so cube rows align with the epsilon cubes) and stores, for the two
delivery objectives (NYC, NJ), the per-unit failing-week counts AND the §1
weekly reliability at each factor in **{0.95, 0.98, 0.99, 1.00}** — the factor
axis is computed from each realization's weekly sums inside the single
simulation pass. Post-hoc reductions
(`scripts/supplemental/satisfaction_factor_figures.py`), per design:

1. weekly satisfaction fraction (the §1 base metric) and failure-year frequency
   at the shipped k, vs factor, across the feasible-policy population;
2. τ_b of the policy ranking at each factor against the 0.99 ranking;
3. the default-FFMP baseline row (`sample_ids = -1`) anchoring the factor curves
   at current operations.

**Gates / output.** The 0.99 constant, shared by the §1 weekly-reliability
metrics and the §2 failing-week counts. The delivery entitlement is a
running-average right whose shortfalls are whole-season curtailments, so the
expected finding is definitional insensitivity; a materially factor-dependent
ranking would instead make the factor a calibrated quantity set from this
curve. SI: one small figure or table.

**Cost.** One evaluation pass of the policy population per ensemble design
(same cost as the epsilon calibration, ~26 SU each); the factor axis itself
adds nothing.

---

## 3. Satisficing-criterion diagnostics (re-evaluation thresholds)

**Purpose.** The main-text stringency sweep (`compare_designs.py`) moves every
criterion in lockstep with one knob s; these diagnostics resolve *which*
criterion drives any threshold dependence of the design ranking, and how much
margin each objective carries against its criterion.

**Method.** Both are pure offline re-scorings of the persisted
(solution × realization × objective) re-evaluation cube — zero simulation.

**(a) One-at-a-time stringency tightening.** Reusing the compare machinery
(`pooled_cells`, `thresholds_at`, `robustness.satisficing_multivariate`): for
each base objective j and each stringency s on the `NYCOPT_COMPARE_STRINGENCY`
grid, build the threshold dict with all criteria i ≠ j at their registry
defaults and only criterion j at stringency s, recompute the multivariate
satisficing metric on every run's cube, and report (i) the per-design
robustness curves and (ii) Kendall τ_b of the design ranking at (j, s) against
the all-default ranking. Read against the lockstep sweep: a ranking flip that
appears under one tightened criterion but not under joint tightening (or vice
versa) localizes the threshold dependence that Quinn et al. (2020) predict at
the conservative end.

**(b) Threshold-margin CDFs.** Per base objective, the empirical CDF of the
re-evaluated E_test cells — per design, for the best-by-primary-metric solution
with the median solution overlaid, plus the step-05 default-FFMP baseline —
with the registry-default criterion drawn as a vertical line (Gold et al. 2023,
Fig. 5). The margin (horizontal distance from the mass to the criterion) shows
whether a criterion sits in a steep region of the distribution, which is the
mechanism behind any fragility found in (a); it also exposes criteria that are
saturated (all mass on one side) before they can distort the comparison.

**Gates / output.** Final registry thresholds (`_DEFAULT_THRESHOLDS` /
`NYCOPT_SAT_THRESHOLDS`) and the identification of ranking-critical criteria to
be flagged in the manuscript. SI by default; promoted to main text if the OAT
sweep shows the design ranking hinges on a single criterion.

**Cost.** Zero simulation; minutes of cube re-scoring (the cube loads
dominate; the grid is ~8 objectives × the stringency grid × runs).

---

## 4. Flood attribution and controllability

**Purpose.** Bound how much of the flood objective is controllable by NYC
release decisions versus set exogenously by the hydrology, pre-empting the
critique that the search gradient on an exogenously dominated objective is
too weak to be meaningful.

**Method.** A post-hoc reduction of the epsilon cubes' per-unit values of the
ACTIVE flood objective (`downstream_flood_exceedance_annual`; older cubes
carry the day-count diagnostic in that role), held for the full
feasible-policy population **and** the default-FFMP baseline row on the same
realizations, per design. For each pooled unit-year u:

- **empirical exogenous floor** F_min(u) = the minimum annual value across
  the policy population ∪ baseline;
- **across-policy spread** = the distribution of F_p(u) − F_min(u) across
  policies.

Summaries, per design: the policy-population distribution of the §2
unit-operator scale with the baseline and the pooled floor Σ_u F_min(u) / NL
marked; the **floor share** Σ_u F_min(u) / Σ_u F_base(u) (the
policy-invariant fraction of the baseline's flood burden); and its
complement, the controllable fraction, contrasted between the ensemble
designs (the wet-enriched hazard-filling composition tests controllability
under stress).

The floor is empirical — a minimum over the evaluated policy sample, not an
oracle — so it is an **upper bound on the exogenous component and the quoted
controllable fraction is a lower bound**; this is stated wherever the figure
is used.

**Gates / output.** One SI figure per design, plus the floor-share summary —
the quantitative reply to the weak-gradient critique.

**Cost.** Zero simulation; seconds of post-processing.

---

## Citations

| Diagnostic element | Citation(s) |
|---|---|
| Saturated criteria tie everything (saturation bands) | Bonham et al. 2024 |
| Rank stability via Kendall τ_b | Herman et al. 2015; McPhail et al. 2018 |
| Design-ranking agreement degrades with stringency | Quinn et al. 2020 |
| Threshold-margin CDF with the criterion overlaid | Gold et al. 2023 (Fig. 5) |
| Expectation can mask floods (mean vs P99) | Quinn et al. 2017 |
