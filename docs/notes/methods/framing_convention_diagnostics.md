# Framing-Convention Diagnostics (SI)

*Last updated: 2026-07-27. Specification of four light diagnostics that test the
sensitivity of the study's framing conventions — the annual failure-week count
k, the weekly satisfaction factor, the re-evaluation satisficing criteria, and
the controllability of the flood-days objective. All four are offline
re-scorings of artifacts that already exist or are produced once by machinery
already specified: the stored per-unit matrix of
`ensemble_objective_sensitivity_experiment.md`, the persisted re-evaluation
cube (`reeval_raw.parquet`), and the workflow step-05 default-FFMP baseline.
None launches a new simulation campaign. Terminology per
`docs/notes/terminology.md`; the two ensemble compositions of the sensitivity
experiment are referred to by that note's own labels, arm P (fixed
probabilistic) and arm H (hazard-filled).*

---

## 0. Scope and cost model

Every diagnostic obeys the sensitivity experiment's efficiency architecture:
**simulate once, subsample outputs**. The expensive step (an ensemble
simulation per DV) is never repeated for a convention sweep; each diagnostic is
a post-hoc reduction of a stored matrix or persisted cube.

| # | Diagnostic | Input artifact | New simulation | Deliverable |
|---|---|---|---|---|
| 1 | Failure-week count k | sensitivity-experiment stored matrix (arms P, H) | none | SI figure + CSV; gates `_DEFAULT_FAILURE_K` |
| 2 | Weekly satisfaction factor | same matrix (factor axis stored in the one-time DV sweep) | none | SI table/figure; gates the 0.99 constant |
| 3 | Satisficing-criterion diagnostics | persisted re-evaluation cube + step-05 baseline | none | SI (candidate main-text) figures; gates `_DEFAULT_THRESHOLDS` |
| 4 | Flood-days controllability | sensitivity-experiment stored matrix + baseline row | none | one SI figure; informs the flood objective's standing |

Diagnostics 1, 2, and 4 are figure-script reductions of the sensitivity
experiment's matrix HDF5 (`outputs/supplemental/ensemble_objective_sensitivity/
matrix/*.h5`); diagnostic 3 is a re-scoring of the re-evaluation cube via
`src/robustness.py`, extending the pooled stringency sweep of
`scripts/main/compare_designs.py`.

---

## 1. Failure-week count k

**Purpose.** Quantify how the frequency objectives' saturation and the policy
rankings they induce depend on the annual failure criterion "≥ k failing weeks
per unit-year", confirming or revising the shipped values (k = 3 for NYC
delivery and Montague, k = 1 for Trenton and NJ;
`objectives_ensemble.py::_DEFAULT_FAILURE_K`).

**Method.** The stored matrix's stage-(i) annual metric for every frequency
objective is the unit-year's **failing-week count**, not a binary failure
indicator, so every candidate k is a free post-hoc reduction: apply
`FailureFrequencyOp(k)` for k ∈ {1, 2, 3, 4} — a grid that contains the shipped
values — to the pooled unit-years of each DV × set. For each objective × arm ×
k, report:

1. **Saturation** — the fraction of the random-DV population whose failure-year
   frequency is ≤ 0.05 or ≥ 0.95 (the sensitivity experiment's saturation
   bands; a saturated criterion ties everything, Bonham et al. 2024).
2. **Ranking sensitivity** — Kendall τ_b between the DV ranking induced at k and
   at the shipped k (Herman et al. 2015), per objective × arm.

Both arms are screened: a k that discriminates in arm P but saturates in arm H
(or vice versa) is exactly the operator-composition interaction the sensitivity
experiment exists to catch.

**Gates / output.** Final `_DEFAULT_FAILURE_K` per objective: a shipped k that
saturates in either arm, or whose ranking is unstable to k ± 1, is revised (via
`NYCOPT_FAILURE_K` and a re-screen). SI: one panel figure (saturation fraction
vs k and τ_b-vs-shipped-k, per objective × arm) plus the CSV.

**Cost.** Zero simulation; seconds of post-processing in the figure script.

---

## 2. Weekly satisfaction factor (0.99)

**Purpose.** The delivery reliability metrics count a week as satisfied when
weekly-total delivery ≥ 0.99 × the weekly-total entitlement; the 0.99 factor is
a convention and its influence on the weekly satisfaction fraction and the
induced failure-years must be bounded.

**Method.** The factor sits inside the weekly reduction
(`src/objectives.py::_weekly_delivery_ok`), upstream of the stored failing-week
counts, so it cannot be recovered from the counts alone. The sensitivity
experiment's DV-sweep therefore stores, for the two delivery objectives (NYC,
NJ), the per-unit failing-week counts at each factor in
**{0.95, 0.98, 0.99, 1.00}** — one small extra matrix axis computed inside the
single simulation pass, at no additional simulation cost. Post-hoc reductions,
per arm:

1. weekly satisfaction fraction (the §1 base metric) and failure-year frequency
   at the shipped k, vs factor, across the random-DV population;
2. τ_b of the DV ranking at each factor against the 0.99 ranking;
3. the default-FFMP baseline row (`sample_ids = -1`) anchoring the factor curves
   at current operations.

Where a stored matrix predates the factor axis, the fallback is an offline
recomputation of the weekly series from the persisted step-05 baseline outputs
(the baseline-only factor curve, zero simulation); the full DV-population sweep
then waits for the next matrix build.

**Gates / output.** The 0.99 constant, shared by the §1 weekly-reliability
metrics and the §2 failing-week counts. The delivery entitlement is a
running-average right whose shortfalls are whole-season curtailments, so the
expected finding is definitional insensitivity; a materially factor-dependent
ranking would instead make the factor a calibrated quantity set from this
curve. SI: one small figure or table.

**Cost.** Negligible extra compute inside the existing one-time sweep; zero
extra simulation.

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

## 4. Flood-days attribution and controllability

**Purpose.** Bound how much of the flood-days objective (mean annual days at
NWS minor stage, days/yr) is controllable by NYC release decisions versus set
exogenously by the hydrology, pre-empting the critique that the search
gradient on an exogenously dominated objective is too weak to be meaningful.

**Method.** A post-hoc reduction of the stored matrix's
`downstream_flood_days_annual` per-unit values, which the sensitivity
experiment already computes for the full random-DV population **and** the
default-FFMP baseline row on the same realizations, per arm. For each pooled
unit-year u:

- **empirical exogenous floor** F_min(u) = the minimum annual flood-day count
  across the DV population ∪ baseline;
- **across-policy spread** = the distribution of F_dv(u) − F_min(u) across DVs.

Summaries, per arm:

1. the distribution of the ensemble-mean annual flood days (days/yr, the §2
   unit-operator scale) across the DV population, with the baseline and the
   pooled floor Σ_u F_min(u) / NL marked;
2. the **floor share** Σ_u F_min(u) / Σ_u F_base(u) — the policy-invariant
   fraction of the baseline's flood days;
3. its complement, the controllable fraction, contrasted between arms (the
   wet-enriched arm H tests controllability under stress-enriched
   composition).

The floor is empirical — a minimum over the evaluated policy sample, not an
oracle — so it is an **upper bound on the exogenous component and the quoted
controllable fraction is a lower bound**; this is stated wherever the figure is
used. The same reduction reports the spread of the P99 unit-operator variant
alongside the mean, feeding the flood unit-operator decision (expectation can
mask floods; Quinn et al. 2017).

**Gates / output.** One SI figure (per arm: DV-population distribution of mean
annual flood days with the baseline and empirical-floor lines, plus the
floor-share summary). Informs whether flood days remains in the active search
set with the mean operator, and supplies the quantitative reply to the
weak-gradient critique.

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
