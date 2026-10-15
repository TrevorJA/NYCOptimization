# Hazard-Selector Diagnostics (SI experiment)

*Design of the supplemental selector and sizing diagnostics for the hazard-filling scenario design, measured on the campaign selection axes. Machinery: `scengen/selector_diagnostics.py` (selectors + metric battery), `src/hazard_metric_image.py` (the 21-metric matrix and its pure helpers) and `scripts/supplemental/diagnose_hazard_selectors.py` (driver; settings `SELDIAG_*` in `supplemental_config.py`). Selection recipe under test: `scenario_design_methods.md` §4.3; the candidate metrics: §3.3; the choice of the selection axes: `hazard_axis_selection.md`. Outputs: `outputs/supplemental/hazard_selector_diagnostics/{pool_slug}/`.*

---

## 1. Purpose

The selection axes are chosen by `hazard_axis_selection.md`. Given that choice, three campaign conventions inside the hazard-filling design remain conventions unless measured: the **selection rule** that places N members over the hazard manifold, the **normalization bounds** that define the absolute selection geometry, and the **ensemble size N**. This experiment measures them on the adopted selection axes of a real candidate ensemble, entirely at the selection level — no system simulation — so it runs on a laptop test pool and scales unchanged to the production pool on HPC. It backs four SI claims:

1. The campaign selector administers the intervention at strength (coverage, tail enrichment) without pathologies (near-duplicates, outlier fixation, atom mis-handling) relative to defensible alternatives.
2. The robust normalization bounds (p1/p99) stabilize the selection geometry, and the design's headline properties are not artifacts of the bounds choice.
3. The selection axes deliver the design's per-axis marginal coverage guarantee, the selection is not hostage to any single axis, and the implicit weighting that correlated axes induce in the selection distance is characterized (a disclosed non-issue, not a correction).
4. The redundancy of the 21 candidate metrics and the tail coverage the selection delivers on every metric, including those it does not select on, are reported; window-edge truncation of the scored drought events is reported alongside.

## 2. Selection rules compared

All rules select N members from the same pool sub-image, normalized once with the campaign robust bounds, so differences are attributable to the rule alone.

| Rule | Construction | Role |
|---|---|---|
| `random` | Without replacement | The null every designed rule must beat (many-seed null band). |
| `lhs_nn` | LHS targets + greedy nearest-unused-neighbor snap | The wired selector. |
| `lhs_assign` | Same targets, optimal one-to-one assignment (Hungarian) | Isolates the greedy snap's order-dependence: same plan, globally optimal pairing. |
| `maximin` | Greedy maximin distance (Kennard–Stone type; Johnson et al. 1990) | The DOE-standard comparator; target-free but known to load the hull. |
| `eps_cell` | Grid the unit box at the coarsest resolution with ≥ N occupied cells; draw N occupied cells uniformly; one representative per cell (nearest cell center) | Uniform over the manifold's *occupied support* at resolution ε — no target can land off the manifold; one-per-cell separation guarantee. Coverage analogue of ε-dominance archiving (Laumanns et al. 2002). |

Target-based rules face the manifold-support problem: hazard axes are structurally dependent (run theory: deficit ≈ duration × intensity), so part of the unit box is unoccupied and targets placed there must snap. The target-displacement distribution measures that cost; `maximin` and `eps_cell` are the target-free comparators.

## 3. Metric battery

The design's defining metrics (minimax distance relative to the pool, per-axis Kolmogorov–Smirnov distance, minimum-spanning-tree edge lengths, per-axis span, nearest-member redistribution and effective sample size) and the certified exact-assignment gap of the `lhs_nn` rule are defined in `hf_design_metrics.md`. The battery below is the selector-comparison instrument; its cube-based L2-star discrepancy is kept for the rule comparison only. Target displacements are those of the exact target-to-member pairing of the greedy rule (`subsample.lhs_nn_assignment`).

Per (rule, seed), on the screened pool sub-image (`selection_metrics`, `per_axis_selection_metrics`):

- **Coverage uniformity**: L2-star discrepancy in the absolute (campaign) and rank geometries, placed against the many-seed random null; MST edge statistics and minimum pairwise separation (near-duplicate guard) in absolute geometry.
- **Per-axis marginal coverage** — the mechanism metric: LHS targets stratify every axis into N bins regardless of dimension, so the design's coverage guarantee is per-axis marginal, not joint. Per axis in the campaign scaled coordinates: KS distance of the selected marginal to uniform, 1-D L2-star discrepancy, largest marginal gap, and the tail share above the pool P90 (unbiased ≈ 0.10).
- **Tail enrichment**: mean per-axis share above the pool P90 and the any-axis P90 corner share — the deliberate distribution shift, quantified.
- **Displacement vs dimension**: target-displacement distribution and the distance-concentration ratio (mean target displacement / mean random pool-pair distance in the same space) — raw displacements are not comparable across dimensions, the ratio is.
- **Marginal distortion**: mean KS distance to the pool marginals.
- **Dry zero-event atom**: pool share of windows with no SSI-6 ≤ −1 event, and each rule's selected share.
- **Stability**: across-seed selected-set Jaccard per rule; target displacements for the LHS rules.
- **Selection invariance / implicit weighting**: Jaccard overlap of selected member IDs between the full-axis-set selection and leave-one-axis-out / add-one-axis-back variants; per-axis (and dry-vs-wet group) mean share of the squared target displacement.

On the 21 candidate metrics (the two truncation flags excluded):

- **Metric redundancy**: the Spearman matrix; average-linkage clusters on $1-\lvert\rho_S\rvert$ cut so every pair at $\lvert\rho_S\rvert \ge 0.7$ shares a cluster (0.7, the level above which collinearity matters, Dormann et al. 2013); a principal component analysis of the correlation matrix of the normal scores (each metric's ranks mapped through the standard normal quantile), reporting the eigenvalues, the participation ratio $(\sum\lambda)^2/\sum\lambda^2$, the number of components that reach 90% of the variance, and the highest-loading metric of each of those components (index reduction by principal components, Olden & Poff 2003).
- **Hazard-direction tail share** of a selection on every metric: the share of selected members beyond the pool p90, or below the pool p10 for the four low-flow minima, where low values are hazardous (i.i.d. share 0.10).
- **Attainment** of a selection axis: its tail share over the share an exact snap to uniform targets on the clipped p1–p99 range would give, $(p_{99} - p_{90})/(p_{99} - p_1)$; the minimum over the set's own axes is reported.
- **Target displacement and reach**: the mean target-to-member displacement of the `lhs_nn` pairing, and the share of targets farther than 0.25 (scaled units) from every pool member.
- **Nearest-member weights**: $n_{\text{eff}}/N$ of the Voronoi masses the pool assigns to the selected members (the measure statement of `hf_design_metrics.md` §5, computed with its helpers).
- **Truncation**: the fractions of windows whose largest drought event has a truncated onset or termination (the two truncation flags).

On the historical record:

- **SSI fit check**: the record's SSI-6 under the reference fit, the two-parameter gamma per calendar month that every realization is transformed with. Per calendar month it reports the standard deviation, the count of values at or below −1 (the level a run-theory event must reach to qualify) and the count a standard normal gives, $n\,\Phi(-1)$. A fit that reproduces the record gives a standard deviation of 1 and a count near the expected one in every month.

## 4. Analysis blocks

A. **Selection-axis screen report and metric redundancy**: the axis screen (degenerate drop + near-duplicate dedupe at |ρ_S| ≥ 0.95) of the selection axes on the pool image, and the redundancy statistics of §3 on the 21 candidate metrics (Spearman matrix, |ρ_S| ≥ 0.7 clusters, normal-score principal components) — a diagnostic. Table `descriptor_redundancy.csv`. The SSI fit check of §3 on the historical record is table `ssi_fit_check.csv`.
1. **Selector comparison** at the campaign bounds on the selection axes: designed rules × S seeds + a wide random null.
2. **Normalization-bounds sweep**: designed rules re-run under (0, 100), (0.5, 99.5), (1, 99), (2, 98). The campaign choice is where tail enrichment and coverage stabilize; the full-range column documents the outlier-fixation failure mode.
3. **Sub-pool draw stability**: the pool is randomly partitioned into disjoint halves — independent i.i.d. pools, since the pool is i.i.d. — and block 1 re-runs per half. Between-half spread is a zero-generation-cost stand-in for pool-re-roll (construction) variance.
B. **Per-axis marginal coverage + tail enrichment** on the selection axes, `lhs_nn` seeds vs the random null band.
C. **Selection-set profile**: per seed, one `lhs_nn` selection on the selection axes gives the snap statistics and the profile statistics of §3: the hazard-direction tail share on every candidate metric, the minimum own-axis tail share and attainment (within-seed minima), the mean target displacement, the share of far targets, and $n_{\text{eff}}/N$; all are averaged over the seed ladder, and `lhs_nn` vs `lhs_assign` order-dependence is measured. Table `selection_set_profile.csv` (one row per statistic and metric: seed mean and SD).
D. **N-sweep**: an N ladder from 50 to 500 (`NYCOPT_SELDIAG_N_SWEEP`) on the selection axes — per-axis tail enrichment and stratification + joint L2-star vs the matched random null. The reported statistic: the minimum over **every** selection axis of the per-axis tail share above the pool P90 (within-seed minimum, averaged over seeds), against the 0.10 share of an i.i.d. selection; no threshold is applied to it.
E. **Selection invariance**: leave-one-axis-out and add-one-metric-back (each other candidate metric) Jaccard overlaps vs the selection on all selection axes; per-axis and dry/wet-group snap-distance contributions.

**Truncation summary**: the pool fractions of windows with a truncated onset and with a truncated termination, the same fractions among the selected members (seed mean), and among the pool members in the top decile (above the pool p90) of each drought metric. Table `truncation_summary.csv`; no figure.

**Figures** (SI): F1 selected members on the (dry, wet) magnitude plane per rule; F2 coverage vs the random null in both geometries; F3 tail enrichment + atom treatment; F4 target displacements + minimum separation; F5 the bounds sweep; F6 metric redundancy (|ρ_S| heatmap over the 21 candidate metrics, cluster tree with the near-duplicate and 0.7 cuts, normal-score PCA spectrum); F7 per-axis coverage and tail enrichment vs the null; F9 the N ladder; F10 selection invariance + implicit weighting; F11 the selection-set profile (tail share on every metric and the set statistics).

## 5. Findings

The findings below are re-measured on the adopted selection axes (`hazard_axis_selection.md` §7) by the production run of the battery on `statpool_10yr_n1000000_d{0,1}` at N = 300 (10 seeds + 50-seed null); only the record-level and pool-level statements that do not depend on the axes are retained.

- **The SSI fit reproduces the historical record.** The record's SSI-6 has a
  standard deviation of 1.00 in every calendar month, and 12 to 14 of the 78
  or 79 values per month lie at or below −1 against 12.4 to 12.5 expected.
- **Metric redundancy (production pools).** 14 clusters at |ρ_S| ≥ 0.7 over the
  21 candidate metrics, identical on both draws; participation ratio 6.27; 9
  normal-score components reach 90% of the variance, led by the total deficit.
- **Dry zero-event atom.** 0.33% of the pool's windows have no qualifying
  SSI-6 event; the truncated-onset and truncated-termination fractions of the
  pool are 8.2% and 10.1%.
- **Selector comparison on the selection axes** (coverage of the campaign
  geometry, tail enrichment, order-dependence of the greedy snap, the atom's
  treatment): [value].
- **Robust bounds (p1/p99)** on the selection axes: [value].
- **Per-axis mechanism** (every selection axis better stratified than the null
  and tail-enriched above it): [value].
- **Enrichment vs N** at P = 10⁶ on the selection axes: [value].
- **Selection-set profile** (tail share on every candidate metric, own-axis
  minimum share and attainment, target displacement, far-target share,
  $n_{\text{eff}}/N$): [value].
- **Truncation** among the selected members and in the drought-metric top
  deciles: [value].
- **Selection invariance and implicit weighting** (leave-one-axis-out and
  add-one-metric-back overlaps; per-axis and dry/wet shares of the squared
  snap displacement): [value].
- **Sub-pool stability** (between-half means of the selector comparison): [value].

Caveat for reading the tables: box-based L2-star structurally favors
target/box-filling rules over manifold-support-filling rules, so F1 (the
selection scatter) is the fair visual comparison.

## 6. Sizing

| Scale | Pool | Use |
|---|---|---|
| Laptop (test) | P ≈ 2,000–5,000, L = 10, stream-only (hazard image only) | Selector + bounds + N evidence; SI draft figures |
| HPC (production) | The production candidate ensemble (P = 10⁶; §5) | Final SI figures on the campaign pool; records the per-axis tail share on the selection axes (N = 300) on every draw |

The experiment reads only `hazard_image.npz` (never pool timeseries) and, for the SSI fit check, the historical record named in the pool's `_meta.json`. The full battery on a synthetic 10⁶-row image at N = 300 (10 seeds, 50 null seeds, the default N ladder) took 45 min and a 3.2 GB peak working set on one workstation, 21 min of it in the bounds sweep and about 1 min in the metric blocks. N, seed counts, and the pool slug are environment-configured in the driver.
