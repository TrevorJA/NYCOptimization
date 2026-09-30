# Hazard-Selector Diagnostics (SI experiment)

*Design of the supplemental selector, axis-set, and sizing diagnostics for the hazard-filling scenario design. Machinery: `scengen/selector_diagnostics.py` (selectors + metric battery) and `scripts/supplemental/diagnose_hazard_selectors.py` (driver; descriptor settings `SELDIAG_*` in `supplemental_config.py`). Selection recipe under test: `scenario_design_methods.md` §4.3; axis policy and the hazard-image supplement: §3.3. Outputs: `outputs/supplemental/hazard_selector_diagnostics/{pool_slug}/`.*

---

## 1. Purpose

Five campaign choices inside the hazard-filling design are conventions unless measured: the **selection rule** that places N members over the hazard manifold, the **normalization bounds** that define the absolute selection geometry, the **retained axis set** (all non-degenerate descriptors minus near-duplicates at |ρ_S| ≥ 0.95), the **selection axis set** among named alternatives, and the **ensemble size N**. This experiment measures all five on a real candidate ensemble, entirely at the selection level — no system simulation — so it runs on a laptop test pool and scales unchanged to the production pool on HPC. It backs four SI claims:

1. The campaign selector administers the intervention at strength (coverage, tail enrichment) without pathologies (near-duplicates, outlier fixation, atom mis-handling) relative to defensible alternatives.
2. The robust normalization bounds (p1/p99) stabilize the selection geometry, and the design's headline properties are not artifacts of the bounds choice.
3. The full retained axis set delivers the design's per-axis marginal coverage guarantee, the selection is not hostage to any single correlated axis, and the implicit weighting that correlated axes induce in the selection distance is characterized (a disclosed non-issue, not a correction).
4. The selection axis set is chosen on measured redundancy of the full descriptor set (the candidate axes plus the hazard-image supplement, `scenario_design_methods.md` §3.3) and on the tail coverage each named set delivers on every descriptor, including those it does not select on; window-edge truncation of the scored drought events is reported alongside.

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

On the descriptor set — the 8 candidate axes plus the 13 supplement descriptors (the two truncation flags excluded):

- **Descriptor redundancy**: the Spearman matrix; average-linkage clusters on $1-\lvert\rho_S\rvert$ cut so every pair at $\lvert\rho_S\rvert \ge 0.7$ shares a cluster (0.7, the level above which collinearity matters, Dormann et al. 2013); a principal component analysis of the correlation matrix of the normal scores (each descriptor's ranks mapped through the standard normal quantile), reporting the eigenvalues, the participation ratio $(\sum\lambda)^2/\sum\lambda^2$, the number of components that reach 90% of the variance, and the highest-loading descriptor of each of those components (index reduction by principal components, Olden & Poff 2003).
- **Hazard-direction tail share** of a selection on every descriptor: the share of selected members beyond the pool p90, or below the pool p10 for the four low-flow minima, where low values are hazardous (i.i.d. share 0.10).
- **Attainment** of a selection axis: its tail share over the share an exact snap to uniform targets on the clipped p1–p99 range would give, $(p_{99} - p_{90})/(p_{99} - p_1)$; the minimum over the set's own axes is reported.
- **Target displacement and reach**: the mean target-to-member displacement of the `lhs_nn` pairing, and the share of targets farther than 0.25 (scaled units) from every pool member.
- **Nearest-member weights**: $n_{\text{eff}}/N$ of the Voronoi masses the pool assigns to the selected members (the measure statement of `hf_design_metrics.md` §5, computed with its helpers).
- **Truncation**: the fractions of windows whose largest drought event has a truncated onset or termination (the two supplement flags).

On the historical record:

- **SSI fit check**: the record's SSI-6 under the reference fit, the two-parameter gamma per calendar month that every realization is transformed with. Per calendar month it reports the standard deviation, the count of values at or below −1 (the level a run-theory event must reach to qualify) and the count a standard normal gives, $n\,\Phi(-1)$. A fit that reproduces the record gives a standard deviation of 1 and a count near the expected one in every month.

## 4. Analysis blocks

A. **Retained-set report and descriptor redundancy**: the axis screen (degenerate drop + near-duplicate dedupe at |ρ_S| ≥ 0.95) on the pool image, and the descriptor-redundancy statistics of §3 on the 21 descriptors (Spearman matrix, |ρ_S| ≥ 0.7 clusters, normal-score principal components) — a diagnostic, never used to reduce the set further. Table `descriptor_redundancy.csv`. The SSI fit check of §3 on the historical record is table `ssi_fit_check.csv`.
1. **Selector comparison** at the campaign bounds on the full retained set: designed rules × S seeds + a wide random null.
2. **Normalization-bounds sweep**: designed rules re-run under (0, 100), (0.5, 99.5), (1, 99), (2, 98). The campaign choice is where tail enrichment and coverage stabilize; the full-range column documents the outlier-fixation failure mode.
3. **Sub-pool draw stability**: the pool is randomly partitioned into disjoint halves — independent i.i.d. pools, since the pool is i.i.d. — and block 1 re-runs per half. Between-half spread is a zero-generation-cost stand-in for pool-re-roll (construction) variance.
B. **Per-axis marginal coverage + tail enrichment** at the full retained set, `lhs_nn` seeds vs the random null band.
C. **Snap behavior vs dimension and the axis-set comparison** over the named axis sets of `supplemental_config.seldiag_axis_sets`: the campaign selection set (`config.HAZARD_SELECTION_AXES`, m = 6), the full retained set, `four_axis` (magnitude, severity, peak discharge, pulse duration), `four_axis_rate` (magnitude, development rate, peak discharge, pulse duration) and `five_axis` (the campaign set without severity), each restricted to the retained axes. Per set and seed, one `lhs_nn` selection gives the snap statistics and the comparison statistics of §3: the hazard-direction tail share on every descriptor, the minimum own-axis tail share and attainment (within-seed minima), the mean target displacement, the share of far targets, and $n_{	ext{eff}}/N$; all are averaged over the seed ladder. `lhs_nn` vs `lhs_assign` order-dependence is measured at the campaign and full sets. The two diagnostic axis sets of blocks D and E and of saturation mode remain the campaign and full sets. Table `axis_set_comparison.csv` (one row per set, statistic and descriptor: seed mean and SD).
D. **N-sweep**: an N ladder from 50 to 500 (`NYCOPT_SELDIAG_N_SWEEP`) × the block-C axis sets — per-axis tail enrichment and stratification + joint L2-star vs the matched random null. The reported statistic: the minimum over **every** selection axis of the per-axis tail share above the pool P90 (within-seed minimum, averaged over seeds), against the 0.10 share of an i.i.d. selection; no threshold is applied to it.
E. **Selection invariance**: leave-one-axis-out and add-one-axis-back (campaign base) Jaccard overlaps vs the full-set selection; per-axis and dry/wet-group snap-distance contributions.

**Truncation summary**: the pool fractions of windows with a truncated onset and with a truncated termination, the same fractions among the block-C selections of each named set (seed mean), and among the pool members in the top decile (above the pool p90) of each drought axis. Table `truncation_summary.csv`; no figure.

**Figures** (SI): F1 selected members on the (dry, wet) magnitude plane per rule; F2 coverage vs the random null in both geometries; F3 tail enrichment + atom treatment; F4 target displacements + minimum separation; F5 the bounds sweep; F6 descriptor redundancy (|ρ_S| heatmap over the 21 descriptors, cluster tree with the near-duplicate and 0.7 cuts, normal-score PCA spectrum); F7 per-axis coverage and tail enrichment vs the null; F8 snap behavior vs dimension; F9 the (N × axis set) sizing surface; F10 selection invariance + implicit weighting; F11 the axis-set comparison (tail share of every named set on every descriptor, and the per-set statistics).

## 5. Findings

Pilot battery: P = 2,000, L = 10, N = 100; 10 seeds + 50-seed null
(`outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/`).
Production axis-set evidence: nested-P saturation rungs {2k, 5k, 20k, 10⁵,
3×10⁵, 10⁶} of one stream-only P = 10⁶ pool (`workflow/supplemental/nestedp_ladder.sh`,
`scripts/supplemental/nestedp_saturation_analysis.py`, which writes
`nested_P_saturation.md` under the diagnostics output root; prefixes are
honest i.i.d. pools by the global-index seeding).

- **Axis screen: all 8 candidates retained (m = 8).** No degenerate axes; no
  near-duplicate pair — the largest |ρ_S| is 0.88 (drought magnitude ↔
  duration), below the 0.95 dedupe cut and still below a 0.90 tightening.
  The cluster tree shows the expected concept groups with all between-group
  |ρ_S| ≤ 0.61.
- **Campaign selector confirmed (`lhs_nn`).** Best coverage of the campaign
  geometry (L2-star 0.023 vs 0.132 random) and tail enrichment at strength
  (mean per-axis share above pool P90 = 0.259 vs 0.10 unbiased; any-axis
  corner share 0.95 vs 0.51 random). `lhs_assign` is metric-indistinguishable
  and selects nearly the same members (per-seed Jaccard 0.83), so the greedy
  snap's order-dependence is immaterial. No near-duplicate pathology; the dry
  zero-event atom is 0.7% of windows and `lhs_nn` selects none of it.
  `maximin` concentrates on the hull and over-selects the sparse zero-event
  corner; `eps_cell` under-enriches the tails.
- **Per-axis mechanism holds on every axis**: every retained axis is both
  better stratified than the null and tail-enriched above it. Target displacement
  grows with dimension at fixed P (the expected target-to-nearest-member
  distance scales as P^(−1/m)).
- **Enrichment is flat in N at the production pool size.** At P = 10⁶ the
  campaign-set minimum tail share is flat in N (0.27–0.29 from N = 50 to
  500) and joint L2-star improves with N (0.019 → 0.011); a decline of
  tail share with N appears only on small prefixes P′ ≤ 2·10⁴, where the
  pool holds only ~P′/10 members above P90 per axis, and does not apply at
  the production scale. Pool size therefore does not bound N from above, and
  the campaign N = 300 is set by the ensemble-size diagnostics
  (`ensemble_size_diagnostics.md`, `campaign_design.md`). On the production
  pool d0 the minimum share is 0.283 (drought_magnitude binding; per-axis
  0.284–0.513, mean 0.385).
- **Robust bounds confirmed (p1/p99).** Full-range (0, 100) bounds degrade
  realized coverage ~2.5–3× (outlier fixation); tail enrichment moves
  smoothly across (2, 98)–(0.5, 99.5) with no cliff at the campaign default.
- **Selection axis set: the campaign selects on m = 6**
  ({drought magnitude, severity, development rate, termination rate, peak discharge,
  pulse duration} = `config.HAZARD_SELECTION_AXES`, consumed by the step-03
  selection). The full 8-axis set's minimum per-axis tail share is
  geometry-limited, not supply-limited: ~0.22 at P = 10⁶, with the nested-P
  rungs showing an improvement exponent ~0.04, far below the P^(−1/8)
  bound, so no affordable pool lifts it. The m = 6 set sits at 0.27–0.29 on
  the production P = 10⁶ pools (≈ 3× the 0.10 i.i.d. share; recorded per
  production draw), saturated in pool size and flat in N; the measured
  alternatives (duration for severity; duration + rise rate both in)
  enriched less on the same pools. The dropped descriptors stay computed in every hazard
  image and reportable post-hoc. Blocks D and E score two axis sets —
  campaign and full — the full set serving as the measured evidence for
  restricting selection; block C compares the five named sets.
- **The SSI fit reproduces the historical record.** The record's SSI-6 has a
  standard deviation of 1.00 in every calendar month, and 12 to 14 of the 78
  or 79 values per month lie at or below −1 against 12.4 to 12.5 expected.
- **Descriptor redundancy (production pools).** [value] clusters at
  |ρ_S| ≥ 0.7 over the 21 descriptors; participation ratio [value];
  [value] normal-score components reach 90% of the variance, led by
  [value].
- **Axis-set comparison (production pools, N = 300).** Minimum own-axis
  tail share and attainment per named set [value]; hazard-direction tail
  share on the descriptors each set does not select on [value]; mean
  target displacement, far-target share and $n_{\text{eff}}/N$ per set
  [value].
- **Truncation (production pools).** Pool fractions of windows with a
  truncated onset / termination [value]; the same among the selected
  members per set [value] and in the drought-axis top deciles [value].
- **Not hostage to any single axis; implicit weighting disclosed.**
  Leave-one-axis-out selections overlap the full-set selection at Jaccard
  0.18–0.27 with no outlier axis; per-axis shares of the squared snap
  displacement are near-equal, and the dry group carries 0.67 vs the 0.625 of
  pure axis-count proportionality — the dry:wet axis count, not hidden
  concept-doubling, sets the weighting.
- **Sub-pool stability**: between-half means identical to three digits for
  `lhs_nn` — the seed/construction-stability SI evidence.

Caveat for reading the tables: box-based L2-star structurally favors
target/box-filling rules over manifold-support-filling rules, so F1 (the
selection scatter) is the fair visual comparison.

## 6. Sizing

| Scale | Pool | Use |
|---|---|---|
| Laptop (test) | P ≈ 2,000–5,000, L = 10, stream-only (hazard image only) | Selector + bounds + N evidence; SI draft figures |
| HPC (production) | The production candidate ensemble (P = 10⁶; §5) | Final SI figures on the campaign pool; records the per-axis tail share at the campaign selection set (m = 6, N = 300) on every draw |

The experiment reads only `hazard_image.npz` (never pool timeseries) and, for the SSI fit check, the historical record named in the pool's `_meta.json`. The full battery on a synthetic 10⁶-row image at N = 300 (10 seeds, 50 null seeds, the default N ladder) took 45 min and a 3.2 GB peak working set on one workstation, 21 min of it in the bounds sweep and about 1 min in the descriptor blocks. N, seed counts, and the pool slug are environment-configured in the driver.
