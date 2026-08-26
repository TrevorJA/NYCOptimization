# Hazard-Selector Diagnostics (SI experiment)

*Design of the supplemental selector, axis-set, and sizing diagnostics for the hazard-filling scenario design. Machinery: `scengen/selector_diagnostics.py` (selectors + metric battery) and `scripts/supplemental/diagnose_hazard_selectors.py` (driver). Selection recipe under test: `scenario_design_methods.md` §4.3; axis policy: §3.3. Outputs: `outputs/supplemental/hazard_selector_diagnostics/{pool_slug}/`.*

---

## 1. Purpose

Four campaign choices inside the hazard-filling design are conventions unless measured: the **selection rule** that places N members over the hazard manifold, the **normalization bounds** that define the absolute selection geometry, the **retained axis set** (all non-degenerate descriptors minus near-duplicates at |ρ_S| ≥ 0.95), and the **ensemble size N**. This experiment measures all four on a real candidate pool, entirely at the selection level — no system simulation — so it runs on a laptop test pool and scales unchanged to the production pool on HPC. It backs three SI claims:

1. The campaign selector administers the intervention at strength (coverage, tail enrichment) without pathologies (near-duplicates, outlier fixation, atom mis-handling) relative to defensible alternatives.
2. The robust normalization bounds (p1/p99) stabilize the selection geometry, and the design's headline properties are not artifacts of the bounds choice.
3. The full retained axis set delivers the design's per-axis marginal coverage guarantee, the selection is not hostage to any single correlated axis, and the implicit weighting that correlated axes induce in the snap distance is characterized (a disclosed non-issue, not a correction).

## 2. Selection rules compared

All rules select N members from the same pool sub-image, normalized once with the campaign robust bounds, so differences are attributable to the rule alone.

| Rule | Construction | Role |
|---|---|---|
| `random` | Without replacement | The null every designed rule must beat (many-seed null band). |
| `lhs_nn` | LHS anchors + greedy nearest-unused-neighbor snap | The wired status-quo selector. |
| `lhs_assign` | Same anchors, optimal one-to-one assignment (Hungarian) | Isolates the greedy snap's order-dependence: same plan, globally optimal pairing. |
| `maximin` | Greedy maximin distance (Kennard–Stone type; Johnson et al. 1990) | The DOE-standard comparator; anchor-free but known to load the hull. |
| `eps_cell` | Grid the unit box at the coarsest resolution with ≥ N occupied cells; draw N occupied cells uniformly; one representative per cell (nearest cell center) | Uniform over the manifold's *occupied support* at resolution ε — no anchor can land off the manifold; one-per-cell separation guarantee. Coverage analogue of ε-dominance archiving (Laumanns et al. 2002). |

Anchor-based rules face the manifold-support problem: hazard axes are structurally dependent (run theory: deficit ≈ duration × intensity), so part of the unit box is unoccupied and anchors placed there must snap. The snap-distance distribution measures that cost; `maximin` and `eps_cell` are the anchor-free comparators.

## 3. Metric battery

Per (rule, seed), on the screened pool sub-image (`selection_metrics`, `per_axis_selection_metrics`):

- **Coverage uniformity**: L2-star discrepancy in the absolute (campaign) and rank geometries, placed against the many-seed random null; MST edge statistics and minimum pairwise separation (near-duplicate guard) in absolute geometry.
- **Per-axis marginal coverage** — the mechanism metric: LHS anchors stratify every axis into N bins regardless of dimension, so the design's coverage guarantee is per-axis marginal, not joint. Per axis in the campaign scaled coordinates: KS distance of the selected marginal to uniform, 1-D L2-star discrepancy, largest marginal gap, and the tail share above the pool P90 (unbiased ≈ 0.10).
- **Tail enrichment**: mean per-axis share above the pool P90 and the any-axis P90 corner share — the deliberate distribution shift, quantified.
- **Snap behavior vs dimension**: snap-distance distribution and the distance-concentration ratio (mean snap distance / mean random pool-pair distance in the same space) — raw snap distances are not comparable across dimensions, the ratio is.
- **Marginal distortion**: mean KS distance to the pool marginals.
- **Dry zero-event atom**: pool share of windows with no SSI-6 ≤ −1 event, and each rule's selected share.
- **Stability**: across-seed selected-set Jaccard per rule; anchor snap distances for the LHS rules.
- **Selection invariance / implicit weighting**: Jaccard overlap of selected member IDs between the full-axis-set selection and leave-one-axis-out / add-one-axis-back variants; per-axis (and dry-vs-wet group) mean share of the squared snap displacement.

## 4. Analysis blocks

A. **Retained-set report**: the axis screen (degenerate drop + near-duplicate dedupe at |ρ_S| ≥ 0.95) on the pool image, with the Spearman matrix and 1−|ρ_S| cluster tree (Olden & Poff 2003 framing) reported as a redundancy diagnostic — never used to reduce the set further.
1. **Selector comparison** at the campaign bounds on the full retained set: designed rules × S seeds + a wide random null.
2. **Normalization-bounds sweep**: designed rules re-run under (0, 100), (0.5, 99.5), (1, 99), (2, 98). The campaign choice is where tail enrichment and coverage stabilize; the full-range column documents the outlier-fixation failure mode.
3. **Sub-pool draw stability**: the pool is randomly partitioned into disjoint halves — independent i.i.d. pools, since the pool is i.i.d. — and block 1 re-runs per half. Between-half spread is a zero-generation-cost stand-in for pool-re-roll (construction) variance.
B. **Per-axis marginal coverage + tail enrichment** at the full retained set, `lhs_nn` seeds vs the random null band.
C. **Snap behavior vs dimension** at the two diagnostic axis sets — the campaign selection set (`config.HAZARD_SELECTION_AXES`, m = 6) and the full retained set — including `lhs_nn` vs `lhs_assign` order-dependence at full m.
D. **N-sweep**: N ∈ {100, 150, 200, 300} × the block-C axis sets — per-axis tail enrichment and stratification + joint L2-star vs the matched random null. The adequacy criterion, stated before results: minimum per-axis tail share ≥ ~0.30 (≥ ~3× the unbiased 0.10) on **every** axis.
E. **Selection invariance**: leave-one-axis-out and add-one-axis-back (campaign base) Jaccard overlaps vs the full-set selection; per-axis and dry/wet-group snap-distance contributions.

**Figures** (SI): F1 selected members on the (dry, wet) magnitude plane per rule; F2 coverage vs the random null in both geometries; F3 tail enrichment + atom treatment; F4 snap distances + minimum separation; F5 the bounds sweep; F6 Spearman heatmap + cluster tree; F7 per-axis coverage and tail enrichment vs the null; F8 snap behavior vs dimension; F9 the (N × axis set) sizing surface; F10 selection invariance + implicit weighting.

## 5. Findings

Laptop battery: P = 2,000, L = 10, N = 100; 10 seeds + 50-seed null
(`outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/`).
Production axis-set decision: nested-P saturation rungs {2k, 5k, 20k, 10⁵,
3×10⁵, 10⁶} of ONE stream-only P = 10⁶ pool
(`nested_P_saturation.md` + `m6_axis_set_assessment.json`; prefixes are
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
  better stratified than the null and tail-enriched above it. Snap distance
  dilutes with dimension at fixed P (the expected anchor-to-nearest-member
  distance scales as P^(−1/m)).
- **N = 100 confirmed at P = 2,000; raising N does not buy enrichment at
  fixed P.** Across N = 100 → 300 per-axis tail shares are flat-to-declining
  at every axis set (the pool holds only ~P/10 members above P90 per axis),
  and joint L2-star degrades mildly. N = 100 is the best value in the sweep,
  and larger N costs linearly in every search. CORRECTION 2026-08-25
  (`ensemble_size_diagnostics.md` §7.1, block D extended to N = 50…500 on
  the production P = 10⁶ pools, `NYCOPT_SELDIAG_N_SWEEP`): the decline is a
  small-pool effect. At P = 10⁶ the campaign-set minimum tail share is flat
  in N (0.27–0.29 from N = 50 to 500) and joint L2-star improves with N
  (0.019 → 0.011); the decline appears only on prefixes P′ ≤ 2·10⁴. The
  same run re-gated the regenerated pool d0 at 0.283 (drought_magnitude
  binding) — a thin miss of the 0.30 gate that the pre-regeneration pools
  passed at 0.311.
- **Robust bounds confirmed (p1/p99).** Full-range (0, 100) bounds degrade
  realized coverage ~2.5–3× (outlier fixation); tail enrichment moves
  smoothly across (2, 98)–(0.5, 99.5) with no cliff at the campaign default.
- **Selection axis set: the campaign selects on m = 6**
  ({drought magnitude, severity, onset rate, recovery rate, peak discharge,
  pulse duration} = `config.HAZARD_SELECTION_AXES`, consumed by the step-03
  selection). The full 8-axis set cannot pass the pre-stated adequacy gate
  (min per-axis tail share ≥ ~0.30) at any affordable pool size — the
  nested-P rungs show an improvement exponent ~0.04, far below the P^(−1/8)
  bound, i.e. geometry-limited, not supply-limited. The m = 6 set passes at
  P = 10⁶ (min 0.311; thin margin — re-confirm per production draw); the
  measured alternatives (duration for severity; duration + rise rate both
  in) fail the gate. The dropped descriptors stay computed in every hazard
  image and reportable post-hoc. The battery scores exactly two axis sets —
  campaign and full — the full set serving as the measured evidence for
  restricting selection.
- **Not hostage to any single axis; implicit weighting disclosed.**
  Leave-one-axis-out selections overlap the full-set selection at Jaccard
  0.18–0.27 with no outlier axis; per-axis shares of the squared snap
  displacement are near-equal, and the dry group carries 0.67 vs the 0.625 of
  pure axis-count proportionality — the dry:wet axis count, not hidden
  concept-doubling, sets the weighting.
- **Sub-pool stability**: between-half means identical to three digits for
  `lhs_nn` — the seed/construction-stability SI evidence.

Caveat for reading the tables: box-based L2-star structurally favors
anchor/box-filling rules over manifold-support-filling rules, so F1 (the
selection scatter) is the fair visual comparison.

## 6. Sizing

| Scale | Pool | Use |
|---|---|---|
| Laptop (test) | P ≈ 2,000–5,000, L = 10, stream-only (hazard image only) | Selector + bounds + N evidence; SI draft figures |
| HPC (production) | The production candidate pool (P = 10⁶; §5) | Final SI figures on the campaign pool; must confirm the per-axis adequacy criterion at the campaign selection set (m = 6, N = 100) on every draw |

The experiment reads only `hazard_image.npz` (never pool timeseries), so production-scale cost is minutes. N, seed counts, and the pool slug are environment-configured in the driver.
