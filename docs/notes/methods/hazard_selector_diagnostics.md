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
C. **Snap behavior vs dimension** at nested axis sets m4 ⊂ m6 ⊂ full (m4 = deficit volume, onset rate, peak magnitude, pulse duration — the benchmark subset carried for continuity with the earlier battery; m6 adds the next canonical-priority member per tail), including `lhs_nn` vs `lhs_assign` order-dependence at full m.
D. **N-sweep**: N ∈ {100, 150, 200, 300} × the block-C axis sets — per-axis tail enrichment and stratification + joint L2-star vs the matched random null. The adequacy criterion, stated before results: minimum per-axis tail share ≥ ~0.30 (≥ ~3× the unbiased 0.10) on **every** axis.
E. **Selection invariance**: leave-one-axis-out and add-one-axis-back Jaccard overlaps vs the full-set selection; per-axis and dry/wet-group snap-distance contributions.

**Figures** (SI): F1 selected members on the (dry, wet) magnitude plane per rule; F2 coverage vs the random null in both geometries; F3 tail enrichment + atom treatment; F4 snap distances + minimum separation; F5 the bounds sweep; F6 Spearman heatmap + cluster tree; F7 per-axis coverage and tail enrichment vs the null; F8 snap behavior vs dimension; F9 the (N × axis set) sizing surface; F10 selection invariance + implicit weighting.

## 5. Findings at laptop scale (P = 2,000, L = 10, N = 100; 10 seeds + 50-seed null)

Results in `outputs/supplemental/hazard_selector_diagnostics/statpool_10yr_n2000_d0/`.

- **Axis screen: all 8 candidates retained (m = 8).** No degenerate axes; no near-duplicate pair — the largest |ρ_S| is 0.88 (deficit volume ↔ duration), below the 0.95 dedupe cut and still below a 0.90 tightening, so the retained set is insensitive to the threshold over that range. The cluster tree shows the expected concept groups (integrated-drought {deficit volume, duration, peak depth}; drought rates; flood magnitude/rise; pulse duration) with all between-group |ρ_S| ≤ 0.61.
- **Campaign selector re-confirmed (`lhs_nn`) at the full axis set.** Best coverage of the campaign geometry (L2-star 0.023 vs 0.132 random) and tail enrichment at strength (mean share above pool P90 = 0.259 vs the 0.10 of an unbiased rule; any-axis corner share 0.95 vs 0.51 random). `lhs_assign` is metric-indistinguishable (L2-star 0.024, tail 0.256) and selects nearly the same members (per-seed Jaccard 0.83 at full m), so the greedy snap's order-dependence remains immaterial at eight axes. No near-duplicate pathology (minimum separation 0.164 vs 0.104 random). The dry zero-event atom is 0.7% of windows and `lhs_nn` selects none of it.
- **Per-axis mechanism holds on every axis.** At full m and N = 100 every retained axis is both better stratified than the null (KS to uniform below the null mean on all 8 axes) and tail-enriched above it (per-axis tail shares 0.14–0.37 vs ≈ 0.10): the marginal-coverage guarantee survives the move to eight axes. Enrichment is not uniform across axes — weakest on drought duration (0.151) and flood rise rate (0.195), strongest on drought onset rate (0.368).
- **Dimension dilutes the snap at this pool size.** From m4 to the full set the mean anchor snap distance grows 0.205 → 0.606 and the distance-concentration ratio 0.39 → 0.80, i.e. at P = 2,000 in eight dimensions a snapped member is barely closer to its anchor than a random pool point; per-axis mean tail share falls 0.354 → 0.259 accordingly. This is a finite-pool effect — the expected anchor-to-nearest-member distance scales as P^(−1/m) — not an intrinsic property of the axis set, so the production pool size P is the binding lever on full-set enrichment. The **pool-size saturation diagnostic (nested P) and the production-pool rerun decide it**; the laptop numbers are the conservative floor.
- **Raising N does not buy enrichment at fixed P.** Across N = 100 → 300, per-axis tail shares are flat-to-declining at every axis set (full-set mean 0.259 → 0.228; m4 0.354 → 0.300): the pool holds only ~P/10 members above P90 per axis, so additional selections increasingly draw from the bulk. Joint L2-star also degrades mildly with N. Under the pre-stated adequacy criterion (min per-axis tail share ≥ ~0.30 on every axis) **no (axis set, N) combination passes at P = 2,000** — the m4 benchmark itself reaches only 0.256 at N = 100 — and N = 100 is the best value in the sweep at every axis set. The criterion is therefore a gate on the production pool size, not on N.
- **Robust bounds re-confirmed (p1/p99) at full m.** Full-range (0, 100) bounds degrade realized coverage ~2.5–3× (`lhs_nn` L2-star 0.031 → 0.076 between (0.5, 99.5) and (0, 100)) — the outlier-fixation failure mode — while tail enrichment moves smoothly across (2, 98)–(0.5, 99.5) (0.219–0.282) with no sensitivity cliff at the campaign default.
- **The selection is not hostage to any single axis, and the implicit weighting is disclosed.** Leave-one-axis-out selections overlap the full-set selection at Jaccard 0.18–0.27 with no outlier axis (add-one-back from the m4 base: 0.18–0.23): every axis moves the selection somewhat and none dominates it. Per-axis shares of the squared snap displacement are 0.08–0.16 (equal weighting = 0.125); the dry group carries 0.67 vs the 0.625 of pure axis-count proportionality — the 5:3 dry:wet count, not hidden concept-doubling, sets the weighting.
- **Alternatives characterized.** `maximin` concentrates on the hull and over-selects the sparse zero-event corner (4% of its members); `eps_cell` under-enriches the tails (0.191). Caveat for reading the tables: box-based L2-star structurally favors anchor/box-filling rules over manifold-support-filling rules, so F1 (the selection scatter) is the fair visual comparison.
- **Sub-pool stability**: between-half means are identical to three digits for `lhs_nn` (tail share 0.229 / 0.229) — the seed/construction-stability SI evidence.

**Recommendation from this battery** (final call and any budget change are the user's): retain the **full descriptor set (m = 8)** — the screen finds no near-duplicates to prune, no axis exhibits a pathology, and the per-axis guarantee holds on all eight; keep **N = 100** — larger N is strictly worse on enrichment at fixed P and costs linearly (173.8 s/eval at N = 100; N = 150 would raise every search cost 1.5×). The open lever is the **production pool size P**: the production-pool rerun must show min per-axis tail share ≥ ~0.30 at (m = 8, N = 100) — expected, since tail supply and snap density both improve with P — before the campaign gates.

## 5b. Nested-P saturation outcome (HPC, 2026-07-29/30) and the axis-set decision

The pool-size saturation diagnostic ran on Anvil as prefix rungs {2k, 5k, 20k, 10⁵, 3×10⁵, 10⁶} of ONE stream-only P = 10⁶ pool (`outputs/supplemental/hazard_selector_diagnostics/nested_P_saturation.md`; prefixes are honest i.i.d. pools by the global-index seeding). **The (m = 8, N = 100) gate fails at every rung** (min per-axis tail share 0.144 → 0.256; improvement exponent ~0.04, far below the P^(−1/8) bound from the intrinsic dimension — geometry-limited, so P is not the lever), activating the pre-registered fallback ladder. A follow-on candidate-set assessment (`m6_axis_set_assessment.json`) measured the fallback options: the **campaign decision (2026-07-30) is the m = 6 set** {deficit volume, peak depth, onset rate, recovery rate, peak magnitude, pulse duration}, which passes at P = 10⁶ (min 0.311; thin margin — re-confirm per production draw); keeping duration instead of peak depth fails (0.265, pinned by duration's quasi-discrete tail), as does the m4+duration+rise-rate nesting (0.249, pinned by the flood-group-entangled rise rate). The m4 benchmark passes from P ≈ 10⁵ with margin (0.353 at 10⁶). Wired as `config.HAZARD_SELECTION_AXES`, consumed by the step-03 selection; the dropped descriptors stay computed and reportable.

## 6. Sizing

| Scale | Pool | Use |
|---|---|---|
| Laptop (test) | P ≈ 2,000–5,000, L = 10, stream-only (hazard image only) | Selector + bounds + axis-set + N decisions for the campaign; SI draft figures |
| HPC (production) | The production candidate pool (P = 10⁶, decided §5b) | Final SI figures on the campaign pool; must confirm the per-axis adequacy criterion at the campaign selection set (m = 6, N = 100) on every draw |

The experiment reads only `hazard_image.npz` (never pool timeseries), so production-scale cost is minutes. N, seed counts, and the pool slug are environment-configured in the driver.
