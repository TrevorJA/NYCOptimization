# Hazard-Selector Diagnostics (SI experiment)

*Design of the supplemental selector + normalization diagnostics for the hazard-filling scenario design. Machinery: `scengen/selector_diagnostics.py` (selectors + metric battery) and `scripts/supplemental/diagnose_hazard_selectors.py` (driver). Selection recipe under test: `scenario_design_methods.md` §4.3. Outputs: `outputs/supplemental/hazard_selector_diagnostics/{pool_slug}/`.*

---

## 1. Purpose

Two campaign choices inside the hazard-filling design are conventions unless measured: the **selection rule** that places N members over the hazard manifold, and the **normalization bounds** that define the absolute selection geometry. This experiment measures both on a real candidate pool, entirely at the selection level — no system simulation — so it runs on a laptop test pool and scales unchanged to the production pool on HPC. It backs two SI claims:

1. The campaign selector administers the intervention at strength (coverage, tail enrichment) without pathologies (near-duplicates, outlier fixation, atom mis-handling) relative to defensible alternatives.
2. The robust normalization bounds (p1/p99) stabilize the selection geometry, and the design's headline properties are not artifacts of the bounds choice.

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

Per (rule, seed), on the screened pool sub-image (`selection_metrics`):

- **Coverage uniformity**: L2-star discrepancy in the absolute (campaign) and rank geometries, placed against the many-seed random null; MST edge statistics and minimum pairwise separation (near-duplicate guard) in absolute geometry.
- **Tail enrichment**: mean per-axis share above the pool P90 (unbiased ≈ 0.10) and the any-axis P90 corner share — the deliberate distribution shift, quantified.
- **Marginal distortion**: mean KS distance to the pool marginals.
- **Dry zero-event atom**: pool share of windows with no SSI-6 ≤ −1 event, and each rule's selected share — how each rule treats the atom at the dry origin.
- **Stability**: across-seed selected-set Jaccard per rule (deterministic rules report 1 by construction); anchor snap distances for the LHS rules.

## 4. Analysis blocks

1. **Selector comparison** at the campaign bounds: designed rules × S seeds + a wide random null.
2. **Normalization-bounds sweep**: designed rules re-run under (0, 100), (0.5, 99.5), (1, 99), (2, 98). The campaign choice is where tail enrichment and coverage stabilize; the full-range column documents the outlier-fixation failure mode.
3. **Sub-pool draw stability**: the pool is randomly partitioned into disjoint halves — independent i.i.d. pools, since the pool is i.i.d. — and block 1 re-runs per half. Between-half spread is a zero-generation-cost stand-in for pool-re-roll (construction) variance, and doubles as the ensemble-stability-across-generation-seeds SI diagnostic.
4. **Figures** (SI): F1 selected members on the (dry, wet) magnitude plane per rule; F2 coverage vs the random null in both geometries; F3 tail enrichment + atom treatment; F4 snap distances + minimum separation; F5 the bounds sweep.

## 5. Sizing

| Scale | Pool | Use |
|---|---|---|
| Laptop (test) | P ≈ 2,000–5,000, L = 10, stream-only (hazard image only) | Selector + bounds decision for the campaign; SI draft figures |
| HPC (production) | The production candidate pool (P = 10⁵–10⁶) | Final SI figures on the campaign pool; confirms the laptop-scale decision |

The experiment reads only `hazard_image.npz` (never pool timeseries), so production-scale cost is minutes. N, seed counts, and the pool slug are environment-configured in the driver.
