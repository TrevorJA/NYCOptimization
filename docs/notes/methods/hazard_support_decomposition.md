# Hazard-Support Decomposition of the E_test Design Contrast (SI)

*Last updated: 2026-08-25. Supplemental diagnostic (SI Text S10) decomposing the
`hazard_filling` − `fixed_probabilistic` difference on E_test by where each state
of the world (SOW) sits relative to the stationary candidate pool's hazard
support. Zero simulation: every deliverable reduces persisted artifacts. Code:
`scripts/supplemental/hazard_support_run.py` (stage A support/membership tables,
stage B design contrast) and `scripts/supplemental/hazard_support_figures.py`,
configuration in `supplemental_config.py` (`HSD_*`), wrapper
`workflow/supplemental/hazard_support_decomposition.sh`, tests
`tests/test_hazard_support_decomposition.py`. Outputs under
`outputs/supplemental/hazard_support_decomposition/{tables,figures}/`.*

**PRE-REGISTRATION.** The support definition, strata cut points, endpoints, and
decision rules in this note were fixed on 2026-08-25 BEFORE stage B was run on
any cube. Stage A (support membership) is policy-free and runs on the
regenerated pool and E_test hazard images; stage B consumes the stage-A stratum
labels unchanged.

**THIS IS A PRE-CAMPAIGN DECISION INSTRUMENT.** Its job is to settle, before
any campaign search SU is spent, whether the hazard-filling candidate pool
should carry a climate-forcing signal. It therefore runs on the re-evaluated
cubes that exist before the campaign — the first-round go/no-go sets scored on
the interim 200-SOW `etest_kn_50yr_n25000_first10ch` subset — and is read with
that substrate's limits stated (§6). Re-running it on the production cubes
after the campaign would answer the question too late to act on. The decision
is recorded in §7.

---

## 1. Question and scope

The study's central claim is a generalization claim (`experimental_design.md`,
"The re-evaluation is a generalization test"): hazard-space coverage of the
stationary natural-variability manifold during search produces policies that
generalize to forced climate conditions the search never saw, better than i.i.d.
sampling from the same generator. E_test spans a CMIP6-informed forcing envelope;
neither search design contains any climate signal. The obvious extension —
putting climate-forced sub-ensembles inside the hazard-filling candidate pool —
is NOT run in this study, because it would confound the selection-rule factor
with search-population breadth and leak the test distribution into one arm.
Instead the question is answered with the instruments already in the design, at
zero simulation cost:

> Decompose the HF − PS difference on E_test by whether each SOW's hazard image
> lies inside, at the fringe of, or outside the stationary pool's hazard
> support. If HF's advantage persists on out-of-support SOWs, that is the
> generalization result. If it vanishes there, that bounds the claim and
> motivates the climate-augmented extension with evidence.

This is the hazard-restricted composition-sensitivity re-scoring promised in
`experimental_design.md` and SI Text S10, made concrete. **Scope guards:** it is
supplemental, never a new primary endpoint; hazard-space coverage remains method
verification; the one hazard-space inference stays the step-11 coverage-deficit
mechanism test; no policy-architecture comparison is introduced.

**Claim scoping, stated in advance.** The six campaign selection axes
(`config.HAZARD_SELECTION_AXES`) encode drought/flood event *magnitudes* on the
aggregate NYC inflow. E_test's forcing also moves seasonal structure (the
harmonic amplitudes r1, r2), which the axes do not encode, so a seasonal
excursion is out-of-support for BOTH arms and invisible to this decomposition.
The diagnostic therefore bounds hazard-*magnitude* generalization; the
seasonal-structure dimension is a scope limitation shared by both designs, not a
bias between them.

## 2. Support membership (stage A — policy-free)

**Substrate.** The E_test sub-window hazard image
(`hazard_image_subwindows.npz`: 25,000 realizations × 5 disjoint 10-yr
December-aligned sub-windows = 125,000 rows × 8 candidate axes, scored on the
pool's exact metric span — `scenario_design_methods.md` §5.4), and the candidate
pool images (`statpool_10yr_n1000000_d{0,1,2}/hazard_image.npz`, P = 10⁶ rows
each). Both carry current date-convention provenance (`reference_start`,
`scenario_stamp_start`), asserted at load.

**Coordinates.** The six campaign selection axes, scaled to the selector's own
geometry: each axis mapped by the pool draw's robust p1/p99 range
(`scenario_design_methods.md` §4.3), recomputed from the pool image and
cross-checked against the `normalization` block persisted in the corresponding
`hazfill_stat_abs_10yr_n100_d{k}` `_meta.json`. Scaling is linear and UNCLIPPED
(clipping is a selector construction detail; support scoring must see the
excursion beyond the box).

**Primary support definition (D1, nearest-neighbour).** For each E_test
sub-window coordinate `z`, the support statistic is the Euclidean
nearest-neighbour distance `d(z)` to the pool in scaled coordinates. The
exceedance threshold is the q = 0.99 quantile of the pool's OWN self-NN distance
distribution (each pool member's distance to its nearest other member): a
sub-window farther from every pool member than 99% of pool members are from
their own nearest neighbours has no stationary analogue at pool resolution. A
SOW's **support score** is `out_frac` = the fraction of its 125 sub-windows
exceeding the threshold. Sensitivity: q ∈ {0.95, 0.999} reported alongside.

**Sensitivity support definition (D2, selection box).** `box_frac` = the
fraction of a SOW's sub-windows falling outside the pool's p1/p99 selection box
on ANY selection axis. The pool's own any-axis-outside fraction (≤ ~6 × 2%,
less under correlation) is reported beside it as the null calibration. D2 is
coarser than D1 (a point can re-enter the box far from any pool member, and the
box ignores the joint structure); it is the sensitivity, not the primary.

**Strata (pre-declared cut points on the primary score).**

| Stratum | Rule | Rationale |
|---|---|---|
| `in_support` | `out_frac` ≤ 0.05 | Under the null that a SOW's sub-windows are draws from the pool law, E[out_frac] ≈ 0.01 by construction of q = 0.99; Binomial(125, 0.01) puts ≥ 7/125 (= 0.056) below probability 2 × 10⁻⁴, so 0.05 is the chance-excursion ceiling for a stationary SOW. |
| `boundary` | 0.05 < `out_frac` < 0.50 | A real but minority excursion: the SOW's typical decade still has stationary analogues. |
| `out_of_support` | `out_frac` ≥ 0.50 | The majority of the SOW's own natural-variability windows lie beyond the pool fringe — search under the stationary pool cannot have presented this SOW's typical hazard content. |

**Fallback rule (pre-declared):** if any stratum holds fewer than 30 SOWs, the
headline contrast is reported on the two-way split `in_support` vs the merged
complement, with the three-way table still emitted and the merge disclosed.

**Pool-draw handling.** The population object is the stationary law; the three
regenerated pools are i.i.d. samples of it. The **primary stratum labels come
from pool d0**; scores against d1 and d2 are computed identically and reported
as a cross-draw agreement table (expected near-1 at P = 10⁶). No pooling across
draws (a union pool would change the reference size and the threshold's
meaning).

**Attribution.** For every above-threshold sub-window, the per-axis scaled
exceedance beyond the p1/p99 interval is recorded; each SOW reports the axis
carrying its largest mean exceedance (`dominant_axis`) and per-axis exceedance
shares. Membership is mapped onto the forcing coordinates θ = (m, r1, r2) via
the sub-window image's `theta_index` (a label join, never positional), with the
`em = exp(m)` display convention of `src.factor_mapping.theta_features`.
Expected structure (a sanity check, not a result): dry-forced corners (low m)
excurse on the drought axes; near-θ=0 SOWs land in support.

**Pool-level coverage deficit (the step-11 complement).** Step 11
(`scripts/main/scenario_discovery.py`) tests failure against distance to the
nearest member of a design's SEARCH ensemble in E_test's SOW-level
empirical-CDF/rank space. Stage A additionally persists, per SOW, the distance
to the nearest member of the POOL, computed in exactly that space (SOW-level
coordinates = within-SOW mean of realization descriptors; screened axes via
`src.factor_mapping.screen_hazard_axes`; `cdf_transform` anchored on E_test;
`coverage_deficit` reused). This separates "unreachable by any stationary
design" (pool deficit) from "under-covered by this design" (step 11's search
deficit); its association with failure is drawn in stage B beside the step-11
result and duplicates none of it.

## 3. Design contrast by support stratum (stage B — on the available cubes)

**Inputs.** The persisted per-SOW cubes of every (design, draw, seed) re-eval
run on `HSD_REEVAL_TAG` (`compare_designs.discover_runs` /
`src.results_data`) — before the campaign, the go/no-go sets on the interim
first10ch subset — the incumbent cube beside each run (`baseline/`), and the
stage-A stratum labels, consumed unchanged.

**Endpoints, per run × stratum, computed on the stratum's SOWs only:**

1. **Primary: the multivariate Starr satisficing fraction** of the run's
   **full-E_test best policy** — the policy the run would actually report
   (highest full-set satisficing under the focal criterion set, ties broken by
   distance-to-ideal, exactly `factor_mapping.select_compromise`) — decomposed
   over the strata. Fixing the policy across strata is deliberate: a per-stratum
   re-selection would let a design present a different policy to each stratum,
   which no adoption process does.
2. **Companion: the within-stratum best-attained fraction** (the study's
   run-level max-over-set endpoint re-scored on the stratum), carrying the
   max-over-set bias disclosure.
3. **No-harm frequency** `no_harm_freq_tau` at the adopted tolerance vector
   (`NYCOPT_REGRET_TAU` from the sourced production env file — never re-chosen
   here), restricted to the stratum, for the same fixed policy and as the
   within-stratum best.

Criteria: the focal criterion set of the active variant
(`NYCOPT_CRITERIA_VARIANT`) and `reference_all8` (the adopted snapshot from the
cube's own `reeval_raw_meta.json` — the moving-measuring-stick guard). Both are
labelled in every table; thresholds are never re-chosen.

**Headline figure and statistic.** HF − PS difference of endpoint 1 per
stratum: per-(draw, seed) points, design summarized by the draw-level mean
(seeds are pseudoreplicates), uncertainty by a SOW-level bootstrap (resample
SOWs with replacement WITHIN the stratum, recompute both design means, 2,000
replicates, percentile 95% CI — the designs share the SOWs, so the difference is
bootstrapped as a pair).

**Unification with the forcing-tercile decomposition.** The same script emits
the identical contrast under terciles of the dominant forcing factor
(`m`, matching `compare_designs.regret_by_severity`'s severity axis) and a
stratum × tercile cross-tabulation, so the hazard-support and forcing-space
partitions of the same cubes are read together.

**Decision rules (pre-registered).**

- *Generalization supported:* the out-of-support HF − PS difference on endpoint
  1 is positive, its SOW-bootstrap 95% CI excludes zero, and the per-draw
  differences agree in sign in at least 2 of 3 draws.
- *Claim bounded:* the in-support difference is positive by the same standard
  while the out-of-support CI includes zero or is negative — reported as "the
  advantage does not demonstrably extend beyond the stationary support",
  motivating the climate-augmented pool extension as future work.
- *Uninformative:* the out-of-support stratum is smaller than the fallback
  floor, or the focal criterion is degenerate (fraction within 0.01 of 0/1 for
  every design) on it; reported as such, with the tercile partition carrying
  the descriptive story.
- No new primary endpoint is created in any case; RQ2's endpoint remains the
  full-E_test comparison.

## 4. Figures

| Figure | What the reader learns |
|---|---|
| `F1_support_map_theta` | Which corners of the CMIP6 forcing box leave the stationary hazard support (SOWs in the (e^m, r1) plane colored by stratum, score colorbar companion panel). |
| `F2_support_score_distribution` | How E_test divides across the strata and that the division is stable across the three pool re-rolls (ECDF of `out_frac` per pool draw, cut points marked). |
| `F3_axis_excursion` | Which hazard axes carry the excursion, by forcing tercile (expected: drought axes under dry forcing) — and that seasonal structure is not an axis. |
| `F4_reach_by_tercile` | Per selection axis, where E_test's sub-window quantiles sit against the pool's p1/p99 band, by forcing tercile (the reach view of the `etest_hazard_overlay` contract). |
| `F5_contrast_by_stratum` (stage B) | The headline: HF − PS satisficing and no-harm differences vs support stratum, draw-level points with SOW-bootstrap CIs. |
| `F6_partition_agreement` (stage B) | Whether the hazard-support and forcing-tercile partitions tell the same story about where the difference lives. |
| `F7_pool_vs_design_deficit` (stage B) | Failure rate vs pool-deficit decile beside the step-11 search-ensemble deficit: unreachable-by-any-stationary-design vs under-covered-by-this-design. |

Tables mirror every figure (`hsd_*.csv`); figures follow `src/plotting/style.py`
(PNG only during iteration, no in-panel text annotations, no footer boxes).

## 5. Run sequence

1. **Stage A now** (post-regeneration, pre-search): smoke with `HSD_SMOKE=1`
   (P = 2,000 smoke pools, first 200 SOWs — the first10ch subset size, so a
   smoke stage B can label the interim cubes), then the full run on `shared`
   (pool self-NN at 10⁶ × 6 via `cKDTree` with threaded query is minutes;
   memory is dominated by the three 10⁶ × 8 images, < 2 GB).
2. **Stage B on the pre-campaign cubes** (the go/no-go sets on the first10ch
   subset): the same wrapper with a production env file sourced
   (`NYCOPT_ENV_FILE` supplies the adopted `NYCOPT_REGRET_TAU` and
   `NYCOPT_CRITERIA_VARIANT`); stage A tables are reused, not recomputed.
   Stage-B artifacts carry the tag in their filenames.
3. Figures regenerate from tables alone (`hazard_support_figures.py`).
4. Decide (§7) BEFORE the campaign launches. Re-running stage B on the
   eventual production cubes is optional SI material, not part of the
   decision.

## 6. Stage-A result (measured 2026-08-25; job 20142248, 30 s)

Recorded AFTER the pre-registered definitions above were fixed; no cut point,
quantile, or rule was changed in response to these numbers.

- **Strata (pool d0, primary): in_support 611 / boundary 389 /
  out_of_support 0.** No SOW has a majority of its sub-windows beyond the
  pool's q0.99 self-NN fringe (max `out_frac` = 0.352), so the §2 fallback
  rule governs stage B: the headline contrast runs on the two-way split
  `in_support` (611) vs `beyond_support` (389). Overall 5.7% of the 125,000
  sub-windows lie beyond the fringe, against the 1% pool null the threshold
  is built on; the D2 box sensitivity agrees in direction (12.9% of
  sub-windows outside the p1/p99 box on some axis vs the pool's own 6.4%).
- **Cross-draw stability:** stratum agreement with d0 is 0.910 (d1) and
  0.902 (d2); the disagreements are SOWs near the 0.05 cut, and the split
  sizes move by < 20 SOWs. The primary labels remain d0's.
- **Direction of the excursion — measured, and opposite to the §2 dry-axes
  expectation in the aggregate:** `out_frac` correlates POSITIVELY with the
  volume multiplier (Spearman-scale r ≈ 0.57 with e^m, 0.45 with r1), and the
  dominant excursion axis is `flood_peak_discharge` for 447 of 1,000 SOWs
  (then `flood_pulse_duration` 158). Wet-forced SOWs push flood peaks and
  pulse durations beyond the stationary p99 broadly, while dry-forced drought
  excursions are real but thinner: the dry tercile's sub-window q99 exceeds
  the pool band on drought magnitude (60.2 vs 49.0) and onset rate (2.26 vs
  1.98), i.e. only the top few percent of dry-tercile windows leave support.
  The stationary pool's 10⁶ natural-variability windows already contain very
  severe droughts; multiplicative wet forcing escapes the pool ceiling more
  readily than dry forcing escapes its drought extremes.
- **Rank-space pool deficit is near-orthogonal to the support score**
  (r ≈ −0.06), as the construction implies: the step-11 empirical-CDF space
  deliberately compresses tails (a point beyond every pool member and the
  pool's own maximum both map to rank ≈ 1), so the pool deficit measures
  density coverage inside the cloud while `out_frac` measures magnitude
  exceedance beyond it. The two are complementary, not redundant.

Implication for stage B, stated in advance of any cube: the generalization
question is answered on the `beyond_support` stratum, and the axis attribution
says that stratum is predominantly a flood-side, wet-forced excursion with a
dry-side minority — so the stage-B readout is read beside the flood/drought
criterion decomposition rather than as a drought story by default.

### Stage-B result on the pre-campaign cubes (job 20142525)

**Substrate and its limits, stated first.** The three first-round go/no-go
sets (one draw, one seed per design) re-evaluated on the interim 200-SOW
`etest_kn_50yr_n25000_first10ch` subset, scored against the archived
pre-regeneration incumbent cube. These cubes predate the 2026-08-18
seasonal-rotation fix (the searches used the retired water-year unit), so the
absolute numbers are not manuscript results; the draw-level replication of
§3's decision rule (sign agreement in ≥ 2 of 3 draws) is not estimable at
K = 1; and 200 SOWs put the worst-case SE of a stratum fraction at ±5 pp
(in-support, n = 117) to ±5.5 pp (beyond, n = 83). What the substrate CAN
show is the sign and rough size of the HF − PS contrast inside vs beyond the
stationary support, on the same cubes the go/no-go decision was already read
from. Two-way fallback applied (117 in-support / 83 beyond-support).

**Focal criterion set (`compromise`), fixed full-set best policy (endpoint 1):**

| Group | n | PS | HF | HF − PS [SOW-bootstrap 95% CI] |
|---|---|---|---|---|
| in_support | 117 | 0.530 | 0.624 | **+0.09 [+0.01, +0.17]** |
| beyond_support | 83 | 0.072 | 0.301 | **+0.23 [+0.14, +0.33]** |
| dry tercile | 63 | 0.333 | 0.349 | +0.02 [−0.11, +0.14] |
| middle tercile | 60 | 0.700 | 0.817 | +0.12 [+0.03, +0.22] |
| wet tercile | 77 | 0.065 | 0.351 | +0.29 [+0.20, +0.39] |

The within-stratum best (endpoint 2) tells the same story (beyond-support best
0.157 PS vs 0.361 HF). The HF advantage does not vanish beyond the stationary
support; it is larger there, and the tercile view locates it in the wet
tercile — exactly the region stage A identified as the flood-side excursion.
`reference_all8` is 0 for every design in every group (the known conjunction
degeneracy). The fixed policy's no-harm frequency at the adopted tau is 0 for
every design in every group (the most-satisficing policy of each set degrades
at least one objective beyond tolerance relative to the incumbent in every
SOW, while within-stratum no-harm-best reaches 0.4–1.0), so endpoint 3 is
uninformative for the fixed policy on this substrate; the archived incumbent
cube is the likely reason and this is re-read on the regenerated incumbent
whenever a post-regeneration policy cube exists.

Pool-deficit companion (F7): the HF fixed policy's failure rate falls from 0.8
in the lowest pool-deficit decile to 0.3 in the highest, PS from 0.8 to 0.5
and `historic` stays ≥ 0.85 throughout — i.e. failures for every design
concentrate where E_test is DENSELY covered by the stationary pool, not where
it is unreachable, which is the opposite of what a pool-coverage deficit would
produce (n = 20 per decile; descriptive only).

## 7. Decision: the hazard-filling candidate pool stays stationary

**Question.** Should climate-forced sub-ensembles be added to the HF candidate
pool before the campaign runs?

**Answer: no — exclude the climate signal.** The available evidence supports
this on three independent grounds, in decreasing order of weight:

1. **The stationary pool already spans nearly all of the hazard magnitude
   E_test presents (stage A, full scale, definitive).** 94.3% of E_test's
   125,000 forced sub-windows fall within the P = 10⁶ stationary pool's own
   q0.99 nearest-neighbour fringe; no SOW has a majority of its windows
   beyond it; 61% of SOWs are indistinguishable from stationary at the 5%
   chance ceiling. A climate-augmented pool could add candidates only in the
   thin remaining fringe, which is predominantly wet/flood-side. On the dry
   side — the side the advisor's question implicitly targets — the stationary
   pool's natural-variability extremes already exceed the dry tercile's q90 on
   every drought axis and are exceeded only by its top ~1–5% of windows. A
   climate signal would therefore buy little drought coverage that 10⁶
   stationary windows do not already supply.
2. **HF's advantage does not collapse beyond the stationary support (stage
   B, interim, directional).** On the go/no-go cubes the HF − PS satisficing
   difference is positive inside the support and larger beyond it, with the
   wet tercile carrying the gain. Under the §3 rules this is the
   "generalization supported" branch (CI excludes zero; draw agreement not
   estimable at K = 1). There is no signal that the stationary HF pool leaves
   HF policies exposed in the region a climate-augmented pool would fill.
3. **The design argument, independent of the data.** Adding climate forcing
   to one arm's pool confounds the selection-rule factor with search-population
   breadth and leaks the test distribution into that arm, voiding the i.i.d.
   control that makes PS the exact null for HF. That would trade the study's
   one clean contrast for an uninterpretable one.

**What this decision does NOT establish, and is recorded as scope.** (i)
Seasonal-structure forcing (r1, r2) is not encoded by the six selection axes
and is out-of-support for both arms; a climate-augmented pool would be the
only way to present it during search, and that is a different experiment
(population breadth), registered as future work. (ii) The dry tercile shows no
HF advantage on the interim cubes (+0.02, CI spans zero); stage A says this is
not a pool-coverage gap (dry droughts are in support), so it is a search or
criterion question, not an argument for climate candidates. (iii) The interim
substrate is K = 1, pre-regeneration; the direction is what is relied on, not
the magnitudes. (iv) Whether HF's advantage is specifically a flood-side
result is the natural first question for the production cubes, and the
tagged stage-B run reproduces on them at zero simulation cost if wanted for
the SI.

## Citations

Bryant & Lempert (2010); Eker & Kwakkel (2018); Giuliani & Castelletti (2016);
Gold et al. (2022, 2023); Herman et al. (2014, 2015); Lamontagne et al. (2018);
McPhail et al. (2018) (T2 scenario-subset re-scoring); Quinn et al. (2020);
Starr (1962); Trindade et al. (2017). Resolved via `docs/notes/literature/`.
