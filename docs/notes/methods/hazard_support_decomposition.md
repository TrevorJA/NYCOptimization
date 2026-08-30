# Hazard-Support Decomposition of the E_test Design Contrast (SI)

*Supplemental re-scoring (SI Text S10) decomposing the `hazard_filling_stationary`
(HF) − `monte_carlo` (MC) difference on E_test by where each state of
the world (SOW) sits relative to the stationary candidate pool's hazard support.
Zero simulation, every deliverable reduces persisted artifacts. Code:
`scripts/supplemental/hazard_support_run.py` (stage A support membership, stage
B design contrast) and `scripts/supplemental/hazard_support_figures.py`,
configuration in `supplemental_config.py` (`HSD_*`), wrapper
`workflow/supplemental/hazard_support_decomposition.sh`, tests
`tests/test_hazard_support_decomposition.py`. Outputs under
`outputs/supplemental/hazard_support_decomposition/{tables,figures}/`.*

---

## 1. Question and scope

The study's central claim is a generalization claim (`experimental_design.md`).
Hazard-space coverage of the stationary natural-variability manifold during
search produces policies that generalize to forced climate conditions the
search never saw, better than i.i.d. sampling from the same generator. E_test
spans a CMIP6-informed forcing envelope and neither search design contains a
climate signal. Putting climate-forced sub-ensembles inside the HF candidate
pool is not run, because it would confound the selection-rule factor with
search-population breadth and leak the test distribution into one design,
voiding the i.i.d. control that makes MC the exact null for HF. The question is
instead answered from the persisted re-evaluation output.

> Decompose the HF − MC difference on E_test by whether each SOW's hazard image
> lies inside, at the fringe of, or outside the stationary pool's hazard
> support. A difference that persists on out-of-support SOWs states the
> generalization claim directly. A difference confined to the supported region
> bounds it and motivates the climate-augmented extension as future work.

Scope guards. The decomposition is supplemental and never a new primary
endpoint, hazard-space coverage remains method verification, the one
hazard-space inference stays the step-11 coverage-deficit mechanism test, and
no policy-architecture comparison is introduced.

Claim scoping. The six selection axes (`config.HAZARD_SELECTION_AXES`) encode
drought and flood event magnitudes on the aggregate NYC inflow. E_test's
forcing also moves seasonal structure (the harmonic amplitudes r1, r2), which
the axes do not encode, so a seasonal excursion is out of support for both
designs alike and invisible here. The decomposition bounds hazard-magnitude
generalization, and the seasonal-structure dimension is a scope limitation
shared by both designs, not a bias between them.

## 2. Stage A, support membership (policy-free)

**Substrate.** The E_test sub-window hazard image
(`hazard_image_subwindows.npz`, 25,000 realizations × 5 disjoint 10-yr
December-aligned sub-windows = 125,000 rows × 8 candidate axes, scored on the
pool's exact metric span, `scenario_design_methods.md` §5.4) and the candidate
pool images (`statpool_10yr_n1000000_d{0,1,2}/hazard_image.npz`, P = 10⁶ rows
each). Both carry date-convention provenance (`reference_start`,
`scenario_stamp_start`), asserted at load.

**Coordinates.** The six selection axes scaled to the selector's own geometry.
Each axis is mapped by the pool draw's robust p1/p99 range
(`HSD_BOUND_PCT`, `scenario_design_methods.md` §4.3), recomputed from the pool
image and cross-checked against the `normalization` block persisted in the
corresponding staged HF `_meta.json`. Scaling is linear and unclipped, because
support scoring must see the excursion beyond the box.

**Primary support definition (D1, nearest neighbour).** For each E_test
sub-window coordinate z the support statistic is the Euclidean
nearest-neighbour distance d(z) to the pool in scaled coordinates. The
exceedance threshold is the q = 0.99 quantile (`HSD_SELF_NN_QUANTILE`) of the
pool's own self-NN distance distribution, so a sub-window farther from every
pool member than 99 % of pool members are from their own nearest neighbour has
no stationary analogue at pool resolution. A SOW's support score `out_frac` is
the fraction of its 125 sub-windows exceeding the threshold. Sensitivity
quantiles q ∈ {0.95, 0.999} (`HSD_SELF_NN_QUANTILES_SENS`) are reported
alongside.

**Sensitivity support definition (D2, selection box).** `box_frac` is the
fraction of a SOW's sub-windows falling outside the pool's p1/p99 selection box
on any selection axis, with the pool's own any-axis-outside fraction reported
beside it as the null calibration. D2 is coarser than D1 (a point can re-enter
the box far from any pool member, and the box ignores the joint structure).

**Strata** (`HSD_STRATA_CUTS`, `HSD_STRATA_NAMES`).

| Stratum | Rule | Rationale |
|---|---|---|
| `in_support` | `out_frac` ≤ 0.05 | Under the null that a SOW's sub-windows are draws from the pool law, E[out_frac] ≈ 0.01 by construction of q = 0.99; Binomial(125, 0.01) puts ≥ 7/125 (= 0.056) below probability 2 × 10⁻⁴, so 0.05 is the chance-excursion ceiling for a stationary SOW. |
| `boundary` | 0.05 < `out_frac` < 0.50 | A real but minority excursion; the SOW's typical decade still has stationary analogues. |
| `out_of_support` | `out_frac` ≥ 0.50 | The majority of the SOW's own natural-variability windows lie beyond the pool fringe, so search under the stationary pool cannot have presented this SOW's typical hazard content. |

**Fallback rule.** If any stratum holds fewer than `HSD_MIN_STRATUM_SOW` = 30
SOWs, the headline contrast is reported on the two-way split `in_support` vs
the merged complement `beyond_support`, with the three-way table still emitted
and the merge disclosed.

**Pool-draw handling.** The population object is the stationary law and the
three regenerated pools are i.i.d. samples of it. The primary stratum labels
come from pool d0, and scores against d1 and d2 are computed identically and
reported as a cross-draw agreement table. Draws are never pooled, because a
union pool would change the reference size and the threshold's meaning.

**Attribution.** For every above-threshold sub-window the per-axis scaled
exceedance beyond the p1/p99 interval is recorded, and each SOW reports the
axis carrying its largest mean exceedance (`dominant_axis`) with per-axis
exceedance shares. Membership is mapped onto the forcing coordinates
θ = (m, r1, r2) via the sub-window image's `theta_index` (a label join, never
positional), with the `em = exp(m)` display convention of
`src.factor_mapping.theta_features`.

**Pool-level coverage deficit (the step-11 complement).** Step 11
(`scripts/main/scenario_discovery.py`) tests failure against the distance to
the nearest member of a design's search ensemble in E_test's SOW-level
empirical-CDF/rank space. Stage A additionally persists, per SOW, the distance
to the nearest member of the pool in exactly that space (SOW-level coordinates
as the within-SOW mean of realization descriptors, screened axes via
`src.factor_mapping.screen_hazard_axes`, `cdf_transform` anchored on E_test,
`coverage_deficit` reused). This separates unreachable by any stationary design
(pool deficit) from under-covered by this design (step 11's search deficit).

## 3. Stage B, design contrast by support stratum

**Inputs.** The persisted per-SOW cubes of every (design, draw, seed)
re-evaluation run on the campaign re-evaluation preset
(`HSD_REEVAL_TAG`, the `etest_kn_50yr_n25000_first25ch` subset named by
`src/etest.py::campaign_reeval_preset()`), discovered by
`compare_designs.discover_runs` / `src.results_data`, the incumbent cube
beside each run (`baseline/`), and the stage-A stratum labels consumed
unchanged. Stage A labels all 1,000 generated SOWs and the re-evaluated 500 are
their nested prefix.

**Endpoints, per run × stratum, computed on the stratum's SOWs only.**

1. Primary. The multivariate Starr satisficing fraction of the run's
   full-E_test best policy (highest full-set satisficing under the focal
   criterion set, ties broken by distance to ideal, exactly
   `factor_mapping.select_compromise`), decomposed over the strata. Fixing the
   policy across strata is deliberate, because a per-stratum re-selection would
   let a design present a different policy to each stratum, which no adoption
   process does.
2. Companion. The within-stratum best-attained fraction (the run-level
   max-over-set endpoint re-scored on the stratum), carrying the max-over-set
   bias disclosure.
3. No-harm frequency `no_harm_freq_tau` at the adopted tolerance vector
   (`NYCOPT_REGRET_TAU` from the sourced production env file, never re-chosen
   here), restricted to the stratum, for the same fixed policy and as the
   within-stratum best.

Criteria are the focal criterion set of the active variant
(`NYCOPT_CRITERIA_VARIANT`) and `reference_all8` (the adopted snapshot from
the cube's own `reeval_raw_meta.json`, the moving-measuring-stick guard). Both
are labelled in every table and thresholds are never re-chosen.

**Headline statistic.** The HF − MC difference of endpoint 1 per stratum, with
per-seed points on each design's searched draw and the design summarized by
the seed mean (the seed is the unit of analysis, one searched draw per
design). Uncertainty comes from a SOW-level bootstrap that resamples SOWs with
replacement within the stratum and recomputes both design means
(`HSD_BOOTSTRAP_N` = 2,000 replicates, percentile 95 % CI). The designs share
the SOWs, so the difference is bootstrapped as a pair.

**Unification with the forcing-tercile decomposition.** The same script emits
the identical contrast under terciles of the dominant forcing factor
(`HSD_TERCILE_AXIS` = `m`, matching `compare_designs.regret_by_severity`) and
a stratum × tercile cross-tabulation, so the hazard-support and forcing-space
partitions of the same cubes are read together.

**Decision rules.**

- Generalization supported. The out-of-support HF − MC difference on endpoint
  1 is positive, its SOW-bootstrap 95 % CI excludes zero, and both seeds of
  each design agree in sign.
- Claim bounded. The in-support difference is positive by the same standard
  while the out-of-support CI includes zero or is negative, reported as the
  advantage not demonstrably extending beyond the stationary support and
  motivating the climate-augmented pool extension as future work.
- Uninformative. The out-of-support stratum is smaller than the fallback
  floor, or the focal criterion is degenerate on it (fraction within 0.01 of 0
  or 1 for every design), reported as such with the tercile partition carrying
  the descriptive story.
- No new primary endpoint is created in any case. RQ2's endpoint remains the
  full-E_test comparison.

## 4. Figures

| Figure | What the reader learns |
|---|---|
| `F1_support_map_theta` | Which corners of the CMIP6 forcing box leave the stationary hazard support (SOWs in the (e^m, r1) plane coloured by stratum, score colourbar companion panel). |
| `F2_support_score_distribution` | How E_test divides across the strata and that the division is stable across the three pool draws (ECDF of `out_frac` per draw, cut points marked). |
| `F3_axis_excursion` | Which hazard axes carry the excursion, by forcing tercile, and that seasonal structure is not an axis. |
| `F4_reach_by_tercile` | Per selection axis, where E_test's sub-window quantiles sit against the pool's p1/p99 band, by forcing tercile. |
| `F5_contrast_by_stratum` (stage B) | The headline, HF − MC satisficing and no-harm differences vs support stratum, seed-level points with SOW-bootstrap CIs. |
| `F6_partition_agreement` (stage B) | Whether the hazard-support and forcing-tercile partitions tell the same story about where the difference lives. |
| `F7_pool_vs_design_deficit` (stage B) | Failure rate vs pool-deficit decile (`HSD_DEFICIT_BINS` = 10) beside the step-11 search-ensemble deficit. |

Tables mirror every figure (`hsd_*.csv`) and figures follow
`src/plotting/style.py`. Stage A runs once after the pools are staged (smoke
with `HSD_SMOKE=1` on the P = 2,000 smoke pools and the first `HSD_SMOKE_N_SOW`
SOWs, then the full run, minutes on `shared`). Stage B runs on the production
cubes with a production env file sourced (`NYCOPT_ENV_FILE` supplies
`NYCOPT_REGRET_TAU` and `NYCOPT_CRITERIA_VARIANT`) and reuses the stage-A
tables. Figures regenerate from tables alone.

## 5. Support membership on the regenerated pools

Stage A on the P = 10⁶ pools gives the strata **in_support 611 / boundary 389 /
out_of_support 0** (pool d0, primary). No SOW has a majority of its
sub-windows beyond the pool's q0.99 self-NN fringe (max `out_frac` = 0.352),
so the fallback rule governs stage B and the headline contrast runs on the
two-way split `in_support` (611) vs `beyond_support` (389). Overall 5.7 % of
the 125,000 sub-windows lie beyond the fringe against the 1 % pool null the
threshold is built on, and the D2 box sensitivity agrees in direction (12.9 %
of sub-windows outside the p1/p99 box on some axis vs the pool's own 6.4 %).
Stratum agreement with d0 is 0.910 (d1) and 0.902 (d2). The disagreements are
SOWs near the 0.05 cut and the split sizes move by fewer than 20 SOWs.

The excursion is predominantly wet-forced and flood-side. `out_frac`
correlates positively with the volume multiplier (Spearman r ≈ 0.57 with e^m,
0.45 with r1), and the dominant excursion axis is `flood_peak_discharge` for
447 of 1,000 SOWs (then `flood_pulse_duration`, 158). Dry-forced drought
excursions are real but thin. The dry tercile's sub-window q99 exceeds the pool
band on drought magnitude (60.2 vs 49.0) and onset rate (2.26 vs 1.98), so only
the top few percent of dry-tercile windows leave support. The stationary pool's
10⁶ natural-variability windows already contain very severe droughts, and
multiplicative wet forcing escapes the pool ceiling more readily than dry
forcing escapes its drought extremes. The rank-space pool deficit is
near-orthogonal to the support score (r ≈ −0.06), as the construction implies,
because the step-11 empirical-CDF space compresses tails while `out_frac`
measures magnitude exceedance beyond the cloud. The two are complementary.

Stage B is therefore read on the `beyond_support` stratum beside the
flood/drought criterion decomposition rather than as a drought story by
default.

## 6. Design consequence

The hazard-filling candidate pool is stationary. The stationary P = 10⁶ pool
spans nearly all of the hazard magnitude E_test presents. 94.3 % of E_test's
125,000 forced sub-windows fall within the pool's own q0.99 nearest-neighbour
fringe, no SOW has a majority of its windows beyond it, and 61 % of SOWs are
indistinguishable from stationary at the 5 % chance ceiling. A climate-augmented
pool could add candidates only in the thin remaining fringe, which is
predominantly wet and flood-side, while on the dry side the pool's
natural-variability extremes already exceed the dry tercile's q90 on every
drought axis. Adding climate forcing to one design's pool would also trade the
study's one clean contrast for a confounded one (§1). Whether HF's advantage
extends beyond the stationary support is measured by stage B on the production
cubes. Seasonal-structure forcing (r1, r2) is not encoded by the selection axes
and a climate-augmented pool would be the only way to present it during
search, and that is a different experiment (population breadth), registered
as future work.

## Citations

Bryant & Lempert (2010); Eker & Kwakkel (2018); Giuliani & Castelletti (2016);
Gold et al. (2022, 2023); Herman et al. (2014, 2015); Lamontagne et al. (2018);
McPhail et al. (2018) (T2 scenario-subset re-scoring); Quinn et al. (2020);
Starr (1962); Trindade et al. (2017). Resolved via `docs/notes/literature/`.
