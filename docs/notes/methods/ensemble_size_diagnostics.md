# Ensemble-Size Diagnostics: the Statistical Basis of N = 300 (SI)

*Supplemental diagnostic (SI Texts S4 and S5) that sets the realization count N
of both matched search designs at the fixed record length L = 10 yr. Code:
`scripts/supplemental/ensemble_size_hazard.py` (Layer A, selection level),
`scripts/supplemental/ensemble_size_library_run.py` (Layer B library build),
`scripts/supplemental/ensemble_size_figures.py` (statistics, decision table,
NFE reading, figures), pure statistics in `src/ensemble_size_stats.py`,
configuration in `supplemental_config.py` (`ESD_*`), wrappers
`workflow/supplemental/ensemble_size_{hazard,library_stage,library_eval,analysis}.sh`,
env file `workflow/envs/ensemble_size_diagnostics.env`, tests
`tests/test_ensemble_size_diagnostics.py`. Outputs under
`outputs/supplemental/ensemble_size_diagnostics/{tables,figures,library}/`.
The campaign itself is stated in `campaign_design.md`.*

---

## 1. What N must deliver

The search ensemble is N = 300 realizations × L = 10 yr for both matched
designs. N is set by the statistical adequacy of the ensemble as the sample the
optimizer scores every policy on. Neither compute budget nor the selection level
sets it. Hazard-filling tail enrichment is flat in N at the production pool size
(§4), and budget is a constraint on the campaign rather than a criterion for N.
L = 10 is fixed because the selection contrast needs a common L
(`scenario_design_methods.md` §6), only N moves, and N stays common to both
designs.

Representativeness is a different property for the two designs.

- `fixed_probabilistic` (PS) is an i.i.d. sample from the stationary generator,
  so every objective is a sample statistic of a population quantity (frequency,
  mean, or pooled percentile of annual units). Representativeness is estimator
  precision and estimator bias versus N (order-statistic operators are
  finite-sample biased, means and frequencies are not).
- `hazard_filling_stationary` (HF) is a designed exploration, LHS anchors in the
  six-axis robust range-scaled hazard space snapped to a 10⁶-member i.i.d.
  pool. Its objective values estimate nothing probabilistic, the distribution
  shift is the design effect, so bias is undefined. Representativeness is the
  hazard-range coverage attained (per-axis stratification and tail enrichment
  above the 0.10 i.i.d. share) and the reproducibility of the induced objective
  vectors across independent constructions (fresh anchor plan on the same pool,
  fresh pool and anchor plan across production draws).
- The common criterion is the one the optimizer cares about. The noise in paired
  dominance decisions between policies must sit below the ε-dominance
  precisions the archive resolves at (reliabilities 0.05, deficit severities
  10.0, flood 0.3, storage 5.0, `config.get_epsilons()`). An ε-archive cannot
  use precision the sample does not deliver (Kasprzyk et al. 2013), and paired
  differences on common samples are far more precise than levels (Linderoth et
  al. 2006).

The unit of independence is the realization, never the annual unit. Annual
units within a realization are serially dependent (`objective_definitions.md`
§2), so every standard error, bootstrap, and subsample resamples realizations,
and effective sample sizes are reported rather than assumed.

## 2. The per-realization annual-unit library

Stage (i) of the objective scheme yields one annual metric per (realization,
unit-year). It is policy-dependent but ensemble-independent, because every
realization is simulated independently from the same initial storage. A fixed
policy set is therefore evaluated once per unique realization, the
(policy × realization × objective × unit-year) tensor is persisted, and the
objective vector of any (design, N, replicate) ensemble is composed offline by
pooling its members' unit-years through the registered stage-(ii) operators
(`src.objectives_ensemble`). No search is needed. The composition is asserted
equal to the driver-computed scalar (`compute_for_borg_from_units`), and the
library rows for the staged production `hazfill_stat_abs_10yr_n100_d0` members
are compared end to end with a fresh simulation of that ensemble (agreement to
≤ 7·10⁻⁴ ε on every composed objective, LP jitter only). Built on
`src.simulation.evaluate_annual_units`, the MPI task farm of
`scripts/supplemental/epsilon_calibration_run.py`,
`src.sensitivity_common.apply_operator_rows`, and
`src.ensembles.materialize_subset` / `with_indices_override`.

**Policy set (ten, fixed by rule before any result).** From the union U of the
two matched designs' ε-filtered merged pilot Pareto-approximate sets at
N = 100 (`ESD_POLICY_SET_FILES`), PS rows before HF rows:

| id | rule |
|---|---|
| P0 | the FFMP incumbent (`get_baseline_values("ffmp")`) |
| P1–P4 | per-objective best in U for NYC delivery reliability, Montague flow reliability, downstream flood exceedance, NYC storage P01 (`solution_selection.best_single`, ties to the lowest row) |
| P5, P6 | best-satisficing compromise per design (`factor_mapping.select_compromise`, rule `best_satisficing`, thresholds from the cube's own snapshot) on that design's pilot re-evaluation cube |
| P7, P8 | nearest neighbour of each compromise in U's direction-oriented, min–max-scaled objective space (Euclidean over the 8 active objectives) |
| P9 | the PS design's `min_dist_ideal` compromise |

If a rule returns an already-selected DV vector, the next-best under that rule
is taken. The set spans the response regimes the operators see. It is not a
population and nothing ranks it.

**Reference, library, replicates.**

- PS reference. The first P_ref = 5,000 rows of `statpool_10yr_n1000000_d0`.
  A prefix of an i.i.d. pool is an exact i.i.d. sample (SI Text S2), so 5,000
  rows give ⌊5000/N⌋ disjoint replicates (100, 66, 50, 33, 25, 16, 12, 10 over
  the ladder). Where that count falls below 20 (N ≥ 300) the disjoint set is
  supplemented to 20 with uniformly random overlapping subsets, flagged in
  every table.
- HF constructions. The campaign selector on pool d0 at each ladder N for three
  anchor plans (draws 0, 101, 102, where draw 0 at N = 100 is the production
  `hazfill_stat_abs_10yr_n100_d0` ensemble exactly), plus at N = 100 the
  production constructions on pools d1 and d2 (fresh pool and fresh anchors).
- PS fresh draws. The staged `fixprob_10yr_n100_d{0,1,2}` ensembles.
- Ladder N ∈ {50, 75, 100, 150, 200, 300, 400, 500} (`ESD_N_LADDER`), and
  every figure marks the campaign N (`ESD_N_CAMPAIGN`).
- Library realized. 10 policies × 9,571 realizations (8,971 regenerated pool
  members plus the six staged production draws) × 10 registry objectives × 9
  unit-years, about 50 SU. Pool members are regenerated from their global
  indices into chunked staged ensembles (≤ 1,000 realizations each, step-04
  inputs per chunk) and are bit-identical to the staged production members.
  Replicates realized are PS 100/66/53/33/25/20/20/20 for N = 50…500 and HF 3
  constructions at every N with 5 at N = 100. HF standard errors rest on 2–4
  degrees of freedom and are read as orders of magnitude.
- Smoke mode (`NYCOPT_ESD_SMOKE=1`) runs every stage on the staged P = 2,000
  smoke pool with a `smoke_` artifact prefix.

## 3. Statistics and decision criteria

Let J_p(S) be policy p's objective composed from member set S, with replicates
r = 1..R per (design, N). Signs are oriented so that better is positive where a
sign matters.

| # | Statistic | Definition | Criterion | Citation |
|---|---|---|---|---|
| 1 | Level SE | SD over replicates of J_p(S_r), per policy; max and median over policies. Tail operators additionally get the replicate 5/50/95 band and a realization-level bootstrap (B = 1,000), never σ/√N | ≤ ε on every objective | Kasprzyk et al. 2013; Reed et al. 2013 |
| 2 | **Paired SE (binding)** | SD over replicates of J_a(S_r) − J_b(S_r) for every policy pair on common realizations; max over pairs (median and P90 reported) | ≤ ε/2 on every objective (two policies one box apart are separated at ≥ 2 SE) | Linderoth et al. 2006; Homem-de-Mello & Bayraksan 2014 |
| 3 | Flip rate | fraction of pairs whose ε-dominance relation at replicate r differs from the reference relation (PS: the 5,000-member reference; HF: the across-replicate majority at that N), averaged over replicates | ≤ 0.05 | Zatarain Salazar et al. 2017 |
| 4a | Optimism (PS) | mean over replicates of sign × (J_p(S_r) − J_p(S_ref)), per policy; max over policies. For fixed policies this is estimator bias, not selection optimism | ≤ ε/2 | Kaut & Wallace 2007 |
| 4b | Construction SD and shift (HF) | SD over constructions of J_p(S_r) − J_p(S_ref^PS) (the noise) and its mean (the intended design effect, reported) | SD ≤ ε | Kaut & Wallace 2007; Bonham et al. 2024 |
| 5 | Effective sample size | per operator at the campaign N, n_eff/(N(L−1)) = (SD_unit-bootstrap / SD_realization-bootstrap)² | reported (SI Text S5) | Quinn et al. 2017; Hamilton et al. 2022 |

The criterion fractions are `ESD_LEVEL_SE_EPS_FRAC`, `ESD_PAIRED_SE_EPS_FRAC`,
`ESD_FLIP_RATE_MAX`, and `ESD_OPTIMISM_EPS_FRAC`.

**Decision rule.** N_min(design) is the smallest ladder N at which criteria
1–4 hold on every objective, with the binding objective and criterion named.
N_common is the larger of the two. Where one design's crossing is not located
on the ladder, N_common is set by the design whose crossing is, and the other
design's residual is carried as a measured property of that construction (§5).
N_common is priced with the measured cost surface (SU per search ∝ N^0.951 at
L = 10, `ESD_SU_PER_SEARCH_N100`, `ESD_COST_N_EXPONENT`) and the per-rank
memory model against 128-rank packing on 256 GB nodes.

**Cross-check.** The epsilon-calibration cubes (513 random feasible policies ×
N = 100 per design) are subsampled on the realization axis at N ∈ {25, 50, 75}
(20 subsets each) and their level-SE decay is overlaid on the library's. The
policy-median level SEs decay at the same rate and sit within a factor of about
1.5 of the library's on every objective, so the ten front policies are not
atypical (figure B7).

## 4. Layer A, hazard-space representativeness (selection level)

`ensemble_size_hazard.py` uses the campaign selector itself
(`select_from_candidate_image` with `config.HAZARD_SELECTION_AXES`) and the
`scengen` primitives, so the HF selections it scores are exactly the selections
step 03 stages (the anchor-plan-0 selection at N = 100 on pool d0 reproduces
the staged production member list 100/100).

| Block | Statistic | Ladder / replicates |
|---|---|---|
| A-HF | per-axis tail share above pool P90 and P99, per-axis KS to uniform, joint L2-star, MST edge statistics, minimum separation, each as a ratio to a matched random design | N ladder × pools d0, d1, d2 × 10 anchor plans (draws 0, 101–109); 50-seed random null |
| A-NP | min per-axis tail share above P90 (seed-mean convention) on nested prefixes P′ of pool d0 | P′ ∈ {5·10³, 2·10⁴, 10⁵, 3·10⁵, 10⁶} × N ladder × 10 plans |
| A-PS | sampling distribution of per-axis counts above pool P90/P99, relative error of subset quantiles vs the pool, closed-form P(≥ 1 member beyond quantile q) = 1 − qᴺ | 200 uniform size-N subsets of the pool image |
| A-CV | convergence of the descriptors themselves, pooled mean per axis vs ensemble extreme per axis | 5/50/95 bands over the PS subsets; HF plans |

The tail-share statistic is the min per-axis share above P90 over the selection
axes, seed-mean convention, reported against the 0.10 share of an i.i.d.
selection with no threshold (`hazard_selector_diagnostics.md`). Layer A
reports mechanism (Bonham et al. 2024, maximum-based statistics never converge,
and Zatarain Salazar et al. 2017, tail representation is what low sampling
rates lose). It does not size N by itself.

**Measured.** HF tail enrichment is flat in N at P = 10⁶. Seed-mean over 10
plans × 3 pools, the min per-axis share above P90 is 0.287, 0.277, 0.278,
0.288, 0.288, 0.285, 0.291, 0.283 for N = 50…500 (drought magnitude binding),
the mean per-axis share is 0.386–0.388 at every N, and the min share above P99
is 0.011–0.016 against 0.000–0.005 for a random design. The pool holds ~10⁵
members above P90 per axis, so selecting 500 does not deplete the supply. The
(N, P′) ladder shows the small-pool effect directly. At P′ = 5,000 the min
share falls from 0.22 (N = 100) to 0.18 (N = 500), at P′ = 2·10⁴ from 0.25 to
0.22, and at P′ ≥ 10⁵ it is flat in N. The statistic is the same on d0, d1, d2
(0.271, 0.276, 0.271 at N = 100) and saturates in P′ from 3·10⁵ at every N, so
its value is set by the m = 6 geometry of the pool's joint support rather than
by pool supply. HF joint L2-star improves slowly with N (0.019 → 0.011) while a
random design stays at 0.17, and minimum separation stays 1.4–2× the random
design's, so no near-duplicate pathology appears up to N = 500. For PS the
exact i.i.d. law holds. A size-N sample holds on average 0.01·N members above
the pool P99 per axis (0.97 at N = 100, 5.0 at N = 500), the closed form
1 − 0.99ᴺ matches the empirical probability of holding at least one, and the
RMS relative error of the subset P99 is 0.16 at N = 100 and 0.08 at N = 500.
This is the mechanism behind the Layer-B tail-operator noise. PS pooled-mean
descriptors are unbiased with a 5–95 % width shrinking from 27 % (N = 50) to
9 % (N = 500), the ensemble maxima drift monotonically and never converge, and
HF pooled means sit at 1.57–1.59× the pool mean at every N, so the design
effect is N-independent. Nothing at the selection level penalizes a larger N,
and the cost of N is compute alone.

## 5. Layer B decision table and the adopted N

Worst-pair paired SE and companion statistics in units of the objective's ε
(`tables/decision_table.csv`, `n_min.csv`, figures B1–B6):

| design | N | level SE max / ε (worst obj.) | paired SE max / ε (worst obj.) | flip rate | bias (PS) or construction SD (HF) / ε | passes |
|---|---|---|---|---|---|---|
| PS | 50 | 1.14 (Montague def.) | 1.12 (Montague def.) | 0.051 | 0.52 | no |
| PS | 75 | 0.87 | 0.86 (Montague def.) | 0.040 | 0.32 | no (paired) |
| PS | 100 | 0.82 | 0.82 (NYC def.), 0.79 (Montague def.), 0.71 (storage) | 0.037 | 0.27 | no (paired) |
| PS | 150 | 0.63 | 0.62 (Montague def.), 0.52 (storage) | 0.034 | 0.15 | no (paired) |
| PS | 200 | 0.64 | 0.63 (Montague def.), 0.61 (NYC def.), 0.51 (storage) | 0.026 | 0.16 | no (paired) |
| PS | **300** | 0.45 | 0.45 (Montague def.), 0.43 (NYC def.), 0.40 (storage) | 0.026 | 0.09 | **yes** |
| PS | 400 | 0.46 | 0.45 | 0.019 | 0.08 | yes |
| PS | 500 | 0.40 | 0.40 | 0.011 | 0.05 | yes |
| HF | 50 | 1.00 (Montague def.) | 1.10 (storage), 1.00 (Montague def.), 0.93 (NYC def.) | 0.067 | 1.00 | no |
| HF | 75 | 4.2 (NYC def.) | 4.2 (NYC def.), 1.3 (Montague def.) | 0.059 | 4.2 | no |
| HF | 100 (5 constructions) | 2.0 (NYC def.) | 2.0 (NYC def.), 0.70 (Montague def.) | 0.058 | 2.0 | no |
| HF | 150 | 1.9 | 1.9 (NYC def.), 0.59 (Montague def.), 0.58 (storage) | 0.044 | 1.9 | no |
| HF | 200 | 1.3 | 1.3 (NYC def.), 0.70 (storage) | 0.007 | 1.3 | no |
| HF | 300 | 1.7 | 1.7 (NYC def.) | 0.015 | 1.7 | no |
| HF | 400 | 1.6 | 1.7 (NYC def.) | 0.044 | 1.6 | no |
| HF | 500 | 0.68 | 0.69 (NYC def.) | 0.052 | 0.68 | no |

Every reliability objective and the flood objective pass every criterion at
every N ≥ 75 for both designs (paired SE ≤ 0.34 ε at N = 100, flood ≤ 0.10 ε).
The decision is carried by the three pooled-percentile operators.

**N_min(PS) = 300.** The binding statistic is the worst-pair paired SE and the
binding objectives are the two deficit-P99 operators and storage P01. At
N = 100 the worst pair differs by 0.7–0.8 ε per replicate on those axes, so two
policies one ε-box apart on a deficit axis are re-ordered by the sample about
one time in five, and at N = 300 the paired SE is 0.40–0.45 ε. The pair-median
paired SE is far smaller (0.11–0.41 ε at N = 100, 0.02–0.2 ε at 300), and the
binding pairs involve the per-objective-best extremes (P3, the flood-optimal
policy, and the P5/P7 PS compromise pair), which the policy rule was meant to
expose.

**N_min(HF) is not located on the ladder.** HF meets every criterion on seven
objectives (Montague deficit and storage from N = 200–400) and fails the
paired-SE criterion on NYC deficit P99 at every N. Its between-construction SD
on that objective is 4.2, 2.0, 1.9, 1.3, 1.7, 1.6 ε for N = 75…400 and 0.68 ε
at N = 500, and the flip rate (0.058 at N = 100, 0.052 at 500 on 3–5
constructions) sits at the criterion. The mechanism (figure B5,
`layer_b_policy_bands.csv`) is that the worst-1 % annual CVaR90 of NYC deficit
takes values on discrete shelves set by the FFMP drought-stage delivery cuts.
Seven of the ten policies read exactly the same value in every PS replicate
(SD = 0), and a replicate that pushes the worst unit onto the next shelf moves
the objective by 10–35 percentage points, one to three ε at once. PS crosses a
shelf rarely and its paired SE decays as N^(−1/2) (0.82 ε → 0.43 ε from N = 100
to 300). HF selects precisely the extreme members that decide the shelf, so
different anchor plans land on different shelves and the construction SD does
not fall with N until the selection itself stabilizes. With three constructions
per rung (about 50 % relative error on an SD) the ladder cannot separate a slow
√N-type decay from a plateau. This is Bonham et al. (2024)'s non-convergence of
maximum-type statistics one order statistic in.

**Decay fits** (log worst-pair paired SE/ε on log N, slope β with SE, fitted
ε/2 crossing N*):

| objective | PS β | PS N* | HF β | HF N* |
|---|---|---|---|---|
| NYC def P99 | −0.48 (0.04) | 232 | −0.24 (0.25) | undetermined |
| Montague def P99 | −0.44 (0.03) | 283 | −0.64 (0.12) | 184 |
| storage P01 | −0.41 (0.03) | 210 | −0.68 (0.15) | 158 |
| NYC rel | −0.58 (0.05) | 55 | −0.45 (0.10) | 45 |
| Montague rel | −0.55 (0.04) | 35 | −0.43 (0.07) | 44 |
| flood | −0.61 (0.02) | 7 | −0.46 (0.14) | 4 |

The PS exponents are the √N law of an i.i.d. sample and place the crossing
between the 200 and 300 rungs (ladder answer 300). The HF exponents on
Montague deficit and storage are steeper than √N (the shelf variance there
averages out, crossing near N ≈ 160–180). Only NYC deficit P99 is unresolved,
and its SE of β is as large as β.

**Bias and design effect (figure B4).** PS estimator bias is ≤ 0.27 ε at
N = 100 (identically zero for the frequency and mean operators) and ≤ 0.09 ε at
N = 300. The HF construction shift from the PS reference is N-independent
(median over policies −0.94 ε on NYC reliability, −0.75 ε Montague reliability,
−0.70 ε NYC deficit, −0.47 ε Montague deficit, −2.3 ε flood exceedance, −1.4 ε
storage P01, ≈ 0 on Trenton and NJ reliability).

**Effective sample size (figure B6, `n_eff.csv`).** n_eff/N(L−1) is 0.68
(NYC deficit P99), 0.73 (NYC reliability), 0.79 (NJ), 0.82 (storage P01), 0.84
(Montague reliability), 0.90 (Montague deficit), 0.93 (Trenton), 1.02 (flood
mean) for PS and 0.70–1.23 for HF. Serial dependence within a realization costs
at most about 30 % of the pooled units, so the 2,700 pooled unit-years of one
evaluation at N = 300 act as roughly 1,840–2,700 independent units, above the
1,000 one-year realizations of Quinn et al. (2017). The naive unit-level
bootstrap understates the tail operators' SE by up to 20 %, so every SE here
is realization-level.

**Adopted.** N_common = 300, the i.i.d. control's crossing (the smallest
ladder N at which every criterion holds for PS, missed at N = 100 only by the
paired-SE criterion on the three tail operators by a factor 1.4–1.6). The
hazard-filling design's NYC-deficit P99 residual (construction SD 1.7 ε at
N = 300, 1.3–4.2 ε across N ≤ 400 on three constructions) is disclosed as a
measured property of that construction rather than resolved by a larger N,
and its draw-dependence is measured by the SI draw-sensitivity re-evaluation,
in which each matched design's final set is re-simulated on its own draws d1
and d2 at N = 300 (`campaign_design.md` §5). The ε floors are re-measured on
the N = 300 ensembles and the adopted vector stands provided every entry lies
above its floor (`epsilon_calibration_experiment.md`).

## 6. NFE and cost

The runtime archives read by `ESD_NFE_ARCHIVES` (N = 100 searches at 500k NFE,
4 islands, figure B8, `tables/nfe_asymptote.csv`) show no NFE slack. No island
reaches 95 % of its final hypervolume before about 75 % of its budget, every
island still gains 3–8 % of its final hypervolume in the last fifth of the run
with its ε-progress rate within a factor of about 2 of the early-run rate, and
the archive is still growing at the end. NFE therefore cannot be traded for N
at equal SU, and raising N is a pure cost increase of (N/100)^0.951 per search
(figure B9, `cost_pricing.csv`, on the measured basis of
`ESD_SU_PER_SEARCH_N100`). Full 128-rank packing at 85 % of node memory is
feasible only up to N ≈ 280 unbatched, so N = 300 runs with a 150-realization
batch, and the 96 h wall on eight nodes binds near N = 300 for a 750k-NFE
search, so the campaign runs on 12 nodes (`campaign_design.md` §3, §6).

## 7. Run sequence

```
# 0. smoke (shared; minutes)                       NYCOPT_ESD_SMOKE=1 on every wrapper
# 1. Layer A + library plan (shared, 1 task)        workflow/supplemental/ensemble_size_hazard.sh
# 2. regenerate + prep chunks (shared array)        workflow/supplemental/ensemble_size_library_stage.sh
# 3. library evaluation (wholenode, 1 node)         workflow/supplemental/ensemble_size_library_eval.sh
# 4. statistics, decision table, figures (shared)   workflow/supplemental/ensemble_size_analysis.sh
```

Every wrapper sources `workflow/envs/ensemble_size_diagnostics.env`, all
settings are in `supplemental_config.py` (`ESD_*`), and there are no CLI value
flags. The analysis refuses a library whose `library_qc.json` is not `library_valid`.
Figures are A1–A5 (`hf_tail_share_vs_n`, `hf_coverage_vs_n`, `np_ladder`,
`ps_tail_sampling`, `descriptor_convergence`) and B1–B9 (`level_se_vs_n`,
`paired_se_vs_n`, `flip_rate_vs_n`, `optimism_vs_n`, `tail_replicate_bands`,
`effective_sample_size`, `epscube_crosscheck`, `nfe_asymptote`,
`cost_pricing`).

## Citations

Bonham et al. (2024); Hamilton et al. (2022); Homem-de-Mello & Bayraksan
(2014); Kasprzyk et al. (2013); Kaut & Wallace (2007); Linderoth et al.
(2006); Quinn et al. (2017); Reed et al. (2013); Zatarain Salazar et al.
(2017). Resolved via `docs/notes/literature/`.
