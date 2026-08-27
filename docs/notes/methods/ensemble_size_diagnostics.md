# Ensemble-Size Diagnostics: a Statistically Grounded Minimum N (SI)

*Last updated: 2026-08-26 (results in §7). Supplemental diagnostic (SI Texts S4 and S5) deriving a
pre-registered, statistically based minimum realization count N for both matched
search designs at the fixed record length L = 10 yr. Code:
`scripts/supplemental/ensemble_size_hazard.py` (Layer A, selection level),
`scripts/supplemental/ensemble_size_library_run.py` (Layer B library build),
`scripts/supplemental/ensemble_size_figures.py` (all statistics, the decision
table, the NFE-asymptote reading, and every figure — pure post-processing),
pure statistics in `src/ensemble_size_stats.py`, configuration in
`supplemental_config.py` (`ESD_*`), wrappers `workflow/supplemental/ensemble_size_*.sh`,
env file `workflow/envs/ensemble_size_diagnostics.env`, tests
`tests/test_ensemble_size_diagnostics.py`. Outputs under
`outputs/supplemental/ensemble_size_diagnostics/{tables,figures,library}/`.*

**PRE-REGISTRATION.** The Layer-B ε criteria, ladders, replicate counts, and
selection rules in §§2–5 were fixed on 2026-08-25 before any Layer-A or Layer-B
result was inspected and are unchanged. The Layer-A tail-share statistic (§3)
is reported without a threshold. §7 records what was measured.

---

## 1. The decision and what "representative" means

The campaign search ensemble is N = 300 realizations × L = 10 yr for both
matched designs (`campaign_design.md`), set by this diagnostic from the
statistical adequacy of the ensemble as the *sample the optimizer scores every
policy on*. Neither compute budget nor the selection level sets it: hazard-filling
tail enrichment is flat in N at the production pool size (§7.1), and budget is a
constraint on the campaign, not a criterion for N. L = 10 is fixed (the
selection contrast needs a common L; `scenario_design_methods.md` §6); only N
moves; N stays common to both designs, so the recommendation is
N_common = max over designs of N_min(design).

Representativeness is a different property for the two designs, and the note
scores each on its own terms:

- **`fixed_probabilistic` (PS)** is an i.i.d. sample from the stationary
  generator, so every objective is a sample statistic of a population quantity
  (frequency, mean, or pooled percentile of annual units). Representativeness is
  estimator precision (standard error versus N) and estimator bias versus N
  (order-statistic operators are finite-sample biased; means and frequencies
  are not).
- **`hazard_filling_stationary` (HF)** is a designed exploration: LHS anchors in
  the six-axis robust range-scaled hazard space snapped to a 10⁶-member i.i.d.
  pool. Its objective values estimate nothing probabilistic — the distribution
  shift is the treatment — so "bias" is undefined. Representativeness is (a)
  the hazard-range coverage attained (per-axis stratification and tail
  enrichment above the 0.10 i.i.d. share) and (b) reproducibility of the induced
  objective vectors across independent constructions (fresh anchor plan on the
  same pool; fresh pool and anchor plan across production draws).
- **The common criterion, the one the optimizer cares about:** the noise in
  paired dominance decisions between policies must sit below the
  ε-dominance precisions the archive resolves at (reliabilities 0.05, deficit
  severities 10.0, flood 0.3, storage 5.0; `config.get_epsilons()`). An
  ε-archive cannot use precision the sample does not deliver (Kasprzyk et al.
  2013), and paired differences on common samples are the precise quantity,
  far more so than levels (Linderoth et al. 2006, Table 10).

**The unit of independence is the realization, never the annual unit.** Annual
units within a realization are serially dependent (`objective_definitions.md`
§2), so every standard error, bootstrap, and subsample in this note resamples
realizations. Effective sample sizes are reported, not assumed.

## 2. The central efficiency: a per-realization annual-unit library

Stage (i) of the objective scheme yields one annual metric per (realization,
unit-year); it is policy-dependent but ensemble-independent, because every
realization is simulated independently from the same initial storage. So a
fixed policy set is evaluated ONCE per unique realization, the
(policy × realization × objective × unit-year) tensor is persisted, and the
objective vector of any (design, N, replicate) ensemble is composed offline by
pooling its members' unit-years through the registered stage-(ii) operators
(`src.objectives_ensemble`). No search is needed for Layers A and B. The
composition is asserted equal to the driver-computed scalar
(`compute_for_borg_from_units` on the pooled block, as the epsilon calibration
does) and, end to end, the library's rows for the production `hazfill_..._d0`
members are compared with a fresh simulation of that staged ensemble
(regeneration determinism; reported at the pywrdrb LP-jitter tolerance).

Built on `src.simulation.evaluate_annual_units` (the `(R, M, U)` tensor for one
policy on an `EnsembleSpec`, honoring the realization batch), the MPI
task-farm-and-merge pattern of `scripts/supplemental/epsilon_calibration_run.py`,
`src.sensitivity_common.apply_operator_rows` (vectorized operators on pooled
unit rows), and `src.ensembles.materialize_subset` / `with_indices_override`
(regenerate pool members into staged chunks; subset a staged chunk without
re-staging).

## 3. Layer A — hazard-space representativeness (selection level, ~0 SU)

Extends block D of `scripts/supplemental/diagnose_hazard_selectors.py`
(`N_SWEEP` is now `NYCOPT_SELDIAG_N_SWEEP`; block D records gain the P99 tail
share, minimum separation, and MST edge statistics; saturation mode can run
block D via `NYCOPT_SELDIAG_SATURATION_NSWEEP=1` so the nested-prefix ladder
carries an N sweep). `ensemble_size_hazard.py` consumes the same `scengen`
primitives (`absolute_filling_subsample`, `random_subsample`,
`per_axis_selection_metrics`, `selection_metrics`) and the campaign selector
itself (`select_from_candidate_image` with `config.HAZARD_SELECTION_AXES`), so
the HF selections it scores are exactly the selections step 03 would stage.

| Block | Statistic | Design | Ladder / replicates |
|---|---|---|---|
| A-HF | per-axis tail share above pool P90 and P99; per-axis KS to uniform; joint L2-star; MST mean/min edge; minimum pairwise separation — each as a ratio to the matched random design (same N, m) | HF | N ∈ {50, 75, 100, 150, 200, 300, 400, 500}; P = 10⁶ pools d0, d1, d2; 10 anchor plans per (pool, N) (`design.selector_seed(draw)` for draws 0, 101–109; draw 0 is the production plan); 50-seed random null |
| A-NP | min per-axis tail share above P90 (seed-mean convention) on nested prefixes P′ of pool d0 | HF | P′ ∈ {5·10³, 2·10⁴, 10⁵, 3·10⁵, 10⁶} × the N ladder × 10 anchor plans; reports how the statistic saturates with P′ at each N |
| A-PS | sampling distribution of per-axis counts above pool P90/P99; relative error of subset P50/P90/P99 vs the pool; closed-form P(≥ 1 member beyond pool quantile q) = 1 − qᴺ | PS | uniform size-N subsets of the pool image (exact i.i.d. law); 200 subsets per N |
| A-CV | convergence of the descriptors themselves: pooled mean per axis (expected to flatten) vs ensemble extreme per axis (expected to move monotonically, never converge) | PS, HF | 5/50/95 bands over the 200 PS subsets; HF seeds; N ladder |

**Tail-share statistic (convention unchanged from
`hazard_selector_diagnostics.md`):** min per-axis tail share above P90 over the
selection axes, seed-mean convention, reported against the 0.10 share of an
i.i.d. selection with no threshold. Layer A *reports* how the statistic moves
with N at P = 10⁶ and how it saturates with P′ at each N; it does not size N
by itself — it explains mechanism (Bonham et al. 2024: maximum-based
statistics never converge; Zatarain Salazar et al. 2017: tail representation
is what low sampling rates lose).

## 4. Layer B — objective-estimator stability on fixed policies (the sizing criterion)

### 4.1 Policy set (10, fixed by rule before any result)

From the union U of the two matched designs' ε-filtered merged reference sets
(`ESD_POLICY_SET_FILES`; currently the pilot `ffmp_obj8_merged_eps20260812.set`
of each design — pre-regeneration substrate, disclosed; the regenerated
go/no-go sets replace them under the identical rule when they land), PS rows
before HF rows:

| id | rule |
|---|---|
| P0 | the FFMP incumbent (`get_baseline_values("ffmp")`) |
| P1–P4 | per-objective best in U for NYC delivery reliability, Montague flow reliability, downstream flood exceedance, NYC storage P01 (`solution_selection.best_single`; ties → lowest row) |
| P5, P6 | best-satisficing compromise per design (`factor_mapping.select_compromise`, rule `best_satisficing`, thresholds = the cube's own snapshot) on that design's `etest_kn_50yr_n25000_first10ch` cube |
| P7, P8 | nearest neighbour of each compromise in U's direction-oriented, min–max-scaled objective space (Euclidean over the 8 active objectives) |
| P9 | the PS design's `min_dist_ideal` compromise |

If a rule returns an already-selected DV vector, the next-best under that rule
is taken. The set spans the response regimes the operators see; it is not a
population and nothing here ranks it.

### 4.2 Reference, library, replicates

- **PS reference:** the first P_ref = 5,000 rows of the production candidate
  pool `statpool_10yr_n1000000_d0`. A prefix of an i.i.d. pool is an exact
  i.i.d. sample (global-index child streams; SI Text S2), so 5,000 rows give
  R = ⌊5000/N⌋ disjoint replicates: 100, 66, 50, 33, 25, 16, 12, 10 for the
  ladder. Where R < 20 (N ≥ 300) the disjoint set is supplemented to 20 with
  uniformly random overlapping subsets, flagged in every table.
- **HF constructions:** the campaign selector on pool d0 at each ladder N for
  the three anchor plans draws 0, 101, 102 (the first three Layer-A plans;
  draw 0 at N = 100 is the production `hazfill_stat_abs_10yr_n100_d0`
  ensemble exactly); plus, at N = 100 only, the production constructions on
  pools d1 and d2 (fresh pool AND fresh anchors — the draw-level replicate).
- **PS fresh draws:** the staged `fixprob_10yr_n100_d{0,1,2}` ensembles at
  N = 100 (the draws as staged when the library was built; cross-checks the
  prefix-subset law).
- **Library:** every policy × every unique realization in the union of the
  above. The pool members are regenerated from their global indices into
  chunked staged ensembles (`esd_lib_10yr_d0__chunkJJJ`, ≤ 1,000 realizations
  each, step-04 inputs staged per chunk); the staged production ensembles are
  read as they are. Expected size ≈ 5,000 + ~4,500 HF-union + 600 staged ≈
  10,000 realizations × 10 policies ≈ 1,000 N = 100-evaluation equivalents.
- **Cost (measured surface, SI Text S8):** 173.8 s per N = 100 evaluation at
  128 ranks per node ⇒ ≈ 48 SU per 1,000 realization-policies-of-100 ⇒
  ≈ 50 SU for the library; regeneration (≈ 2.1 s per realization, serial
  array tasks) ≈ 6 core-h; the one-time full-model presim per chunk
  (202 s per 100 realizations) ≈ 6 core-h; Layer A and the analysis are
  minutes on one core. Total ≈ 70–80 SU plus ≈ 5 SU of smoke. Per-rank memory
  at a 100-realization block is 601 + 0.394 × 1,000 ≈ 1.0 GB (trimmed RSS
  model, `supplemental_config.ENSEMBLE_COST_RSS_MB`), ≈ 130 GB per node at
  128 ranks.
- **Smoke first** on `statpool_10yr_n2000_d0` (the staged P = 2,000 smoke
  pool; the n4000 pool named in the kickoff is not staged): P_ref = 200,
  N ∈ {25, 50}, two policies, one chunk, a few ranks on `shared`.

### 4.3 Statistics (per objective and N; every figure carries ε and the campaign N)

Let J_p(S) be policy p's objective composed from member set S. Replicates
r = 1..R per (design, N). Signs are oriented so that "better" is positive
where a sign matters.

| # | Statistic | Definition | Threshold (pre-registered) | Citation |
|---|---|---|---|---|
| 1 | Level SE | SD over replicates of J_p(S_r), per policy; reported as the max and median over policies. Tail operators (worst-1st-percentile unit, within-year CVaR90 → P99; storage P01) additionally get the replicate 5/50/95 band and a realization-level bootstrap (B = 1,000, resampling realizations inside each replicate) — never σ/√N | ≤ ε on every objective (archive resolution must exceed estimator noise) | Kasprzyk et al. 2013; Reed et al. 2013 |
| 2 | **Paired SE (binding)** | SD over replicates of J_a(S_r) − J_b(S_r) for every policy pair (a, b) on common realizations; max over pairs (median and P90 reported) | **≤ ε/2 on every objective: N_noise = smallest ladder N passing** (two policies one box apart are separated at ≥ 2 SE) | Linderoth et al. 2006 Table 10; Homem-de-Mello & Bayraksan 2014 |
| 3 | Flip rate | fraction of pairs whose ε-dominance relation ({a ≻ b, b ≻ a, incomparable}, Borg box convention) at replicate r differs from the reference relation — PS: the 5,000-member reference; HF: the across-replicate majority at that N — averaged over replicates | ≤ 0.05 | Zatarain Salazar et al. 2017 §5.3 (inconsistent trade-off portrayal) |
| 4a | Optimism (PS) | mean over replicates of sign × (J_p(S_r) − J_p(S_ref)), per policy; max |·| over policies. For fixed policies this is estimator bias (order-statistic operators), NOT selection optimism — that part belongs to Layer C | |·| ≤ ε/2 | Kaut & Wallace 2007 §4 (in-sample vs out-of-sample) |
| 4b | Construction SD and shift (HF) | SD over constructions of J_p(S_r) − J_p(S_ref^PS) (the noise) and its mean (the intended design effect; reported, not gated) | SD ≤ ε (identical in form to #1 for HF, kept as its own row so the shift is read beside it) | Kaut & Wallace 2007 (out-of-sample stability); Bonham et al. 2024 (replicate ensembles per size are mandatory) |
| 5 | Effective sample size | per operator at the campaign N (`ESD_N_CAMPAIGN`): n_eff/(N(L−1)) = (SD_unit-bootstrap / SD_realization-bootstrap)², the naive i.i.d.-unit bootstrap against the realization-level block bootstrap | report only (SI Text S5) | Quinn et al. 2017 §3.1.2 (1,000-unit convention); Hamilton et al. 2022 |

**Decision table.** Per design and objective: ε, level SE(N), paired SE(N),
flip rate(N), optimism or construction SD(N), n_eff. **N_min(design)** is the
smallest ladder N at which criteria 1–4 hold on every objective;
**N_common = max over designs**; the binding objective and the binding
criterion are named. N_common is priced with the measured cost surface
(SU per search ∝ N^0.951 at L = 10, trimmed; `scaling_fits.csv`) and the
memory model (601 + 0.394·N·L MB per rank), and the per-node memory at full
128-rank packing is checked against 256 GB.

**Free cross-check.** The epsilon-calibration cubes (513 policies × N = 100 per
design; pre-regeneration substrate, disclosed) are subsampled on the
realization axis at N ∈ {25, 50, 75} (20 random subsets each) and their
level-SE decay is overlaid on the library's at the campaign N; agreement
in shape confirms the library's ten policies are not atypical.

## 5. Layer C — search-level confirmation (design and cost only; NOT run)

Only searches show that the optimizer's output changes with N. The confirmatory
design, to be run only if a larger N is adopted, follows Zatarain Salazar et al.
(2017) and Quinn et al. (2017) §3.4:

- Arms: the N = 100 go/no-go cells (draw 0, seed 1, 500k NFE) and the N = 300
  campaign (one draw, two seeds, reported at 500k NFE), both designs — the
  equal-NFE variant (cost scales with N). An equal-SU variant (NFE scaled by
  100/N, so N × NFE is constant) was designed and is not run.
- Compared solely on E_test: per-N-level reference sets (hypervolume is never
  compared across N levels — Zatarain Salazar et al. normalize within level);
  search-value vs test-value 1:1 plots per solution with the attribution rule
  (all solutions shift on an objective → the operator is unstable; only the
  solutions optimized under it shift → overfitting); ordering preservation
  under re-evaluation (Kendall τ_b between search and test ranks per
  objective); seed attainment probability; seed variance components. Reuses `src/diagnostics.py`, `src/plotting/hypervolume_convergence.py`,
  `src/plotting/seed_reliability.py`, `src/results_data.py`, `src/robustness.py`.
- The production campaign at N = 300 *is* the confirmation; the N = 100 go/no-go
  cells are the comparison arm.

sbatch plan and SU: §7.5.

## 6. The NFE side

At fixed SU, N × NFE is constant. From the existing runtime archives (the
2026-08-10 go/no-go cells: 500k NFE, seed 1, 4 islands, both matched designs;
the 2026-08-05 historic `mm_full` at 50k NFE, two seeds) the note reports where
runtime hypervolume and ε-progress (Borg `Improvements`) reach their asymptote
per island: the NFE at which each island attains 95% and 99% of its final
hypervolume, the ε-progress rate over the last fifth of the run, and archive
growth. If the searches converge well before 500k, that NFE can be traded for
N at equal SU; otherwise raising N is a pure cost increase. Both readings are
stated.

## 7. Findings

*(filled after the runs; §§2–5 were not edited afterwards)*

### 7.1 Layer A (measured 2026-08-25; job 20142729, 24 min, 1 core; tables `hf_ladder`, `np_ladder`/`np_tail_share`, `ps_tail_sampling`, `descriptor_convergence`; figures A1–A5)

**Identity check passed:** the anchor-plan-0 selection at N = 100 on pool d0
reproduces the staged production `hazfill_stat_abs_10yr_n100_d0` member list
exactly (100/100), so every HF number below is the campaign selector's.

**HF tail enrichment is flat in N at P = 10⁶.** Seed-mean over 10 anchor
plans × 3 regenerated pools, min per-axis share above the pool P90:
0.287, 0.277, 0.278, 0.288, 0.288, 0.285, 0.291, 0.283 for N = 50…500 (SD over
plans and pools 0.03 → 0.01); the mean per-axis share is 0.386–0.388 at every
N, and the min per-axis share above P99 is 0.011–0.016 (mean 0.05) against
0.000–0.005 (mean 0.010) for a random design. The pool holds ~10⁵ members
above P90 per axis, so selecting 500 does not deplete the supply: the
"raising N lowers enrichment" reading of the P = 2,000 laptop battery
(`hazard_selector_diagnostics.md` §5) is a small-pool effect, and it does not
carry to the production pool. The (N, P′) ladder (A3) shows exactly that:
at P′ = 5,000 the min share falls from 0.22 (N = 100) to 0.18 (N = 500),
at P′ = 2·10⁴ from 0.25 to 0.22, at P′ ≥ 10⁵ it is flat (0.27–0.29) in N.
Consequence for the decision: nothing at the selection level penalizes a
larger N at P = 10⁶; the cost of N is compute alone.

**The tail share is a size-invariant property of the selector on the
regenerated pools.** The same statistic is 0.27–0.29 on d0, d1, d2 (min over N
of the seed mean: 0.271, 0.276, 0.271 at N = 100). The prefix ladder saturates
(0.268 → 0.282 from P′ = 10⁵ to 10⁶ at N = 100; flat from P′ = 3·10⁵ at every
N, A3b), so the value is set by the m = 6 geometry of the pool's joint
support, not by pool supply: anchors placed where the joint support is empty
snap to boundary members nearer the bulk, which compresses the upper
marginals. No larger pool moves it and it is independent of N; the per-axis
record from the diagnostics driver's own seeds 0–9 is in §7.1a.

**Joint geometry (A2).** HF joint L2-star improves slowly with N (0.019 →
0.011) while a random design of the same size stays at 0.17; the MST mean
edge and minimum separation shrink as N grows (0.43 → 0.25 and 0.16 → 0.04 in
unit-box units) but remain 1.6× and 1.4–2× the random design's, so no
near-duplicate pathology appears up to N = 500. Per-axis KS to uniform falls
0.16 → 0.13 (random 0.42 → 0.39): stratification keeps improving with N.

**PS, the exact i.i.d. law (A4).** A size-N i.i.d. sample holds on average
0.01·N members above the pool P99 per axis: 0.97 at N = 100 (5–95 %: 0–3), 2.0
at N = 200, 5.0 at N = 500; the closed form 1 − 0.99ᴺ (0.63 at N = 100, 0.87
at 200, 0.99 at 500) matches the empirical probability of holding at least
one. Quantile error against the pool falls as N^(−1/2): RMS relative error
of the subset P99 is 0.16 at N = 100 and 0.08 at N = 500 (P50: 0.076 → 0.037).
This is the mechanism behind the Layer-B tail-operator noise: at N = 100 an
i.i.d. ensemble contains roughly one severe member per axis and its P99
descriptor is uncertain to ~16 %.

**Descriptor convergence (A5).** PS pooled-mean descriptors are unbiased
(median 0.99–1.00 of the pool mean) with a 5–95 % width shrinking from 27 %
(N = 50) to 9 % (N = 500) of the pool mean; the ensemble maxima drift
monotonically (median 0.47 → 0.66 of the pool maximum from N = 50 to 500) and
never converge, as Bonham et al. (2024) found for maximum-based statistics.
HF pooled means sit at 1.57–1.59 × the pool mean at every N — the design
effect is N-independent — with the HF maxima near the pool extremes at every N.

### 7.1a Per-axis tail-share record of the regenerated pool d0 (job 20143066; `diagnose_hazard_selectors.py`, saturation mode + N sweep, seeds 0–9, 50-seed null)

Build-QC record, no verdict attached. The diagnostics driver's own convention
gives, on `statpool_10yr_n1000000_d0` at N = 100: campaign six-axis set min
per-axis tail share 0.283, per axis drought_magnitude 0.284 (binding),
flood_pulse_duration 0.334, drought_severity 0.342, flood_peak_discharge
0.389, drought_recovery_rate 0.445, drought_onset_rate 0.513 (mean 0.385);
full eight-axis set 0.220 (drought_duration 0.239). Across the N sweep the
campaign-set minimum is 0.273–0.290 with no trend (0.284, 0.273, 0.283, 0.283,
0.290, 0.289, 0.286, 0.288 for N = 50…500), the P99 minimum 0.009–0.017
(random 0.000–0.005), and joint L2-star 0.019 → 0.011. The statistic is
independent of N, so the ensemble-size decision does not turn on it.
### 7.2 Layer B library and QC

**Staging (job array 20143034, 9 tasks × 4 cores):** 8,971 pool members
regenerated from their global indices in 36 min per 1,000-member chunk
(2.16 s per realization, serial) plus ~1 min of step-04 prep per chunk;
≈ 2.3 GB per chunk on projects space, symlinked into the staged tree. The
regenerated daily flows are bit-identical to the production
`hazfill_stat_abs_10yr_n100_d0` members (checked per site, per member).

**A defect caught by the end-to-end check (first evaluation, job 20143273,
quarantined as `unit_library_ffmp_INVALID_cachebug_20143273.h5`):** the
library rows for the staged `hazfill_d0` members disagreed with the
regenerated rows of the same pool members on 84 of 100 realizations, with
unrelated failure-week patterns — not solver jitter. Cause:
`src.simulation._get_cached_model_dict` keys the cached model dict on the
ensemble preset name (+ DU signature), and the cached dict carries the
block's `inflow_ensemble_indices`; the library's per-task specs shared their
chunk slug, so a rank's later blocks of the same chunk re-simulated the first
block's realizations. The composition QC cannot see this (it compares a
task's stored scalars with the same rows) and the smoke run placed one task
per rank, so only the end-to-end comparison exposed it. Fix: a unique
`preset_name` per block (the convention `run_simulation_ensemble_batched`
already uses), plus two new fatal guards in the merge — identical unit
tensors across two blocks of one (policy, source), and the end-to-end
mismatch itself — and the analysis script refuses a library whose
`library_qc.json` is not `library_valid`. The re-run is job 20143847.

**The valid library (job 20143847; 64 ranks on `shared`, 49 min, 48.7
core-hours ≈ 49 SU):** 10 policies × 9,571 realizations (8,971 regenerated
pool members + the six staged production draws) × 10 registry objectives × 9
unit-years, 960 tasks, 0 failures, 182 s median per 100-realization block.
QC: composition exact (max relative deviation 0); no duplicate blocks; the 100
staged `hazfill_d0` members agree with their regenerated rows to ≤ 7·10⁻⁴ ε on
every composed objective (338 of 90,000 unit-years differ — LP jitter of ≤ 2
failing weeks or ≤ 0.45 % of capacity on annual storage minima — and no
composed reliability or deficit value moves at all). Cost of the whole
diagnostic: ≈ 12 SU staging + 49 SU library + ≈ 55 SU for the quarantined
first evaluation + ≈ 8 SU smoke/analysis ≈ 125 SU.

**Replicates realized:** PS 100/66/53/33/25/20/20/20 at N = 50…500 (the
N = 100 count includes the three staged `fixprob` draws; N ≥ 300 carry
4–10 supplemented random subsets, flagged); HF 3 anchor plans at every N and
5 constructions at N = 100 (plans 0/101/102 on pool d0 plus the production
d1/d2 draws on fresh pools). HF's standard errors therefore rest on 2–4
degrees of freedom and are read as orders of magnitude, not to two digits.
### 7.3 Decision table (`tables/decision_table.csv`, `n_min.csv`; figures B1–B7)

Worst-pair paired SE and other statistics in units of the objective's ε
(ε = 0.05 reliabilities, 10.0 deficit P99s, 0.3 flood, 5.0 storage):

| design | N | level SE max / ε (worst obj.) | paired SE max / ε (worst obj.) | flip rate | bias (PS) or construction SD (HF) / ε (worst) | passes |
|---|---|---|---|---|---|---|
| PS | 50 | 1.14 (Montague def.) | 1.12 (Montague def.) | 0.051 | 0.52 (NYC def.) | no |
| PS | 75 | 0.87 | 0.86 (Montague def.) | 0.040 | 0.32 | no (paired) |
| PS | **100** | 0.82 | 0.82 (NYC def.), 0.79 (Montague def.), 0.71 (storage) | 0.037 | 0.27 | no (paired) |
| PS | 150 | 0.63 | 0.62 (Montague def.), 0.52 (storage) | 0.034 | 0.15 | no (paired) |
| PS | 200 | 0.64 | 0.63 (Montague def.), 0.61 (NYC def.), 0.51 (storage) | 0.026 | 0.16 | no (paired) |
| PS | **300** | 0.45 | 0.45 (Montague def.), 0.43 (NYC def.), 0.40 (storage) | 0.026 | 0.09 | **yes** |
| PS | 400 | 0.46 | 0.45 | 0.019 | 0.08 | yes |
| PS | 500 | 0.40 | 0.40 | 0.011 | 0.05 | yes |
| HF | 50 | 1.00 (Montague def.) | 1.10 (storage), 1.00 (Montague def.), 0.93 (NYC def.) | 0.067 | 1.00 | no |
| HF | 75 | 4.2 (NYC def.) | 4.2 (NYC def.), 1.3 (Montague def.) | 0.059 | 4.2 | no |
| HF | **100** (5 constructions) | 2.0 (NYC def.) | 2.0 (NYC def.), 0.70 (Montague def.) | 0.058 | 2.0 | no |
| HF | 150 | 1.9 | 1.9 (NYC def.), 0.59 (Montague def.), 0.58 (storage) | 0.044 | 1.9 | no |
| HF | 200 | 1.3 | 1.3 (NYC def.), 0.70 (storage) | 0.007 | 1.3 | no |
| HF | 300 | 1.7 | 1.7 (NYC def.) | 0.015 | 1.7 | no |
| HF | 400 | 1.6 | 1.7 (NYC def.) | 0.044 | 1.6 | no |
| HF | 500 | 0.68 | 0.69 (NYC def.) | 0.052 | 0.68 | no |

Every reliability objective and the flood objective pass every criterion at
every N ≥ 75 for both designs (paired SE ≤ 0.34 ε at N = 100; flood ≤ 0.10 ε).
The whole decision is carried by the three pooled-percentile operators.

**N_min(PS) = 300**, binding statistic the worst-pair paired SE, binding
objectives the two deficit-P99 operators and storage P01 in that order
(N_noise = 300). At N = 100 the worst pair differs by 0.7–0.8 ε
per replicate on those axes, i.e. two policies one ε-box apart on a deficit
axis are re-ordered by the sample about one time in five; at N = 300 the
paired SE is 0.4–0.45 ε. The pair-median paired SE is far smaller (0.11–0.41 ε
at N = 100, 0.02–0.2 ε at 300): the binding pairs are the ones involving the
per-objective-best extremes (P3, the flood-optimal policy, and P5/P7, the PS
compromise pair), which is what the policy rule was meant to expose.

**N_min(HF) is undefined on the ladder**: the between-construction SD of NYC
deficit P99 is 2–4 ε from N = 75 to 400 and 0.7 ε at N = 500, so the paired-SE
criterion fails on that objective at every N, and the flip rate (0.058 at
N = 100, 0.052 at 500 on 3–5 constructions) sits at the criterion. Montague
deficit P99 and storage P01 pass from N = 200–400. **Mechanism (figure B5,
`layer_b_policy_bands.csv`):** the worst-1 % annual CVaR90 of NYC deficit
takes values on discrete shelves set by the FFMP drought-stage delivery cuts
— seven of the ten policies read exactly the same value in every PS replicate
(35.00, 49.20, 48.82, 28.92, 30.05, 23.81, 0.29; SD = 0) — and a replicate
that pushes the worst unit onto the next shelf moves the objective by 10–35
percentage points, one to three ε at once (P3: 64.6–85.4 across PS replicates
at N = 100; P5: 19.4 → 31.7). PS crosses a shelf rarely and its paired SE
decays as N^(−1/2) (0.82 ε → 0.43 ε from N = 100 to 300); HF selects precisely
the extreme members that decide the shelf, so different anchor plans land on
different shelves (P1: 0.29 in PS vs 9.95 ± 20 in HF; P5: 19.4 vs 55 ± 10) and
the construction SD does not fall with N until the selection itself
stabilizes. Whether that shelf variance averages out with N is exactly what
the ladder must show, and with three constructions per rung it cannot yet:
the fitted decay on that axis is β = −0.24 ± 0.25 (table below), which
does not separate a slow √N-type decay (crossing ε/2 near N ≈ 1,000–2,000)
from a plateau (never). (Bonham et al. 2024's finding that maximum-type
statistics do not converge at any tested size is the same phenomenon one
order statistic in.)

**Decay fits** (log of the worst-pair paired SE/ε on log N over the ladder;
slope β with its SE and the fitted ε/2 crossing N*):

| objective | PS β | PS N* | HF β | HF N* |
|---|---|---|---|---|
| NYC def P99 | −0.48 (0.04) | 232 | −0.24 (0.25) | undetermined (~2·10⁴ by point estimate) |
| Montague def P99 | −0.44 (0.03) | 283 | −0.64 (0.12) | 184 |
| storage P01 | −0.41 (0.03) | 210 | −0.68 (0.15) | 158 |
| NYC rel | −0.58 (0.05) | 55 | −0.45 (0.10) | 45 |
| Montague rel | −0.55 (0.04) | 35 | −0.43 (0.07) | 44 |
| flood | −0.61 (0.02) | 7 | −0.46 (0.14) | 4 |

The PS exponents are the √N law of an i.i.d. sample and put N_min(PS)
between the 200 and 300 rungs (ladder answer 300, ±1 rung). The HF
exponents on Montague deficit and storage are if anything steeper than √N
(the shelf variance there does average out, crossing near N ≈ 160–180);
only NYC deficit P99 is unresolved, and its SE of β is as large as β.

**Bias / design effect (figure B4).** PS estimator bias is ≤ 0.27 ε at N = 100
(0.24 ε on storage P01, 0.22–0.27 ε on the deficit P99s; identically zero for
the frequency and mean operators, which the disjoint partition of the
reference makes exact) and ≤ 0.09 ε at N = 300. The HF construction shift from
the PS reference — the intended design effect — is N-independent: median over
policies −0.94 ε on NYC reliability, −0.75 ε Montague reliability, −0.70 ε
NYC deficit, −0.47 ε Montague deficit, −2.3 ε flood exceedance, −1.4 ε
storage P01, ≈ 0 on Trenton and NJ reliability.

**Effective sample size (figure B6, `n_eff.csv`, N = 100).** n_eff/N(L−1):
PS 0.68 (NYC deficit P99), 0.73 (NYC reliability), 0.79 (NJ), 0.82 (storage
P01), 0.84 (Montague reliability), 0.90 (Montague deficit), 0.93 (Trenton),
1.02 (flood mean); HF 0.70–1.23 in the same order of magnitude. Serial
dependence within a realization costs at most ~30 % of the pooled units: the
900 pooled unit-years of one evaluation act as ≈ 610–900 independent units,
below but of the same order as the 1,000 one-year realizations of Quinn et al.
(2017). The naive unit-level bootstrap therefore understates the tail
operators' SE by up to 20 %; every SE in this note is realization-level.

**Cross-check (figure B7).** Subsampling the epsilon-calibration cubes (513
random feasible policies, pre-regeneration substrate) at N = 25/50/75 gives
policy-median level SEs that decay at the same rate and sit within a factor
of ~1.5 of the library's policy medians on every objective (the library's ten
front policies are noisier on the deficit axes, as intended).

**Verdict under the pre-registered rule (statistical; cost is a separate
fact in §7.4/§7.6).** (i) **N_min(PS) = 300**: the smallest ladder N at which
every criterion holds; N = 100 misses only the paired-SE criterion, on the
three tail operators, by a factor 1.4–1.6. (ii) **N_min(HF) is not
established by this ladder.** HF meets every criterion on seven objectives
(the tail operators Montague deficit and storage from N = 200–400) and
fails on NYC deficit P99 at every N ≤ 500; the present evidence — three
constructions per rung (≈ 50 % relative error on an SD) and a ladder that
ends at 500 — cannot say whether that axis converges at a larger N or never
does. (iii) **N_common ≥ 300, upper end open.** The rule N_common = max over
designs bounds the common size from below and leaves it unfixed until
N_min(HF) is measured. The statistically required next step is incremental
on the persisted library: raise the HF constructions to ≥ 10 anchor plans
per N (relative SE of the SD ≈ 20 %), extend the ladder to 750, 1,000 and
1,500 for both designs (P_ref raised to 15,000 so PS keeps ≥ 10 disjoint
replicates), and read each crossing from the fitted decay with its
confidence band rather than from the rung grid. Only after that is
N_common a number; whether the budget accommodates it is a decision outside
this note.
### 7.4 NFE asymptote (measured 2026-08-25 from the existing archives; `tables/nfe_asymptote.csv`, figure B8)

Per island of the 2026-08-10 go/no-go cells (500k NFE = 125k per island, seed 1):

| design | island | NFE at 95 % of final HV | NFE at 99 % | HV gained in the last fifth (frac. of final) | ε-progress rate, last fifth / first fifth |
|---|---|---|---|---|---|
| PS | 0–3 | 92.5k–105k | 115k–122.5k | 0.03–0.05 | 0.58–2.3 |
| HF | 0–3 | 100k–107.5k | 115k–125k | 0.05–0.08 | 0.58–0.90 |
| HIST (`mm_full`, 12.5k/island, 2 seeds) | 0–3 | 11k–12k | 12k–12.5k | 0.05–0.13 | 0.65–7.9 |

**Reading 1 (asymptote):** no island reaches 95 % of its final hypervolume
before ~75 % of its budget, and every island still gains 3–8 % of its final
hypervolume in the last fifth of the run while its ε-progress rate remains
within a factor of ~2 of the early-run rate. The searches are budget-limited,
not converged; the archive is still growing (3.3k–4.3k members at the end).
**Reading 2 (the trade):** because there is no NFE slack, N × NFE cannot be
held constant by shortening the search without giving up hypervolume the
current budget still buys, so raising N is a pure cost increase of
(N/100)^0.951 per search: on the measured production basis (21,850 SU per
N = 100 / 500k search) ≈ 32k SU at N = 150, 42k at N = 200, 62k at N = 300
before the batching penalty (Table `cost_pricing.csv` and figure B9 price the
same curve on the model basis, 1.53× higher; `campaign_design.md` §6). Wall
time at 8 nodes scales the same way, and the 96 h queue cap binds at N ≈ 300
for a 750k-NFE search, which is why the campaign runs on 12 nodes. Memory at
full packing (601 + 0.394·N·L MB per rank on the K = 1 probe fit; the step-06
pre-flight uses the more conservative 600 + 0.49·N·L envelope) leaves 128-rank
packing feasible at 85 % of node memory only up to N ≈ 280; N = 300 runs with
a 150-realization batch, N = 500 would need 86 ranks per node or a smaller
batch.

### 7.5 Layer C: the campaign as the confirmation

The production campaign at N = 300 (`campaign_design.md`: one draw and two seeds
per matched design, 500k NFE reported from seed 1's runtime snapshot) is the
larger-N arm; the N = 100 go/no-go cells (draw 0, seed 1, 500k NFE, both
matched designs) are the comparison arm. Both are compared solely on E_test,
reusing the step 07/08 machinery (`src/diagnostics.py`,
`src/plotting/hypervolume_convergence.py`, `src/plotting/seed_reliability.py`,
`src/results_data.py`, `src/robustness.py`) plus one supplemental script:
(i) per-N-level ε-filtered reference sets and runtime hypervolume normalized
within level (never across levels); (ii) search-value vs E_test-value 1:1
plots per solution and objective with the attribution rule — every solution
shifts on an objective → the operator is unstable at that N; only the
solutions optimized under one N shift → overfitting; (iii) Kendall τ_b between
search and re-evaluated ranks per objective (ordering preservation); (iv) seed
attainment probability of the joint-Starr endpoint and (v) seed variance of the
endpoint at each N. Reported, not gating: whether the N = 300 solutions'
re-evaluated joint Starr fraction is at least the N = 100 cells' at equal NFE,
and whether the search-vs-test 1:1 shift on the tail operators shrinks. Staging
before the campaign: steps 02–04 at N = 300 (the per-axis tail share recorded
per hazard-filling draw at the new N; §7.1 shows it flat in N at P = 10⁶), step
05, and the ε re-verification of §7.6.

### 7.6 Consequences of N_common = 300 (adopted; `campaign_design.md` states the campaign)

1. **Epsilon floors re-verified at N = 300 before any search.** The adopted
   vector [0.05, 10.0, 0.05, 10.0, 0.05, 0.3, 5.0, 0.05] was calibrated on
   N = 100 ensembles (`epsilon_calibration_experiment.md`); bootstrap noise
   floors fall as N rises, so the vector stands unless an entry falls below its
   N = 300 floor. `workflow/supplemental/epsilon_calibration.sh` per design on
   the N = 300 ensembles (`EPS_REALIZATION_BATCH = 150`; no JAR rebuild for
   ε-only changes), the regret tolerance τ = k·max(ε, floor) re-pinned in every
   env file only if ε changes, and the archive-cardinality re-filter repeated
   on the first N = 300 archives after seed 1.
2. **HF enrichment at fixed P = 10⁶:** §7.1's (N, P′) ladder shows the min
   per-axis tail share flat in N (0.27–0.29 from N = 50 to 500) and saturated
   in P′ at the production pool size, so N = 300 needs no larger pool; the
   per-axis shares are recorded per draw at N = 300 as build QC.
3. **E_test resolution sentence.** E_test's per-SOW precision is stated in its
   own terms (`src/etest.py` SIZING; `scenario_design_methods.md` §5.4): 1,225
   pooled units per SOW against the 675-unit level-SE crossing of the i.i.d.
   library and the per-SOW noise measured directly on E_test. R = 25 is
   unchanged.
4. **Search geometry and memory.** `src/moea_config.py::production` is 4 × 382
   workers = 1,533 ranks on 12 nodes at 128 per node with
   `NYCOPT_SEARCH_REALIZATION_BATCH=150` in the matched env files
   (`config.search_node_rss_gb`: ~259 GB/node unbatched, ~167 GB batched,
   against a 217 GB line; the step-06 pre-flight enforces it). Wall time
   scales as (N/100)^0.951: ~66–77 h for the 750k seed on 12 nodes.
5. **Sizing constants.** `src/scenario_designs.SEARCH_ENSEMBLE_N = 300` (the
   single source; every design's `n_realizations`), the staged slugs
   (`fixprob_10yr_n300_d{k}`, `hazfill_stat_abs_10yr_n300_d{k}` — steps 02–04
   per draw), `supplemental_config.ESD_N_CAMPAIGN = 300` (the figures' campaign
   marker; re-render), and every note and manuscript passage that states the
   campaign N, unit count (2,700 annual units per evaluation), geometry, or
   budget (`campaign_design.md` is the reference).
6. **Budget.** One searched draw and two seeds per matched design, seed 1 at
   750k NFE and seed 2 at 500k, on the measured cost basis: ~489k SU against
   the ~600k remaining balance (`campaign_design.md` §6).

## 8. Run sequence

```
# 0. smoke (shared; minutes)                       NYCOPT_ESD_SMOKE=1 on every wrapper
# 1. Layer A + library plan (shared, 1 task)        workflow/supplemental/ensemble_size_hazard.sh
# 2. regenerate + prep chunks (shared array)        workflow/supplemental/ensemble_size_library_stage.sh
# 3. library evaluation (wholenode, 1 node)         workflow/supplemental/ensemble_size_library_eval.sh
# 4. statistics, decision table, figures (shared)   workflow/supplemental/ensemble_size_analysis.sh
```

Every wrapper sources `workflow/envs/ensemble_size_diagnostics.env`; all
settings are in `supplemental_config.py` (`ESD_*`); no CLI value flags.

## Figures (one sentence each on what a reader learns)

- **A1 `hf_tail_share_vs_n`** — how much of HF's tail enrichment survives as N
  grows at P = 10⁶, per axis, against the random null and its 0.10 expectation.
- **A2 `hf_coverage_vs_n`** — whether the joint geometry (L2-star, MST edge,
  minimum separation) of the HF selection stays ahead of a random design of the
  same size as N grows.
- **A3 `np_ladder`** — how the tail share saturates with pool size P′ at each N.
- **A4 `ps_tail_sampling`** — how many severe members an i.i.d. sample of N
  contains, its spread across draws, the quantile error of the sample against
  the pool, and the closed-form chance of holding at least one member beyond
  the pool P99.
- **A5 `descriptor_convergence`** — pooled-mean descriptors flatten with N while
  ensemble extremes drift monotonically, separating what N can and cannot buy.
- **B1 `level_se_vs_n`** — per objective, whether estimator noise is below ε at
  each N for both designs.
- **B2 `paired_se_vs_n`** — the decision figure: the worst-pair paired SE
  against ε/2, per objective and design, marking N_noise.
- **B3 `flip_rate_vs_n`** — how often a sample of N reverses an ε-dominance
  verdict relative to the reference.
- **B4 `optimism_vs_n`** — PS estimator bias versus N per operator, and the HF
  construction shift ± SD beside it.
- **B5 `tail_replicate_bands`** — the replicate distribution of the tail
  operators at each N, policy by policy, with ε for scale.
- **B6 `effective_sample_size`** — how many independent unit-years the 900
  serially dependent units are worth, per operator.
- **B7 `epscube_crosscheck`** — the epsilon-cube subsampling decay overlaid on
  the library's, confirming the ten policies are representative.
- **B8 `nfe_asymptote`** — where hypervolume and ε-progress flatten in the
  existing 500k-NFE runs, i.e. whether NFE can be traded for N.
- **B9 `cost_pricing`** — SU per search and per-node memory versus N with the
  ladder and the campaign N marked.

## Citations

- Bonham, N., Kasprzyk, J., Zagona, E., & Rajagopalan, B. (2024). *Environmental Modelling & Software*, 172, 105933.
- Hamilton, A. L., et al. (2022). Two-layer objective vocabulary (within-record aggregation + across-record noise filtering).
- Homem-de-Mello, T., & Bayraksan, G. (2014). *Surveys in Operations Research and Management Science*, 19, 56–85.
- Kasprzyk, J. R., Nataraj, S., Reed, P. M., & Lempert, R. J. (2013). *Environmental Modelling & Software*, 42, 55–71.
- Kaut, M., & Wallace, S. W. (2007). *Pacific Journal of Optimization*, 3(2), 257–271.
- Linderoth, J., Shapiro, A., & Wright, S. (2006). *Annals of Operations Research*, 142, 215–241.
- Quinn, J. D., Reed, P. M., Giuliani, M., & Castelletti, A. (2017). *Water Resources Research*, 53, 7208–7233.
- Reed, P. M., Hadka, D., Herman, J. D., Kasprzyk, J. R., & Kollat, J. B. (2013). *Advances in Water Resources*, 51, 438–456.
- Zatarain Salazar, J., Reed, P. M., Quinn, J. D., Giuliani, M., & Castelletti, A. (2017). *Advances in Water Resources*, 109, 196–210.
