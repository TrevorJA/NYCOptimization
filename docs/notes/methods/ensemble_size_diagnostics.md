# Ensemble-Size Diagnostics: a Statistically Grounded Minimum N (SI)

*Last updated: 2026-08-25. Supplemental diagnostic (SI Texts S4 and S5) deriving a
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

**PRE-REGISTRATION.** Every criterion, threshold, ladder, replicate count, and
selection rule in §§2–5 was fixed on 2026-08-25 before any Layer-A or Layer-B
result was inspected. §7 records what was measured; nothing in §§2–5 was edited
after §7 was written.

---

## 1. The decision and what "representative" means

The campaign search ensemble is N = 100 realizations × L = 10 yr for both
matched designs. That value was set by compute budget and by a selection-level
argument (hazard-filling tail enrichment falls as N rises at fixed pool size P;
`hazard_selector_diagnostics.md` §5). It was never derived from the statistical
adequacy of the ensemble as the *sample the optimizer scores every policy on*.
This diagnostic derives a minimum N from that adequacy. L = 10 is fixed (the
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
  enrichment above the adequacy gate) and (b) reproducibility of the induced
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
| A-NP | min per-axis tail share above P90 (seed-mean convention) on nested prefixes P′ of pool d0 | HF | P′ ∈ {5·10³, 2·10⁴, 10⁵, 3·10⁵, 10⁶} × the N ladder × 10 anchor plans; reports the smallest P′ holding the gate at each N |
| A-PS | sampling distribution of per-axis counts above pool P90/P99; relative error of subset P50/P90/P99 vs the pool; closed-form P(≥ 1 member beyond pool quantile q) = 1 − qᴺ | PS | uniform size-N subsets of the pool image (exact i.i.d. law); 200 subsets per N |
| A-CV | convergence of the descriptors themselves: pooled mean per axis (expected to flatten) vs ensemble extreme per axis (expected to move monotonically, never converge) | PS, HF | 5/50/95 bands over the 200 PS subsets; HF seeds; N ladder |

**Gate (unchanged from `hazard_selector_diagnostics.md`):** min per-axis tail
share above P90 ≥ 0.30 on every selection axis, seed-mean convention. Layer A
*reports* the N at which P = 10⁶ stops holding the gate and the P′ each N
needs; it does not size N by itself — it explains mechanism (Bonham et al.
2024: maximum-based statistics never converge; Zatarain Salazar et al. 2017:
tail representation is what low sampling rates lose).

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
  N = 100 (the design as staged; cross-checks the prefix-subset law).
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

### 4.3 Statistics (per objective and N; every figure carries ε and the N = 100 point)

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
| 5 | Effective sample size | per operator at the campaign point: n_eff/(N(L−1)) = (SD_unit-bootstrap / SD_realization-bootstrap)², the naive i.i.d.-unit bootstrap against the realization-level block bootstrap | report only (SI Text S5) | Quinn et al. 2017 §3.1.2 (1,000-unit convention); Hamilton et al. 2022 |

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
level-SE decay is overlaid on the library's at the campaign point; agreement
in shape confirms the library's ten policies are not atypical.

## 5. Layer C — search-level confirmation (design and cost only; NOT run)

Only searches show that the optimizer's output changes with N. The confirmatory
design, to be run only if a larger N is adopted, follows Zatarain Salazar et al.
(2017) and Quinn et al. (2017) §3.4:

- Arms: N = 100 (campaign) and N = N_common, both designs, K = 3 draws × S = 2
  seeds; two budget variants — equal NFE (500k, so cost scales with N) and
  equal SU (NFE scaled by 100/N_common, so N × NFE is constant).
- Compared solely on E_test: per-N-level reference sets (hypervolume is never
  compared across N levels — Zatarain Salazar et al. normalize within level);
  search-value vs test-value 1:1 plots per solution with the attribution rule
  (all solutions shift on an objective → the operator is unstable; only the
  solutions optimized under it shift → overfitting); ordering preservation
  under re-evaluation (Kendall τ_b between search and test ranks per
  objective); seed attainment probability; draw-versus-seed variance
  components. Reuses `src/diagnostics.py`, `src/plotting/hypervolume_convergence.py`,
  `src/plotting/seed_reliability.py`, `src/results_data.py`, `src/robustness.py`.
- If N_common > 100 is adopted, the production campaign at N_common *is* the
  confirmation and only the N = 100 go/no-go cells are the comparison arm.

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

### 7.1 Layer A (measured 2026-08-25; job 20142729, 24 min, 1 core; tables `hf_ladder`, `np_ladder`/`np_gate`, `ps_tail_sampling`, `descriptor_convergence`; figures A1–A5)

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

**The regenerated pools sit just under the adequacy gate at every N.** The
same statistic is 0.27–0.29 on d0, d1, d2 (min over N of the seed mean:
0.271, 0.276, 0.271 at N = 100), against the 0.311/0.306/0.303 recorded for the
pre-regeneration pools (`TODO.md` §2). The prefix ladder saturates
(0.268 → 0.282 from P′ = 10⁵ to 10⁶ at N = 100), so this is the m = 6
geometry on the December-epoch hazard images, not pool supply; the
**smallest P′ holding the gate is none at any N** (A3b). This is a
pre-registered gate miss of ~0.02 on the production draws that the
post-regeneration re-gate flagged in `TODO.md` §1 had not yet caught; the
official block-D re-gate with the diagnostics driver's own seeds 0–9 is job
20143046 (result in §7.1a below). It does not change the N decision (the
statistic is N-flat), but it is a build-QC finding for the campaign.

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

### 7.1a Official re-gate of the regenerated pool d0 (job 20143066; `diagnose_hazard_selectors.py`, saturation mode + N sweep, seeds 0–9, 50-seed null)

The diagnostics driver's own convention gives, on `statpool_10yr_n1000000_d0`
at N = 100: campaign six-axis set **min per-axis tail share 0.283 (FAIL, gate
0.30)**, worst axis drought_magnitude (0.284), then flood_pulse_duration 0.334,
drought_severity 0.342, flood_peak_discharge 0.389, drought_recovery_rate
0.445, drought_onset_rate 0.513 (mean 0.385); full eight-axis set 0.220
(drought_duration 0.239). Across the N sweep the campaign-set minimum is
0.273–0.290 with no trend (0.284, 0.273, 0.283, 0.283, 0.290, 0.289, 0.286,
0.288 for N = 50…500), the P99 minimum 0.009–0.017, and joint L2-star 0.019 →
0.011. Two conclusions: (i) the thin-margin gate pass recorded for the
pre-regeneration pools (0.311) has become a thin-margin miss on the
December-epoch images — drought_magnitude is the binding axis; the campaign
draws d0–d2 need this recorded as build QC and the gate either re-affirmed
with its ~0.02 seed-level tolerance or the axis set/P revisited (Trevor's
call; `TODO.md`); (ii) the N ladder is flat, so the miss is independent of N
and the ensemble-size decision does not turn on it.
### 7.2 Layer B library and QC
### 7.3 Decision table
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
current budget still buys, so raising N is a pure cost increase at
33,400 × (N/100)^0.951 SU per search (Table `cost_pricing.csv`; figure B9):
≈ 49k SU at N = 150, 65k at N = 200, 95k at N = 300 — i.e. the 12-search
matched campaign moves from 401k SU to 589k / 775k / 1,139k SU, against a
750k-SU allocation of which ~247k is reserve. Wall time at 8 nodes scales the
same way (48 h at N = 150, 63 h at N = 200; the 96 h queue cap binds at
N ≈ 300). Memory at full packing (601 + 0.394·N·L MB per rank) leaves
128-rank packing feasible at 85 % of node memory only up to N ≈ 280; N = 300
needs 124 ranks per node or an explicit realization batch, N = 500 needs 86.

### 7.5 Layer C plan and cost (design only; not run)

The confirmation follows Zatarain Salazar et al. (2017) at two N levels,
compared solely on E_test. Let s = (N_common/100)^0.951 be the measured cost
scale.

| arm | geometry | searches | SU (searches) | SU (E_test re-eval of the new fronts) |
|---|---|---|---|---|
| N = 100 (campaign) | `production` env files as they stand | the K = 3 × S = 2 campaign cells (already budgeted) | 0 extra | 0 extra |
| N_common, equal NFE (500k) | new env files `ffmp_obj8_{fixedprob,hazfill_stat}_production_n{N}.env` with `NYCOPT_SEARCH_N=N_common`; `--time` scaled by s; `NYCOPT_SEARCH_REALIZATION_BATCH=100` (or ≤ 124 ranks/node) when N_common ≥ 300 | 2 designs × K = 3 × S = 2 = 12 | 12 × 33,400 × s | ≈ 80k (≈ 73 SU per merged policy; the re-filter keeps the merged sets near 1,100) |
| N_common, equal SU | as above with `max_evaluations` = 125,000 / s per island (a new `production_equal_su_n{N}` MOEA config) | 12 | 12 × 33,400 = 401k | ≈ 80k |
| minimum confirmatory subset | HF only (the arm whose enrichment falls with N), draw 0, S = 2, equal NFE | 2 | 2 × 33,400 × s | ≈ 15k |

Staging before any search: steps 02 (PS draws at N_common), 03 (HF selections at
N_common — the adequacy gate re-passed per draw at the new N; §7.1 says
whether P = 10⁶ still holds it), 04, and the epsilon re-calibration at N_common
(§7.6). Analysis, reusing step 07/08 machinery (`src/diagnostics.py`,
`src/plotting/hypervolume_convergence.py`, `src/plotting/seed_reliability.py`,
`src/results_data.py`, `src/robustness.py`) plus one new supplemental script:
(i) per-N-level ε-filtered reference sets and runtime hypervolume normalized
within level (never across levels); (ii) search-value vs E_test-value 1:1
plots per solution and objective with the attribution rule — every solution
shifts on an objective → the operator is unstable at that N; only the
solutions optimized under one N shift → overfitting; (iii) Kendall τ_b between
search and re-evaluated ranks per objective (ordering preservation);
(iv) seed attainment probability of the joint-Starr endpoint and (v) draw-vs-
seed variance components of the endpoint at each N. Decision rule, stated
now: the larger N is confirmed if its solutions' re-evaluated joint Starr
fraction is not lower than N = 100's at equal NFE AND the search-vs-test 1:1
shift on the tail operators shrinks; otherwise the N = 100 campaign stands and
N_common is reported as the precision-only recommendation. If N_common > 100
is adopted for the campaign outright, the production campaign at N_common is
the confirmation and only the existing N = 100 go/no-go cells are the
comparison arm (cost: the campaign delta of §7.4 alone).
### 7.6 Consequences of adopting N_common (everything that must change if N_common > 100)

1. **Epsilon calibration is re-run at N_common before any search.** The
   adopted vector [0.05, 10.0, 0.05, 10.0, 0.05, 0.3, 5.0, 0.05] was calibrated
   on N = 100 ensembles (`epsilon_calibration_experiment.md`); the bootstrap
   noise floors fall as N rises, so the floor-bound entries (reliabilities,
   flood, storage) may loosen and the archive-cardinality re-filter must be
   repeated on the first N_common archives. Re-run
   `workflow/supplemental/epsilon_calibration.sh` per design on the
   N_common ensembles (no JAR rebuild for ε-only changes), then the regret
   tolerance τ = k·max(ε, floor) re-pin in every env file.
2. **HF enrichment at fixed P = 10⁶:** §7.1's (N, P) ladder states whether the
   adequacy gate (min per-axis tail share ≥ 0.30) still holds at N_common on
   every production draw; if not, P rises (≈ 600 core-hours per 10⁶-member
   pool; `gen_pool_shards.sh`/`gen_pool_merge.sh`) and the nested-P verdict is
   re-issued.
3. **E_test resolution sentence.** E_test resolves each SOW on
   R × (L_test − 1) = 1,225 units, claimed to be at least one search evaluation
   (900 units). The claim inverts once 9 × N_common > 1,225 (N_common > 136):
   either R_test rises to ⌈9·N_common/49⌉ (e.g. R = 28 at N = 150, 37 at
   N = 200 — re-generation of E_test, ~80k SU of re-evaluation per full pass)
   or the manuscript sentence (Section 3.4.1; `src/etest.py` SIZING) is
   rewritten to state the per-SOW unit count as its own precision argument.
4. **Search geometry and memory.** `src/moea_config.py::production` (1,021
   ranks, 8 nodes), `workflow/_common.sh::NYCOPT_RANKS_PER_NODE=128`, and the
   `--time` requests in the production env files assume N = 100. Per-rank RSS
   is 601 + 0.394·N·L MB: full 128-rank packing stays under 85 % of node
   memory to N ≈ 280; beyond that set `NYCOPT_SEARCH_REALIZATION_BATCH=100` or
   drop to ≤ 124 ranks/node (N = 300) / 86 (N = 500). Wall time scales as
   (N/100)^0.951: 48 h at N = 150, 63 h at 200, 93 h at 300 (the 96-h cap).
5. **Sizing constants.** `NYCOPT_SEARCH_N` / `src/scenario_designs.SEARCH_ENSEMBLE_N`
   (the single source; every design's `n_realizations`), the staged slugs
   (`fixprob_10yr_n{N}_d{k}`, `hazfill_stat_abs_10yr_n{N}_d{k}` — steps 02–04
   re-run per draw), the smoke/moderate env files, and every figure or table
   that prints "N = 100" or "900 annual units": `scenario_design_methods.md` §6
   budget table (33,400 SU/search, 503k campaign total, 247k reserve),
   SI Text S8.5, main-text Sections 3.1.1 and 3.2.2, `TODO.md` §2 sizing,
   `supplemental_config.ENSEMBLE_COST_DESIGN_POINT`, the epsilon-calibration
   and framing-convention SI numbers quoted at 900 units.
6. **Budget.** The 12-search matched campaign costs 401k × (N_common/100)^0.951
   SU (589k at 150, 775k at 200); the 750k allocation holds N_common = 150
   only by consuming the reserve (no additional draw, no RQ3 sweep) and
   cannot hold N_common ≥ 200 at K = 3 × S = 2. The exchange rate is
   explicit: at equal SU, N = 150 buys back K = 2 draws per design.
7. **Layer C** (§7.5) becomes the campaign itself if N_common is adopted
   outright; otherwise the minimum confirmatory subset runs first.

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
  grows at P = 10⁶, per axis, against the gate and the random null.
- **A2 `hf_coverage_vs_n`** — whether the joint geometry (L2-star, MST edge,
  minimum separation) of the HF selection stays ahead of a random design of the
  same size as N grows.
- **A3 `np_ladder`** — the pool size P′ each N needs to hold the adequacy gate.
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
  ladder, N = 100, and N_common marked.

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
