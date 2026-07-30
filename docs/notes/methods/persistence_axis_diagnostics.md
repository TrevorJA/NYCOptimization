# Interannual-persistence DU axis: mechanism design + feasibility diagnostics

**Status:** closed 2026-07-29 — **no persistence axis; no supplementary persistence
ensemble**. E_test is the sole re-evaluation ensemble; the persistence gap is disclosed
and the affected claims are scoped (see Decision).
**Date:** 2026-07-29. **Code:** `scengen.persistence` (mechanism),
`scripts/supplemental/diagnose_persistence_axis.py` (diagnostic driver),
`scripts/supplemental/verify_kirsch_baseline_persistence.py` (baseline verification),
`scripts/supplemental/check_cmip6_rho1_anchor_noise.py` (anchor sampling-noise check).
**Evidence:** `outputs/supplemental/persistence_axis/` (gitignored, regenerable).

## Why

The Kirsch hybrid bootstrap draws its bootstrap index matrix `M` i.i.d. uniform per
(synthetic year, month) cell, so synthetic interannual wet/dry sequencing carries no memory
beyond the cross-year-boundary Cholesky correlation. Measured on the full-record fit: the
unperturbed generator's annual lag-1 autocorrelation of aggregate NYC inflow is
**ρ₁ = 0.02 ± 0.02**, against **0.22** on the fitted record (calendar-year; 0.26 water-year).
Multi-year drought persistence is therefore structurally absent from both the search
populations and the default E_test, and no adjustment of the fitted moments (the mean/CV
forcing axes) can create it. A persistence axis must enter through the generator's dependence
structure. The HMM E_test variant is not the vehicle for this: it swaps the whole generator
(a second DU factor set), rather than adding one interpretable axis to the existing one.

## Baseline: where interannual dependence can and cannot live in Kirsch–Nowak

The generation pipeline (`synhydro.methods.generation.hybrid.kirsch.KirschGenerator`) has
five stages: (1) fit per-(month, site) moments and normal-score transforms on the
aggregated record; (2) draw the bootstrap index matrix `M` — one historical-year index per
(synthetic year, month) cell, i.i.d. uniform (`_get_bootstrap_indices`:
`rng.choice(H, size=(n_years, 12))`); (3) gather the standardized residual tensor
`Y[M[i,m], m, :]` and multiply by each site's within-year 12×12 Cholesky factor; (4) stitch
the calendar-year block to a half-year-shifted block (`Y'`), whose own Cholesky factor
correlates the months of each Jul–Jun window, so month-to-month correlation survives the
Dec–Jan boundary; (5) inverse-NST and destandardize. The correlation operator is strictly
intra-annual (two 12×12 factors per site); the only cross-year device is the half-year
stitch. Nowak disaggregation then samples a daily proportion vector per coarse period and
rescales to the period total — annual totals pass through exactly, so disaggregation can
neither add nor remove interannual dependence. Interannual persistence can therefore enter
this generator family in exactly two places: the distribution of `M` (dependence between
rows — Mechanism A and the bootstrap variants below), or a cross-year term in the
correlation operator itself (Mechanism B).

Verified on the public `generate()` ensemble path (64 × 50-yr realizations,
`verify_kirsch_baseline_persistence.py`), aggregate NYC inflow:

| annual window | fitted record | unperturbed generator |
|---|---|---|
| calendar year | +0.216 | **+0.014 ± 0.020** |
| water year (Oct–Sep) | +0.263 | **+0.099 ± 0.017** |

The nonzero water-year value is the half-year stitch: annual windows that cut inside the
Jul–Jun shifted block inherit its cross-boundary correlation (start-month sweep: ρ₁ ≈ 0.01
for Jan/Apr starts, 0.06 for Jul, 0.10 for Oct). Since the study's annual units are water
years, the operative persistence gap is **0.26 (record) vs 0.10 (generator)** — smaller
than the calendar-year framing suggests, though the multi-year drought proxies are
unaffected (max below-median run 5.1 yr; p10 min 3-yr total 0.66; annual CV ratio
synthetic/record 0.94). An independent measurement in the generator-comparison study
(`SynHydroGeneratorComparison`, New England basins) puts pooled Kirsch annual lag-1 ACF at
0.0001 (calendar-year), the lowest of 13 generator families, corroborating the baseline.

Elsewhere in SynHydro, tunable or fitted interannual persistence already exists — SMARTA
(any-range target annual ACF), ARFIMA (long memory `d`), the multi-site HMM (regime
transitions), phase randomization (spectrum-preserving), HMM-KNN (regime chain + lag-1
analog conditioning) — but each is a different generator family with different marginal
structure and validation properties; none is a drop-in persistence dial for the fitted
Kirsch marginals, so swapping families would reintroduce the second-DU-factor-set problem
the HMM E_test variant was rejected for.

## CMIP6 anchoring

Annual (water-year) totals of aggregate NYC inflow per CMIP6 run
(`CMIP6_multimodel_streamflow/pywrdrb/inputs/*/catchment_inflow_mgd.csv`, 39-yr records),
sibling-matched future vs each run's own 1980–2019 baseline:

| quantity | p5 | p50 | p95 | min/max |
|---|---|---|---|---|
| future-run ρ₁ | −0.10 | 0.08 | 0.40 | −0.17 / 0.49 |
| baseline-run ρ₁ | −0.29 | 0.00 | 0.34 | |
| sibling Δρ₁ (future − baseline) | −0.12 | +0.08 | +0.42 | |

The projections tend toward *more* interannual persistence (late-century SSP370 runs supply
the +0.4–0.5 deltas; the small negative CMIP6 tail is not worth a signed mechanism).

### Anchor vs sampling noise

Each ρ₁ above is estimated from a 39-water-year record, so a single estimate carries
se(ρ̂₁) ≈ 0.15 and a sibling difference ≈ 0.22 (per-pair 90% noise band ±0.41).
`check_cmip6_rho1_anchor_noise.py` reproduces the anchor table from the raw per-run CSVs
and tests it against Monte Carlo nulls with the ensemble's exact dependence structure
(54 future runs sharing 14 hydro-model × GCM baselines; no true change in ρ₁):

- **The central tendency is a real but modest signal.** Ensemble mean Δρ₁ = +0.107
  (p ≈ 0.012–0.014 against both nulls); median +0.080 (p ≈ 0.06); 6 of 7 GCMs have
  positive mean Δ (sign test p = 0.06). Bias-corrected central estimates: baseline true
  ρ₁ ≈ 0.03, future true ρ₁ ≈ 0.15.
- **The upper tail is sampling noise.** The spread of future-run ρ̂₁ (sd = 0.154) equals
  the sampling se of a common-ρ null (0.153; p = 0.48): there is no evidence of run-to-run
  heterogeneity in true persistence, i.e. no evidence that any run truly has ρ₁ ≈ 0.4–0.5.
  The p95 = 0.40 anchor value is the expected extreme order statistic of 54 noisy
  estimates around a modest common mean.
- **Observed persistence is epochal, and the CMIP6 chain matches its own window.** The
  same statistic on the reconstruction: ρ₁ = +0.26 over WY1946–2022, but **+0.06 over
  WY1981–2019** (the CMIP6 baseline window) and **+0.51 over WY1946–1980** (the epoch
  containing the 1960s drought). The near-zero CMIP6 baselines are consistent with the
  observed record in their own window — the chain is not discredited on persistence — but
  a 39-yr window is nearly uninformative about ρ₁ (se ≈ 0.16 spans the full observed
  epochal range), and the full-record 0.26 is dominated by the pre-pluvial epoch.

The evidence-supported anchor is therefore a **modest central shift** (Δρ₁ ≈ +0.1;
future central ρ₁ ≈ 0.15, upper bound ≈ historical full-record 0.26), not the earlier
noise-driven **[0, ~0.5]** reading: a defensible axis range is **ρ₁\* ∈ [0, ~0.25]**.

## Mechanism A — latent-annual-state tilted bootstrap (prototyped)

`scengen.persistence.persistent_bootstrap_indices`: a standard-normal AR(1) latent annual
state `w_t` (coefficient `φ`) drives each cell's bootstrap draw through a Gaussian copula
with loading `λ`, mapped to a historical year via the annual-wetness rank order of the
residual tensor. Every cell's marginal stays *exactly* uniform over historical years for any
`(φ, λ)` — the per-month bootstrap source distribution, the NST, and the shared-`M`
cross-site structure are preserved in distribution — and `λ = 0` reduces exactly to the
unperturbed generator. What changes is only the year-to-year dependence of the source years.

Measured response and distortion (full-record fit, 64 × 50-yr monthly realizations per
setting, `φ = 0.8`; distortions are vs the `λ = 0` control):

| λ | ρ₁ (±se) | max dry run (yr) | p10 min 3-yr total | adj-month corr | log-std drift p50 / p90 / max | cross-site corr |
|---|---|---|---|---|---|---|
| 0 | 0.02 (0.02) | 5.0 | 0.66 | 0.305 | — | 0.855 |
| 0.10 | 0.18 (0.02) | 6.1 | 0.63 | 0.334 | 0.02 / 0.04 / 0.06 | 0.862 |
| 0.20 | 0.23 (0.02) | 6.7 | 0.57 | 0.344 | 0.04 / 0.09 / 0.20 | 0.865 |
| 0.35 | 0.30 (0.02) | 7.1 | 0.56 | 0.379 | (max 0.37) | 0.871 |
| 0.50 | 0.37 (0.02) | 7.9 | 0.50 | 0.393 | (max 0.40) | 0.878 |
| 1.00 | 0.50 (0.02) | 8.8 | 0.47 | 0.434 | (max 1.24) | 0.865 |

(`φ = 0.5/0.9` at `λ = 0.35` give ρ₁ = 0.21/0.29 — `φ` sets the memory length, `λ` the
strength; the sweep above is the operative dial.)

The mechanism spans the full CMIP6-anchored range and produces exactly the intended
multi-year drought stress (run lengths, 3-yr deficit tails). The distortion source is
structural: the Cholesky step assumes independent bootstrap cells, so the within-year
coupling `λ` induces post-Cholesky variance inflation and adjacent-month correlation
inflation that grow with `λ`.

**Admissibility.** At `λ ≤ 0.1` (ρ₁ ≈ 0.18, i.e. restore-historical persistence) the
generator's validation properties hold to ≤6% worst-case monthly log-std drift, +0.03
adjacent-month correlation, +0.01 cross-site correlation. By `λ = 0.2` the worst-month
std drift reaches 20%, and beyond that the moment distortion is inadmissible. **Mechanism A
alone can restore historical-level persistence but cannot span the CMIP6 upper tail
(ρ₁ ≈ 0.4–0.5) while preserving the generator's validation properties.**

## Mechanism B — AR(1) annual-factor re-attribution (designed, not implemented)

The moment-preserving route is to *re-attribute* the year-scale common variance the fitted
within-year correlation already contains, rather than add to it: fit a one-factor model
`z_m = β_m f + e_m` to the intra-annual correlation matrix, give the factor AR(1) dynamics
across years (`φ` = the DU axis), and impose the *residual* correlation `C − ββᵀ` through
the Cholesky. For any `φ` this reproduces the historical monthly moments and within-year
correlation by construction; only cross-year dependence changes. The reachable range is
`ρ₁ ≤ φ·κ²` where `κ²` is the year-factor share of annual variance — measured **κ² ≈ 1.0**
on the historical aggregate (the annual total is dominated by the year-common component
because idiosyncratic monthly noise averages out over 12 months), so the full CMIP6 range
is reachable with `φ ≈ ρ₁*`. The cost is structural: the factor is parametric
(score-space Gaussian), so part of what the bootstrap kept non-parametric becomes
Gaussian-copula, and it requires a SynHydro generator extension plus its own validation
battery (monthly moments, within-year and boundary correlation, cross-site dependence,
drought/flood tails vs the unperturbed generator).

## Candidate mechanisms beyond A/B (parsimony screen)

Simple alternatives assessed against Mechanisms A/B on reachable ρ₁, preservation of the
fitted structure, and method-complexity cost:

- **Year-block / per-year bootstrap** (draw one historical-year index per synthetic year,
  or whole-year blocks of consecutive historical years). The per-year case is Mechanism A's
  `λ = 1, φ = 0` corner (all 12 cells share one source year): maximal within-year
  distortion, no cross-year memory. Consecutive-year blocks restore at most the record's
  own persistence, are not tunable, and inherit the same distortion inside each block.
  Dominated by Mechanism A, which subsumes both with a dial.
- **Two-state (wet/dry) Markov year sampling** over the wetness-ranked years. A discrete
  latent state with a transition-persistence dial — Mechanism A with a 2-point instead of
  Gaussian latent state. Same within-year coupling distortion, coarser control, marginal
  uniformity harder to preserve exactly. No advantage over the implemented A.
- **Weighted bootstrap toward dry years** (Herman et al. 2016 style). Tilts every cell's
  *marginal* toward dryness — a drought-frequency/intensity stress, not a sequencing
  stress — and breaks the fitted monthly moments that the mean axis `m` already perturbs
  in a controlled way. Not a persistence mechanism; rejected for this purpose.
- **Post-hoc hazard-space characterization/stratification of E_test.** No generator
  change: the SSI-6 run-theory descriptors already computed for hazard images quantify
  how much multi-year-drought stress E_test actually expresses, and per-stratum
  breakdowns of the (full-ensemble) re-evaluation can report performance on the
  worst-expressed multi-year events. Cannot *extend* the expressed tail beyond what the
  i.i.d. population generates, so it is disclosure, not stress — but it is nearly free
  and pairs with any option below (the E_test hazard overlay is already planned).
- **Longer L_test only.** Longer windows let the i.i.d. population express longer worst
  dry runs by chance, but the underlying run-length distribution stays thin-tailed
  relative to a persistent process (compounding via ρ₁ is what deepens multi-year
  deficits, cf. the λ-response table). A partial, linear-cost palliative, not a
  substitute.
- **Fixed persistence stress-test ensemble (Mechanism A at a fixed setting, no DU
  axis).** One supplementary ensemble at `(φ, λ) = (0.8, 0.1)` (ρ₁ ≈ 0.18, inside the
  admissible region) re-evaluated alongside E_test. Rejected on workflow grounds: the
  campaign endpoint is single-ensemble by design — every Pareto set is re-evaluated on
  the full E_test and nothing else — and a second evaluation ensemble would stand up a
  parallel evidence base beside the pre-registered endpoint, method accretion
  disproportionate to the claim it would support.

## Decision (2026-07-29)

**No persistence axis, and no supplementary persistence ensemble. E_test remains the
sole re-evaluation ensemble; the persistence gap is disclosed and the affected claims
are scoped.** Basis:

- *The projection case for an axis fails.* The CMIP6 sibling-Δρ₁ anchor's upper tail
  (ρ₁ ≈ 0.4–0.5) is statistically indistinguishable from 39-yr sampling noise (anchor
  check above); the surviving signal is a modest central shift (Δρ₁ ≈ +0.1) from a chain
  class that systematically underestimates drought persistence (Moon et al. 2018; Vieira
  & Stadnyk 2023) and does not reproduce the internal-variability mechanism behind the
  region's actual persistence epochs (Seager et al. 2012). Projections for the NE point
  toward wetting and *flash* drought, not multi-year persistence (Cook et al. 2020; Xue &
  Ullrich 2021, 2022). A DU axis parameterizing "future persistence change" would be
  anchored on noise; the full-range Mechanism B extension has no scientific warrant, and
  a bounded axis has no defensible range beyond "restore what the record already shows."
- *What remains is a disclosure, not a design change.* The observed record's persistence
  is real but epochal (WY ρ₁ 0.51 pre-1980 vs 0.06 after; 0.26 full record), the record
  itself sits in an unprecedented ~43-yr pluvial (Pederson et al. 2013), and the
  generator expresses ρ₁ ≈ 0.10 in the operative water-year frame. Because every
  scenario design *and* E_test share the same generator, this limitation does not touch
  the design comparison — the paper's contribution — at all; it bounds only the
  absolute robustness claims, which are scoped to "under historical-level (pluvial-era)
  interannual persistence" with the measured numbers stated
  (`scenario_design_methods.md` §5.1/§6). The planned E_test hazard overlay
  (simulation-free) quantifies how much multi-year-dry stress E_test actually expresses,
  completing the disclosure.

Consequences: E_test stays 3-axis with L_test = 10 yr (the §2 sizing gate resolves to
the no-axis branch). Mechanism A remains prototyped and validated in `scengen.persistence`
as future-work material (a persistence-stressed test ensemble is the natural follow-on
study); Mechanism B is rejected outright.
