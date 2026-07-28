# Interannual-persistence DU axis: mechanism design + feasibility diagnostics

**Status:** designed and prototyped; **not adopted** — adoption is an open campaign decision.
**Date:** 2026-07-28. **Code:** `scengen.persistence` (mechanism),
`scripts/supplemental/diagnose_persistence_axis.py` (diagnostic driver). **Evidence:**
`outputs/supplemental/persistence_axis/` (gitignored, regenerable).

## Why

The Kirsch hybrid bootstrap draws its bootstrap index matrix `M` i.i.d. uniform per
(synthetic year, month) cell, so synthetic interannual wet/dry sequencing carries no memory
beyond the cross-year-boundary Cholesky correlation. Measured on the full-record fit: the
unperturbed generator's annual lag-1 autocorrelation of aggregate NYC inflow is
**ρ₁ = 0.02 ± 0.02**, against **0.22** on the fitted record (calendar-year; 0.27 water-year).
Multi-year drought persistence is therefore structurally absent from both the search
populations and the default E_test, and no adjustment of the fitted moments (the mean/CV
forcing axes) can create it. A persistence axis must enter through the generator's dependence
structure. The HMM E_test variant is not the vehicle for this: it swaps the whole generator
(a second DU factor set), rather than adding one interpretable axis to the existing one.

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
the +0.4–0.5 deltas). A CMIP6-plausible axis range is **ρ₁\* ∈ [0, ~0.5]** (the base
generator sits at ~0; the small negative CMIP6 tail is not worth a signed mechanism).

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

## Verdict

A persistence DU axis is **feasible**, on one of two terms:

1. **Bounded axis now** (Mechanism A, `λ ∈ [0, ~0.1–0.2]` ≙ ρ₁ ∈ [0, ~0.2]): restores
   historical-level interannual persistence with small, quantified distortion; no generator
   changes beyond injecting `M`. Spans "no persistence → historical persistence", not the
   CMIP6 upper tail.
2. **Full CMIP6 range** (ρ₁ ∈ [0, ~0.5]): requires Mechanism B (SynHydro factor
   extension + validation). Mechanism A cannot get there admissibly.

Whether either enters the campaign (E_test axis; the search populations stay stationary) is
an open scope decision — the study's current claims are scoped to "under historical
persistence", and E_test currently does not stress persistence at all
(`scenario_design_methods.md` §6, flagged uncertainties).
