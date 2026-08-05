# Interannual persistence: decision and disclosure

**Decision: no persistence DU axis and no supplementary persistence
ensemble.** E_test is the sole re-evaluation ensemble; the persistence gap is
a manuscript disclosure with the affected claims scoped accordingly
(manuscript SI Text S2; literature base in
`docs/notes/literature/persistence_and_low_frequency_variability.md`).

**The disclosed gap.** The Kirsch hybrid bootstrap draws its bootstrap index
matrix i.i.d. per (synthetic year, month) cell, so synthetic interannual
wet/dry sequencing carries no memory beyond the cross-year-boundary Cholesky
correlation. On the water-year frame (the study's annual unit) the operative
gap is **ρ₁ = 0.26 (fitted record) vs 0.10 (generator)**. The record value is
epochal, not stationary: **ρ₁ = +0.51 over WY1946–1980** but **+0.06 over
WY1981–2019** (the CMIP6 baseline window), and the record itself sits in an
unprecedented ~43-yr pluvial (Pederson et al. 2013), so the full-record 0.26
is dominated by the pre-pluvial epoch. The CMIP6 Δρ₁ signal is a modest
central shift (ensemble mean ≈ +0.1) whose upper tail is statistically
indistinguishable from 39-yr sampling noise — a DU axis parameterizing
"future persistence change" would be anchored on noise.

**Why the comparison is untouched.** Every scenario design *and* E_test share
the same generator, so the limitation does not affect the design comparison —
the paper's contribution; it bounds only the absolute robustness claims,
which are scoped to "under historical-level (pluvial-era) interannual
persistence" with the measured numbers stated
(`scenario_design_methods.md` §5.1/§6). Multi-year drought stress still
enters through the hazard axes (SSI-6 run-theory descriptors span multi-year
events) and the 50-yr E_test realizations.

**Evidence artifacts** (all regenerable):
`scripts/supplemental/diagnose_persistence_axis.py`,
`scripts/supplemental/verify_kirsch_baseline_persistence.py`,
`scripts/supplemental/check_cmip6_rho1_anchor_noise.py` →
`outputs/supplemental/persistence_axis/`. The tilted-bootstrap mechanism
remains prototyped and validated in `scengen.persistence` as future-work
material (a persistence-stressed test ensemble is the natural follow-on
study).
