# Manuscript Acronym Conventions

Working aid for the NYC re-optimization manuscript. Rule: every acronym is
defined exactly once, at its first use in the main text, and used consistently
thereafter. Never redefine. When the Introduction is drafted, first-use
locations will shift upstream; update this table at that pass.

| Acronym | Expansion | Current first use (definition site) |
|---|---|---|
| DRB | Delaware River Basin | Section 2.1 |
| NYC | New York City | Introduction P6 (drafted) |
| NJ | New Jersey | Section 2.1 |
| FFMP | Flexible Flow Management Program | Introduction P6 (drafted) |
| MGD | million gallons per day | Section 2.1 |
| cfs | cubic feet per second | Section 2.1 |
| HIST | Historical design | Section 3.1.1 |
| MC | Monte Carlo Sampling design | Section 3.1.1 |
| HF | Hazard Filling design | Section 3.1.1 |
| i.i.d. | independent and identically distributed | Introduction P6 (drafted) |
| SSI-6 | six-month Standardized Streamflow Index | Section 3.1.3 |
| CVaR | conditional value-at-risk | Section 3.2.2 |
| NWS | National Weather Service | Section 3.2.2 |
| SOW | state of the world | Section 3.2.2 (will move to Introduction P4 when drafted) |
| MM Borg | multi-master Borg multiobjective evolutionary algorithm | Section 3 overview |
| NFE | number of function evaluations | Section 3.3 |
| CMIP6 | Coupled Model Intercomparison Project Phase 6 | Section 3.4.1 (used without expansion; standard) |

## Terminology conventions (non-acronym)

- **hazard metrics** — the six quantities computed on each realization; never
  "descriptors", "candidate metrics", or "indicators". "Hazard coordinates"
  refers only to a realization's position vector in the hazard space.
- **satisficing** — the robustness family term ("satisficing criteria",
  "satisficing fraction", "satisficing robustness"). "Domain criterion"
  appears once, as attribution to Starr (1962), and never again.
- **SOW** is the unit of all robustness and regret fractions; "per-SOW", not
  "per-state" or "per-scenario".
- **incumbent** — the default 2017 FFMP policy used as the regret reference.
- **years** — simulated record lengths are stated in plain years with the
  partition into SOWs and realizations made explicit; never "scenario-years"
  or "ensemble-years".
- **nodes / cores** — parallel computing configuration language; never "MPI
  ranks" or other code-level terms ("campaign", "smoke", "build diagnostic",
  "substrate", "hard error") in manuscript text.
- Tables carry a *Note* only to explain a symbol flagged in the table itself;
  substantive content belongs in a paragraph of the main text.
