# Random-DV Objective Sensitivity Experiment

*Last updated: 2026-06-17. Companion to
`docs/notes/methods/objective_definitions.md`. Selects among objective
formulations by running many random decision-variable (DV) vectors through the
model on a **single historical reference trace** and measuring how the objectives
respond. Ensemble- and scenario-noise questions are deliberately out of scope and
deferred to a separate supplemental experiment (see §4).*

## Purpose

Before committing compute to the full MOEA, confirm which objective formulations
are worth carrying: (a) does each objective discriminate between policies, and
(b) which objectives are redundant? This is purely an **objective-definition**
question, so it is run on one fixed hydrologic input — changes in objective values
come only from changes in the decision variables, not from streamflow variability.

## Setup

- **Single historical reference trace.** Run the (trimmed) model on the historical
  reference inflow `pub_nhmv10_BC_withObsScaled` — the historic scenario design,
  one continuous trace, no realization axis. This isolates objective behavior from
  ensemble/scenario noise.
- **Configurable objective set.** The diagnostic must be able to evaluate **all
  objective formulations in `src/objectives.py` (`OBJECTIVES` registry)** — the
  extreme case, and the default for the redundancy screen, because it compares the
  recommended metrics against the diagnostics they replace (CVaR vs max, p5 vs
  min, minor vs action/major flood stage, Trenton vs salt-front) — **and/or a named
  subset** (e.g. the current recommended set). The objective list is a config
  setting, not a CLI flag.
- Salinity/temperature objectives (`salt_front_intrusion_max_rm`,
  `lordville_temp_exceedance_days`) return NaN unless their LSTMs are enabled;
  evaluating the full registry therefore requires the salinity model on (and the
  reference period extended to cover it). NaN objectives are reported, not silently
  dropped.
- Build on the existing harness `scripts/supplemental/random_sample_mpi.py` (LHS/
  uniform DV sampling within `get_bounds`, FFMP baseline row, in-memory simulation
  via `run_simulation_inmemory`, MPI parallel across DV samples). No realization
  loop is needed — one simulation per DV vector.

## Procedure

1. **Sampling.** Latin hypercube over the DV bounds
   (`scipy.stats.qmc.LatinHypercube` scaled to `get_bounds`), N ≈ 200–500 vectors
   for stable correlation estimates; retain the FFMP baseline as a reference row.

2. **Discrimination.** Per objective: min, p5, p25, median, p75, p95, max, IQR,
   and the fraction of samples that are NaN or pinned at a DV bound. An objective
   with no spread across random DVs carries no Pareto gradient and must be dropped
   or reformulated. This also compares each replaced worst-case metric against its
   stable replacement (`*_deficit_max_pct` vs `*_deficit_cvar90_pct`;
   `nyc_storage_min_pct` vs `nyc_storage_p5_pct`) to confirm the stable form
   discriminates at least as well across policies.

3. **Redundancy screen (Olden & Poff 2003 style).** Spearman rank-correlation
   matrix over all evaluated objectives (recommended + the diagnostics they
   replace). Flag `|ρ| > 0.8`. Priority tests:
   - each reliability ↔ CVaR-deficit pair (NYC, Montague) — whether both legs
     survive;
   - **`trenton_flow_reliability_weekly` ↔ `montague_flow_reliability_weekly`** —
     both are flow-Decree frequencies and Montague is upstream of Trenton, so they
     may be collinear; decides whether both flow axes are retained;
   - `trenton_flow_reliability_weekly` ↔ `nj_delivery_reliability_weekly` —
     decides the optional NJ-delivery objective.
   From each `|ρ| > 0.8` cluster keep the more stable, more stakeholder-
   interpretable member (favoring the satisficing/frequency form, Bonham et al.
   2024).

## How results feed objective selection

- Drop no-gradient objectives (step 2).
- Prune each `|ρ| > 0.8` cluster to one member (step 3); this decides whether NJ
  delivery is added and whether both Montague/Trenton flow axes survive.
- Set each objective's satisficing level `θ_i` from its discrimination
  distribution, and each epsilon to ≈ (random-DV IQR / 10) so the archive
  resolution matches the signal scale (Reed et al. 2013).

## §4. Deferred to a separate supplemental experiment (ensemble / scenario noise)

The following concern **scenario-design construction**, not objective definition,
and require ensembles of streamflow realizations rather than a single historical
trace — so they are held for a separate supplemental experiment:

- **Across-realization operator stability** — recomputing each objective under
  satisficing / mean / p90 / CVaR₉₀ over an ensemble and comparing the
  operator-induced DV rankings (Kendall τ_b; Herman et al. 2015; McPhail et al.
  2020).
- **Resampled-draw noise** — repeatedly redrawing a stochastic/historic ensemble
  and measuring the Monte-Carlo coefficient of variation of each objective, to size
  the realization count K for the resampled scenario design (Brodeur et al. 2020).
