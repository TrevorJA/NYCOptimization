# First full workflow run — historic NYC re-optimization + DU re-evaluation

**Run:** `ffmp_obj7_mm_moderate` · scenario `historic`
(`pub_nhmv10_BC_withObsScaled`, 1945-10-01 → 2022-09-30) · Anvil.
First end-to-end optimization → diagnostics → DU re-evaluation, at a modest scale,
with a stakeholder-acceptability screen and robustness postprocessing.

## What ran

| Step | Where | Wall | Result |
|------|-------|------|--------|
| 00 build JAR (new DVs) | login | ~min | `drb_ffmp.jar` (24 DVs, 7 obj) |
| 12 build E_test | shared | 8 min | `etest_kn_10yr_n200` (50 SOWs × 4 reps × 10 yr, wide DU box) |
| 04 stage E_test inputs | shared | 10 min | pywrdrb HDF5 inputs |
| 05 baseline + reeval | shared | 6 min | FFMP baseline matrix on E_test |
| **06 MM-Borg search** | wholenode 5×33 | **62 min** | 20,000 NFE, 1 seed, **31.1 s/eval** |
| 07 diagnostics | shared | 2 min | HV/GD + merged reference set |
| 08 DU re-eval + robustness | shared 32 | 90 min | 500-policy subset scored on E_test |

Per-eval cost calibrated on Anvil at **31 s** (vs the stale ~150 s comment from a
slower machine) — the number to size future runs with.

## Optimization results

- Borg archive **4,549** solutions → epsilon-merged Pareto reference set **3,171**,
  thinned to an evenly-spaced **500-policy** subset for the DU re-eval (to keep the
  first run modest; the full front is preserved).
- Hypervolume rose monotonically across all 4 islands and was **still climbing** at
  the 5,000-NFE/island budget — expected for a modest first run.

## Stakeholder screen (new)

The search trades NYC delivery against downstream flow **hard**: the median NYC
weekly delivery reliability across the 500-policy front is only **0.10**. Policies
below **0.5** reliability are unacceptable to DRB stakeholders regardless of how
they score elsewhere, so a reusable screen (`src/pareto_filter.py`,
`nyc_delivery_reliability_weekly >= 0.5`) is applied before every figure:

> **214 of 500 policies (43%) are acceptable.** The screened-out 57% (grey in fig 01)
> collapse to ~0 NYC reliability to win the other objectives.

This is a postprocessing screen for now; it is the natural definition for a formal
reliability **constraint** in the next search.

## DU robustness (the headline)

The acceptable policies are re-simulated on the held-out **E_test** deep-uncertainty
ensemble (50 SOWs × 4 reps over a *wider* DU box than any design searched in) and
scored with the **multivariate satisficing (Starr 1962) domain criterion on the SOW
unit** — the fraction of the 50 deeply-uncertain states in which a policy meets **all
seven** thresholds jointly (Herman et al. 2015; Trindade et al. 2017; Gold et al.
2023) — with the status-quo FFMP baseline scored on the same ensemble.

- **Status-quo FFMP is not robust: it satisfices all 7 thresholds in 0 of 50 SOWs
  (sat_sow = 0.00).**
- The **most-robust acceptable policy (id 422) reaches 0.42** (21/50 SOWs); the
  median acceptable policy is still ~0.00.
- The **binding objective is NYC delivery reliability** (fig 04B): its single-criterion
  satisficing is the lowest of the seven, and the joint criterion can be no larger than
  its smallest component. NYC deficit is the next tightest. Trenton reliability, minor
  flood days, and storage-p5 are almost always satisficed, so they are not what limits
  robustness.

**Reading:** policies optimized on the single historic trace generalize poorly to a
wide DU box on the NYC-delivery objectives specifically — an overfitting-to-historic
signal that motivates the DU-aware search designs, and one the status quo shares.

## Scenario discovery (new)

For the most-robust policy (id 422), each of the 50 SOWs is plotted pass/fail across
the sampled CMIP6 forcing factors θ = (m, r₁, r₂) — no boosted-tree/PRIM box, just the
raw success/failure scatter (fig 05). **Failures concentrate in drier states of the
world** (low water-year volume multiplier eᵐ < ~1.05); passes cluster at wetter
futures. Vulnerability is driven by the drought/volume factor, not the seasonal-shape
amplitudes.

## Operating rules (new)

Three representative acceptable policies vs the FFMP baseline (fig 06), from their 24
decision variables: **most-robust (id 422)**, **NYC-diversion priority (id 315)**, and
**Montague-flow priority (id 45)**. The core contrast is the NYC drought-level delivery
schedule (panel 1): the Montague-flow policy cuts NYC delivery to ~0.30 at L5 to buy
downstream flow, while the NYC-diversion policy barely curtails (~0.90 at L5). The
panels also compare MRF flow targets, NYC diversion cap, storage-zone shifts,
flood-release caps, and seasonal MRF scaling.

## Figures (`figures/historic/ffmp_obj7_mm_moderate/`)

1. `01_pareto_parallel_coords.png` — Pareto front, acceptable (blue) vs screened-out
   (grey) vs FFMP baseline, all 7 objectives.
2. `02_hypervolume_convergence.png` — HV vs NFE, 4 islands.
3. `03_du_performance_distributions.png` — each objective's DU-expected value across
   the acceptable policies vs baseline **and its satisficing threshold** (raw magnitudes
   co-reported, Huang et al. 2025).
4. `04_du_robustness.png` — **primary metric**: multivariate satisficing (SOW unit)
   distribution with the FFMP baseline overlaid, plus the per-objective decomposition
   flagging the binding constraint.
5. `05_scenario_discovery.png` — most-robust policy pass/fail across the DU forcing
   factors (m, r₁, r₂).
6. `06_operating_rules.png` — operating rules (24 DVs) of the 3 representative policies
   vs baseline.

## Reproduce

```bash
python3 -m scripts.main.plot_run_results \
    --slug ffmp_obj7_mm_moderate --scenario historic \
    --preset etest_kn_10yr_n200 --formulation ffmp
```

Reusable postprocessing added this pass: `src/pareto_filter.py` (stakeholder screen),
`src/plotting/robustness_summary.py`, `src/plotting/scenario_discovery.py`,
`src/plotting/operating_rules.py`. The `pareto_evolution` figure and its module were
removed.

## Scale up next

Raise NFE (`mm_full` = 50k; ~3.5–4 h/seed on Anvil at 31 s/eval), add seeds
(`--array=1-10`), re-evaluate the **full** 3,171-policy front (drop the 500 cap), and
promote the reliability screen to a formal **search constraint**.
