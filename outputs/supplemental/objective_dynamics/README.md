# Objective-dynamics diagnostics (historical baseline)

Round-one diagnostic figures that visualize the operational **dynamics behind
each of the 7 performance measures** on the historical single trace
(1945-10-01 → 2022-09-30). Purpose: confirm each measure is a faithful, readable
reduction of a real reservoir-operations dynamic before the full campaign.

**Terminology (important).** The main panels report the **§1 whole-trace
performance metrics** — weekly reliability, CVaR90 weekly deficit, whole-record
flood days, daily-storage 5th percentile. These are the interpretable historical
narrative and the per-realization base of the re-evaluation layer; they are
**NOT the optimization objectives**. The optimizer targets the **annual-unit
(§2)** versions of the same seven measures even on the historic trace
(`simulation.evaluate` → `compute_for_borg_ensemble`, treating the trace as
N = 1), shown here as the per-water-year "annual-unit view" strips. So this suite
reserves the word *objective* for the §2 annual-unit metrics; the whole-trace
quantities are called *performance metrics*.

Each figure shows the default **FFMP baseline** and one interpretable
**storage-conservative** contrasting policy, so the metric can be seen to move with
operations. Guiding principle: *the figures compute nothing the reduction helpers
don't* — every threshold, weekly-resample basis, metric-window cut and CVaR tail is
imported from `src.objectives` / `src.objectives_ensemble`, and every annotated
value is asserted equal (within epsilon) to
`build_objective_set(config.ACTIVE_OBJECTIVES).compute(data)`.

## Files

| File | Contents |
|------|----------|
| `make_objective_dynamics_figures.py` | Driver: loads/simulates the two policies, computes scores, renders figures. |
| `objective_dynamics.py` | Plot module: shared panel helpers + the five `plot_*` functions. |
| `figures/figA..figE_*.png` | The five performance-metric anatomy figures (version-controlled). |
| `baseline_ffmp.hdf5`, `contrast_*.hdf5` | Cached full-model simulations (git-ignored, regenerable). |

## Figures

- **A — NYC delivery** (reliability + CVaR90 deficit): weekly delivery vs the
  running-average entitlement (shortfalls shaded), the CVaR90 deficit tail
  (● = score), and the §2 per-water-year failing-weeks strip.
- **B — Montague flow** (reliability + CVaR90 deficit): weekly flow vs the 1131 MGD
  Decree target (near-threshold band; floods clipped), CVaR90 deficit tail, §2 strip.
- **C — Trenton flow** (reliability): weekly flow vs the 1939 MGD Decree target, §2 strip.
- **D — NYC storage** (daily-storage p5): daily combined storage % with the p5 line,
  the storage duration curve (● = p5 score), and the §2 annual-minimum-storage strip.
- **E — Downstream flooding** (minor-flood days): baseline tail-gauge stages vs each
  gauge's NWS minor-flood line, and the §2 per-water-year minor-flood-day count.
- **F — Policy rules** (storage-conservative): the contrast policy's FFMP
  operating rules (storage-zone curves, diversion limits, release schedules, flow
  targets, flood-zone releases) with the baseline dashed underneath, via
  `src.plotting.policy_rules.plot_policy_rules`. Shows the two contrast levers
  directly — raised storage-zone curves (a) and lowered Montague/Trenton flow
  targets (d).
- **G — Performance-metric parallel axes**
  (`figG_performance_metric_parallel_axes.png`): baseline vs storage-conservative
  on native-scaled parallel axes (top = better). These are the whole-trace **§1
  performance metrics** — a diagnostic head-to-head, **not** the optimization
  objectives; the crossings show the trade-off.

Each anatomy figure's bottom-right (`annual-unit view`) panel shows the
per-water-year annual-unit metric that Borg actually optimizes on the historic
trace (the §2 objective, N = 1 realization over its ~76 water years); the §1
whole-trace performance metric is the direct narrative above it. Figures A–E are
the anatomy set; F and G tie the metrics back to the operating rules (F) and to a
head-to-head performance view (G).

## Run

From the repo root with the project venv active:

```
python outputs/supplemental/objective_dynamics/make_objective_dynamics_figures.py
```

The baseline reuses the artifact-of-record `outputs/baseline/ffmp_baseline.hdf5`
when its objective columns match the active set, else simulates a fresh full-model
baseline and caches it here (run `workflow/05_run_baseline.sh` to refresh the
artifact of record). The contrast is simulated once and cached. Re-runs reuse the
caches and only re-plot. Env toggle: `NYCOPT_OBJDYN_NOSTRICT=1` disables the
figure-vs-score self-check assertions.

## Contrast policy

A storage-conservative perturbation of the FFMP baseline: lowers the
Montague/Trenton flow-target scaling (`mrf_target_scale_*` → 0.80) and raises the
storage-zone boundary curves (`zone_vshift_*` → +0.03 of capacity), so the system
holds more NYC storage at the cost of downstream Decree flow. Defined in
`build_contrast_dv()`; feasibility is checked with `compute_constraint_violations`.
On the historical trace it raises storage-p5 (≈33% → 38%) and roughly doubles the
Montague deficit CVaR90 (≈4.7% → 11%), a legible, attributable trade-off.
