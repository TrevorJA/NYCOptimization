# Objective-dynamics diagnostics (ensemble evaluation)

The ensemble analog of `../objective_dynamics/` (historical single trace). Same
two policies — the default **FFMP baseline** and one interpretable
**storage-conservative** contrast — but evaluated under a small **stationary
Kirsch-Nowak ensemble** (5 realizations × 50 years) and scored the way the
optimizer scores them during an ensemble search: the **two-layer annual-unit
(§2) objectives** (`src/objectives_ensemble.py`,
`docs/notes/methods/objective_definitions.md` §2).

Purpose: see how each objective reads when the score is a statistic **pooled
across realizations** rather than a single trace — the reduction Borg actually
optimizes under an ensemble.

## The two-layer reduction each figure shows

- **Stage (i)** — one annual metric per (realization × water-year) unit. A
  50-year realization yields 49 metric-bearing water years, so the ensemble
  pools 5 × 49 = **245 ensemble water years**.
- **Stage (ii)** — one operator over the pooled ensemble water years: *failure
  frequency* (reliability), *pooled P99* (deficit tails), *pooled P01*
  (annual-minimum storage), or *pooled mean* (annual flood days).

Each figure has two rows mirroring the two layers:

- **Row 1 — seasonal distribution.** The realizations are independent random
  synthetic sequences, so their calendar dates are not comparable — a time series
  would imply a spurious alignment. Instead the underlying dynamic is shown as a
  **distribution across the standard water year**: every ensemble water year
  pooled by day-of-water-year, drawn as a median line with an inter-quartile band
  per policy, against the static Decree threshold. (Flooding is shown as its
  per-month contribution to the annual objective, whose bars sum to the score.)
- **Row 2 — pooled reduction.** The empirical CDF of the pooled annual metric
  per policy, with the stage-(ii) operator's cut marked: a **dot** at the
  percentile/mean (score = its x-value), or — for a reliability objective — a
  dot at the failure threshold *k* whose **height** is the score. Each policy's
  true score is annotated in the legend.

Guiding principle (kept from the single-trace suite): *the figures compute
nothing the objective functions don't.* Row-2 metrics come from the registered
stage-(i) functions and row-2 scores are the registered stage-(ii) operators
applied to the pooled units, each asserted equal to
`build_ensemble_objective_set(config.ACTIVE_OBJECTIVES).compute(data_per_real)`.

## Files

| File | Contents |
|------|----------|
| `make_ensemble_objective_dynamics_figures.py` | Driver: stages the ensemble, simulates the two policies across 5 realizations, computes §2 scores, renders figures. |
| `ensemble_objective_dynamics.py` | Plot module: the seasonal-band panels + the pooled-reduction panel + the five `plot_*` functions. |
| `figures/figA..figE_*.png` | The five objective figures (version-controlled). |
| `figures/figF_*`, `figures/figG_*` | Policy rules and §2 objective parallel axes. |
| `cache/*.pkl` | Cached per-realization simulations (git-ignored, regenerable). |

## Figures

- **A — NYC delivery**: seasonal delivery distribution vs the demand cap; pooled
  reliability (no-failing-week fraction) and pooled deficit P99 (within-year CVaR90).
- **B — Montague flow**: seasonal flow distribution vs the 1131 MGD Decree
  target; pooled reliability and pooled deficit P99.
- **C — Trenton flow**: seasonal flow distribution vs the 1939 MGD Decree
  target; pooled reliability.
- **D — NYC storage**: seasonal storage distribution; pooled annual-minimum
  storage P01 (vulnerability).
- **E — Downstream flooding**: per-month contribution to expected annual flood
  days (bars sum to the objective); pooled **mean** annual flood days.
- **F — Policy rules** (storage-conservative): the contrast policy's FFMP
  operating rules with the baseline dashed underneath (DV-agnostic; identical to
  the single-trace suite).
- **G — Objective parallel axes**: baseline vs storage-conservative on the §2
  ensemble scores (top = preferred); the crossings show the trade-off.

## The ensemble

A fixed **5-realization × 50-year stationary Kirsch-Nowak** ensemble, staged
once by `src.local_test_ensemble.ensure_local_test_ensemble()` under
`config.STAGED_ENSEMBLE_DIR/kn_50yr_n5/` and **reusable by any local experiment
or test** (resolve it anywhere with `get_ensemble_spec("kn_50yr_n5")`). The
driver generates + stages it on first run (a couple of minutes) and reuses it
thereafter. Evaluation uses the **trimmed** search model (the ensemble path Borg
runs), so these figures reflect how the ensemble is scored during optimization.

> Local-filesystem note: `outputs/synthetic_ensembles` is a git-tracked symlink
> to Anvil scratch that does not resolve on a local checkout. It must be a real
> local directory for staging to work here; do not commit that change (it would
> break the symlink on the cluster).

## Run

From the repo root with the project venv active:

```
python outputs/supplemental/ensemble_objective_dynamics/make_ensemble_objective_dynamics_figures.py
```

Env toggles (no CLI value flags, per project convention):
- `NYCOPT_ENSOBJDYN_NOSTRICT=1` — disable the figure-vs-score self-check assertions.
- `NYCOPT_ENSOBJDYN_REFRESH=1` — ignore the simulation cache and re-simulate.

## Contrast policy

Identical to the single-trace suite so the two are directly comparable: lowers
the Montague/Trenton flow-target scaling (`mrf_target_scale_*` → 0.80) and raises
the storage-zone boundary curves (`zone_vshift_*` → +0.03 of capacity), holding
more NYC storage at the cost of downstream Decree flow. Defined in
`build_contrast_dv()`; feasibility is checked with `compute_constraint_violations`.
