# Forcing-space parameterization (method)

**Status:** method (adopted). The climate forcing space is parameterized by an **interpretable
low-order harmonic model** of the monthly change factor, sampled as a **deeply-uncertain (DMDU)
hypercube** whose bounds are taken from the CMIP6 ensemble. **Date:** 2026-06-24. **Evidence /
figures:** `outputs/diagnostics/forcing_parameterization/` (gitignored, regenerable):
`SI_harmonic_fit.png`, `SI_harmonic_param_space.png`, `SI_harmonic_lhs_sampling.png`,
`SI_harmonic_best_worst_fits.png`, `SI_harmonic_monthly_flow_comparison.png`, `cmip6_harmonic_params.csv`.

## Forcing space

The deeply-uncertain forcing is a **12-month multiplicative change factor** `a_j` applied to NYC-inflow
monthly mean flow, with an optional independent **CV (variance) factor** `v_j`. Both are injected into
the fitted Kirsch generator via the Kirsch et al. (2013) eqs. 10–11 lognormal moment-matching transform
(`scengen.forcing_space.apply_climate_adjustment`), where `a` is a *real-space mean* factor and
`c = a·v` a *real-space SD* factor (CV-preserving baseline `v=1`):

```
arg     = (c/a)²·(e^{σ²} − 1) + 1
Ȳ_new   = ln a + Ȳ + σ²/2 − ½·ln(arg)
σ_new   = sqrt( ln(arg) )
```

The generator is fit once; per forcing profile the adjusted `(Ȳ_new, σ_new)` are set and a realization
is generated (`generate_input_spanning_master`).

## Harmonic parameterization

Each monthly change factor is the exponential of a low-order harmonic (Fourier) series in the
**log** change factor, with `ω = 2π/12` and water-year month index `t = 0..11`. The harmonic
**phases are fixed at the canonical CMIP6 shape** `(τ₁*, τ₂*)` — the phases of the harmonic fit to the
CMIP6 ensemble-mean change profile (`canonical_phases`) — and only the **amplitudes** are sampled
(the Quinn et al. 2018 structure: hold baseline phases fixed, perturb amplitudes):

```
ln a(t) = m  +  r₁·cos(ω(t − τ₁*))  +  r₂·cos(2ω(t − τ₂*))      # τ₁*, τ₂* FIXED (canonical)
ln v(t) = m_v +  r_v·cos(ω(t − τ₁*))                            # independent CV (variance) axis
```

Sampled (magnitude) parameters — CMIP6-fitted ranges, [p5, p50, p95] across the 54 future runs:

| param | meaning | range | role |
|---|---|---|---|
| `m` | log annual-mean change; `eᵐ` = volume multiplier | `eᵐ` [0.96, 1.08, 1.19] | total water-year volume |
| `r₁` | annual-harmonic amplitude (log) | [0.11, 0.20, 0.30] | **seasonal amplitude = winter-wettening / summer-drying** |
| `r₂` | semiannual-harmonic amplitude (log) | [0.04, 0.14, 0.26] | **snowmelt-shoulder / bimodal shape** |
| `m_v, r_v` | CV mean / seasonal amplitude | from std envelope | drought/flood-tail variability |

Phases are **not** sampled: `τ₁` is nearly constant across CMIP6 (≈ January, 0.58-month std) and `τ₂`
is ill-determined (4.8-month std — its phase is meaningless when its amplitude is small), so sampling
the phases independently *scrambles* the seasonal shape (only ~39% of independently-sampled profiles
peak in winter, vs ~96% of CMIP6). Fixing them at the canonical shape makes **every** profile peak in
the correct month with the correct rise/shoulder asymmetry, scaled (100% peak in winter; ~½ the
median-profile shape error — see "Improving shape fidelity" below). Net: **3 magnitude axes** (`m, r₁,
r₂`) + ≤2 variance axes — *fewer* parameters than the all-free form, with strictly better shape fidelity.

**Why this parameterization.** It is interpretable (every axis is a named hydrologic quantity),
low-dimensional, and a faithful description of the CMIP6 change: a 2-harmonic fit explains a **median
per-profile shape-R² of 0.85** (p5–p95 = 0.63–0.96; `SI_harmonic_fit.png`,
`SI_harmonic_best_worst_fits.png`), and the change is ~90% seasonal *shape* / ~10% annual-mean level —
so a shape-capturing parameterization is essential. The forward map is closed-form (parameters → `a(t)`
→ the climate transform above), trivial to communicate, and easy to reason about for extrapolation.

The fitted CMIP6 parameters reveal a legible climate gradient (`SI_harmonic_param_space.png`):
warming (end-century, higher SSP) raises the seasonal amplitude `r₁` markedly (median 0.15 → 0.27) with
only modest volume change — i.e. *warming amplifies seasonality more than it shifts total volume* — and
the canonical seasonal-change phase peaks in **January**.

### Improving shape fidelity (why phases are fixed)

Sampling the amplitudes `{m, r₁, r₂}` over the CMIP6 box with phases fixed at the canonical shape
(default `fix_phase=True`) was chosen after a direct comparison against the all-free 5-parameter form
(`SI_harmonic_monthly_flow_comparison.png`):

| sampler | params | over-extension | 5–95 band match | median shape RMSE | % peaking in winter |
|---|---|---|---|---|---|
| independent phases (5-param) | 5 | 0.28 | 0.295 | 0.194 | 0.39 |
| **fixed-phase amplitudes (3-param)** | **3** | **0.13** | **0.170** | **0.079** | **1.00** |

Fixing the phases improves every metric *and* uses fewer parameters. It also curbs the over-dispersion
that independent phase sampling causes, so the box needs no inward trim (`margin = 0` = the CMIP6
amplitude min/max). A bounded timing perturbation, if ever wanted, is a *single shared* phase-shift `dτ`
applied rigidly to both harmonics (a 4th axis, still ≤ the all-free form) — kept as an optional
supplementary sensitivity, consistent with the weak DRB snowmelt-timing signal.

## CMIP6-based DMDU hypercube sampling

Per DMDU practice, forcing profiles are drawn by **Latin hypercube over an independent box** of the
**amplitude** parameters, whose per-parameter bounds are the **empirical 90% range** (per-axis 5th–95th
percentile) of the CMIP6 fits (`sample_harmonic_forcing` → `fit_harmonic_params` → `harmonic_param_box`
→ `reconstruct_harmonic`; phases fixed via `canonical_phases`). The box is sampled **independently per
axis** — the joint climate-model correlation is *deliberately not preserved* — so the design admits
deeply-uncertain combinations the GCMs did not jointly produce, while staying anchored to the CMIP6
plausible range. This matches the exploratory intent (and the sampling design of Quinn et al. 2018).

**Box width.** The default `bound_pct = (5, 95)` (the empirical 90% range) trims the most extreme tails
so a single outlier GCM run does not drive the box; with the phases fixed the reconstructed monthly
envelope does not over-disperse, so this already keeps the LHS monthly envelope tightly bounded to the
CMIP6 monthly range (`SI_harmonic_monthly_flow_comparison.png`). An optional `margin` widens (`>0`,
extrapolation) or tightens (`<0`) the box.

**Phase / spring-runoff timing.** Phases are fixed at the canonical CMIP6 shape (above). `τ₁` varies
only ~0.6 month across CMIP6, and the DRB (~39–42°N) lies south of the ~44°N snowmelt-timing band
(Hodgkins & Dudley 2006), so timing is held at the canonical value rather than treated as a
load-bearing free axis. A deliberate earlier-melt ±1-month perturbation (Stewart et al. 2005; Gnann et
al. 2020; 1 month = 30°) — a *single shared* phase-shift on both harmonics — belongs in a separately
labeled supplementary sensitivity, not the primary design.

**Variance axis.** `v_j` is derived from the CMIP6 monthly-std change (`derive_variance_envelope`,
sibling-matched CV change) and sampled with the same harmonic-hypercube machinery; `c = a·v` decouples
the real-space mean and SD effects through the Kirsch transform.

## Validation and limitations

- **Hazard-coverage check.** Because hazards (drought_deficit_volume, drought_onset_rate,
  flood_peak_magnitude, flood_pulse_duration) are nonlinear tail functionals, the seasonal-mean harmonic
  slightly smooths the late-summer high-flow tail; coverage of the flood/drought corners is checked in
  **hazard space** (hull volume, tail percentiles) rather than on forcing-space fit alone. If a corner
  is under-covered, add the 3rd harmonic (shape-R² → 0.93) or a bounded per-month residual.
- **Scope.** The forcing perturbs only the monthly mean/CV of the Kirsch marginals; it does not perturb
  interannual **persistence** (which dominates multi-year drought) or **daily-extreme** structure (Nowak
  tail → flood metrics). These are the higher-leverage forcing enrichments and are deferred (see the
  input-vs-hazard diagnostic note).

## Implementation (`scengen.forcing_space`)

`fit_harmonic_params(envelope, order)` → per-anchor `{m, amp, phase}`; `harmonic_param_box(fit, margin)`
→ DMDU bounds; `reconstruct_harmonic(params, order)` → `a(t)`; `sample_harmonic_forcing(n, envelope,
seed, margin)` → LHS over the box → profiles. Used for both the mean (`a`) and variance (`v`) envelopes
by `generate_input_spanning_master`.

## Relationship to Quinn et al. (2018) — note

This parameterization is a deliberate adaptation of the monsoon-dynamics method of Quinn et al. (2018,
WRR 54:4638–4662; same Kirsch–Nowak generator, same two-harmonic log-space seasonal model, same
independent DMDU hypercube). The substantive differences:

- **Harmonics of the *change factor* vs the *baseline state*.** Quinn fits harmonics to the baseline
  log-mean-flow `ȳ₁(i)` and perturbs *its* coefficients (`mC₁·C₁`, phase `φ₁−dφ₁`), so the change is
  tethered to the baseline monsoon shape. We fit harmonics to the **change factor** `ln a` (anchored to
  CMIP6), so amplitude `r₁` and peak-month `τ₁` are free of the baseline shape.
- **Real-space vs log-space multipliers.** Quinn multiplies the *log* mean (`μ_j → M_{μ,j}·μ_j`), which
  is level-dependent and couples mean and variance; we specify the *real-space* mean factor `a` (and SD
  factor `c`) via the lognormal eqs. 10–11, decoupling them.
- **Bounds.** Quinn's box is expert-assumed; ours is read from the CMIP6 harmonic fits.
- **Variance & injection.** We carry an independent (optionally seasonal) CV axis — Quinn's flagged
  future work — and regenerate per profile rather than rescaling a fixed stationary series.

The DMDU hypercube-sampling philosophy is identical, which makes Quinn et al. (2018) the natural citation
for both the seasonal harmonic parameterization and the independent-box sampling design.

## References
Quinn et al. 2018 (WRR, monsoon dynamics; Kirsch–Nowak); Prudhomme et al. 2010 `7NTGSBJP`
(mean–amplitude–phase harmonic change factors); Nazemi et al. 2013 `VMHQUFPM` (timing+volume regime);
Matonse et al. 2011 `8A8ZXLTN` (NYC snowmelt timing); Stewart, Cayan & Dettinger 2005; Hodgkins & Dudley
2006; Gnann et al. 2020 (center-of-timing / harmonic phase); Kirsch et al. 2013 `GF8MRHNE` (log-space
adjustment); Nowak et al. 2010 (daily disaggregation).
