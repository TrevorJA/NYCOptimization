# Scenario-Design Generation Methods

*Last updated: 2026-06-18. Working draft. Terminology per `docs/notes/terminology.md`. Companion to `docs/notes/methods/experimental_design.md` (rev. 5). Citations author-year, resolved via `docs/notes/literature/`; §8 lists references to add to the Zotero collection "Paper 3 NYC Reoptimization" (ISYGLK35). This note specifies **only** how the streamflow scenario ensembles are constructed; objective formulations, MOEA internals, and robustness-metric mathematics are out of scope and named only where a construction choice depends on them. The methodology here was hardened through three rounds of independent adversarial review (methodology/literature, experimental design/statistics, implementation/reproducibility).*

---

## 1. Purpose and scope

This note gives the construction recipe for every streamflow scenario set in the scenario-design comparison (Pywr-DRB simulation + MM-Borg search): six **evaluation ensembles** (one per scenario design), a **support-point** supplementary design, and one held-out **test ensemble** for re-evaluation. The methodological contribution is the **hazard-filling** design — an in-the-loop evaluation ensemble built by space-filling subsampling over the **empirical hazard manifold** of a large master ensemble — contrasted against prevailing input-space and probability-space designs.

A standing caveat governs the document: several designs deliberately depart from the scenario probabilities of the master ensemble (§4.6). This is the design choice under study, not an artifact; the held-out re-evaluation (§4.7) is the common, probability-faithful basis of cross-design comparison.

**Out of scope** (decided elsewhere; named only where downstream-coupled): objective formulations and across-scenario aggregation (`objective_definitions.md`) — these gate whether the hazard-filling probability distortion is benign and set ensemble *size*; MOEA algorithm internals (the `MOEAConfig` axis); robustness-metric mathematics beyond naming.

**Novelty relative to Bonham et al. (2024).** Bonham uses space-filling subsampling and the same space-filling diagnostics (L2-star discrepancy, minimum-spanning-tree edge statistics) to *size post-hoc robustness-ranking ensembles*. We adopt those diagnostics directly (§6); they are not novel here. The precise delta is twofold: (i) we subsample to build the **in-the-loop optimization evaluation ensemble** used to compute fitness during search, not a post-hoc ranking set; and (ii) we subsample **in hazard space**, with axes that are event/hazard metrics (drought intensity / duration / severity, low- and high-flow statistics), not input forcing factors or raw outcomes. The novelty is the *in-loop, hazard-space construction*, which separates training-composition (selection) effects from evaluation (test-ensemble) effects.

---

## 2. Notation and the 2×2 frame

A single notation set is used by every design.

| Symbol | Meaning |
|---|---|
| $Q_{\text{obs}}$ | Observed/reconstructed daily multi-site historical record (`pub_nhmv10_BC_withObsScaled`, ~1945–2022). |
| $\theta$ | Generator forcing / deeply-uncertain factor vector: a 12-month multiplicative change-factor profile (§3.1) plus generator hyper-parameters. $\theta \in \Theta$. |
| $\mathcal{M}$ | **Master ensemble**: a large set of short synthetic realizations, $\lvert\mathcal{M}\rvert = N_{\mathcal{M}}$ (target $\sim 10^6$). Never simulated in full — only its hazard image is stored (§3.2). |
| $r$ | A **realization**: one generator output sequence, seed-deterministic given $\theta$ and a realization seed. |
| $s = \text{window}(r)$ | A **scenario**: one $L$-year window of a realization presented to Pywr-DRB in one evaluation. Disjoint windows ⇒ $s \equiv r$ at the chosen $L$. |
| $h(s) \in \mathbb{R}^{m}$ | **Hazard-metric vector** of a scenario: $m$ low-redundancy drought/low-flow/high-flow indices (§3.3). |
| $\mathcal{H}$ | The $N_{\mathcal{M}} \times m$ **empirical hazard manifold** — the image of $\mathcal{M}$ in hazard space. The object the hazard-filling design fills. |
| $E_d$ | **Evaluation ensemble** (= search ensemble) for design $d$: size $N_d$, length $L_d$. |
| $K_d$ | Number of independent **ensemble draws** (replicate constructions) for design $d$. |
| $S$ | MOEA random **seeds** per (design, draw). |
| $E_{\text{test}}$ | The held-out **test ensemble** (= re-evaluation ensemble), §4.7. |
| $B$ | Per-design budget in **scenario-years** (§5). |
| $\text{NFE}$ | Function evaluations. |
| $a_j = 1 + \Delta_j$ | Per-month multiplicative change factor ($\Delta_j$ a fraction; all `_pct`/`_frac` quantities are 0–1 fractions). |
| $\bar Y_j,\ \sigma_j$ | Log-space per-month mean and SD on the fitted `KirschGenerator` (`mean_period`, `std_period`). |

**Spaces** (per `terminology.md`): **input space** ($\Theta$); **hazard space** = the *empirical hazard manifold of $\mathcal{M}$* (cLHS-style strata are defined on the realized marginals of $\mathcal{M}$, not on an a-priori hazard domain); **outcome space** (simulated objectives). We never use exposure/parametric/performance, or arm/treatment/ablation. Sequence length is always stated in years with explicit window construction.

**The 2×2 frame** crosses *sampling philosophy* × *selection mechanism*:

|  | representative-in-probability | uniform-in-hazard-space |
|---|---|---|
| **random / undesigned** | `fixed_probabilistic_short`, `resampled_probabilistic`, `historic` | — |
| **designed** | `support_points` | `hazard_filling` |

The **faithful arm** = {`historic`, `fixed_probabilistic_short`, `fixed_probabilistic_long`, `resampled_probabilistic`, `support_points`} (all targeting the master-ensemble/empirical measure); the **distorted arm** = {`hazard_filling`} (deliberately reweighting toward uniform hazard coverage). `support_points` sits explicitly in the **faithful × designed** cell.

### 2.1 Design → construction map

| Design (`ScenarioDesign.name`) | `selection` | `source_kind` | $N_d$ | $L_d$ (yr) | $K_d$ | `resample_per_eval` | Mimics |
|---|---|---|---|---|---|---|---|
| `historic` | none | historic | 1 | full (~77) | 1 | False | Giuliani et al. (2016) |
| `fixed_probabilistic_short` | random | synhydro_kn | $N$ | $L$ | $K$ | False | Kleywegt et al. (2002); Herman et al. (2014) |
| `fixed_probabilistic_long` | random | synhydro_kn | $N'=\lfloor NL/L'\rfloor$ | $L'\gg L$ | $K$ | False | Quinn et al. (2017) |
| `resampled_probabilistic` | random (per generation) | synhydro_kn | $N$ | $L$ | 1 | **True** | Trindade et al. (2017) |
| `input_stratified` | lhs_input | synhydro_kn | $N$ | $L$ | $K$ | False | Quinn et al. (2017, *Rival Framings*); Bartholomew & Kwakkel (2020) |
| `hazard_filling` | hazard_fill | synhydro_kn | $N$ | $L$ | $K$ | False | Minasny & McBratney (2006); Morris & Mitchell (1995); Johnson et al. (1990) |
| `support_points` *(suppl.)* | support | synhydro_kn | $N$ | $L$ | $K$ | False | Mak & Joseph (2018) |
| `E_test` (re-eval) | — | synhydro_kn | $N_{\text{test}}\gg N$ | $L_{\text{test}}$ | 1 | — | design-conditional reference (Quinn et al. 2020) |

All non-historical designs draw from the **same** master ensemble $\mathcal{M}$, so differences are attributable to *selection*, not to a different generator. The historical record cannot be size-matched and is a reference for prevailing practice, not a controlled comparison.

---

## 3. Shared upstream pipeline

Stages 3.1–3.2 are `workflow/02`; stage 3.3 is the front half of `workflow/03`.

### 3.1 Forcing space: CMIP6-based interpretable harmonic-parameter hypercube

The deeply-uncertain forcing space is the set of plausible monthly climate perturbations applied to the Kirsch generator's log-space statistics. The full parameterization, CMIP6 fits, sampler, and relationship to Quinn et al. (2018) are in `docs/notes/methods/forcing_parameterization.md`; the essentials:

1. **CMIP6 anchors.** Take the CMIP6 multimodel cloud (GCM × SSP × period hydrologic runs), each a 12-month multiplicative change-factor profile $a_j$ for the NYC inflow gages.
2. **Interpretable fixed-phase harmonic parameterization + DMDU hypercube.** Decompose each anchor's *log* change-factor profile into a low-order harmonic (Fourier) series and, following Quinn et al. (2018), **hold the harmonic phases fixed at the canonical CMIP6 shape** (the phases of the fit to the ensemble-mean profile) while sampling only the **amplitudes** — annual-mean level $m$, annual amplitude $r_1$, semiannual amplitude $r_2$ (all magnitudes; volume / seasonal amplitude / shoulder shape). The forcing space is the deeply-uncertain hypercube of those amplitudes across the CMIP6 anchors — bounded by the **empirical 90% range** (per-axis 5th–95th percentile, robust to outlier runs) — sampled independently by LHS (`scengen.forcing_space.sample_harmonic_forcing`, `fix_phase=True`). Fixing the phases anchors every profile to the characteristic CMIP6 seasonal shape (correct winter peak + asymmetry), which independent phase sampling otherwise scrambles, while still admitting deeply-uncertain magnitude combinations the GCMs did not jointly produce. *(The earlier PC-rotated-LHS variant has been retired in favor of this interpretable parameterization.)*

The CMIP6 profiles are read from the multiplicative twin product
`CMIP6_multimodel_streamflow/stats/diff_relative_to_dataset_baseline/nyc_inflow_monthly_mean_frac_by_dataset_ssp_and_period.csv`
(already $a_j$; the baseline column equals 1.0). The CSV month index is **calendar (1 = Jan)** while Kirsch/Pywr-DRB are **water-year aligned (Oct start)**: re-index calendar → water-year before applying. (The `_prc_change_` sibling holds percentages and would require $a_j = 1 + \Delta_j/100$; the `_frac_` file avoids that conversion and is the `diff_relative_to_dataset_baseline` variant consistent with the baseline-fit generator of §3.2.)

**Climate adjustment is applied externally** to the fitted generator's `mean_period`/`std_period` before generation (as in `StochasticExploratoryExperiment/methods/generate.py`). Per Kirsch et al. (2013, eqs. 10–11), with log-space baseline $(\bar Y_j,\sigma_j)$, multiplicative real-space mean factor $a_j$, and $c_j=1$ (preserve absolute real-space SD; CV not preserved — a flagged choice):

$$
\bar Y_j^{\text{new}} = \ln a_j + \bar Y_j + \tfrac{1}{2}\sigma_j^2 - \tfrac{1}{2}\ln\!\Big[\tfrac{1}{a_j^2}\big(e^{\sigma_j^2}-1\big)+1\Big],
\qquad
\sigma_j^{\text{new}} = \sqrt{\ln\!\Big[\tfrac{1}{a_j^2}\big(e^{\sigma_j^2}-1\big)+1\Big]}.
$$

This extends the current single-profile implementation to a **sampled continuous space** of profiles. The same forcing space drives both the deeply-uncertain/resampled generation and the test ensemble.

### 3.2 Master-ensemble generation (length, windows, initial storage, storage strategy)

$\mathcal{M}$ is generated by the fitted Kirsch–Nowak pipeline: monthly Kirsch generation (log-space, Cholesky correlation restoration), Nowak daily disaggregation, KDE regression for non-major nodes, and `_subtract_upstream_catchment_inflows` to recover marginal catchment inflows. Applying §3.1 requires fitting **two** Kirsch generators — one on the **baseline period** (supplying $\bar Y_j,\sigma_j$ for eqs. 10–11) and one on the full record — mirroring `StochasticExploratoryExperiment/methods/generate.py`.

**Cardinality.** $\mathcal{M}$ must be dense enough for subsampling/stratification (Minasny & McBratney 2006; Mak & Joseph 2018 assume a dense candidate set) and generated under enough distinct $\theta$ to represent the envelope. Construct $\mathcal{M}$ as $N_\Theta$ forcing profiles × $n_{\text{per}\Theta}$ realizations, targeting $N_{\mathcal{M}}\approx 10^6$.

**Window construction.** Disjoint, non-overlapping $L$-year blocks (avoids the effective-sample-size inflation and autocorrelation of overlapping windows). $L$ (default 5–10 yr) **must exceed the longest within-window drought**: check against the 1960s DRB drought of record (~4–5 yr). $L$ is held constant across designs compared on composition; the long-record design (§4.3) varies $L$ and is analyzed as a separate length contrast.

**Initial storage and warm-up.** Each scenario starts from a fixed initial reservoir storage (`INITIAL_VOLUME_FRAC = 0.80`) with a 365-day warm-up excluded from metric computation — a *terminating-simulation* design (Law 2015), not a steady-state/regenerative one (not Whitt 1991). Fixing initial storage is a deliberate, reported boundary condition (it makes within-window severity depend only on within-window forcing); sampling it is a flagged alternative. Two truncation caveats are stated: (a) a drought event straddling a window boundary is split, biasing whole-event metrics — bounded by the disjoint-window choice, the $L$-vs-design-drought check, and long $L_{\text{test}}$ where rankings are decided (McKee et al. 1993 on timescale dependence); (b) warm-up-year metrics are discarded.

**Dependence structure.** Disjoint windows are **not** independent under a shared generator: scenarios are i.i.d. *conditional on* $\theta$ but positively correlated marginally (hierarchical/cluster structure). Effective sample size is discounted for within-$\theta$ correlation, and standard errors on ensemble-level quantities are clustered/blocked by $\theta$ (§6).

**Storage.** Storing all daily traces ($\sim 10^6 \times 30$ nodes $\times \sim 1825$ days $\times$ float32 $\approx 440$ GB) is infeasible and exceeds the in-memory `Ensemble.to_hdf5` path. Instead: persist only the hazard image $\mathcal{H}$ ($N_{\mathcal{M}}\times m$, small), the generator parameters, and the **per-realization seeds**, and **regenerate selected realizations on demand** (Kirsch–Nowak is seed-deterministic *after the §7 determinism upgrade*). Metric computation over $\mathcal{M}$ is **streaming**: generate a block, compute $h(\cdot)$, discard daily traces, append to $\mathcal{H}$. $\mathcal{M}$ and $\mathcal{H}$ are design-independent and staged under `STAGED_ENSEMBLE_DIR`, not under per-run directories. A determinism-independent safety option is retained: store the daily HDF5 only for the ~hundreds of *selected* realizations actually simulated.

### 3.3 Hazard metrics and redundancy screening

Hazard axes are **reused and re-screened** from the MOEA-FIND drought-metric library. The candidate pool is `MOEA-FIND/src/metrics/{drought_metrics,short_block,extended}.py`: SSI-3 event intensity/duration/severity, low-flow indices (e.g. negated Q10), high-flow/flood-corner metrics, seasonal totals (DJF/OND), recession slopes. Per scenario, $h_{\text{full}}(s)$ is computed via the SSI characterization (fit once on $Q_{\text{obs}}$).

A redundancy screen on $\mathcal{H}$ selects the $m$ axes, reusing `MOEA-FIND/src/metrics/screening.py`:

1. `per_metric_spread` — drop degenerate axes ($\sigma<10^{-9}$, zero IQR, $\lvert\text{skew}\rvert>3$).
2. `cluster_metrics` — average-linkage clustering on $1-\lvert\rho_S\rvert$ (Spearman), cut at distance 0.30 so any pair with $\lvert\rho_S\rvert\ge 0.7$ is clustered as redundant (Olden & Poff 2003); representative = highest robust spread.
3. `enumerate_k_sets` / `relax_until_nonempty` — the Kennard et al. (2010) dual filter: the chosen $m$-set spans distinct redundancy clusters **and** distinct concept tags.
4. **Multivariate redundancy check** (new): a VIF/PCA screen on the retained set, because pairwise-uncorrelated metrics can be jointly collinear. Report retained $m$, per-axis VIF, and sensitivity of coverage to $m$.

The screen runs on the master ensemble's hazard image $\mathcal{H}$ (the population the designs subsample). Output: the hazard-axis set $h(\cdot)\in\mathbb{R}^m$, target $m\in[3,6]$ (large enough to separate drought/low-flow/high-flow concepts; small enough that uniform filling is achievable at feasible $N$). $\mathcal{H}$ is persisted alongside $\mathcal{M}$.

---

## 4. Per-design generation plans

Each design implements a common contract `build(spec, draw, seed) -> EnsembleSpec` (§7). Inputs: frozen $\mathcal{M}$ and (where needed) $\mathcal{H}$, draw index, seed. Outputs: staged HDF5 ensembles addressed by a provenance-bearing slug.

### 4.1 Historical record (`historic`) — concise

Already wired (`historic_single` preset, `is_ensemble=False`). The observed record as one continuous trace; $N=1$, full window; reference for prevailing applied practice (Giuliani et al. 2016). $K=1$; composition variance is zero by construction. Cannot be budget-matched (§5).

### 4.2 Fixed probabilistic, short (`fixed_probabilistic_short`) — concise

Sample-average-approximation baseline (Kleywegt et al. 2002; Herman et al. 2014). For draw $k$, draw $N$ realizations uniformly without replacement from $\mathcal{M}$; stage as ensemble $k$; fixed across the search. Replicated over $K$ draws × $S$ seeds (draw = ensemble-sampling variance; seed = MOEA variance). *Stand-up status:* wired via direct Kirsch-Nowak generation of the named `kn_{L}yr_n{N}` ensemble (no subsample-from-master step yet); provisional small test size $N{=}10$, $L{=}5$.

### 4.3 Fixed probabilistic, long records (`fixed_probabilistic_long`) — concise

Few multi-decadal records at **equal total simulated years** (Quinn et al. 2017). Generate (or draw from a long-$L'$ block) $N'=\lfloor NL/L'\rfloor$ realizations of length $L'=50$ yr so $N'L'\approx NL$; random draw, fixed across search; $K$ draws × $S$ seeds. Analyzed as a *length contrast* against §4.2, never pooled with the constant-$L$ composition comparison (Brodeur et al. 2020 on overfitting in policy search). *Stand-up status:* wired via direct Kirsch-Nowak generation of the named `kn_{L'}yr_n{N'}` ensemble; provisional small test size $N'{=}2$, $L'{=}25$ (matched to §4.2 at 50 scenario-years).

### 4.4 Resampled probabilistic (`resampled_probabilistic`) — concise

Per-evaluation re-randomization (Trindade et al. 2017); `resample_per_eval=True`. At every evaluation, draw $N$ realizations uniformly at random from the **pre-staged master pool** using an evaluation-indexed RNG stream, so the draw is reproducible but differs each evaluation. This is **index resampling into a pre-staged pool**, not a per-evaluation callback and not flow regeneration mid-evaluation. No fixed ensemble exists; replication is **seeds only** ($S$), and the per-evaluation draw-variance folds into fitness noise (noisy-fitness archive: Fieldsend & Everson 2015; Branke). In-search objective values are not commensurable across evaluations or designs — which is why re-evaluation (§4.7) is the comparison point. *Stand-up status:* wired. `resolve_search_spec` returns the master-pool spec (`kn_{L}yr_n{N_\text{master}}`) marked `resample_per_eval=True` with `resample_size=N`; `src/simulation.py::evaluate` redraws $N$ of the $N_\text{master}$ pool indices each evaluation (keyed by base-salt, MPI rank, eval counter). Provisional small test sizes: draw $N{=}10$ from an $N_\text{master}{=}50$ pool, $L{=}5$.

### 4.5 Input-stratified (`input_stratified`) — concise

LHS over generator **parameters** $\theta$, one realization per parameter set (Quinn et al. 2017, *Rival Framings*, WRR 10.1002/2017WR020524 — distinct from the 2017 EMS lake paper; Bartholomew & Kwakkel 2020 for the fixed-in-search LHS set). Draw $N$ forcing vectors by LHS over $\Theta$; generate one realization per $\theta_i$ (deterministic block seed); fixed across search; $K$ draws × $S$ seeds. Contrasting this with `hazard_filling` isolates the central claim: uniform coverage in *input* space need not produce uniform coverage in *hazard* space, because distinct $\theta$ often yield hydrologically redundant realizations (Quinn et al. 2020; Guo et al. 2018).

### 4.6 Hazard-filling (`hazard_filling`) — full depth

**Goal.** Select $E_d\subset\mathcal{M}$, $\lvert E_d\rvert=N$, whose hazard coordinates are approximately **uniform and well-separated** over the empirical hazard manifold $\mathcal{H}$.

**Method and naming.** The selector is a **stratified maximin space-filling design with cLHS-style marginal (quantile-stratum) conditioning** — *not* cLHS. Conditioned Latin hypercube sampling (Minasny & McBratney 2006) is defined by a correlation-matching term that *preserves* the parent dependence structure; we deliberately omit that term (it is antithetical to uniform hazard coverage), so the design is not cLHS and is not described as such. Citations are partitioned by component: **Minasny & McBratney (2006)** for the marginal quantile-stratification only; **Morris & Mitchell (1995)** and **Johnson et al. (1990)** for the maximin / $\phi_p$ separation term.

Let $X=\mathcal{H}$ be normalized per axis to $[0,1]$ by its empirical CDF (so each axis is uniform; "uniform in hazard space" is well-defined under skewed marginals). Minimize, by simulated annealing over subsets $T\subset X$, $\lvert T\rvert=N$:

$$
\Phi(T) = w_1 \underbrace{\sum_{a=1}^{m}\sum_{q=1}^{N}\big\lvert \eta_{a,q}(T)-1\big\rvert}_{\text{marginal quantile-stratum occupancy}} \;+\; \lambda \underbrace{\Big(\sum_{i<j}\lVert h_i-h_j\rVert^{-p}\Big)^{1/p}}_{\text{maximin } \phi_p \text{ separation}},
$$

with $\eta_{a,q}$ the count of $T$-members in stratum $q$ of axis $a$ (target 1 per stratum), and **no parent-correlation term by construction**. Simulated annealing swaps a $T$-member for a non-member each step, accepting by the Metropolis rule under a geometric cooling schedule.

**Distortion, its motivation, and its corrective.** Hazard-filling deliberately distorts scenario probabilities toward uniform hazard coverage: rare-but-severe corners are over-represented relative to their frequency in $\mathcal{M}$. The motivation for uniform hazard *coverage* of the response surface comes from the bottom-up / decision-scaling tradition (Brown et al. 2012; Culley et al. 2016; Herman et al. 2016) — cited as a *sampling/coverage* rationale, **not** as a precedent for biasing the optimization objective (decision scaling re-imposes probabilities post hoc and does not distort the search measure). The contrast with **representative-in-probability** reduction (Dupačová et al. 2003; support points, Mak & Joseph 2018) is the role of §4.6.1. *Hilbers et al. (2019)* (importance subsampling) is probability-faithful and belongs to the faithful arm, not here.

**Selection vs evaluation bias.** The distortion biases the **search trajectory**, not only the reported number: any objective formulated as an expectation/quantile *over scenarios* is computed under the distorted measure and biases *which policies the optimizer selects*. Re-evaluation (§4.7) gives an **unbiased comparison of the solutions each design produced, conditional on a fixed $E_{\text{test}}$**; it corrects *evaluation* bias but **not** *selection* bias, and it does **not** make hazard-filling unbiased. Two principled options (deferred to the objective-formulation decision):

1. **Importance-sampling reweighting** — carry per-scenario weights $\mathrm{d}P_{\mathcal{M}}/\mathrm{d}P_{\text{uniform}}$ so expectation-type objectives are unbiased in search. The estimator variance **explodes** as coverage → uniform (extreme weights in the low-probability corners the design targets), so this route is likely undefensible for aggressive filling.
2. **Heuristic, no-expectation interpretation** — optimize a **coverage-weighted** aggregate directly; the objectives must then be **reported as coverage-weighted quantities, never as estimates of $\mathbb{E}[\cdot]$ under any measure**, or the §6 overfitting-gap regression tests a confounded quantity. Expected to be the only defensible path.

**Coverage in context.** Report achieved L2-star discrepancy against the **expected discrepancy of a random design at the same $(N,m)$**, so the $m$-vs-$N$ tension (uniform filling is sparse at high $m$, fixed $N$) is visible rather than asserted.

**Implementation.** Net-new code (`subsample.py`) seeded by MOEA-FIND scaffolding: `coverage_metrics` and `generate_lhs_samples` are reused; the selection algorithm is *not* a port (`pick_space_filling_subset` is LHS + greedy nearest-neighbor, retained only as a diagnostic baseline). Replicated over $K$ independent annealer runs × $S$ seeds — the annealer is itself stochastic, and the $K$ replicates expose its construction variance (without them, hazard-filling's lower sampling variability would falsely read as stability).

*Stand-up status:* initial draft wired. The selector, the hazard metrics, and the driver live in the sibling `scengen` package (`subsample.py`, `hazard_metrics.py`, `hazard_filling.py`) — MOEA-FIND's coverage/LHS primitives and its "primary" SSI metric set (`mean_severity`, `mean_magnitude`, `time_in_drought_fraction`, SSI-3 on the aggregate NYC inflow) are **copied**, not imported, so there is no MOEA-FIND dependency. The original (subsampled-from) ensemble and the hazard axes are both parameterized to change later. Initial draft: subsample $N{=}10$ from a $200$-realization **stationary** Kirsch-Nowak pool of 5-yr records (same generator as `fixed_probabilistic_short`). The subset is computed offline (`scripts/main/subsample_hazard_filling.py`) and written as a manifest under the staged pool dir; `resolve_search_spec` loads it and overrides the pool spec's realization indices. $K$ annealer-draw replication and the CMIP6-forced master are deferred.

#### 4.6.1 Support-point supplement (`support_points`) — full depth

Recommended by reviewers to isolate **whether benefits stem from uniform coverage specifically, or from designed subsampling generally.** Support points (Mak & Joseph 2018) minimize the energy distance between the subsample's empirical distribution and the parent $\mathcal{H}$ — a *representative-in-probability, designed* selection (faithful × designed cell):

$$
\mathcal{E}(T,\mathcal{H}) = \tfrac{2}{N N_{\mathcal{M}}}\!\sum_{i\in T}\sum_{j\in\mathcal{H}}\!\lVert h_i-h_j\rVert \;-\; \tfrac{1}{N^2}\!\sum_{i,i'\in T}\!\lVert h_i-h_{i'}\rVert \;-\; \tfrac{1}{N_{\mathcal{M}}^2}\!\sum_{j,j'}\!\lVert h_j-h_{j'}\rVert,
$$

minimized over $T\subset\mathcal{H}$ via the difference-of-convex/block-update algorithm restricted to the candidate set. Net-new selector in `subsample.py`; replicated $K$ × $S$. If hazard-filling beats both `fixed_probabilistic_short` and `support_points` on re-eval, the benefit is attributable to uniform coverage; if only `fixed_probabilistic_short`, to designed subsampling generally.

### 4.7 Re-evaluation / test ensemble (`E_test`) — full depth

**Role.** The single, common, held-out basis of cross-design comparison. Never used during search; every design's final Pareto-approximate set is re-simulated on $E_{\text{test}}$ and nondominated sets are recomputed from re-evaluated values for all designs alike.

**Design principles.**
- **Largest and most uncertainty-encompassing.** $N_{\text{test}}\gg\max_d N_d$ (target $\ge 1000$), spanning the full §3.1 hybrid forcing space, with realizations allocated across $\theta$ rather than concentrated.
- **NOT scenario-neutral.** Following Quinn et al. (2020, *"Can exploratory modeling … be scenario neutral?"*) — whose thesis is that no design is truly neutral, because uniformity/independence assumptions belie neutrality — $E_{\text{test}}$ is honestly **one deliberately-broad, deeply-uncertain, design-conditional reference**; rankings are conditional on it. This conditioning is a limitation to be bounded (Quinn et al. 2020; McPhail et al. 2018, 2020), so we **commit to ≥2 alternative $E_{\text{test}}$ constructions** (e.g. a wider-envelope variant and a historically-anchored-tail variant) and report ranking stability; where budget allows, $E_{\text{test}}$ draw is treated as an additional random factor (two correlated draws agreeing does not establish robustness).
- **Long records where rankings are decided.** $L_{\text{test}}\ge 30$ yr, to contain whole multi-year droughts (avoiding §3.2 truncation bias).
- **Independent seed.** Generation seed independent of every search ensemble (selection-bias guard, Bonham et al. 2024; `config.py` warns on preset collisions).

**Reference-set construction and precision.** Recomputing one nondominated set from re-eval values pooled across designs induces winner's-curse / self-reference bias (a design contributes points to the frontier it is scored against; per §5 the design handed more NFE searches hardest). Therefore report metrics against **both** the pooled reference set **and** a **design-leave-one-out** reference (the gap is the self-reference inflation), and report the **test-set Monte-Carlo standard error** clustered by $\theta$ ($N_{\text{test}}\ge 1000$ is not itself a precision statement; precision is governed by the number of $\theta$ draws — §6c).

**Construction.** Generate $N_{\text{test}}$ realizations of length $L_{\text{test}}$ across the §3.1 forcing space, stage via `stage_pywrdrb_ensemble_inputs`, register as a `PRESETS` entry (e.g. `reeval_hybrid_n{N_test}`), select by `NYCOPT_REEVAL_ENSEMBLE_PRESET`. It is an `EnsembleSpec`, **not** a `ScenarioDesign` (it never enters search).

---

## 5. Budget and control

**Two control axes, run on a common paired/blocked set of ensemble draws and optimizer seeds** so the arm contrast is within-draw (not confounded by draw/seed variance).

**Primary (deployment) — equal scenario-years.** Define $B_d = \text{NFE}_d \times N_d \times L_d$ and hold $B_d=B$ constant across budget-matched designs. Scenario-years are the dominant cost driver (one reservoir-year of Pywr-DRB integration each), so equal $B$ makes raw compute commensurable and, under identical parallel config, approximates equal wall-clock. Designs with cheaper evaluations complete more NFE within $B$ — a real deployment consequence, not a confound to remove. Solving: $\text{NFE}_d = B/(N_d L_d)$, setting `MOEAConfig.max_evaluations`; `budget_scenario_years` carries $B$. (Justified on its own footing; **Zatarain Salazar et al. 2017** is cited only for the ensemble-size-vs-fidelity-vs-cost tradeoff it actually establishes — not as the source of equal-scenario-years or of 1-D stratification.)

**Companion (mechanism) — equal NFE (required).** Hold $\text{NFE}_d=\text{NFE}$ constant; let scenario-years float. This is the **only** condition under which the composition effect is identifiable: *does ensemble composition change policy quality at fixed search effort?* Equal-$B$ alone confounds composition with search effort (cheaper designs are defined to get more NFE), so a hazard-filling win could not otherwise be attributed to composition versus extra NFE.

**Identifiability concession.** There are **three** candidate held-fixed quantities — (a) scenario-years $B$; (b) NFE; (c) fitness-estimator variance / per-evaluation scenario information — and **no single arm isolates composition from estimator precision**. Equal-NFE isolates only *composition at fixed search effort*; quantity (c) is an acknowledged, unremoved confound (small-$N$ designs give low-variance/biased fitness; `resampled_probabilistic` gives unbiased/noisy fitness). **Sweep ≥2 budget levels** in each arm to expose the budget × design interaction.

**Worked example.** With $B=2.0\times10^{8}$ scenario-years and $N L = 1000$ (e.g. $N=200,L=5$ or $N'=20,L'=50$): $\text{NFE}=2.0\times10^{5}$. The many-short-vs-few-long contrast is then purely within-evaluation composition. A design choosing $NL=500$ instead receives $\text{NFE}=4\times10^{5}$ — the efficiency dividend made explicit.

**Replication (mixed-effects variance components).** Model $\text{outcome} \sim \text{design (fixed)} + \text{draw(design) (random)} + \text{seed(draw) (random)}$. The unit of analysis for between-design tests is the **ensemble draw**; seeds within a draw are pseudoreplicates, so effective $n\approx K$, not $KS$. Target **$K=10$, $S=2$–$3$** (more draws, fewer seeds). `historic` (structural-zero draw variance) and `resampled_probabilistic` (draw variance folded into within-generation fitness noise) have $K=1$ and enter contrasts on the within-draw scale with between-draw uncertainty declared zero/omitted-and-flagged — every contrast has a defined denominator, with the between-draw component supplied only by $K\ge 2$ designs. State a pre-registered **target effect size** (in test-set hypervolume and additive-$\epsilon$ units, $\theta$-clustered SE). Cite a components-of-variance/mixed-models reference for this (not Kaut & Wallace, retained for stability in §6).

| Design | $K_d$ | $S$ | Replication rationale |
|---|---|---|---|
| `historic` | 1 | $S$ | No ensemble; structural-zero draw variance. |
| `fixed_probabilistic_short` / `_long` | $K$ | $S$ | Random-draw + MOEA variance. |
| `resampled_probabilistic` | 1 | $S$ | Per-generation draw folded into fitness noise. |
| `input_stratified` | $K$ | $S$ | LHS-design + MOEA variance. |
| `hazard_filling` | $K$ | $S$ | SA-selector + MOEA variance (essential). |
| `support_points` | $K$ | $S$ | Energy-optimizer + MOEA variance. |

---

## 6. Ensemble-quality diagnostics

Tiered into build-QC gates, falsifiable outcome hypotheses, and a few-clusters inference rule. Implemented in `diagnostics.py`, reusing `coverage_metrics`.

### 6a. Build-QC gates (verify the ensemble was built as specified; not evidence of advantage)
- **L2-star discrepancy / MST / effective sample size** on normalized hazard coordinates of $E_d$. Discrepancy *is* the hazard-filling objective, so it is **demoted to a build-verification gate** (confirms the annealer converged), not a result. ESS is $\theta$-discounted.
- **Scenario redundancy** — run the §3.3 clustering on $E_d$; report distinct clusters represented (+ VIF/PCA). Pass: spans $\ge m$ clusters.
- **Coverage-in-context** — achieved discrepancy vs the expected discrepancy of a random design at the same $(N,m)$.
- **Statistical fidelity to $Q_{\text{obs}}$/$\mathcal{M}$** (monthly mean/SD, lag-1 and cross-site correlation, flow-duration curve, drought severity–duration–frequency) — a **within-faithful-arm check only**; `hazard_filling` and `support_points` are checked only that each selected scenario is a valid generator output, and `hazard_filling` is **never ranked on fidelity** (it distorts marginals by design).

### 6b. Outcome hypotheses (falsifiable; may be null)
- **Central hypothesis:** higher coverage of the test hazard space by $E_d$ predicts a smaller out-of-sample overfitting gap. Operationalized as a **pre-registered regression** of the gap (search-ensemble metric − re-evaluated metric, in hypervolume and additive-$\epsilon$ units) on coverage: report slope, CI, $R^2$; linear in design, no post-hoc interactions; Holm correction across the ≥2 $E_{\text{test}}$ sets and pooled-vs-leave-one-out reference variants. **Confound guard:** for `hazard_filling` on route (b), the in-sample term is coverage-weighted, the out-of-sample term is probability-faithful, and the gap is never interpreted as expectation-vs-expectation. A null is a reportable result.
- **In-/out-of-sample stability** (Kaut & Wallace 2007) per design, $\theta$-clustered + test-set MC SE.
- **Ranking sensitivity** across the ≥2 $E_{\text{test}}$ constructions and pooled-vs-LOO reference sets.
- **Budget × design interaction** across the swept budget levels.

### 6c. Few-clusters inference
Pre-state the number of distinct $\theta$ draws in $\mathcal{M}$ and $E_{\text{test}}$. $\theta$-clustered SEs are unreliable with few clusters: if **fewer than ~40 $\theta$ clusters**, use **wild-cluster bootstrap** or **hierarchical bootstrap-by-$\theta$** rather than asymptotic cluster-robust SEs. Test-set precision is governed by the number of $\theta$ draws, not $N_{\text{test}}$.

---

## 7. Code and file organization

A clean, by-design layout split across repositories (the optimization-independent generation work lives in a sibling repo behind a shared contract; see the implementation plan). The contract: the generation repo emits staged pywrdrb-format HDF5 + a provenance manifest (`_meta.json`: design, draw, forcing-hash, seeds, hazard-axis definitions, screen results); NYCOptimization consumes by slug via `EnsembleSpec` / `register_ensemble_path`.

```
NYCOptimization_scenario_generation/   # optimization-independent
  forcing_space.py      # §3.1  CMIP6 _frac_ load (calendar->WY), harmonic-param hypercube -> theta; eqs 10-11
  master_ensemble.py    # §3.2  fit baseline + full Kirsch; global-index generation; streaming H; manifest
  hazard_metrics.py     # §3.3  import MOEA-FIND metrics + screening; VIF/PCA; persist H
  subsample.py          # §4.6  net-new SA stratified-maximin + energy-distance selectors
  diagnostics.py        # §6a   ensemble-quality (coverage/redundancy/fidelity)

NYCOptimization/                        # optimization-coupled
  src/ensemble/designs/*.py   # build(spec, draw, seed) -> EnsembleSpec; registry keyed by design name
  src/scenario_designs.py     # resolve_search_spec -> selection-keyed dispatch (closes NotImplementedError)
  src/ensembles.py            # slug grammar enriched (design+draw+forcing-hash); E_test as PRESETS
  src/simulation.py           # per-generation re-index hook for resample_per_eval
  src/ensemble_prep.py        # stage_pywrdrb_ensemble_inputs (flood/STARFIT/predicted)
  diagnostics (§6b)           # outcome hypotheses (need MOEA re-eval results)
```

**Determinism contract** (the regenerate-on-demand precondition). Realization $k$ is fully determined by a single child RNG stream keyed to the **global** index $k$ (MPI- and batch-invariant), driving the Kirsch monthly step, the Nowak daily step, and the KDE downstream fill identically regardless of how the index range is partitioned.

*Upstream half — done in SynHydro (commit `7659704`, "Attempt better seed control across generation schemes"; 12 determinism tests pass).* SynHydro now provides `synhydro/core/seeding.py`: `as_seed_sequence(seed)`, `spawn_realization_seed(master, k)` (reconstructs `SeedSequence(master).spawn(N)[k]` directly from the spawn key, so one realization regenerates without materializing the other $N-1$), and `realization_rng(master, k, stage)`. Each child seed is split into two labeled sub-streams `SUBSTREAM_LABELS = ("generation", "disaggregation")`. `KirschGenerator.generate(..., realization_indices=None)` keys output by global index (pass `[k]` to regenerate one realization); `NowakDisaggregator.disaggregate(..., seed=...)` infers each global index from the ensemble's **integer realization keys**; `KirschNowakPipeline.generate(..., realization_indices=None, seed=...)` forwards the master seed to both stages.

*Downstream half — remaining work in `NYCOptimization/src/ensemble_generation.py` (or the `master_ensemble.py` port).* The Kirsch+Nowak RNG is now deterministic via the SynHydro API, but two NYCOptimization-side steps still break single-realization reproducibility and must be fixed in the port:
- **Stable hash** — replace process-salted `hash(kde_name)` with `zlib.crc32(kde_name.encode())`.
- **Per-realization KDE resample** — replace the single batched `kde.resample(n_days*n_realizations, …)` + reshape (which couples realizations) with a per-realization `kde.resample(n_days, …)` seeded from realization $k$'s child stream, namespaced per downstream pair by `crc32(pair_name)`. (The KDE fill is the only randomness not covered by the SynHydro sub-streams.)
- **Preserve realization keys** — do not re-key/renumber the monthly ensemble between `generate` and `disaggregate`; the Nowak sub-stream is keyed to the integer realization id (SynHydro caveat).
- **Regeneration:** `regenerate_realization(master_seed, k)` calls `KirschGenerator.generate(realization_indices=[k], seed=master_seed)` then `disaggregate(seed=master_seed)` on the key-$k$ ensemble, then the per-$k$ KDE fill.
- **Regression-test gate** — extend the SynHydro-style test through the NYCOptimization KDE fill: generate the same global realization under two partitions and assert array-equality post float32/HDF5 round-trip (gage flows, catchment inflows, regressed downstream gages) before relying on regeneration.

**Integration.** `outputs/{scenario}/{moea_slug}/{artifact}/` with `{scenario} = ScenarioDesign.name`; master ensemble + $\mathcal{H}$ under `STAGED_ENSEMBLE_DIR` (`outputs/synthetic_ensembles/`). `resolve_search_spec(draw)` dispatches into the design registry when `ensemble_preset is None`. Maps to `workflow/02` (forcing + master + $\mathcal{H}$), `workflow/03` (per-design subsample + stage), `workflow/04` (`stage_pywrdrb_ensemble_inputs`).

---

## 8. Open parameters and flagged uncertainties

| Parameter | Symbol | Recommended default | Rationale / citation |
|---|---|---|---|
| Master-ensemble cardinality | $N_{\mathcal{M}}$ | $\sim 10^{6}$ | Dense parent (Minasny & McBratney 2006; Mak & Joseph 2018). |
| Forcing-profile count | $N_\Theta$ | 500–2000 | Populate the CMIP6-based harmonic-parameter hypercube by LHS. |
| CMIP6 envelope band | $[\underline\Delta,\overline\Delta]$ | min–max of anchors | Conservative; percentile band flagged. |
| Box width (param range) | bound_pct | (5, 95) — empirical 90% range | Robust to outlier GCM runs; fixed phases prevent over-dispersion. Optional `margin` widens/tightens. |
| Forcing parameterization | $\{m,r_1,r_2\}$ + fixed canonical phases | fixed-phase harmonic hypercube (DMDU) | Quinn-style: sample amplitudes, fix phases at the canonical CMIP6 shape (best shape fidelity, fewer params). See `forcing_parameterization.md`. |
| Climate-adjustment SD | $c_j$ | $a_j$ (CV-preserving) + indep. variance axis $v_j$ | CV-preserving baseline; $c_j=a_j v_j$ for the variance axis (`derive_variance_envelope`). |
| Scenario length (search) | $L$ | 5–10 yr | Exceed longest within-window drought; vs 1960s DRB record. |
| Window construction | — | disjoint | Scenario independence (conditional on $\theta$). |
| Initial storage | — | 0.80 + 365-day warmup | Terminating-simulation BC (Law 2015); sampling flagged. |
| Hazard-axis count | $m$ | 3–6 | Concept separation vs filling feasibility; VIF/PCA screen. |
| Redundancy cut | $\lvert\rho_S\rvert$ | $\ge 0.7$ (cut 0.30) | Olden & Poff 2003; Kennard 2010. |
| Evaluation size | $N$ | 100–300 | Coverage need vs per-eval cost; gated by objective rule. |
| Long-record length | $L'$ | 50 yr | Quinn et al. 2017. |
| Test-ensemble size / length | $N_{\text{test}},L_{\text{test}}$ | $\ge 1000$, $\ge 30$ yr | Recomputed reference; whole droughts. |
| Alt test ensembles | — | $\ge 2$ | Design-conditional ranking (Quinn 2020). |
| Budget / NFE levels | $B$ | $\ge 2$ each arm | Budget × design interaction. |
| Draws / seeds | $K,S$ | 10 / 2–3 | Draw is the unit of analysis. |
| Maximin weight | $\lambda$ | tuned | $\phi_p$ separation (Morris & Mitchell 1995). |
| Distortion handling | — | route (b), coverage-weighted | IS variance explodes (deferred to objectives). |

**Flagged methodological uncertainties.**
- **Identifiability / estimator precision.** No arm isolates composition from fitness-estimator precision; the mechanism claim is "composition at fixed search effort," and quantity (c) remains a confound.
- **$E_{\text{test}}$ conditioning.** Rankings are conditional; the ≥2-construction sensitivity (ideally $E_{\text{test}}$ as a random factor) bounds but does not eliminate this.
- **Distortion legitimacy.** Defensible only with re-evaluation as the comparison point and objectives reported as coverage-weighted (route b); the IS route is variance-pathological.
- **$m$-vs-$N$ tension.** Uniform filling at $m\ge 4$, $N\sim 200$ is sparse; §6a coverage-in-context must demonstrate, not assert, superiority.
- **Partial-event truncation.** Bounded but not eliminated by disjoint windows + $L$ checks; flagged where any hazard axis is event-based.
- **Generator stationarity under perturbation.** The Kirsch correlation/NST structure is fit on history and reused under shifted `mean_period`/`std_period` (Kirsch et al. 2013).
- **Regeneration determinism.** The upstream SynHydro half is done (commit `7659704`, global-index RNG streams, 12 tests pass); the NYCOptimization-side KDE-fill + stable-hash fixes (§7) and an end-to-end regression-test gate remain before regenerate-on-demand is trusted.
- **CMIP6 calendar alignment.** Month-index convention must be confirmed before applying factors.

**References to add to the Zotero collection (ISYGLK35):** Minasny & McBratney 2006; Mak & Joseph 2018; Kennard et al. 2010; Law 2015; Fieldsend & Everson 2015 / Branke; Herman et al. 2016; Brown et al. 2012; Culley et al. 2016; Morris & Mitchell 1995; Johnson et al. 1990; Quinn et al. 2020; McPhail et al. 2018, 2020; Bonham et al. 2024; Brodeur et al. 2020; McKee et al. 1993; Quinn et al. 2017 (*Rival Framings*, WRR 10.1002/2017WR020524 — confirm distinct from the 2017 EMS lake paper); a components-of-variance/mixed-models reference (for §5; not Kaut & Wallace, retained for stability only); Olden & Poff 2003; Zatarain Salazar et al. 2017 (size-vs-fidelity-vs-cost only).
